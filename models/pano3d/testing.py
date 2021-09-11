import os
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon, Point
from copy import deepcopy
import trimesh

from configs.data_config import igibson_colorbox
from external.ldif.inference.metrics import mesh_chamfer_via_points
from models.testing import BaseTester
from models.eval_metrics import bdb3d_iou, bdb2d_iou, rot_err, classification_metric, AverageMeter, \
    AveragePrecisionMeter, BinaryClassificationMeter, ClassMeanMeter
from utils.layout_utils import manhattan_world_layout_info
from utils.mesh_utils import save_mesh
from utils.relation_utils import relation_from_bins, test_bdb3ds, RelationOptimization
from .dataloader import collate_fn
from .training import Trainer
from utils.transform_utils import bins2bdb3d, IGTransform, bdb3d_corners, points2bdb2d, expand_bdb3d
from utils.igibson_utils import IGScene
from utils.visualize_utils import IGVisualizer
from utils.image_utils import save_image
from external.HorizonNet.eval_general import test_general
from utils.basic_utils import dict_of_array_to_list_of_dict, recursively_to


class Tester(BaseTester, Trainer):

    def __init__(self, cfg, net, device=None):
        super(Tester, self).__init__(cfg, net, device)

    def get_metric_values(self, est_data, data):
        ''' Performs a evaluation step.
        '''
        metrics = {}
        _, gt_data, _, gt_scenes = data
        do_evaluation = not any(s.empty() for s in gt_scenes)

        # LEN
        if do_evaluation and 'layout' in gt_data:
            est_scenes = IGScene.from_batch(est_data, gt_data)

            losses = dict([
                (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse': [], 'delta_1': []})
                for n_corner in ['4', '6', '8', '10+', 'odd', 'overall']
            ])
            for dt_cor_id, gt_cor_id, est_scene in zip(
                    est_data['layout']['manhattan_pix'],
                    gt_data['layout']['manhattan_pix'],
                    est_scenes
            ):
                height, width = est_scene['camera']['height'], est_scene['camera']['width']
                test_general(dt_cor_id, gt_cor_id, height, width, losses)
            for n_corners, ms in losses.items():
                for metric_name, metric in ms.items():
                    metrics[f"layout_{n_corners}_{metric_name}"] = AverageMeter(metric)

        # relations to do relation classification evaluation
        rel_type_scenes = {}

        # BEN
        if do_evaluation and 'objs' in est_data and est_data['objs'] and 'bdb3d' in est_data['objs']:
            est_scenes = IGScene.from_batch(est_data, gt_data)
            metrics.update(self.eval_BEN(est_scenes, gt_scenes))

            if 'layout' in est_data:
                # collision metrics and relation fidelity
                metric_col = defaultdict(AverageMeter)
                rel_scenes = []
                for est_scene, gt_scene in zip(est_scenes, gt_scenes):
                    # generate relation between estimated objects and layout
                    # 0.1m of toleration distance when measuring collision
                    relation_optimization = RelationOptimization(expand_dis=-0.1)
                    est_rel_data = est_scene.data.copy()
                    est_rel_data['objs'] = deepcopy(est_rel_data['objs'])
                    est_rel_scene = IGScene(est_rel_data)
                    relation_optimization.generate_relation(est_rel_scene)
                    relation = est_rel_scene['relation']

                    # collision metrics
                    metric_col['collision_pairs'].append(relation['obj_obj_tch'].sum() / 2)
                    metric_col['collision_objs'].append(relation['obj_obj_tch'].any(axis=0).sum())
                    metric_col['collision_walls'].append(relation['obj_wall_tch'].any(axis=-1).sum())
                    metric_col['collision_ceil'].append(sum(o['ceil_tch'] for o in est_rel_scene['objs']))
                    metric_col['collision_floor'].append(sum(o['floor_tch'] for o in est_rel_scene['objs']))

                    # save reconstructed relations for relation fidelity evaluation
                    relation_optimization = RelationOptimization(expand_dis=self.cfg.config['data'].get('expand_dis', 0.1))
                    relation_optimization.generate_relation(est_rel_scene)
                    est_rel_scene['relation'] = relation_from_bins(relation, None)
                    rel_scenes.append(est_rel_scene)

                rel_type_scenes['fidelity'] = rel_scenes
                metrics.update(metric_col)

        # save classified relations for relation classification evaluation
        if 'objs' in est_data and est_data['objs'] and 'in_room' in est_data['objs']:
            # get label from relation output
            relation_label = relation_from_bins(
                est_data, self.cfg.config['model']['scene_gcn'].get('score_thres'))
            est_data['objs'].update(relation_label['objs'])
            if 'relation' in relation_label:
                est_data['relation'] = relation_label['relation']
            rel_type_scenes['classify'] = IGScene.from_batch(est_data, gt_data)

        # Relation evaluation
        if do_evaluation and 'relation' in gt_data:
            if rel_type_scenes:
                gt_rels = relation_from_bins(gt_data, None)['relation']
                gt_rels = recursively_to(gt_rels, dtype='numpy')

            for rel_type, rel_scenes in rel_type_scenes.items():
                metric_rels = defaultdict(BinaryClassificationMeter)
                metric_rot = defaultdict(AverageMeter)

                for rel_scenes, gt_rel, gt_scene in zip(rel_scenes, gt_rels, gt_scenes):
                    id_gt = np.array([obj['gt'] for obj in rel_scenes['objs']])
                    id_match = id_gt >= 0
                    id_gt = id_gt[id_match]

                    if len(id_gt) <= 0:
                        continue

                    for k in ('floor_tch', 'ceil_tch', 'in_room'):
                        est_v = np.array([o[k] for o in rel_scenes['objs']])[id_match]
                        gt_v = np.array([o[k] for o in gt_scene['objs']])[id_gt]
                        metric_rels[f"{k}_{rel_type}"].add({'est': est_v, 'gt': gt_v})

                    for k in ('obj_obj_rot', 'obj_obj_dis', 'obj_obj_tch', 'obj_wall_rot', 'obj_wall_tch'):
                        # match estimated objects with ground truth
                        if 'relation' not in rel_scenes.data:
                            continue
                        est_v = rel_scenes['relation'][k]
                        est_v = est_v[id_match]
                        gt_v = gt_rel[k][id_gt]
                        if k.startswith('obj_obj'):
                            if len(id_gt) <= 1:
                                continue
                            mask = ~np.eye(len(id_gt), dtype=np.bool)
                            est_v = est_v[:, id_match][mask]
                            gt_v = gt_v[:, id_gt][mask]

                        est_v = est_v.reshape(-1)
                        gt_v = gt_v.reshape(-1)
                        if est_v.dtype == np.bool:
                            metric_rels[f"{k}_{rel_type}"].add({'est': est_v, 'gt': gt_v})
                        elif k.endswith('rot'):
                            metric_rot[f"{k}_{rel_type}_err"].extend(np.rad2deg(rot_err(est_v, gt_v)).tolist())

                metrics.update(metric_rels)
                metrics.update(metric_rot)

            # Scene Reconstruction
            if self.cfg.config['full']:
                if 'objs' in est_data and 'mesh_extractor' in est_data['objs']:
                    est_data['objs']['mesh'] = est_data['objs']['mesh_extractor'].extract_mesh()
                    est_scenes = IGScene.from_batch(est_data)
                    chamfers = AverageMeter()
                    for est_scene, gt_scene in zip(est_scenes, gt_scenes):
                        est_mesh = est_scene.merge_mesh()
                        gt_mesh = gt_scene.merge_mesh()
                        chamfers.append(mesh_chamfer_via_points(est_mesh, gt_mesh))

                        # from utils.mesh_utils import save_mesh
                        # save_mesh(est_mesh, 'out/tmp/est.obj')
                        # save_mesh(gt_mesh, 'out/tmp/gt.obj')
                    metrics['chamfer_scene'] = chamfers

        if 'save_as_dataset' in self.cfg.config['log']:
            est_scenes = IGScene.from_batch(est_data, gt_data, full=True)
            self.save_as_dataset(est_scenes, gt_scenes)

        # est_scenes['objs'] = gt_scenes['objs']
        return metrics, est_data

    def eval_BEN(self, est_scenes, gt_scenes):
        metrics = {}

        # evaluate IoU and bdb3d parameters
        metric_iou = defaultdict(AverageMeter)
        metric_bdb3d = defaultdict(lambda: ClassMeanMeter(AverageMeter))
        for est_scene, gt_scene in zip(est_scenes, gt_scenes):
            for est_obj in est_scene['objs']:
                # match objects prediction and ground truth
                if 'gt' not in est_obj or est_obj['gt'] < 0:
                    continue
                est_bdb3d = est_obj['bdb3d']
                gt_obj = gt_scene['objs'][est_obj['gt']]
                gt_bdb3d = gt_obj['bdb3d']

                # 3D IoU
                est_corners, gt_corners = bdb3d_corners(est_bdb3d), bdb3d_corners(gt_bdb3d)
                metric_iou['bdb3d_3DIoU'].append(bdb3d_iou(est_corners, gt_corners))

                # 2D IoU: project bdb3d to camera plane
                recentered_trans = IGTransform.level_look_at(est_scene.data, gt_bdb3d['centroid'])
                corners2d = recentered_trans.world2campix(est_corners)
                # full_convex = MultiPoint(corners2d).convex_hull
                # pyplot.plot(*full_convex.exterior.xy)
                # pyplot.axis('equal')
                # pyplot.show()
                metric_iou['bdb3d_2DIoU'].append(bdb2d_iou(points2bdb2d(corners2d), gt_obj['bdb2d_from_bdb3d']))

                # bdb3d parameter err
                classname = gt_obj['classname']
                metric_bdb3d['bdb3d_translation'][classname].append(
                    np.mean(np.abs(est_obj['bdb3d']['centroid'] - gt_obj['bdb3d']['centroid'])))
                metric_bdb3d['bdb3d_rotation'][classname].append(
                    np.rad2deg(rot_err(est_obj['bdb3d']['ori'], gt_obj['bdb3d']['ori'])))
                metric_bdb3d['bdb3d_scale'][classname].append(
                    np.mean(np.abs(est_obj['bdb3d']['size'] / gt_obj['bdb3d']['size'] - 1.)))

        metrics.update(metric_iou)
        metrics.update(metric_bdb3d)

        # mAP and bdb3d parameters
        metric_aps = ClassMeanMeter(AveragePrecisionMeter)
        for est_scene, gt_scene in zip(est_scenes, gt_scenes):
            if gt_scene['objs']:
                if 'bdb3d' not in gt_scene['objs'][0]:
                    continue
            else:
                continue

            # sort detections by score
            est_objs = sorted(est_scene['objs'], key=lambda obj: -obj['score'])
            for est_obj in est_objs:
                # match gt with same label and largest iou
                iou_max = -np.inf
                for gt_obj in gt_scene['objs']:
                    if 'matched' in gt_obj or est_obj['label'] != gt_obj['label']:
                        continue
                    est_corners, gt_corners = bdb3d_corners(est_obj['bdb3d']), bdb3d_corners(gt_obj['bdb3d'])
                    iou = bdb3d_iou(est_corners, gt_corners)
                    if iou > iou_max:
                        iou_max = iou
                        matched_gt_obj = gt_obj

                classname = est_obj['classname']
                if iou_max > 0.15:
                    # label gt as 'matched'
                    matched_gt_obj['matched'] = True

                    # label as TP
                    metric_aps[classname]['TP'].append(1.)
                    metric_aps[classname]['FP'].append(0.)
                else:
                    # label as FP
                    metric_aps[classname]['TP'].append(0.)
                    metric_aps[classname]['FP'].append(1.)

                metric_aps[classname]['score'].append(est_obj['score'])

            for gt_obj in gt_scene['objs']:
                if 'matched' in gt_obj:
                    gt_obj.pop('matched')

        if metric_aps:
            metrics['3D_detection_AP'] = metric_aps

        return metrics

    def save_as_dataset(self, est_scenes, gt_scenes):
        # save intermedia results as dataset
        save_path = self.cfg.config['log']['save_as_dataset']
        for est_scene, gt_scene in zip(est_scenes, gt_scenes):
            camera_folder = os.path.join(save_path, est_scene['scene'], est_scene['name'])
            est_scene.to_pickle(os.path.join(camera_folder, 'data.pkl'))
            gt_scene.to_pickle(os.path.join(camera_folder, 'gt.pkl'))
            est_scene.image_io.link(camera_folder)

    def test_step(self, data):
        '''
        test by epoch
        '''
        '''load input and ground-truth data'''
        data = self.to_device(data)

        '''network forwarding'''
        est_data = self.net(data)

        loss, est_scenes = self.get_metric_values(est_data, data)
        return loss, est_scenes

    def visualize_step(self, est_data):
        ''' Performs a visualization step.
        '''
        if 'objs' in est_data and 'mesh' not in est_data['objs'] \
                and 'mesh_extractor' in est_data['objs'] and self.cfg.config['full']:
            est_data['objs']['mesh'] = est_data['objs']['mesh_extractor'].extract_mesh()
        est_scenes = IGScene.from_batch(est_data)

        for est_scene in est_scenes:
            scene_folder = os.path.join(self.cfg.config['log']['vis_path'], est_scene['scene'], est_scene['name'])
            gpu_id = int(self.cfg.config['device']['gpu_ids'].split(',')[0])
            visualizer = IGVisualizer(est_scene, gpu_id=gpu_id)

            # save mesh
            if self.cfg.config['log'].get('save_mesh'):
                scene_mesh = est_scene.merge_mesh(
                    colorbox=igibson_colorbox * 255,
                    separate=False,
                    layout_color=(255, 69, 80),
                    texture=False
                )
                scene_mesh.vertices[:, 1] *= -1
                scene_mesh.vertices = scene_mesh.vertices[:, (0, 2, 1)]
                save_mesh(scene_mesh, os.path.join(scene_folder, 'scene.glb'))

            if self.cfg.config['full'] and est_scene.mesh_io:
                background = visualizer.background(200)
                render = visualizer.render(background=background)
                save_image(render, os.path.join(scene_folder, 'render.png'))

            image = visualizer.image('rgb')
            save_image(image, os.path.join(scene_folder, 'rgb.png'))
            image = visualizer.layout(image, total3d=True)
            image = visualizer.objs3d(image, bbox3d=True, axes=True, centroid=True, info=False)
            save_image(image, os.path.join(scene_folder, 'det3d.png'))
            image = visualizer.bfov(image)
            image = visualizer.bdb2d(image)
            save_image(image, os.path.join(scene_folder, 'visual.png'))

            est_scene.to_pickle(scene_folder)
