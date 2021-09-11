import os
import numpy as np
from collections import defaultdict
from shapely.geometry import Polygon, Point
from copy import deepcopy

from external.ldif.inference.metrics import mesh_chamfer_via_points
from models.testing import BaseTester
from models.eval_metrics import bdb3d_iou, bdb2d_iou, rot_err, classification_metric, AverageMeter, \
    AveragePrecisionMeter, BinaryClassificationMeter, ClassMeanMeter
from utils.layout_utils import manhattan_world_layout_info
from utils.relation_utils import relation_from_bins, test_bdb3ds, RelationOptimization
from .dataloader import collate_fn
from ..pano3d.testing import Tester as PanoTester
from utils.transform_utils import bins2bdb3d, IGTransform, bdb3d_corners, points2bdb2d, expand_bdb3d
from utils.igibson_utils import IGScene
from utils.visualize_utils import IGVisualizer
from utils.image_utils import save_image
from external.HorizonNet.eval_general import test_general
from utils.basic_utils import dict_of_array_to_list_of_dict, recursively_to


class Tester(PanoTester):

    def __init__(self, cfg, net, device=None):
        super(Tester, self).__init__(cfg, net, device)

    def get_metric_values(self, est_data, data):
        ''' Performs a evaluation step.
        '''
        metrics = {}
        gt_scenes = [v[-1][0] for v in data['views']]
        do_evaluation = not any(s.empty() for s in gt_scenes)

        # est_data_views = []
        # for est_data_view, data_view in zip(est_data, data['views']):
        #     metrics_view, est_data_view = super(Tester, self).get_metric_values(est_data_view, data_view)
        #     est_data_views.append(est_data_view)

        # affinity
        est_affinity, gt_affinity = est_data['affinity'], data['affinity']
        metric_affinity = BinaryClassificationMeter()
        est_affinity = np.array(est_affinity.cpu().numpy(), dtype=np.bool)
        gt_affinity = np.array(gt_affinity.cpu().numpy(), dtype=np.bool)
        metric_affinity.add({'est': est_affinity.reshape(-1), 'gt': gt_affinity.reshape(-1)})
        metrics['affinity'] = metric_affinity

        # 3D detection
        stitched = est_data['stitched']
        if do_evaluation and 'objs' in stitched and stitched['objs'] and 'bdb3d' in stitched['objs']:
            est_scene = IGScene.from_batch(stitched)[0]
            metric_aps = ClassMeanMeter(AveragePrecisionMeter)

            # merge GT objs
            gt_objs = gt_scenes[0]['objs'].copy()
            id_view1 = set(obj['id'] for obj in gt_objs)
            for obj_scene2 in gt_scenes[1]['objs']:
                if obj_scene2['id'] not in id_view1:
                    gt_objs.append(obj_scene2)

            # sort detections by score
            est_objs = sorted(est_scene['objs'], key=lambda obj: -obj['score'])
            for est_obj in est_objs:
                # match gt with same label and largest iou
                iou_max = -np.inf
                for gt_obj in gt_objs:
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

            for gt_obj in gt_objs:
                if 'matched' in gt_obj:
                    gt_obj.pop('matched')

            if metric_aps:
                metrics['3D_detection_AP'] = metric_aps

        return metrics, est_data

    def visualize_step(self, est_data):
        ''' Performs a visualization step.
        '''
        stitched = est_data['stitched']
        if 'objs' in stitched and 'mesh' not in stitched['objs']\
                and 'mesh_extractor' in stitched['objs'] and self.cfg.config['full']:
            stitched['objs']['mesh'] = stitched['objs']['mesh_extractor'].extract_mesh()
        est_scene = IGScene.from_batch(stitched)[0]

        scene_folder = os.path.join(
            self.cfg.config['log']['vis_path'], est_scene['scene'][0],
            f"{est_scene['name'][0]}-{est_scene['name'][1]}"
        )
        gpu_id = int(self.cfg.config['device']['gpu_ids'].split(',')[0])
        visualizer = IGVisualizer(est_scene, gpu_id=gpu_id)

        if self.cfg.config['full'] and est_scene.mesh_io:
            render = visualizer.render(background=200, camera='birds_eye')
            save_image(render, os.path.join(scene_folder, 'birds_eye.png'))
            render = visualizer.render(background=200, camera='up_down')
            save_image(render, os.path.join(scene_folder, 'up_down.png'))
