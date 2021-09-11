import torch
from torch import nn
import numpy as np

from external.ldif.representation.structured_implicit_function import StructuredImplicit
from models.mgnet.modules.mgn import MGNetMeshExtractor
from models.pano3d.modules.detector_2d import bdb2d_geometric_feature
from models.registers import METHODS
from models.method import BaseMethod
from models.datasets import Pano3DDataset
from models.eval_metrics import contour_iou
from models.pano3d.dataloader import collate_fn
from utils.igibson_utils import IGScene, reverse_fov_split
from utils.layout_utils import manhattan_world_layout_from_pix_layout, wall_contour_from_manhattan_pix_layout, \
    wall_bdb3d_from_manhattan_world_layout
from utils.relation_utils import RelationOptimization
from utils.basic_utils import dict_of_array_to_list_of_dict, recursively_to
from configs.data_config import IG56CLASSES
from utils.transform_utils import IGTransform, bdb3d_corners, points2bdb2d, size2reg, num2bins, contour2bfov, bins2bdb3d
from configs import data_config


@METHODS.register_module
class Pano3D(BaseMethod):

    _model_order = [
        'detector', 'layout_estimation', 'bdb3d_estimation',
        'shape_encoder', 'shape_decoder', 'scene_gcn',
        'mesh_reconstruction'
    ]

    def __init__(self, cfg):
        super(Pano3D, self).__init__(cfg, self._model_order)

        self.fov_split = self.cfg.config['data'].get('fov_split', 0)
        self.offset_bdb2d = self.cfg.config['data'].get('offset_bdb2d', False)

        # Multi-GPU setting
        # (Note that for bdb3d_estimation,
        # we should extract relational features,
        # thus it does not support parallel training.
        # except its image encoder)
        for model_name in ['shape_encoder', 'shape_decoder', 'mesh_reconstruction']:
            if hasattr(self, model_name):
                setattr(self, model_name, nn.DataParallel(self.__getattr__(model_name)))

    def forward(self, data):
        est_data, gt_data, est_scenes, gt_scenes = data

        if hasattr(self, 'layout_estimation'):
            layout_output = self.layout_estimation(gt_data['image_tensor']['rgb'])
            if isinstance(layout_output, dict):
                layout_output = (layout_output, )
            est_data['layout'] = {k: v for k, v in zip(('horizon', 'manhattan_pix'), layout_output)}

            layout_scenes = dict_of_array_to_list_of_dict(est_data['layout'])
            for est_scene, layout_scene in zip(est_scenes, layout_scenes):
                est_scene['layout'] = layout_scene

            if len(self.model_names) > 1 or self.cfg.config['log'].get('save_as_dataset'):
                self.generate_wall_est(est_scenes, gt_scenes)

        if hasattr(self, 'detector'):
            det_scenes = self.detector(gt_data['image_np']['rgb'], camera=[s['camera']for s in gt_scenes])

            for est_scene, det_scene in zip(est_scenes, det_scenes):
                est_scene.image_io.update(det_scene['image_np'])
                est_scene['objs'] = det_scene['objs']

            if len(self.model_names) > 1 or self.cfg.config['log'].get('save_as_dataset'):
                self.generate_bdb2d_est(est_scenes, gt_scenes)

            if self.cfg.config['log'].get('save_as_dataset'):
                self.generate_bdb3d_gt(est_scenes, gt_scenes)

            if 'layout_estimation' in self.model_names:
                self.generate_relation_gt(est_scenes, gt_scenes)

            est_data, gt_data = collate_fn([[s.data for s in scenes] for scenes in zip(est_scenes, gt_scenes)])
            est_data, gt_data = recursively_to((est_data, gt_data), device='cuda')

        # For FoV experiment: split objects of each scene into perspective cameras with specified FoV
        # split cameras by FoV and rotate objs in each camera into their own camera frame
        if self.fov_split:
            split_scenes = []
            for est_scene, (gt_offset, _) in zip(est_scenes, gt_data['objs']['split']):
                cam_scenes = est_scene.fov_split(
                    self.fov_split, gt_offset=gt_offset.cpu().item(), offset_bdb2d=self.offset_bdb2d)
                n_splits = len(cam_scenes)
                split_scenes.extend(cam_scenes)
            est_data = collate_fn([(s.data, ) for s in split_scenes])
            est_data = recursively_to(est_data, device='cuda')
            est_scenes = split_scenes

        est_objs = est_data.get('objs')
        if est_objs:
            if hasattr(self, 'shape_encoder'):
                lien_output = self.shape_encoder(est_objs['rgb'], est_objs['cls_code'])
                est_objs.update(lien_output)

            if hasattr(self, 'mesh_reconstruction'):
                mgn_output = self.mesh_reconstruction(est_objs['rgb'], est_objs['cls_code'])
                est_objs['mesh_extractor'] = MGNetMeshExtractor(mgn_output)

            if hasattr(self, 'bdb3d_estimation') or hasattr(self, 'scene_gcn'):
                # construct bdb2d_geometric_feature
                g_features = [bdb2d_geometric_feature(
                    [o['bdb2d'] for o in est_scene['objs']], self.cfg.config['data']['g_feature_length']
                ) for est_scene in est_scenes]
                est_data['objs']['g_feature'] = torch.cat(g_features).to(est_objs['rgb'].device)

            if hasattr(self, 'bdb3d_estimation'):
                bdb3d_output = self.bdb3d_estimation(
                    est_objs['rgb'], est_objs['cls_code'], est_objs['g_feature'], est_objs['split'])
                est_objs.update(bdb3d_output)

            if hasattr(self, 'scene_gcn'):
                gcn_output = self.scene_gcn(est_data)
                if isinstance(gcn_output, dict):
                    gcn_output = (gcn_output, )
                for k, v in zip(('objs', 'relation'), gcn_output):
                    if k == 'objs':
                        est_data[k].update(v)
                    else:
                        est_data[k] = v

            if hasattr(self, 'shape_decoder'):
                structured_implicit = StructuredImplicit.from_activation(
                    self.cfg.config['model']['shape_decoder'], lien_output['lien_activation'], self.shape_decoder)
                est_objs['mesh_extractor'] = structured_implicit

            if 'bdb3d' in est_objs:
                self.update_bdb3d(est_data)

        # For FoV experiment: put split objects back together
        # take first camera as center and rotate objs back to their place in the first camera frame
        if self.fov_split:
            est_data = reverse_fov_split(est_data, n_splits, offset_bdb2d=self.offset_bdb2d)

        return est_data

    @staticmethod
    def update_bdb3d(est_data):
        # transform bdb3d raw output to bdb3d estimation
        est_objs = est_data['objs']
        est_bdb3d = est_objs['bdb3d']
        bdb3d_detect_pix = bins2bdb3d(est_data) if 'ori' not in est_bdb3d else est_bdb3d
        est_trans = IGTransform(est_data)
        bdb3d_detect = est_trans.campix2world(bdb3d_detect_pix)
        est_objs['bdb3d'].update(bdb3d_detect_pix)
        est_objs['bdb3d'].update(bdb3d_detect)

    def generate_bdb2d_est(self, est_scenes, gt_scenes):
        for est_scene in est_scenes:
            # crop and preprocess objects
            est_scene.crop_images(
                perspective='K' not in gt_scenes[0]['camera'], short_width=Pano3DDataset.crop_width)
            for o in est_scene['objs']:
                o['rgb'] = Pano3DDataset.crop_transforms['train' if self.training else 'test'](o['rgb'])

            # encode class as one-hot
            for i_est, est_obj in enumerate(est_scene['objs']):
                cls_code = np.zeros(len(IG56CLASSES), dtype=np.float32)
                cls_code[est_obj['label']] = 1.
                est_obj['cls_code'] = cls_code

    def generate_wall_est(self, est_scenes, gt_scenes):
        for est_scene, gt_scene in zip(est_scenes, gt_scenes):
            # transform pixel manhattan world layout to 3d world frame layout
            if 'layout' in gt_scene.data and 'manhattan_world' in gt_scene['layout']:
                floor_z = gt_scene['layout']['manhattan_world'][:, -1].min()
                gt_camera_height = gt_scene['camera']['pos'][-1] - floor_z
            else:
                gt_camera_height = 1.6
            est_scene['layout']['manhattan_world'] = \
                manhattan_world_layout_from_pix_layout(est_scene, gt_camera_height)

            # generate walls from layout
            walls_contour = wall_contour_from_manhattan_pix_layout(
                est_scene['layout']['manhattan_pix'], est_scene.transform)
            walls_bdb3d = wall_bdb3d_from_manhattan_world_layout(
                est_scene['layout']['manhattan_world'])
            walls = []
            for wall, wall_bdb3d in zip(walls_contour, walls_bdb3d):
                contour = wall['contour']
                bfov = contour2bfov(contour, est_scene['camera']['height'], est_scene['camera']['width'])
                bdb2d = points2bdb2d(contour)
                bdb2d = {k: int(v) for k, v in bdb2d.items()}
                wall['bfov'] = bfov
                wall['bdb2d'] = bdb2d
                wall['bdb3d'] = wall_bdb3d['bdb3d']
                wall['bdb3d'].update(est_scene.transform.world2campix(wall['bdb3d']))
                walls.append(wall)
            est_scene['walls'] = walls

    def generate_relation_gt(self, est_scenes, gt_scenes):
        for est_scene, gt_scene in zip(est_scenes, gt_scenes):
            if not('objs' in gt_scene.data and gt_scene['objs'] and 'bdb3d' in gt_scene['objs'][0]):
                continue
            gt_objs = gt_scene['objs']

            # generate relation between gt objects and estimated layout
            relation_optimization = RelationOptimization(
                expand_dis=self.cfg.config['data']['expand_dis'])
            gt_rel_data = est_scene.data.copy()
            gt_rel_data['objs'] = gt_objs
            gt_rel_scene = IGScene(gt_rel_data)
            relation_optimization.generate_relation(gt_rel_scene)
            # from utils.relation_utils import visualize_relation
            # visualize_relation(gt_rel_scene, wall3d=True, show=True)
            gt_scene['relation'] = gt_rel_scene['relation']
            est_scene['walls'] = gt_rel_scene['walls']

    def generate_bdb3d_gt(self, est_scenes, gt_scenes):
        for est_scene, gt_scene in zip(est_scenes, gt_scenes):
            est_objs = est_scene['objs']
            if not ('objs' in gt_scene.data and gt_scene['objs'] and 'bdb3d' in gt_scene['objs'][0]):
                for est_obj in est_objs:
                    est_obj['gt'] = -1
                continue
            gt_objs = gt_scene['objs']

            # generate IoU and label match matrix
            est_gt_iou = np.zeros([len(est_objs), len(gt_objs)])
            label_match = np.zeros_like(est_gt_iou, dtype=np.bool)
            if est_objs and gt_objs:
                for i_est, est_obj in enumerate(est_objs):
                    for i_gt, gt_obj in enumerate(gt_objs):
                        label_match[i_est, i_gt] = est_obj['label'] == gt_obj['label']
                        est_gt_iou[i_est, i_gt] = contour_iou(
                            est_obj['contour'], gt_obj['contour'], est_scene['camera']['width'])

            # match detections with ground truth
            iou_match = est_gt_iou > self.detector.min_iou
            i_est_gt_match = {}
            for i_gt, (iou_match_mask, label_match_mask) in enumerate(zip(iou_match.T, label_match.T)):
                perfect_match_mask = (iou_match_mask & label_match_mask)
                # prefer perfect matched object
                # then consider objects with matched bounding box but no matched label
                for match_mask in (perfect_match_mask, iou_match_mask):
                    if match_mask.any():
                        i_est_cand = np.where(match_mask)[0]
                        est_iou = est_gt_iou[i_est_cand, i_gt]
                        i_est_match = i_est_cand[np.argmax(est_iou)]
                        # if i_est_match in i_est_gt_match:
                        i_est_gt_match[i_est_match] = i_gt
                        break
            # set matched gt index for detected objects
            for i_est, est_obj in enumerate(est_objs):
                est_obj['gt'] = i_est_gt_match.get(i_est, -1)

            # get normalized distance between bfov center and projected bdb3d centroid
            for gt_obj in gt_objs:
                gt_obj['delta2d'] = np.zeros(2, dtype=np.float32)
            for est_obj in est_objs:
                if est_obj['gt'] > 0:
                    gt_obj = gt_objs[est_obj['gt']]
                    bdb3d = gt_obj['bdb3d']
                    if 'K' in est_scene['camera']:
                        bdb2d = est_obj['bdb2d']
                        bdb2d_center = np.array([bdb2d['x1'] + bdb2d['x2'], bdb2d['y1'] + bdb2d['y2']]) / 2
                        bdb2d_wh = np.array([bdb2d['x2'] - bdb2d['x1'], bdb2d['y2'] - bdb2d['y1']])
                        bdb3d_center = gt_scene.transform.world2campix(bdb3d['centroid'])
                        center_delta = (bdb2d_center - bdb3d_center) / bdb2d_wh
                    else:
                        bfov = est_obj['bfov']
                        bfov_center = np.array([bfov['lon'], bfov['lat']], dtype=np.float32)
                        bfov_wh = np.array([bfov['x_fov'], bfov['y_fov']], dtype=np.float32)
                        bdb3d_center = gt_scene.transform.world2camrad(bdb3d['centroid'])
                        center_delta = (bfov_center - bdb3d_center) / bfov_wh
                    gt_obj['delta2d'] = center_delta

            # parameterize gt_bdb3d
            for gt_obj in gt_objs:
                gt_bdb3d = gt_obj['bdb3d']
                bdb3d_pix = gt_scene.transform.world2campix(gt_bdb3d)
                gt_bdb3d.update(bdb3d_pix)
                # parameterize gt_bdb3d as classification and regression
                gt_bdb3d['dis_cls'], gt_bdb3d['dis_reg'] = num2bins(
                    data_config.metadata['dis_bins'], bdb3d_pix['dis'])
                gt_bdb3d['ori_cls'], gt_bdb3d['ori_reg'] = num2bins(
                    data_config.metadata['ori_bins'], bdb3d_pix['ori'])
                gt_bdb3d['size_reg'] = size2reg(gt_bdb3d['size'], gt_obj['label'])
                # project gt_bdb3d to camera plane to get bdb2d_from_3d
                recentered_trans = IGTransform.level_look_at(gt_scene.data, gt_bdb3d['centroid'])
                gt_bdb3d_corners = recentered_trans.world2campix(bdb3d_corners(gt_bdb3d))
                bdb2d_from_gt_bdb3d = points2bdb2d(gt_bdb3d_corners)
                gt_obj['bdb2d_from_bdb3d'] = bdb2d_from_gt_bdb3d

    def load_weight(self, pretrained_model):
        if 'state_dict' in pretrained_model:
            # for compatibility with HorizonNet
            compatible_model = pretrained_model['state_dict']
            compatible_model = {'layout_estimation.module.' + k: v
                                for k, v in compatible_model.items()}
        else:
            # for compatibility with Total3D
            compatible_model = {k.replace('object_detection', 'bdb3d_estimation'): v
                                for k, v in pretrained_model.items()}

        # for compatibility with DataParallel HorizonNet
        old_prefix = 'layout_estimation.module.'
        new_prefix = 'layout_estimation.horizon_net.module.'
        compatible_model = {k.replace(old_prefix, new_prefix): v
                            for k, v in compatible_model.items()}
        super(Pano3D, self).load_weight(compatible_model)
