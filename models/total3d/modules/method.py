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
from utils.transform_utils import IGTransform, bdb3d_corners, points2bdb2d, size2reg, num2bins, contour2bfov, \
    bins2bdb3d, bins2layout, bins2camera
from configs import data_config
from models.pano3d.modules.method import Pano3D


@METHODS.register_module
class Total3D(Pano3D):

    def __init__(self, cfg):
        super(Total3D, self).__init__(cfg)

        for model_name in ['layout_estimation']:
            if hasattr(self, model_name):
                setattr(self, model_name, nn.DataParallel(self.__getattr__(model_name)))

    def forward(self, data):
        est_data, gt_data, est_scenes, gt_scenes = data

        if hasattr(self, 'layout_estimation'):
            layout_output = self.layout_estimation(gt_data['image_tensor']['rgb'])
            est_data['layout'] = layout_output
            self.generate_layout_gt(gt_data)
            self.update_layout(est_data, gt_data)

            layout_scenes = dict_of_array_to_list_of_dict(est_data['layout'])
            for est_scene, layout_scene in zip(est_scenes, layout_scenes):
                est_scene['layout'] = layout_scene

        if hasattr(self, 'detector'):
            det_scenes = self.detector(gt_data['image_np']['rgb'], camera=[s['camera']for s in gt_scenes])

            for est_scene, det_scene in zip(est_scenes, det_scenes):
                est_scene.image_io.update(det_scene['image_np'])
                est_scene['objs'] = det_scene['objs']

            if len(self.model_names) > 1 or self.cfg.config['log'].get('save_as_dataset'):
                self.generate_bdb2d_est(est_scenes, gt_scenes)

            if self.cfg.config['log'].get('save_as_dataset'):
                self.generate_bdb3d_gt(est_scenes, gt_scenes)

            est_data, gt_data = collate_fn([[s.data for s in scenes] for scenes in zip(est_scenes, gt_scenes)])
            est_data, gt_data = recursively_to((est_data, gt_data), device='cuda')

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
                objs_output, layout_output = self.scene_gcn(est_data)
                est_data['objs'].update(objs_output)
                est_data['layout'].update(layout_output)
                self.update_layout(est_data, gt_data)

            if hasattr(self, 'shape_decoder'):
                structured_implicit = StructuredImplicit.from_activation(
                    self.cfg.config['model']['shape_decoder'], lien_output['lien_activation'], self.shape_decoder)
                est_objs['mesh_extractor'] = structured_implicit

            if 'bdb3d' in est_objs:
                self.update_bdb3d(est_data)

        return est_data

    @staticmethod
    def update_layout(est_data, gt_data):
        est_layout = est_data['layout']
        # est_layout = gt_data['layout'] # debug reg/cls GT
        est_layout_total3d = est_layout['total3d']

        # transform layout raw outputs to layout estimation
        layout_bdb3d_ori = bins2layout(est_layout_total3d)
        est_trans = IGTransform(est_data, split='layout')
        layout_bdb3d = est_trans.ori2basis(layout_bdb3d_ori)
        est_data['layout']['total3d'].update(layout_bdb3d)

        # transform camera raw outputs to camera pose
        camera = bins2camera(est_layout_total3d)
        trans = IGTransform(est_data, split='layout')
        yaw, pitch, roll = trans.get_camera_angle()
        pitch, roll = camera['pitch'], camera['roll']
        trans.set_camera_angle(yaw, pitch, roll)

    def generate_layout_gt(self, gt_data):
        layout_bdb3d = gt_data['layout']['total3d']
        bins = recursively_to(data_config.metadata, dtype='tensor', device=layout_bdb3d['ori'].device)

        # generate parameterized classification/regression GT for layout
        layout_bdb3d['centroid_reg'] = layout_bdb3d['centroid_total3d'] - bins['layout_centroid_avg']
        layout_bdb3d['ori_cls'], layout_bdb3d['ori_reg'] = num2bins(bins['layout_ori_bins'], layout_bdb3d['ori'])
        layout_bdb3d['size_reg'] = size2reg(layout_bdb3d['size'], avg_key='layout_size_avg')

        # generate parameterized classification/regression GT for camera pose
        trans = IGTransform(gt_data, split='layout')
        _, pitch, roll = trans.get_camera_angle()
        layout_bdb3d['pitch_cls'], layout_bdb3d['pitch_reg'] = num2bins(bins['pitch_bins'], pitch)
        layout_bdb3d['roll_cls'], layout_bdb3d['roll_reg'] = num2bins(bins['roll_bins'], roll)

    def load_weight(self, pretrained_model):
        # for compatibility with Total3D
        pretrained_model = {k.replace('object_detection', 'bdb3d_estimation'): v
                            for k, v in pretrained_model.items()}
        super(Pano3D, self).load_weight(pretrained_model)
