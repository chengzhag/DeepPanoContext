import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import getitem
import os
from collections import defaultdict
from shapely.geometry import Polygon, Point
import numpy as np

from external.ldif.representation.structured_implicit_function import StructuredImplicit
from models.eval_metrics import bdb2d_iou
from models.pano3d.modules.detector_2d import bdb2d_geometric_feature
from models.registers import MODULES
from configs.data_config import IG56CLASSES
from configs import data_config
from utils.igibson_utils import split_batch_into_patches
from utils.layout_utils import manhattan_2d_from_manhattan_world_layout
from utils.transform_utils import IGTransform, bins2bdb3d, bdb3d_corners, point_polygon_dis, num2bins
from utils.relation_utils import RelationOptimization, relation_from_bins, test_bdb3ds
from utils.basic_utils import dict_of_array_to_list_of_dict, list_of_dict_to_dict_of_array


def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()


class _Collection_Unit(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(_Collection_Unit, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out, bias=True)
        normal_init(self.fc, 0, 0.01)

    def forward(self, target, source, attention_base):
        # assert attention_base.size(0) == source.size(0), "source number must be equal to attention number"
        fc_out = F.relu(self.fc(source))
        collect = torch.mm(attention_base, fc_out)  # Nobj x Nrel Nrel x dim
        collect_avg = collect / (attention_base.sum(1).view(collect.size(0), 1) + 1e-7)
        return collect_avg


class _Update_Unit(nn.Module):
    def __init__(self, dim):
        super(_Update_Unit, self).__init__()
    def forward(self, target, source):
        assert target.size() == source.size(), "source dimension must be equal to target dimension"
        update = target + source
        return update


class _GraphConvolutionLayer_Collect(nn.Module):
    """ graph convolutional layer """
    """ collect information from neighbors """
    def __init__(self, dim_obj, dim_rel):
        super(_GraphConvolutionLayer_Collect, self).__init__()
        self.collect_units = nn.ModuleList()
        self.collect_units.append(_Collection_Unit(dim_rel, dim_obj)) # obj (subject) from rel
        self.collect_units.append(_Collection_Unit(dim_rel, dim_obj)) # obj (object) from rel
        self.collect_units.append(_Collection_Unit(dim_obj, dim_rel)) # rel from obj (subject)
        self.collect_units.append(_Collection_Unit(dim_obj, dim_rel)) # rel from obj (object)
        self.collect_units.append(_Collection_Unit(dim_obj, dim_obj)) # obj from obj

    def forward(self, target, source, attention, unit_id):
        collection = self.collect_units[unit_id](target, source, attention)
        return collection


class _GraphConvolutionLayer_Update(nn.Module):
    """ graph convolutional layer """
    """ update target nodes """
    def __init__(self, dim_obj, dim_rel):
        super(_GraphConvolutionLayer_Update, self).__init__()
        self.update_units = nn.ModuleList()
        self.update_units.append(_Update_Unit(dim_obj)) # obj from others
        self.update_units.append(_Update_Unit(dim_rel)) # rel from others

    def forward(self, target, source, unit_id):
        update = self.update_units[unit_id](target, source)
        return update


class FeatureExtractor:

    def __init__(self, cfg, model_name):
        model_config = cfg.config['model'][model_name]
        width = cfg.config['data'].get('width', 0)
        self.OBJ_ORI_BIN = len(data_config.metadata['ori_bins'])
        self.OBJ_CENTER_BIN = len(data_config.metadata['dis_bins'])
        self.PITCH_BIN = len(data_config.metadata['pitch_bins'])
        self.ROLL_BIN = len(data_config.metadata['roll_bins'])
        self.LO_ORI_BIN = len(data_config.metadata['layout_ori_bins'])
        self.shape_encoder_config = cfg.config['model'].get('shape_encoder', {})
        self.shape_decoder_config = cfg.config['model'].get('shape_decoder', {})

        self.feature_length = {
            # for layout node
            'layout.horizon.bon': width * 2,
            'layout.horizon.cor': width,
            'layout.total3d.pitch_reg': 2,
            'layout.total3d.roll_reg': 2,
            'layout.total3d.pitch_cls': 2,
            'layout.total3d.roll_cls': 2,
            'layout.total3d.ori_reg': 2,
            'layout.total3d.ori_cls': 2,
            'layout.total3d.centroid_reg': 3,
            'layout.total3d.size_reg': 3,
            'layout.afeatures': 2048,
            'camera.K': 3,
            # for wall node
            'walls.bdb3d.centroid': 3,
            'walls.bdb3d.size': 3,
            'walls.bdb3d.center': 2,
            'walls.bdb3d.dis': 1,
            'walls.bdb3d.ori': 1,
            'walls.bfov.lon': 14,
            'walls.bfov.lat': 8,
            'walls.bfov.x_fov': 1,
            'walls.bfov.y_fov': 1,
            # for object/relation node
            'objs.cls_code': len(IG56CLASSES),
            'objs.bdb2d': 4,
            'objs.bdb3d.size_reg': 3,
            'objs.bdb3d.ori_reg': self.OBJ_ORI_BIN,
            'objs.bdb3d.ori_cls': self.OBJ_ORI_BIN,
            'objs.bdb3d.dis_reg': self.OBJ_CENTER_BIN,
            'objs.bdb3d.dis_cls': self.OBJ_CENTER_BIN,
            'objs.bdb3d': 12,
            'objs.delta2d': 2,
            'objs.ben_afeature': 2048,
            'objs.ben_rfeatures': 2048,
            'objs.ben_arfeature': 2048,
            'objs.lien_afeature': self.shape_encoder_config.get('bottleneck_size', 0),
            'objs.lien_activation': self.shape_encoder_config.get('shape_code_length', 0),
            'objs.ldif_blob_center': (self.shape_decoder_config.get('element_count', 0)
                                      + self.shape_decoder_config.get('sym_element_count', 0)) * 3,
            'objs.ldif_analytic_code': (self.shape_decoder_config.get('element_count', 0)) * 10,
            'objs.layout_dis': 8,
            'objs.ceil_dis': 8,
            'objs.floor_dis': 8,
            'objs.bfov.lon': 14,
            'objs.bfov.lat': 8,
            'objs.bfov.x_fov': 1,
            'objs.bfov.y_fov': 1,

            # for relation node
            'objs.g_feature': cfg.config['data']['g_feature_length'] // 2,
            'g_feature': cfg.config['data']['g_feature_length'],
            'bdb2d': 8,
            'iou': 1,
            'bdb3d_test': 3,
            'rel_rot': 10
        }
        
        self.obj_features = model_config.get('obj_features')
        self.obj_features_len = sum([self.feature_length[k] for k in self.obj_features]) if self.obj_features else None

    def _get_single_feature(self, data, key):
        transform = IGTransform(data, split=key.split('.')[0])

        if key.endswith('.bdb3d') or key in (
                'objs.ldif_blob_center', 'objs.ldif_analytic_code',
                'objs.ceil_dis', 'objs.floor_dis', 'objs.layout_dis'
        ):
            bdb3ds = bins2bdb3d(data)
            if key.endswith('.bdb3d'):
                transform_centered = IGTransform.world_centered(data['camera'])
                bdb3ds.update(transform_centered.campix23d(bdb3ds))
                bdb3ds = {k: bdb3ds[k] for k in (
                    'center', 'size', 'dis', 'dis_score', 'ori', 'ori_score', 'centroid')}
                v = torch.cat([v[:, None] if v.dim() == 1 else v for v in bdb3ds.values()], -1)
            elif key in ('objs.ldif_blob_center', 'objs.ldif_analytic_code'):
                structured_implicit = StructuredImplicit.from_activation(
                    self.shape_decoder_config, data['objs']['lien_activation'])
                if key == 'objs.ldif_blob_center':
                    bdb3ds.update(transform.campix23d(bdb3ds))
                    centers = structured_implicit.all_centers
                    v = transform.obj2frame(centers, bdb3ds)
                else:
                    v = structured_implicit.analytic_code
                v = v.reshape(v.shape[0], -1)
            elif key in ('objs.ceil_dis', 'objs.floor_dis', 'objs.layout_dis'):
                objs = data['objs']
                bdb3ds.update(transform.campix2world(bdb3ds))
                corners = bdb3d_corners(bdb3ds)
                v = []
                for layout, (start, end) in zip(data['layout']['manhattan_world'], objs['split']):
                    if end - start > 0:
                        corners_scene = corners[start:end]
                        if key == 'objs.layout_dis':
                            layout_2d = manhattan_2d_from_manhattan_world_layout(layout)
                            corners_2d = corners_scene[..., :2].reshape(-1, 2)
                            v.append(point_polygon_dis(corners_2d, layout_2d).view(-1, 8))
                        elif key == 'objs.ceil_dis':
                            ceil = layout[:, -1].max()
                            v.append(ceil - corners_scene[..., -1])
                        elif key == 'objs.floor_dis':
                            floor = layout[:, -1].min()
                            v.append(corners_scene[..., -1] - floor)
                v = torch.cat(v)
        else:
            v = reduce(getitem, key.split('.'), data)

            if key.endswith('.bdb2d'):
                v = v.copy()
                hheight, hwidth = transform['height'] / 2, transform['width'] / 2
                for boundary in v.keys():
                    v[boundary] = v[boundary] / (hwidth if 'x' in boundary else hheight) - 1
                v = torch.stack(list(v.values()), -1)
            elif key == 'objs.cls_code':
                v = v * data['objs']['score'][..., None]
            elif key.endswith('.bfov.lon') or key.endswith('.bfov.lat'):
                bins_key = 'lon_bins' if key.endswith('.bfov.lon') else 'lat_bins'
                bins = data_config.metadata[bins_key]
                cls, reg = num2bins(bins, v)
                onehot = torch.zeros([len(v), len(bins)], device=v.device)
                onehot[range(len(cls)), cls] = 1
                v = torch.cat([onehot, reg[:, None], v[:, None]], -1)
            elif key == 'camera.K':
                v = v.reshape(v.shape[0], -1)
                v = v.index_select(1, torch.tensor([0, 2, 4, 5], device=v.device))
                v = v[:, :3] / v[:, 3:]

        if isinstance(v, torch.Tensor) and v.dim() == 1:
            v = v[:, None]
        return v

    def _get_object_features(self, data, feature_list):
        features = []
        for key in feature_list:
            v = self._get_single_feature(data, key)
            v = v.view(-1, self.feature_length[key])
            assert self.feature_length[key] == v.shape[-1], f"length of feature '{key}' does not match"
            features.append(v)
        return torch.cat(features, -1)


class SceneGCN(FeatureExtractor, nn.Module):

    def __init__(self, cfg, optim_spec=None):
        FeatureExtractor.__init__(self, cfg, 'scene_gcn')
        nn.Module.__init__(self)

        '''Optimizer parameters used in training'''
        self.optim_spec = optim_spec

        '''configs and params'''
        self.cfg = cfg
        self.model_config = cfg.config['model']['scene_gcn']
        model_config = self.model_config
        self.feature_dim = model_config['feature_dim']
        self.update_groups = model_config['update_groups']
        self.update_steps = model_config['update_steps']
        self.res_output = model_config['res_output']

        feature_dim = self.feature_dim
        self.rel_features = model_config.get('rel_features')
        self.lo_features = model_config.get('lo_features')
        self.rel_features_len = sum([self.feature_length[k] * 2 if k.startswith('objs.') else self.feature_length[k]
                                     for k in self.rel_features]) if self.rel_features else None
        self.lo_features_len = sum([self.feature_length[k] for k in self.lo_features]) if self.lo_features else None

        '''GCN modules (from graph-rcnn)'''

        # representation embedding
        self.obj_embedding = nn.Sequential(
            nn.Linear(self.obj_features_len, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.rel_embedding = nn.Sequential(
            nn.Linear(self.rel_features_len, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim),
        )
        self.lo_embedding = nn.Sequential(
            nn.Linear(self.lo_features_len, feature_dim),
            nn.ReLU(True),
            nn.Linear(feature_dim, feature_dim),
        )

        # graph message passing
        if self.update_steps > 0:
            self.gcn_collect_feat = nn.ModuleList([
                _GraphConvolutionLayer_Collect(feature_dim, feature_dim) for i in range(self.update_groups)])
            self.gcn_update_feat = nn.ModuleList([
                _GraphConvolutionLayer_Update(feature_dim, feature_dim) for i in range(self.update_groups)])

        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.5)

    def initiate_weights(self):
        # initiate weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.zero_()

    def construct_bdb3d_branch(self):
        '''representation to object output (from Total3D object_detection)'''

        # branch to predict the size
        self.fc1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc2 = nn.Linear(self.feature_dim // 2, 3)

        # branch to predict the orientation
        self.fc3 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc4 = nn.Linear(self.feature_dim // 2, self.OBJ_ORI_BIN * 2)

        # branch to predict the centroid
        self.fc5 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_centroid = nn.Linear(self.feature_dim // 2, self.OBJ_CENTER_BIN * 2)

        # branch to predict the 2D offset
        self.fc_off_1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_off_2 = nn.Linear(self.feature_dim // 2, 2)

    def construct_layout_branch(self):
        # feature to output (from Total3D layout_estimation)
        self.fc_1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_2 = nn.Linear(self.feature_dim // 2, (self.PITCH_BIN + self.ROLL_BIN) * 2)
        # fc for layout
        self.fc_layout = nn.Linear(self.feature_dim, self.feature_dim)
        # for layout orientation
        self.fc_3 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_4 = nn.Linear(self.feature_dim // 2, self.LO_ORI_BIN * 2)
        # for layout centroid and coefficients
        self.fc_5 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_6 = nn.Linear(self.feature_dim // 2, 6)

    def _get_relation_features(self, data):
        features = []
        for key in self.rel_features:
            v = self._get_single_feature(data, key)

            # reshape relation feature
            if key != 'objs.g_feature':
                features_rel = []
                for start, end in data['objs']['split']:
                    if end - start > 0:
                        features_rel.append(torch.stack(
                            [torch.cat([loc1, loc2], -1)
                             for loc1 in v[start:end]
                             for loc2 in v[start:end]]))
                v = torch.cat(features_rel, 0)

            feature_length = self.feature_length[key]
            feature_length *= 2 if key.startswith('objs.') else 1
            assert feature_length == v.shape[-1], f"length of feature '{key}' does not match"
            features.append(v)

        return torch.cat(features, -1)

    def message_passing(self, obj_obj_map, obj_pred_map, subj_pred_map, x_obj, x_pred):
        '''feature level agcn'''

        obj_feats = [x_obj]
        pred_feats = [x_pred]
        start = 0
        for group, (gcn_collect_feat, gcn_update_feat) in enumerate(zip(self.gcn_collect_feat, self.gcn_update_feat)):
            for t in range(start, start + self.update_steps):
                '''update object features'''
                # message from other objects
                source_obj = gcn_collect_feat(obj_feats[t], obj_feats[t], obj_obj_map, 4)

                # message from predicate
                source_rel_sub = gcn_collect_feat(obj_feats[t], pred_feats[t], subj_pred_map, 0)
                source_rel_obj = gcn_collect_feat(obj_feats[t], pred_feats[t], obj_pred_map, 1)
                source2obj_all = (source_obj + source_rel_sub + source_rel_obj) / 3
                obj_feats.append(gcn_update_feat(obj_feats[t], source2obj_all, 0))

                '''update predicate features'''
                source_obj_sub = gcn_collect_feat(pred_feats[t], obj_feats[t], subj_pred_map.t(), 2)
                source_obj_obj = gcn_collect_feat(pred_feats[t], obj_feats[t], obj_pred_map.t(), 3)
                source2rel_all = (source_obj_sub + source_obj_obj) / 2
                pred_feats.append(gcn_update_feat(pred_feats[t], source2rel_all, 1))
            start += self.update_steps
        return obj_feats, pred_feats

    def bdb3d_from_obj_feature(self, data, obj_feats_wolo):
        '''representation to layout output (from Total3D layout_estimation)'''

        # branch to predict the size
        size_reg = self.fc1(obj_feats_wolo)
        size_reg = self.relu(size_reg)
        size_reg = self.dropout(size_reg)
        size_reg = self.fc2(size_reg)

        # branch to predict the orientation
        ori = self.fc3(obj_feats_wolo)
        ori = self.relu(ori)
        ori = self.dropout(ori)
        ori = self.fc4(ori)
        ori = ori.view(-1, self.OBJ_ORI_BIN, 2)
        ori_reg = ori[:, :, 0]
        ori_cls = ori[:, :, 1]

        # branch to predict the centroid
        centroid = self.fc5(obj_feats_wolo)
        centroid = self.relu(centroid)
        centroid = self.dropout(centroid)
        centroid = self.fc_centroid(centroid)
        centroid = centroid.view(-1, self.OBJ_CENTER_BIN, 2)
        centroid_cls = centroid[:, :, 0]
        centroid_reg = centroid[:, :, 1]

        # branch to predict the 2D offset
        delta2d = self.fc_off_1(obj_feats_wolo)
        delta2d = self.relu(delta2d)
        delta2d = self.dropout(delta2d)
        delta2d = self.fc_off_2(delta2d)

        if self.res_output:
            objs = data['objs']
            bdb3d = objs['bdb3d']

            size_reg += bdb3d['size_reg']
            ori_reg += bdb3d['ori_reg']
            ori_cls += bdb3d['ori_cls']
            centroid_reg += bdb3d['dis_reg']
            centroid_cls += bdb3d['dis_cls']
            delta2d += objs['delta2d']

        output = {
            'bdb3d': {
                'size_reg': size_reg, 'ori_reg': ori_reg, 'ori_cls': ori_cls,
                'dis_reg': centroid_reg, 'dis_cls': centroid_cls
            },
            'delta2d': delta2d,
        }

        return output

    def layout_from_lo_feature(self, data, obj_feats_lo):
        '''representation to object output (from Total3D object_detection)'''

        # branch for camera parameters
        cam = self.fc_1(obj_feats_lo)
        cam = self.relu(cam)
        cam = self.dropout(cam)
        cam = self.fc_2(cam)
        pitch_reg = cam[:, 0: self.PITCH_BIN]
        pitch_cls = cam[:, self.PITCH_BIN: self.PITCH_BIN * 2]
        roll_reg = cam[:, self.PITCH_BIN * 2: self.PITCH_BIN * 2 + self.ROLL_BIN]
        roll_cls = cam[:, self.PITCH_BIN * 2 + self.ROLL_BIN: self.PITCH_BIN * 2 + self.ROLL_BIN * 2]

        # branch for layout orientation, centroid and coefficients
        lo = self.fc_layout(obj_feats_lo)
        lo = self.relu(lo)
        lo = self.dropout(lo)
        # branch for layout orientation
        lo_ori = self.fc_3(lo)
        lo_ori = self.relu(lo_ori)
        lo_ori = self.dropout(lo_ori)
        lo_ori = self.fc_4(lo_ori)
        lo_ori_reg = lo_ori[:, :self.LO_ORI_BIN]
        lo_ori_cls = lo_ori[:, self.LO_ORI_BIN:]

        # branch for layout centroid and coefficients
        lo_ct = self.fc_5(lo)
        lo_ct = self.relu(lo_ct)
        lo_ct = self.dropout(lo_ct)
        lo_ct = self.fc_6(lo_ct)
        lo_centroid_reg = lo_ct[:, :3]
        lo_size_reg = lo_ct[:, 3:]

        if self.res_output:
            layout = data['layout']['total3d']

            pitch_reg += layout['pitch_reg']
            pitch_cls += layout['pitch_cls']
            roll_reg += layout['roll_reg']
            roll_cls += layout['roll_cls']
            lo_ori_reg += layout['ori_reg']
            lo_ori_cls += layout['ori_cls']
            lo_centroid_reg += layout['centroid_reg']
            lo_size_reg += layout['size_reg']

        output = {
            'total3d':{
                'pitch_reg': pitch_reg,
                'pitch_cls': pitch_cls,
                'roll_reg': roll_reg,
                'roll_cls': roll_cls,
                'ori_reg': lo_ori_reg,
                'ori_cls': lo_ori_cls,
                'centroid_reg': lo_centroid_reg,
                'size_reg': lo_size_reg,
            }
        }

        return output

@MODULES.register_module
class RefineSGCN(SceneGCN):

    def __init__(self, cfg, optim_spec=None):
        super(RefineSGCN, self).__init__(cfg, optim_spec)

        self.rel_obj_lo = 0.001
        self.refine_layout = self.model_config.get('refine_layout')

        # representation to object output (from Total3D object_detection)
        self.construct_bdb3d_branch()
        if self.refine_layout:
            self.construct_layout_branch()

        self.initiate_weights()

    def _get_map(self, split):
        device = split.device
        n_vert = split[-1][-1] + split.shape[0] # number of objects and layouts
        obj_obj_map = torch.zeros([n_vert, n_vert]) # mapping of obj/lo vertices with connections
        rel_inds = [] # indexes of vertices connected by relation nodes
        rel_masks = [] # mask of relation features for obj/lo vertices connected by relation nodes
        obj_masks = torch.zeros(n_vert, dtype=torch.bool) # mask of object vertices
        lo_masks = torch.zeros(n_vert, dtype=torch.bool) # mask of layout vertices
        for i_scene, (start, end) in enumerate(split):
            start = start + i_scene # each subgraph has Ni object vertices and 1 layout vertex
            end = end + i_scene + 1 # consider layout vertex, Ni + 1 vertices in total
            obj_obj_map[start:end, start:end] = 1 # each subgraph is a complete graph with self circle
            obj_ind = torch.arange(start, end, dtype=torch.long)
            subj_ind_i, obj_ind_i = torch.meshgrid(obj_ind, obj_ind) # indexes of each vertex in the subgraph
            rel_ind_i = torch.stack([subj_ind_i.reshape(-1), obj_ind_i.reshape(-1)], -1)
            rel_mask_i = rel_ind_i[:, 0] != rel_ind_i[:, 1] # vertices connected by relation nodes should be different
            rel_inds.append(rel_ind_i[rel_mask_i])
            rel_masks.append(rel_mask_i)
            obj_masks[start:end - 1] = True # for each subgraph, first Ni vertices are objects
            lo_masks[end - 1] = True # for each subgraph, last 1 vertex is layout

        rel_inds = torch.cat(rel_inds, 0)
        rel_masks = torch.cat(rel_masks, 0)

        subj_pred_map = torch.zeros(n_vert, rel_inds.shape[0]) # [sum(Ni + 1), sum((Ni + 1) ** 2)]
        obj_pred_map = torch.zeros(n_vert, rel_inds.shape[0])
        # map from subject (an object or layout vertex) to predicate (a relation vertex)
        subj_pred_map.scatter_(0, (rel_inds[:, 0].view(1, -1)), 1)
        # map from object (an object or layout vertex) to predicate (a relation vertex)
        obj_pred_map.scatter_(0, (rel_inds[:, 1].view(1, -1)), 1)

        return [t.to(device) for t in (rel_masks, obj_masks, lo_masks,
                                       obj_obj_map, subj_pred_map, obj_pred_map)]

    def forward(self, data):
        split = data['objs']['split']
        maps = self._get_map(split)
        rel_masks, obj_masks, lo_masks, obj_obj_map, subj_pred_map, obj_pred_map = maps

        x_obj = self._get_object_features(data, self.obj_features)
        x_pred = self._get_relation_features(data)
        x_obj, x_pred = self.obj_embedding(x_obj), self.rel_embedding(x_pred)
        x_lo = self._get_object_features(data, self.lo_features)
        x_lo = self.lo_embedding(x_lo)

        xs_obj_lo = [] # representation of object and layout vertices
        xs_pred_objlo = [] # representation of relation vertices connecting obj/lo vertices
        rel_pair = torch.cat([torch.tensor([0], device=split.device), torch.cumsum(
            torch.pow(split[:, 1] - split[:, 0], 2), 0)], 0)
        for i_scene, (start, end) in enumerate(split):
            xs_obj_lo.append(x_obj[start:end]) # for each subgraph, first Ni vertices are objects
            xs_obj_lo.append(x_lo[i_scene:i_scene+1]) # for each subgraph, last 1 vertex is layout
            n_objs = end - start
            if n_objs > 0:
                x_pred_objlo = x_pred[rel_pair[i_scene]:rel_pair[i_scene + 1]].reshape(n_objs, n_objs, -1)
                x_pred_objlo = F.pad(
                    x_pred_objlo.permute(2, 0, 1),
                    [0, 1, 0, 1],
                    "constant",
                    self.rel_obj_lo
                ).permute(1, 2, 0)
            else:
                x_pred_objlo = torch.ones([1, 1, self.feature_dim], device=start.device) * self.rel_obj_lo
            x_pred_objlo = x_pred_objlo.reshape((n_objs + 1) ** 2, -1)
            xs_pred_objlo.append(x_pred_objlo)
        x_obj = torch.cat(xs_obj_lo) # from here, for compatibility with graph-rcnn, x_obj corresponds to obj/lo vertices
        x_pred = torch.cat(xs_pred_objlo)
        x_pred = x_pred[rel_masks]

        obj_feats, _ = self.message_passing(obj_obj_map, obj_pred_map, subj_pred_map, x_obj, x_pred)

        obj_feats_wolo = obj_feats[-1][obj_masks]
        objs_output = self.bdb3d_from_obj_feature(data, obj_feats_wolo)
        if self.refine_layout:
            obj_feats_lo = obj_feats[-1][lo_masks]
            layout_output = self.layout_from_lo_feature(data, obj_feats_lo)
            return objs_output, layout_output
        else:
            return objs_output


@MODULES.register_module
class RelationSGCN(SceneGCN):

    def __init__(self, cfg, optim_spec=None):
        super(RelationSGCN, self).__init__(cfg, optim_spec)

        self.rel_obj_wall = 0.001
        self.OBJ_ROT_BIN = len(data_config.metadata['rot_bins'])
        self.output_bdb3d = self.model_config['output_bdb3d']
        self.output_relation = self.model_config['output_relation']
        self.output_label = self.model_config.get('output_label', False)
        self.relation_adjust = self.model_config['relation_adjust']
        self.visualize_adjust = self.model_config['visualize_adjust']
        self.score_weighted = self.model_config['score_weighted']
        self.score_thres = self.model_config['score_thres']
        self.optimize_steps = self.model_config['optimize_steps']
        self.optimize_lr = self.model_config['optimize_lr']
        self.optimize_momentum = self.model_config['optimize_momentum']
        self.toleration_dis = self.model_config['toleration_dis']
        self.loss_weights = self.model_config.get('loss_weights')

        if self.output_bdb3d:
            # representation to object output (from Total3D object_detection)
            self.construct_bdb3d_branch()

        if self.output_relation:
            # representation to relation output
            self.construct_relation_branch()

        if self.output_label:
            # representation to object label
            self.construct_label_branch()

        self.initiate_weights()

    def construct_label_branch(self):
        '''representation to object label'''
        self.fc_obj_label_1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_obj_label_2 = nn.Linear(self.feature_dim // 2, len(IG56CLASSES) + 1)

    def construct_relation_branch(self):
        '''representation to relation output'''

        # branch to predict the obj_obj_rot
        self.fc_obj_obj_rot_1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_obj_obj_rot_2 = nn.Linear(self.feature_dim // 2, self.OBJ_ROT_BIN)

        # branch to predict the obj_obj_dis
        self.fc_obj_obj_dis_1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_obj_obj_dis_2 = nn.Linear(self.feature_dim // 2, 1)

        # branch to predict the obj_obj_tch
        self.fc_obj_obj_tch_1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_obj_obj_tch_2 = nn.Linear(self.feature_dim // 2, 1)

        # branch to predict the obj_wall_rot
        self.fc_obj_wall_rot_1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_obj_wall_rot_2 = nn.Linear(self.feature_dim // 2, self.OBJ_ROT_BIN)

        # branch to predict the obj_wall_tch
        self.fc_obj_wall_tch_1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_obj_wall_tch_2 = nn.Linear(self.feature_dim // 2, 1)

        # branch to predict the floor_tch
        self.fc_obj_floor_tch_1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_obj_floor_tch_2 = nn.Linear(self.feature_dim // 2, 1)

        # branch to predict the ceil_tch
        self.fc_obj_ceil_tch_1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_obj_ceil_tch_2 = nn.Linear(self.feature_dim // 2, 1)

        # branch to predict the in_room
        self.fc_obj_in_room_1 = nn.Linear(self.feature_dim, self.feature_dim // 2)
        self.fc_obj_in_room_2 = nn.Linear(self.feature_dim // 2, 1)

    def _get_map(self, objs_split, walls_split):
        device = objs_split.device
        n_verts = objs_split[-1][-1] + walls_split[-1][-1] # number of objects and walls
        obj_obj_map = torch.zeros([n_verts, n_verts]) # mapping of obj/wall vertices with connections
        rel_inds = [] # indexes of vertices connected by relation nodes
        rel_masks = [] # mask of relation features for obj/lo vertices connected by relation nodes
        obj_masks = torch.zeros(n_verts, dtype=torch.bool) # mask of object vertices
        wall_masks = torch.zeros(n_verts, dtype=torch.bool) # mask of layout vertices
        for (wall_start, wall_end), (obj_start, obj_end) in zip(walls_split, objs_split):
            start = obj_start + wall_start # each subgraph has Ni object vertices and Mi wall vertices
            end = obj_end + wall_end # consider layout vertex, Ni + Mi vertices in total
            n_objs = obj_end - obj_start
            obj_obj_map[start:end, start:end] = 1 # each subgraph is a complete graph with self circle
            obj_ind = torch.arange(start, end, dtype=torch.long)
            subj_ind_i, obj_ind_i = torch.meshgrid(obj_ind, obj_ind) # indexes of each vertex in the subgraph
            rel_ind_i = torch.stack([subj_ind_i.reshape(-1), obj_ind_i.reshape(-1)], -1)
            rel_mask_i = rel_ind_i[:, 0] != rel_ind_i[:, 1] # vertices connected by relation nodes should be different
            rel_inds.append(rel_ind_i[rel_mask_i])
            rel_masks.append(rel_mask_i)
            obj_masks[start:start + n_objs] = True # for each subgraph, first Ni vertices are objects
            wall_masks[start + n_objs: end] = True # for each subgraph, last Mi vertices are walls

        rel_inds = torch.cat(rel_inds, 0)
        rel_masks = torch.cat(rel_masks, 0)

        subj_pred_map = torch.zeros(n_verts, rel_inds.shape[0]) # [sum(Ni + Mi), sum((Ni + Mi) ** 2)]
        obj_pred_map = torch.zeros(n_verts, rel_inds.shape[0])
        # map from subject (an object or wall vertex) to predicate (a relation vertex)
        subj_pred_map.scatter_(0, (rel_inds[:, 0].view(1, -1)), 1)
        # map from object (an object or wall vertex) to predicate (a relation vertex)
        obj_pred_map.scatter_(0, (rel_inds[:, 1].view(1, -1)), 1)

        return [t.to(device) for t in (rel_inds, rel_masks, obj_masks, wall_masks,
                                       obj_obj_map, subj_pred_map, obj_pred_map)]

    def _get_relation_features(self, data):
        features = []
        for key in self.rel_features:
            objs = data['objs']
            walls = data['walls']
            if key in ('g_feature', 'iou'):
                bdb2d_objs = dict_of_array_to_list_of_dict(objs['bdb2d'], split=objs['split'])
                bdb2d_walls = dict_of_array_to_list_of_dict(walls['bdb2d'], split=walls['split'])
                bdb2d_objwalls = [bdb2d_objs_scene + bdb2d_walls_scene
                                  for bdb2d_objs_scene, bdb2d_walls_scene in zip(bdb2d_objs, bdb2d_walls)]
                if key == 'g_feature':
                    g_features = [bdb2d_geometric_feature(
                        bdb2ds, self.cfg.config['data']['g_feature_length']
                    ) for bdb2ds in bdb2d_objwalls]
                    v = torch.cat(g_features).to(objs['split'].device)
                else:
                    v = []
                    for bdb2ds in bdb2d_objwalls:
                        ious = torch.zeros([len(bdb2ds), len(bdb2ds)], device=objs['split'].device)
                        for i_a, bdb2d_a in enumerate(bdb2ds):
                            for i_b, bdb2d_b in enumerate(bdb2ds):
                                ious[i_a, i_b] = bdb2d_iou(bdb2d_a, bdb2d_b)
                        v.append(ious.view(-1))
                    v = torch.cat(v)
            elif key == 'bdb3d_test':
                transform = IGTransform(data)
                obj_bdb3ds = bins2bdb3d(data)
                obj_bdb3ds = transform.campix2world(obj_bdb3ds)
                wall_bdb3ds = data['walls']['bdb3d']
                vs = []
                for (wall_start, wall_end), (obj_start, obj_end) \
                        in zip(walls['split'], objs['split']):
                    all_bdb3d = {k: torch.cat([obj_bdb3ds[k][obj_start:obj_end], wall_bdb3ds[k][wall_start:wall_end]])
                                 for k in ('centroid', 'basis', 'size')}
                    has_collision, collision_err, touch_err = test_bdb3ds(all_bdb3d)
                    vs.append(torch.stack([has_collision, collision_err, touch_err], -1).view(-1, 3))
                v = torch.cat(vs, 0)
            elif key == 'rel_rot':
                obj_bdb3ds = bins2bdb3d(data)
                wall_bdb3ds = data['walls']['bdb3d']
                bins = data_config.metadata['rot_bins']
                bins_width = bins[1] - bins[0]
                bins = np.stack([bins, bins + bins_width]).T
                features_rel = []
                for (wall_start, wall_end), (obj_start, obj_end) \
                        in zip(walls['split'], objs['split']):
                    all_ori = torch.cat([
                        obj_bdb3ds['ori'][obj_start:obj_end],
                        wall_bdb3ds['ori'][wall_start:wall_end]
                    ])
                    ori_b = all_ori.expand(len(all_ori), -1)
                    ori_a = ori_b.T
                    obj_obj_rot = torch.abs(torch.remainder(ori_a - ori_b, np.pi * 2))
                    obj_obj_rot = obj_obj_rot.view(-1)
                    cls, reg = num2bins(bins, obj_obj_rot)
                    onehot = torch.zeros([len(obj_obj_rot), len(bins)], device=obj_obj_rot.device)
                    onehot[range(len(cls)), cls] = 1
                    features_rel.append(torch.cat([onehot, reg[:, None], obj_obj_rot[:, None]], -1))
                v = torch.cat(features_rel, 0)
            else:
                vs = []
                for vtype in ('objs', 'walls'):
                    vs.append(self._get_single_feature(data, f"{vtype}.{key}"))
                v_objs, v_walls = vs

                # reshape relation feature
                features_rel = []
                for (wall_start, wall_end), (obj_start, obj_end) \
                        in zip(walls['split'], objs['split']):
                    start = obj_start + wall_start # each subgraph has Ni object vertices and Mi wall vertices
                    end = obj_end + wall_end # consider layout vertex, Ni + Mi vertices in total
                    if end - start > 0:
                        v_scene = torch.cat([v_objs[obj_start:obj_end], v_walls[wall_start: wall_end]])
                        features_rel.append(torch.stack(
                            [torch.cat([loc1, loc2], -1)
                             for loc1 in v_scene
                             for loc2 in v_scene]))
                v = torch.cat(features_rel, 0)
            if isinstance(v, torch.Tensor) and v.dim() == 1:
                v = v[:, None]
            assert self.feature_length[key] == v.shape[-1], f"length of feature '{key}' does not match"
            features.append(v)

        return torch.cat(features, -1)

    def forward(self, data):
        objs_split = data['objs']['split']
        walls_split = data['walls']['split']
        maps = self._get_map(objs_split, walls_split)
        rel_inds, rel_masks, obj_masks, wall_masks, obj_obj_map, subj_pred_map, obj_pred_map = maps

        x_obj = self._get_object_features(data, self.obj_features)
        x_pred = self._get_relation_features(data)
        x_obj, x_pred = self.obj_embedding(x_obj), self.rel_embedding(x_pred)
        x_wall = self._get_object_features(data, self.lo_features)
        x_wall = self.lo_embedding(x_wall)

        xs_obj_wall = []  # representation of object and wall vertices
        for (wall_start, wall_end), (obj_start, obj_end) \
                in zip(walls_split, objs_split):
            xs_obj_wall.append(x_obj[obj_start:obj_end])
            xs_obj_wall.append(x_wall[wall_start:wall_end])
        x_obj = torch.cat(xs_obj_wall) # from here, for compatibility with graph-rcnn, x_obj corresponds to obj/wall vertices
        x_pred = x_pred[rel_masks]

        obj_feats, pred_feats = self.message_passing(obj_obj_map, obj_pred_map, subj_pred_map, x_obj, x_pred)

        objs_output = defaultdict(dict)
        obj_feats_wowall = obj_feats[-1][obj_masks]

        if self.output_bdb3d:
            objs_output.update(self.bdb3d_from_obj_feature(data, obj_feats_wowall))

        if self.output_label:
            cls_code = self.fc_obj_label_1(obj_feats_wowall)
            cls_code = self.relu(cls_code)
            cls_code = self.dropout(cls_code)
            cls_code = self.fc_obj_label_2(cls_code)
            objs_output['cls_code'] = cls_code
            cls_code = torch.softmax(cls_code, dim=-1)
            objs_output['score'], objs_output['label'] = cls_code.max(-1)

        if self.output_relation:
            '''representation to relation output'''
            pred_feats = pred_feats[-1]
            objs_rel_output, relation_output = self.relation_from_pred_feature(
                data, rel_inds, obj_feats_wowall, pred_feats)
            objs_output.update(objs_rel_output)

            if self.relation_adjust:
                relation_label = relation_from_bins(
                    {'objs': objs_output, 'relation': relation_output}, self.score_thres)

                # copy another scene data for optimization
                optim_data = data.copy()
                optim_objs = optim_data['objs'] = data['objs'].copy()
                optim_objs.update(objs_output)
                optim_data['relation'] = relation_label['relation']
                optim_objs.update(relation_label['objs'])

                # transform parameterized bdb3d to bdb3d_pix
                optim_trans = IGTransform(optim_data)
                bdb3d_optim_pix = bins2bdb3d(optim_data)
                bdb3d_optim_pix.update(optim_trans.campix2world(bdb3d_optim_pix))
                optim_objs['bdb3d'] = bdb3d_optim_pix

                patches = split_batch_into_patches(optim_data)
                optim_bdb3ds = []
                for patch in patches:
                    if len(patch['objs']['label']) == 0:
                        continue
                    # optimize
                    relation_optimization = RelationOptimization(
                        loss_weights=self.loss_weights,
                        visual_path=os.path.join(
                            self.cfg.save_path, 'relation_adjust', patch['scene'][0], patch['name'][0]),
                        toleration_dis=self.toleration_dis,
                        score_weighted=self.score_weighted,
                        score_thres=self.score_thres
                    )
                    optim_bdb3d = relation_optimization.optimize(
                        patch,
                        visual=self.visualize_adjust,
                        steps=self.optimize_steps,
                        lr=self.optimize_lr,
                        momentum=self.optimize_momentum
                    )
                    optim_bdb3ds.append(optim_bdb3d)

                optim_bdb3ds = list_of_dict_to_dict_of_array(optim_bdb3ds, force_cat=True)
                objs_output['bdb3d'].update(optim_bdb3ds)

            return dict(objs_output), relation_output

        return dict(objs_output)

    def relation_from_pred_feature(self, data, rel_inds, obj_feats_wowall, pred_feats):
        '''representation to relation output'''
        relation = {}

        # branch to predict the obj_obj_rot
        obj_obj_rot = self.fc_obj_obj_rot_1(pred_feats)
        obj_obj_rot = self.relu(obj_obj_rot)
        obj_obj_rot = self.dropout(obj_obj_rot)
        relation['obj_obj_rot'] = self.fc_obj_obj_rot_2(obj_obj_rot)

        # branch to predict the obj_obj_dis
        obj_obj_dis = self.fc_obj_obj_dis_1(pred_feats)
        obj_obj_dis = self.relu(obj_obj_dis)
        obj_obj_dis = self.dropout(obj_obj_dis)
        relation['obj_obj_dis'] = self.fc_obj_obj_dis_2(obj_obj_dis)

        # branch to predict the obj_obj_tch
        obj_obj_tch = self.fc_obj_obj_tch_1(pred_feats)
        obj_obj_tch = self.relu(obj_obj_tch)
        obj_obj_tch = self.dropout(obj_obj_tch)
        relation['obj_obj_tch'] = self.fc_obj_obj_tch_2(obj_obj_tch)

        # branch to predict the obj_wall_rot
        obj_wall_rot = self.fc_obj_wall_rot_1(pred_feats)
        obj_wall_rot = self.relu(obj_wall_rot)
        obj_wall_rot = self.dropout(obj_wall_rot)
        relation['obj_wall_rot'] = self.fc_obj_wall_rot_2(obj_wall_rot)

        # branch to predict the obj_wall_tch
        obj_wall_tch = self.fc_obj_wall_tch_1(pred_feats)
        obj_wall_tch = self.relu(obj_wall_tch)
        obj_wall_tch = self.dropout(obj_wall_tch)
        relation['obj_wall_tch'] = self.fc_obj_wall_tch_2(obj_wall_tch)

        # from pred output to matrix
        objs_split = data['objs']['split']
        walls_split = data['walls']['split']
        device = pred_feats.device
        n_verts = objs_split[-1][-1] + walls_split[-1][-1]  # number of objects and walls
        rel_list = [{} for i in range(len(objs_split))]
        for k in relation.keys():
            rel = relation[k]
            rel_mat = torch.zeros([n_verts, n_verts, rel.shape[-1]], device=device)
            rel_mat[rel_inds[:, 0], rel_inds[:, 1]] = rel
            if 'rot' in k:
                rel_mat = (rel_mat + torch.flip(rel_mat.transpose(0, 1), (-1, ))) / 2
            elif k == 'obj_obj_dis':
                rel_mat = (rel_mat - rel_mat.transpose(0, 1)) / 2
            else:
                rel_mat = (rel_mat + rel_mat.transpose(0, 1)) / 2
            for i_scene, ((wall_start, wall_end), (obj_start, obj_end)) in enumerate(zip(walls_split, objs_split)):
                start = obj_start + wall_start # each subgraph has Ni object vertices and Mi wall vertices
                end = obj_end + wall_end # consider layout vertex, Ni + Mi vertices in total
                n_objs = obj_end - obj_start

                if k.startswith('obj_obj'):
                    rel_list[i_scene][k] = rel_mat[start:start + n_objs, start:start + n_objs]
                else:
                    rel_list[i_scene][k] = rel_mat[start:start + n_objs, start + n_objs:end]

        # branch to predict the floor_tch
        floor_tch = self.fc_obj_floor_tch_1(obj_feats_wowall)
        floor_tch = self.relu(floor_tch)
        floor_tch = self.dropout(floor_tch)
        floor_tch = self.fc_obj_floor_tch_2(floor_tch)

        # branch to predict the ceil_tch
        ceil_tch = self.fc_obj_ceil_tch_1(obj_feats_wowall)
        ceil_tch = self.relu(ceil_tch)
        ceil_tch = self.dropout(ceil_tch)
        ceil_tch = self.fc_obj_ceil_tch_2(ceil_tch)

        # branch to predict the in_room
        in_room = self.fc_obj_in_room_1(obj_feats_wowall)
        in_room = self.relu(in_room)
        in_room = self.dropout(in_room)
        in_room = self.fc_obj_in_room_2(in_room)

        objs = {
            'floor_tch': floor_tch,
            'ceil_tch': ceil_tch,
            'in_room': in_room,
        }

        return objs, rel_list

