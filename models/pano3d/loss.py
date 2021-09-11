import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from configs.data_config import IG56CLASSES
from models.loss import BaseLoss
from models.registers import LOSSES
from utils.basic_utils import list_of_dict_to_dict_of_array, recursively_to
from utils.igibson_utils import IGScene
from utils.layout_utils import manhattan_2d_from_manhattan_world_layout
from utils.transform_utils import IGTransform, bins2bdb3d, bdb3d_corners, points2bdb2d, point_polygon_dis

cls_criterion = nn.CrossEntropyLoss(reduction='mean')
reg_criterion = nn.SmoothL1Loss(reduction='mean')
mse_criterion = nn.MSELoss(reduction='mean')
binary_cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')


def cls_reg_loss(cls_result, cls_gt, reg_result, reg_gt):
    cls_loss = cls_criterion(cls_result, cls_gt)
    if len(reg_result.size()) == 3:
        reg_result = torch.gather(
            reg_result, 1,
            cls_gt.view(reg_gt.size(0), 1, 1).expand(reg_gt.size(0), 1, reg_gt.size(1))
        )
    else:
        reg_result = torch.gather(
            reg_result, 1,
            cls_gt.view(reg_gt.size(0), 1).expand(reg_gt.size(0), 1)
        )
    reg_result = reg_result.squeeze(1)
    reg_loss = reg_criterion(reg_result, reg_gt)
    return cls_loss, reg_loss


@LOSSES.register_module
class Bdb3DLoss(BaseLoss):
    def __call__(self, est_data, gt_data):
        est_objs, gt_objs = est_data['objs'], gt_data['objs']
        if est_data['objs']:
            id_gt = est_objs['gt']
            id_match = id_gt >= 0
            id_gt = id_gt[id_match]
            est_bdb3d, gt_bdb3d = est_objs['bdb3d'], gt_objs['bdb3d']
            # only consider detector predictions matched with ground truth
            size_reg_loss = reg_criterion(est_bdb3d['size_reg'][id_match], gt_bdb3d['size_reg'][id_gt])
            ori_cls_loss, ori_reg_loss = cls_reg_loss(
                est_bdb3d['ori_cls'][id_match], gt_bdb3d['ori_cls'][id_gt],
                est_bdb3d['ori_reg'][id_match], gt_bdb3d['ori_reg'][id_gt])
            dis_cls_loss, dis_reg_loss = cls_reg_loss(
                est_bdb3d['dis_cls'][id_match], gt_bdb3d['dis_cls'][id_gt],
                est_bdb3d['dis_reg'][id_match], gt_bdb3d['dis_reg'][id_gt])
            delta2d_loss = reg_criterion(est_objs['delta2d'][id_match], gt_objs['delta2d'][id_gt])
        else:
            size_reg_loss = ori_cls_loss = ori_reg_loss = dis_cls_loss = dis_reg_loss = delta2d_loss = 0.
        return {'size_reg_loss': size_reg_loss,
                'ori_cls_loss': ori_cls_loss, 'ori_reg_loss': ori_reg_loss,
                'dis_cls_loss': dis_cls_loss, 'dis_reg_loss': dis_reg_loss,
                'delta2d_loss': delta2d_loss}


@LOSSES.register_module
class JointLoss(Bdb3DLoss):
    def __call__(self, est_data, gt_data):
        loss_dict = super(JointLoss, self).__call__(est_data, gt_data)

        est_objs, gt_objs = est_data['objs'], gt_data['objs']
        id_gt = est_objs['gt']
        id_match = id_gt >= 0
        est_bdb3ds = bins2bdb3d(est_data)
        transform = IGTransform(est_data)

        # 3D bounding box corner loss
        gt_bdb3ds = gt_data['objs']['bdb3d']
        est_bdb3ds = transform.campix2world(est_bdb3ds)
        gt_bdb3d_corners = bdb3d_corners(gt_bdb3ds)
        est_bdb3d_corners = bdb3d_corners(est_bdb3ds)
        loss_dict['corner_loss'] = reg_criterion(
            est_bdb3d_corners[id_match], gt_bdb3d_corners[id_gt[id_match]])

        # 2D bdb loss
        # construct transformations to corresponding gt
        est_scenes = IGScene.from_batch(est_data, gt_data)
        transforms_to_gt_bdb3d = IGTransform()
        transforms_to_gt_bdb3d.camera = []
        for est_scene, (gt_start, _) in zip(est_scenes, gt_data['objs']['split']):
            for obj in est_scene['objs']:
                if obj['gt'] < 0:
                    transforms_to_gt_bdb3d.camera.append(est_scene['camera'])
                else:
                    target = gt_bdb3ds['centroid'][obj['gt'] + gt_start].cpu().numpy()
                    transform = IGTransform.level_look_at(est_scene.data, target)
                    transform.data = None
                    transforms_to_gt_bdb3d.camera.append(transform.camera)
        transforms_to_gt_bdb3d.camera = list_of_dict_to_dict_of_array(transforms_to_gt_bdb3d.camera, to_tensor=True)
        transforms_to_gt_bdb3d.camera = recursively_to(transforms_to_gt_bdb3d.camera, device=id_gt.device)

        # transform estimated bdb3d into bdb2d
        est_bdb3d_corners_pix = transforms_to_gt_bdb3d.world2campix(est_bdb3d_corners)
        bdb2d_from_est_bdb3d = points2bdb2d(est_bdb3d_corners_pix)
        bdb2d_from_gt_bdb3d = gt_objs['bdb2d_from_bdb3d']

        # stack bdb2d parameters together
        bdb2d_from_est_bdb3d_t = []
        bdb2d_from_gt_bdb3d_t = []
        for k in bdb2d_from_est_bdb3d:
            factor = transform['width'] if 'x' in k else transform['height']
            bdb2d_from_est_bdb3d_t.append(bdb2d_from_est_bdb3d[k] / factor)
            bdb2d_from_gt_bdb3d_t.append(bdb2d_from_gt_bdb3d[k] / factor)
        bdb2d_from_est_bdb3d_t = torch.stack(bdb2d_from_est_bdb3d_t, -1)
        bdb2d_from_gt_bdb3d_t = torch.stack(bdb2d_from_gt_bdb3d_t, -1)

        loss_dict['bdb2D_loss'] = reg_criterion(
            bdb2d_from_est_bdb3d_t[id_match], bdb2d_from_gt_bdb3d_t[id_gt[id_match]])

        # physical violation loss
        dis = []
        for i_scene, (start, end) in enumerate(est_objs['split']):
            if 'layout' not in est_data:
                continue
            layout = est_data['layout']['manhattan_world'][i_scene]
            layout_2d = manhattan_2d_from_manhattan_world_layout(layout)
            corners_2d = est_bdb3d_corners[start:end, :, :2].reshape(-1, 2)
            dis_2d = torch.relu(point_polygon_dis(corners_2d, layout_2d))

            layout_z = layout[:, -1]
            floor = layout_z.min()
            ceil = layout_z.max()
            corners_z = est_bdb3d_corners[start:end, :, -1].reshape(-1)
            dis_floor = torch.relu(floor - corners_z)
            dis_ceil = torch.relu(corners_z - ceil)

            dis.append(torch.cat([dis_2d, dis_floor, dis_ceil]))
        if dis:
            dis = torch.cat(dis)
            loss_dict['phy_loss'] = mse_criterion(dis, torch.zeros_like(dis, device=dis.device))

        return loss_dict


@LOSSES.register_module
class HorizonLoss(BaseLoss):
    def __call__(self, est_data, gt_data):
        layout_output, layout_gt = est_data['layout']['horizon'], gt_data['layout']['horizon']
        return {'bon_loss': F.l1_loss(layout_output['bon'], layout_gt['bon']),
                'cor_loss': F.binary_cross_entropy_with_logits(layout_output['cor'], layout_gt['cor'])}


@LOSSES.register_module
class RelationLoss(BaseLoss):
    def __call__(self, est_data, gt_data):
        loss_dict = {}

        gt_objs = gt_data['objs']
        if est_data['objs']:
            est_objs = est_data['objs']
            id_gt = est_objs['gt']
            id_match = id_gt >= 0

            # object label
            if est_objs['cls_code'].shape[-1] == len(IG56CLASSES) + 1:
                est_code = est_objs['cls_code']
                gt_label = torch.zeros(len(est_code), device=est_code.device, dtype=torch.long)
                gt_label[:] = len(IG56CLASSES)
                gt_label[id_match] = gt_objs['label'][id_gt[id_match]]
                loss_dict['label_loss'] = cls_criterion(est_code, gt_label)

            # optimized bdb3d pix
            if 'ori' in est_objs['bdb3d']:
                for k in ('center', 'dis', 'ori', 'size'):
                    est_v = est_objs['bdb3d'][k][id_match]
                    gt_v = gt_objs['bdb3d'][k][id_gt[id_match]]
                    if k == 'center':
                        # normalize center in pix with image size
                        transform = IGTransform(est_data)
                        wh = torch.stack([transform['width'], transform['height']], -1)[id_match]
                        est_v = est_v / wh * 10
                        gt_v = gt_v / wh * 10
                    if k == 'ori':
                        est_v = torch.remainder(est_v - gt_v, np.pi * 2)
                        est_v[est_v > np.pi] = 2 * np.pi - est_v[est_v > np.pi]
                        gt_v = torch.zeros_like(est_v, device=est_v.device)
                    loss_dict[k + '_loss'] = reg_criterion(est_v, gt_v)

        # object single relation
        for k in ('floor_tch', 'ceil_tch', 'in_room'):
            loss_name = k + '_loss'
            if est_data['objs'] and id_match.any():
                est_rel = est_objs[k][id_match]
                gt_rel = gt_objs[k][id_gt[id_match]]
                loss_dict[loss_name] = binary_cls_criterion(est_rel[:, 0], gt_rel.type(torch.float))
            else:
                loss_dict[loss_name] = 0.

        # object pairwise relation
        for k in ('obj_obj_rot', 'obj_obj_dis', 'obj_obj_tch', 'obj_wall_rot', 'obj_wall_tch'):
            loss_name = k + '_loss'

            est_rel = []
            for scene_rel, (start, end) in zip(est_data['relation'], est_objs['split']):
                scene_rel = scene_rel[k]
                if end - start == 0:
                    continue
                scene_id_match = id_match[start:end]
                if scene_id_match.any():
                    scene_rel = scene_rel[scene_id_match]
                    if k.startswith('obj_obj'):
                        scene_rel = scene_rel[:, scene_id_match]
                    est_rel.append(scene_rel.view(-1, scene_rel.shape[-1]))
            if len(est_rel) == 0:
                loss_dict[loss_name] = 0.
                continue
            est_rel = torch.cat(est_rel)

            gt_rel = []
            for scene_rel, (est_start, est_end), (gt_start, gt_end) \
                    in zip(gt_data['relation'], est_objs['split'], gt_objs['split']):
                scene_rel = scene_rel[k]
                if est_end - est_start == 0:
                    continue
                scene_id_gt = id_gt[est_start:est_end][id_match[est_start:est_end]] - gt_start
                if len(scene_id_gt) > 0:
                    scene_rel = scene_rel[scene_id_gt]
                    if k.startswith('obj_obj'):
                        scene_rel = scene_rel[:, scene_id_gt]
                    gt_rel.append(scene_rel.view(-1))
            gt_rel = torch.cat(gt_rel)

            if 'rot' in k:
                loss_dict[loss_name] = cls_criterion(est_rel, gt_rel.type(torch.long))
            else:
                loss_dict[loss_name] = binary_cls_criterion(est_rel[..., 0], gt_rel.type(torch.float))

        return loss_dict
