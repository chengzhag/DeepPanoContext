import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from configs.data_config import IG56CLASSES
from models.loss import BaseLoss
from models.pano3d.loss import cls_reg_loss
from models.registers import LOSSES
from utils.basic_utils import list_of_dict_to_dict_of_array, recursively_to
from utils.igibson_utils import IGScene
from utils.layout_utils import manhattan_2d_from_manhattan_world_layout
from utils.transform_utils import IGTransform, bins2bdb3d, bdb3d_corners, points2bdb2d, point_polygon_dis

reg_criterion = nn.SmoothL1Loss(reduction='mean')


@LOSSES.register_module
class PoseLoss(BaseLoss):
    def __call__(self, est_data, gt_data):
        est_layout, gt_layout = est_data['layout']['total3d'], gt_data['layout']['total3d']
        pitch_cls_loss, pitch_reg_loss = cls_reg_loss(
            est_layout['pitch_cls'], gt_layout['pitch_cls'], est_layout['pitch_reg'], gt_layout['pitch_reg'])
        roll_cls_loss, roll_reg_loss = cls_reg_loss(
            est_layout['roll_cls'], gt_layout['roll_cls'], est_layout['roll_reg'], gt_layout['roll_reg'])
        lo_ori_cls_loss, lo_ori_reg_loss = cls_reg_loss(
            est_layout['ori_cls'], gt_layout['ori_cls'], est_layout['ori_reg'], gt_layout['ori_reg'])
        lo_centroid_loss = reg_criterion(
            est_layout['centroid_reg'], gt_layout['centroid_reg'])
        lo_size_loss = reg_criterion(
            est_layout['size_reg'], gt_layout['size_reg'])

        # layout bounding box corner loss
        est_corners, gt_corners = bdb3d_corners(est_layout), bdb3d_corners(gt_layout)
        lo_corner_loss = reg_criterion(est_corners, gt_corners)

        return {'pitch_cls_loss':pitch_cls_loss, 'pitch_reg_loss':pitch_reg_loss,
                'roll_cls_loss':roll_cls_loss, 'roll_reg_loss':roll_reg_loss,
                'lo_ori_cls_loss':lo_ori_cls_loss, 'lo_ori_reg_loss':lo_ori_reg_loss,
                'lo_centroid_loss':lo_centroid_loss, 'lo_size_loss':lo_size_loss,
                'lo_corner_loss':lo_corner_loss}

