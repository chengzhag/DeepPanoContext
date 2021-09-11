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


@LOSSES.register_module
class AffinityLoss(BaseLoss):
    def __call__(self, est_data, gt_data):
        est_affinity, gt_affinity = est_data['affinity'], gt_data['affinity']
        positive = gt_affinity > 0.5
        negative = gt_affinity <= 0.5
        affinity_loss = ((est_affinity[positive] - gt_affinity[positive]) ** 2).sum() / positive.sum() \
                        + ((est_affinity[negative] - gt_affinity[negative]) ** 2).sum() / negative.sum()
        return {'affinity_loss': affinity_loss}
