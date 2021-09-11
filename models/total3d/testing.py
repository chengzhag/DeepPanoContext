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
        _, gt_data, _, gt_scenes = data
        do_evaluation = not any(s.empty() for s in gt_scenes)

        # LEN
        if do_evaluation and 'layout' in gt_data and 'layout' in est_data:
            est_scenes = IGScene.from_batch(est_data, gt_data)

            metric_iou = AverageMeter()
            metric_cam = defaultdict(AverageMeter)
            for est_scene, gt_scene in zip(est_scenes, gt_scenes):

                # evaluate Layout IoU
                est_layout = est_scene['layout']['total3d']
                gt_layout = gt_scene['layout']['cuboid_world']
                est_corners, gt_corners = bdb3d_corners(est_layout), bdb3d_corners(gt_layout)
                metric_iou.append(bdb3d_iou(est_corners, gt_corners))

                # evaluate camera pose
                est_pose = est_scene.transform.get_camera_angle()[1:]
                gt_pose = gt_scene.transform.get_camera_angle()[1:]
                for k, est_angle, gt_angle in zip(('pitch', 'roll'), est_pose, gt_pose):
                    cam_err = np.rad2deg(np.abs(est_angle - gt_angle))
                    metric_cam[f"cam_{k}_err"] = cam_err

            metrics['layout_3DIoU'] = metric_iou
            metrics.update(metric_cam)

            if self.cfg.config['full']:
                gt_scenes = IGScene.from_batch(gt_data)
                lo_statistics = defaultdict(AverageMeter)
                for est_scene, gt_scene in zip(est_scenes, gt_scenes):
                    # parameterized output statistics
                    for k, est_angle, gt_angle in zip(('pitch', 'roll'), est_pose, gt_pose):
                        lo_statistics[f"est_{k}"].append(est_angle)
                        lo_statistics[f"gt_{k}"].append(gt_angle)
                    for label, layout in (('est', est_layout), ('gt', gt_scene['layout']['total3d'])):
                        for k, v in layout.items():
                            if np.ndim(v) > 1:
                                continue
                            if np.ndim(v) == 0:
                                lo_statistics[f"{label}_{k}"].append(v)
                            else:
                                for i, num in enumerate(v):
                                    lo_statistics[f"{label}_{k}_{i}"].append(num)
                metrics.update(lo_statistics)

        # BEN
        if do_evaluation and 'objs' in est_data and est_data['objs'] and 'bdb3d' in est_data['objs']:
            est_scenes = IGScene.from_batch(est_data, gt_data)
            metrics.update(self.eval_BEN(est_scenes, gt_scenes))

        if 'save_as_dataset' in self.cfg.config['log']:
            est_scenes = IGScene.from_batch(est_data, gt_data, full=True)
            self.save_as_dataset(est_scenes, gt_scenes)

        return metrics, est_data
