from collections import defaultdict

from shapely.geometry.polygon import Polygon
from copy import deepcopy
import numpy as np
import wandb

from external.frustum_convnet.train.sunrgbd_eval.eval_det import voc_ap


def _change_key(bbox):
    if 'u1' not in bbox.keys() and 'x1' in bbox.keys():
        bbox = deepcopy(bbox)
        bbox['u1'] = bbox['x1']
        bbox['v1'] = bbox['y1']
        bbox['u2'] = bbox['x2']
        bbox['v2'] = bbox['y2']
        bbox.pop('x1', None)
        bbox.pop('x2', None)
        bbox.pop('y1', None)
        bbox.pop('y2', None)
    return bbox


def bdb3d_iou(cu1, cu2):
    """
        Calculate the Intersection over Union (IoU) of two 3D cuboid.

        Parameters
        ----------
        cu1 : numpy array, 8x3
        cu2 : numpy array, 8x3

        Returns
        -------
        float
            in [0, 1]
    """

    # 2D projection on the horizontal plane (x-y plane)
    polygon2D_1 = Polygon(
        [(cu1[0][0], cu1[0][1]), (cu1[1][0], cu1[1][1]), (cu1[3][0], cu1[3][1]), (cu1[2][0], cu1[2][1])])

    polygon2D_2 = Polygon(
        [(cu2[0][0], cu2[0][1]), (cu2[1][0], cu2[1][1]), (cu2[3][0], cu2[3][1]), (cu2[2][0], cu2[2][1])])

    # from matplotlib import pyplot
    # pyplot.plot(*polygon2D_1.exterior.xy)
    # pyplot.plot(*polygon2D_2.exterior.xy)
    # pyplot.axis('equal')
    # pyplot.show()

    # 2D intersection area of the two projections.
    intersect_2D = polygon2D_1.intersection(polygon2D_2).area

    # the volume of the intersection part of cu1 and cu2
    inter_vol = intersect_2D * max(0.0, min(cu1[4][2], cu2[4][2]) - max(cu1[0][2], cu2[0][2]))

    # the volume of cu1 and cu2
    vol1 = polygon2D_1.area * (cu1[4][2]-cu1[0][2])
    vol2 = polygon2D_2.area * (cu2[4][2]-cu2[0][2])

    # return 3D IoU
    return inter_vol / (vol1 + vol2 - inter_vol)


def bdb2d_iou(bb1, bb2):
    """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'u1', 'v1', 'u2', 'v2'}
            The (u1, v1) position is at the top left corner,
            The (u2, v2) position is at the bottom right corner
        bb2 : dict
            Keys: {'u1', 'v1', 'u2', 'v2'}
            The (u1, v1) position is at the top left corner,
            The (u2, v2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
    """
    bb1 = _change_key(bb1)
    bb2 = _change_key(bb2)

    assert bb1['u1'] <= bb1['u2']
    assert bb1['v1'] <= bb1['v2']
    assert bb2['u1'] <= bb2['u2']
    assert bb2['v1'] <= bb2['v2']

    # determine the coordinates of the intersection rectangle
    u_left = max(bb1['u1'], bb2['u1'])
    v_top = max(bb1['v1'], bb2['v1'])
    u_right = min(bb1['u2'], bb2['u2'])
    v_bottom = min(bb1['v2'], bb2['v2'])

    if u_right < u_left or v_bottom < v_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (u_right - u_left) * (v_bottom - v_top)

    # compute the area of both AABBs
    bb1_area = (bb1['u2'] - bb1['u1']) * (bb1['v2'] - bb1['v1'])
    bb2_area = (bb2['u2'] - bb2['u1']) * (bb2['v2'] - bb2['v1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area + np.finfo(np.float64).eps)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def contour_iou(c1, c2, width=None):
    if width is None:
        c1 = Polygon(np.stack([c1['x'], c1['y']]).T).buffer(0)
        c2 = Polygon(np.stack([c2['x'], c2['y']]).T).buffer(0)
        intersection_area = c1.intersection(c2).area
        iou = intersection_area / (c1.area + c2.area - intersection_area + np.finfo(np.float64).eps)
    else:
        iou = contour_iou(c1, c2)
        iou = max(iou, contour_iou(c1, {'x': c2['x'] + width, 'y': c2['y']}))
        iou = max(iou, contour_iou({'x': c1['x'] + width, 'y': c1['y']}, c2))
    return iou


def rot_err(est, gt):
    err = np.mod(np.abs(est - gt), np.pi * 2)
    if err.size > 1:
        mask = err > np.pi
        err[mask] = np.pi * 2 - err[mask]
    else:
        err = np.pi * 2 - err if err > np.pi else err
    return err


def classification_metric(est_v, gt_v):
    metrics = {}

    TP = (est_v & (est_v == gt_v)).sum().astype(np.float)
    FP = (est_v & (est_v != gt_v)).sum().astype(np.float)
    TN = ((~est_v) & (est_v == gt_v)).sum().astype(np.float)
    FN = ((~est_v) & (est_v != gt_v)).sum().astype(np.float)

    metrics['accuracy'] = (TP + TN) / est_v.size

    if TP + FP > 0:
        precision = metrics['precision'] = TP / (TP + FP)

    if TP + FN > 0:
        recall = metrics['recall'] = TP / (TP + FN)
        metrics['TPR'] = TP / (TP + FN)
        metrics['FNR'] = FN / (TP + FN)

    if FP + TN > 0:
        metrics['FPR'] = FP / (FP + TN)
        metrics['TNR'] = TN / (FP + TN)

    if 'precision' in metrics and 'recall' in metrics and precision + recall > 0:
        metrics['f1'] = 2 * precision * recall / (precision + recall)

    return metrics


class MetricMeter:
    def __call__(self):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError

    def add(self, val):
        if isinstance(val, MetricMeter):
            return val
        else:
            return AverageMeter(val)

    def log(self, name):
        raise NotImplementedError


class AverageMeter(list, MetricMeter):
    def __init__(self, l=()):
        MetricMeter.__init__(self)
        if not isinstance(l, (tuple, list)):
            l = [l]
        list.__init__(self, l)

    def __call__(self):
        return sum(self) / len(self) if len(self) > 0 else None

    def __str__(self):
        return f"{self.__call__():.2f}" if len(self) > 0 else 'nan'

    def add(self, val):
        if isinstance(val, (tuple, list)):
            self.extend(val)
        else:
            self.append(val)
        return self

    def log(self, name):
        metric = self.__call__()
        wandb.summary[f"{name}_avg"] = metric
        wandb.summary[f"{name}_hist"] = wandb.Histogram(self)


class AveragePrecisionMeter(defaultdict, MetricMeter):
    def __init__(self):
        MetricMeter.__init__(self)
        defaultdict.__init__(self, AverageMeter)

    def __call__(self):
        score = self['score']
        TP = [x for _, x in sorted(zip(score, self['TP']), key=lambda pair: -pair[0])]
        FP = [x for _, x in sorted(zip(score, self['FP']), key=lambda pair: -pair[0])]

        FP = np.cumsum(FP)
        TP = np.cumsum(TP)
        recall = TP / float(len(TP))

        precision = TP / np.maximum(TP + FP, np.finfo(np.float64).eps)
        ap = voc_ap(recall, precision)
        return ap

    def __str__(self):
        return '{' + ', '.join([f"{k}: {v:.2f}" for k, v in self.items()]) + '}'

    def add(self, data):
        for k, v in data.items():
            self[k].extend(v)
        return self

    def log(self, name):
        wandb.summary[f"{name}_avg"] = self.__call__()


class BinaryClassificationMeter(defaultdict, MetricMeter):
    def __init__(self):
        MetricMeter.__init__(self)
        defaultdict.__init__(self, list)

    def __call__(self):
        est = np.array(self['est'])
        gt = np.array(self['gt'])
        return classification_metric(est, gt)

    def __str__(self):
        return '{' + ', '.join([f"{k}: {v:.2f}" for k, v in self.__call__().items()]) + '}'

    def add(self, val):
        for k, v in val.items():
            self[k].extend(v)
        return self

    def log(self, name):
        wandb.summary.update({f"{name}_{k}": v for k, v in self.__call__().items()})


class ClassMeanMeter(defaultdict, MetricMeter):
    def __init__(self, default):
        MetricMeter.__init__(self)
        defaultdict.__init__(self, default)

    def __call__(self):
        all = {k: v() for k, v in self.items()}
        all['mean'] = sum(all.values()) / len(all)
        if self.default_factory == AverageMeter:
            val = self.val()
            all['avg'] = sum(val) / len(val) if len(val) > 0 else None
        return all

    def __str__(self):
        return '{' + ', '.join([f"{k}: {v:.2f}" for k, v in self.__call__().items()]) + '}'

    def add(self, val):
        for k, v in val.items():
            self[k].add(v)
        return self

    def val(self):
        return [v for l in self.values() for v in l]

    def log(self, name):
        metric = self.__call__()
        table = wandb.Table(columns=['metric'] + [k for k in metric.keys()])
        table.add_data(name, *[f"{v:.4f}" for v in metric.values()])
        wandb.summary[name + '_table'] = table
        wandb.summary[name + '_avg'] = metric['mean']
        if self.default_factory == AverageMeter:
            wandb.summary[f"{name}_hist"] = wandb.Histogram(self.val())


class MetricRecorder(defaultdict):
    def __init__(self):
        super(MetricRecorder, self).__init__(MetricMeter)

    def __str__(self):
        return '\n'.join([f"{k}: {v()}" for k, v in self.items()])

    def add(self, metrics):
        for key, item in metrics.items():
            self[key] = self[key].add(item)

    def log(self):
        for name, meter in self.items():
            meter.log(name)
