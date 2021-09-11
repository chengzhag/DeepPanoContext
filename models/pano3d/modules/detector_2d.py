import torch
import torch.nn as nn
import numpy as np
import math
from collections import defaultdict

from detectron2.engine import DefaultPredictor

from models.detector.dataset import get_cfg, register_igibson_detection_dataset
from configs.data_config import IG56CLASSES, PC2IG, WIMR11CLASSES
from models.registers import MODULES
from utils.render_utils import is_obj_valid, seg2obj
from utils.detector_utils import nms, nms_all_class


@MODULES.register_module
class Detector2D:
    def __init__(self, cfg, _):
        self.cfg = cfg
        model_config = cfg.config['model']['detector']
        self.min_iou = model_config.get('min_iou', 0.1)
        self.real = model_config['real']
        self.pano = model_config['pano']
        self.cf_thresh = model_config['cf_thresh']
        self.nms_thresh = model_config['nms_thresh']
        self.nms_all_thresh = model_config['nms_all_thresh']

        # configs for detectron
        register_igibson_detection_dataset(cfg.config['data']['split'], real=model_config['real'])
        cfg_detectron = get_cfg(cfg.config['data']['split'], model_config['config'])
        cfg_detectron.MODEL.WEIGHTS = model_config['weight']
        cfg_detectron.MODEL.ROI_HEADS.SCORE_THRESH_TEST = model_config['score_thresh']
        self.predictor = DefaultPredictor(cfg_detectron)

    def __call__(self, image_np, camera):
        est_scenes = []

        for image, c in zip(image_np, camera):
            images = [image]
            if self.pano:
                height, width, channel = image.shape
                roll_width = width // 2
                image2 = np.roll(image, roll_width, axis=1)
                images.append(image2)

            id = 0
            objs = []
            for i_image, image in enumerate(images):
                output = self.predictor(image)
                instances = output['instances'].to('cpu')

                for pred in zip(
                        instances.pred_boxes, instances.scores,
                        instances.pred_classes, instances.pred_masks
                ):
                    box, score, label, mask = [p.numpy() if p.numel() > 1 else p.item() for p in pred]

                    if self.real:
                        # map label and classname to WIMR real dataset
                        classname_wimr = WIMR11CLASSES[label]
                        if classname_wimr not in PC2IG:
                            continue
                        obj = {
                            'classname_wimr': classname_wimr,
                            'label_wimr': label
                        }
                        IG_classname = PC2IG[classname_wimr]
                        label = IG56CLASSES.index(IG_classname)
                    else:
                        obj = {}

                    obj.update({
                        'classname': IG56CLASSES[label],
                        'label': label,
                        'score': score,
                    })

                    # use box as mask if mask is empty
                    if not np.any(mask):
                        box = np.round(box).astype(np.int)
                        mask[box[1]:(box[3] + 1), box[0]:(box[2] + 1)] = True

                    if i_image == 1:
                        # shift rolled detection results back
                        shift = roll_width
                        if box[2] >= roll_width:
                            shift *= -1
                        box[0] += shift
                        box[2] += shift
                        mask = np.roll(mask, roll_width, axis=1)

                    obj_info = seg2obj(mask, 1, c)
                    if obj_info is None:
                        continue
                    obj.update(obj_info)
                    # obj['bdb2d'] = {k: v for k, v in zip(('x1', 'y1', 'x2', 'y2'), np.round(box).astype(np.int))}
                    if not is_obj_valid(obj):
                        continue

                    id += 1
                    obj.update({
                        'id': id,
                        'mask': mask,
                        'box': box
                    })
                    objs.append(obj)

            if self.pano:
                # prepare objects information for NMS
                label_objs = defaultdict(lambda :defaultdict(list))
                label_key = 'label_wimr' if self.real else 'label'
                for i, obj in enumerate(objs):
                    cls_objs = label_objs[obj[label_key]]
                    cls_objs["ids"].append(obj['id'])
                    cls_objs["boxs"].append(obj['box'])
                    cls_objs["scores"].append(obj['score'])
                    cls_objs["masks"].append(obj['mask'])
                for label, cls_objs in label_objs.items():
                    for k, v in cls_objs.items():
                        cls_objs[k] = np.stack(v)

                # run NMS on each class to merge cross-edge objects
                nms_objs = []
                for label, cls_objs in label_objs.items():
                    ids, masks = nms(cls_objs, self.cf_thresh, self.nms_thresh)
                    for id, mask in zip(ids, masks):
                        obj = objs[id - 1]
                        obj_info = seg2obj(mask, 1, c)
                        if obj_info is None:
                            continue
                        obj.update(obj_info)
                        obj['mask'] = mask
                        nms_objs.append(obj)

                # run NMS on all class to merge overlapped objects with different labels
                if nms_objs:
                    indices = nms_all_class(nms_objs, self.nms_all_thresh)
                    objs = [nms_objs[i] for i in indices]
                else:
                    objs = []

            # compose instance segmentation
            seg = np.zeros(image.shape[:2], np.uint8)
            for obj in objs:
                seg[obj['mask']] = obj['id']
                # remove intermedia object information
                obj.pop('mask')
                if 'box' in obj:
                    obj.pop('box')

            est_scenes.append({
                'objs': objs,
                'image_np': {'seg': seg}
            })

        return est_scenes


def bdb2d_geometric_feature(boxes, g_feature_length):
    # g_feature: n_objects x n_objects x 4
    # Note that g_feature is not symmetric,
    # g_feature[m, n] is the feature of object m contributes to object n.
    eps = 1e-6
    g_feature = [[((loc2['x1'] + loc2['x2']) / 2. - (loc1['x1'] + loc1['x2']) / 2.) / (loc1['x2'] - loc1['x1'] + eps),
                  ((loc2['y1'] + loc2['y2']) / 2. - (loc1['y1'] + loc1['y2']) / 2.) / (loc1['y2'] - loc1['y1'] + eps),
                  math.log((loc2['x2'] - loc2['x1'] + eps) / (loc1['x2'] - loc1['x1'] + eps)),
                  math.log((loc2['y2'] - loc2['y1'] + eps) / (loc1['y2'] - loc1['y1'] + eps))] \
                 for id1, loc1 in enumerate(boxes)
                 for id2, loc2 in enumerate(boxes)]
    locs = [num for loc in g_feature for num in loc]
    d_model = int(g_feature_length / 4)
    pe = torch.zeros(len(locs), d_model)
    position = torch.from_numpy(np.array(locs)).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    n_objects = len(boxes)
    # g_feature = pe.view(n_objects, n_objects, g_feature_length)
    g_feature = pe.view(n_objects * n_objects, g_feature_length)
    return g_feature
