###
# File: detector_pano.py
# Created Date: 2021-03-12
# Author: Murray Chen
# Contact: <chencai1105@163.com>
# 
# Last Modified: Wednesday March 17th 2021 3:16:01 pm
# 
# Copyright (c) 2021 by Murray Chen. All Rights Reserved.
# Love & Freewill.
# -----
# HISTORY:
# Date      	 By	Comments
# ----------	---	----------------------------------------------------------
###
import numpy as np


def nms(bobj, cf_thresh, nms_thresh):
    """boxes is classified as each class"""
    bboxs = bobj["boxs"]
    scores = bobj["scores"]
    cfvalid_ids = np.where(scores >= cf_thresh)[0]
    if len(cfvalid_ids) == 0:
        return None, None
    bboxs = bobj["boxs"][cfvalid_ids]
    scores = scores[cfvalid_ids]
    ids = bobj["ids"][cfvalid_ids]
    masks = bobj["masks"][cfvalid_ids]
    x1 = bboxs[:, 0]
    y1 = bboxs[:, 1]
    x2 = bboxs[:, 2]
    y2 = bboxs[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # cfvalid_ids = np.where(scores >= cf_thresh)[0]
    # scores = scores[cfvalid_ids]

    # order = scores.argsort()[::-1]
    mask_sizes = np.sum(masks, axis=(1, 2))
    order = mask_sizes.argsort()[::-1]
    keep = []
    suppress = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        # Because of we split the object cross the boundary in the cropped instance,
        # concatenating it to the original instance, thus we need also mask iou condition for nms
        mask_other = masks[order[1:], :, :]
        mask_cur = masks[i, :, :]
        mask_inter = np.sum(mask_cur & mask_other, axis=(1, 2))
        mask_union = np.sum(mask_cur | mask_other, axis=(1, 2))
        mask_iou = mask_inter / mask_union

        suppress_inds = np.where((iou > nms_thresh) | (mask_iou > nms_thresh))[0]
        sup_i = order[1:][suppress_inds] if suppress_inds.size != 0 else np.array([])
        suppress.append(sup_i)

        inds = np.where((iou <= nms_thresh) & (mask_iou <= nms_thresh))[0]
        order = order[inds + 1]

    for i, sup in enumerate(suppress):
        if sup.any():
            for sup_id in sup:
                # sup_id = s + 1
                keep_id = keep[i]
                # union the keep mask and the suppress mask
                masks[keep_id, :, :] = masks[keep_id, :, :] | masks[sup_id, :, :]
    if keep:
        return ids[keep], masks[keep]
    else:
        return [], []


def nms_all_class(bound_corr_objs, nms_thresh):
    """boxes is classified as each class"""
    bboxs, scores, masks, labels = [], [], [], []
    for obj in bound_corr_objs:
        bboxs.append(obj['box'])
        scores.append(obj['score'])
        # masks.append(obj['mask'])
        # labels.append(obj['label'])
    bboxs = np.asarray(bboxs)
    scores = np.asarray(scores)
    # masks = np.asarray(masks)
    # labels = np.asarray(labels)
    x1 = bboxs[:, 0]
    y1 = bboxs[:, 1]
    x2 = bboxs[:, 2]
    y2 = bboxs[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # cfvalid_ids = np.where(scores >= cf_thresh)[0]
    # scores = scores[cfvalid_ids]

    order = scores.argsort()[::-1]
    # mask_sizes = np.sum(masks, axis=(1, 2))
    # order = mask_sizes.argsort()[::-1]
    keep = []
    suppress = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # mask_other = masks[order[1:], :, :]
        # mask_cur = masks[i, :, :]
        # mask_inter = np.sum(mask_cur & mask_other, axis=(1, 2))
        # mask_union = np.sum(mask_cur | mask_other, axis=(1, 2))
        # mask_iou = mask_inter / mask_union

        # inds = np.where((iou <= nms_thresh) & (mask_iou <= nms_thresh))[0]
        inds = np.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    # masks = masks[keep]
    # ids = ids[keep]
    return keep
