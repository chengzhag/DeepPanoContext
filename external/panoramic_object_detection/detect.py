import os
import sys

this_dir = os.path.dirname(__file__)
sys.path.insert(0,os.path.join(this_dir, 'python'))

import numpy as np
import cv2

# x->right, y->down
def wrapped_line(image, p1, p2, colour, thickness, lineType=cv2.LINE_AA):
    if p1[0] > p2[0]:
        p1, p2 = p2, p1

    _p1 = np.array(p1)
    _p2 = np.array(p2)

    dist1 = np.linalg.norm(_p1 - _p2)

    p1b = np.array([p1[0]+image.shape[1], p1[1]])
    p2b = np.array([p2[0]-image.shape[1], p2[1]])

    dist2 = np.linalg.norm(_p1 - p2b)

    if dist1 < dist2:
        cv2.line(image, p1, p2, colour, thickness, lineType=lineType)
    else:
        cv2.line(image, p1, tuple(p2b), colour, thickness, lineType=lineType)
        cv2.line(image, tuple(p1b), p2, colour, thickness, lineType=lineType)
