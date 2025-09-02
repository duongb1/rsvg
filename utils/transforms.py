"""
Image transforms for remote sensing grounding.
"""
import math
import random
from typing import Tuple

import cv2
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def letterbox(img: np.ndarray, mask: np.ndarray, height: int,
              color=(123.7, 116.3, 103.5)) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Resize keeping ratio and pad to square (height x height)."""
    shape = img.shape[:2]  # (h,w)
    ratio = float(height) / max(shape)
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2
    dh = (height - new_shape[1]) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)

    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    if mask is not None:
        mask = cv2.resize(mask, new_shape, interpolation=cv2.INTER_NEAREST)
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=(255, 255, 255))
    return img, mask, ratio, dw, dh


def random_affine(img, mask, targets, degrees=(-10, 10),
                  translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(123.7, 116.3, 103.5)):
    border = 0
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation & Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(center=(img.shape[1] / 2, img.shape[0] / 2), angle=a, scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)

    M = S @ T @ R
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR, borderValue=borderValue)
    if mask is not None:
        maskw = cv2.warpPerspective(mask, M, dsize=(height, height), flags=cv2.INTER_NEAREST, borderValue=255)
    else:
        maskw = None

    if targets is None:
        return imw, maskw, None, M

    def _wrap_points(b):
        points = b.copy()
        area0 = (points[2] - points[0]) * (points[3] - points[1])
        xy = np.ones((4, 3))
        xy[:, :2] = points[[0, 1, 2, 3, 0, 3, 2, 1]].reshape(4, 2)
        xy = (xy @ M.T)[:, :2].reshape(1, 8)
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, 1).T
        radians = a * math.pi / 180
        reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
        cx = (xy[:, 2] + xy[:, 0]) / 2
        cy = (xy[:, 3] + xy[:, 1]) / 2
        w = (xy[:, 2] - xy[:, 0]) * reduction
        h = (xy[:, 3] - xy[:, 1]) * reduction
        xy = np.concatenate((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)).reshape(4, 1).T
        np.clip(xy, 0, height, out=xy)
        return xy[0]

    targets = _wrap_points(targets)
    return imw, maskw, targets, M
