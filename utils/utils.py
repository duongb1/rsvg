import numpy as np
import torch

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """[x1,y1,x2,y2] -> [cx,cy,w,h]"""
    y = torch.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """[cx,cy,w,h] -> [x1,y1,x2,y2]"""
    y = torch.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def bbox_iou_numpy(box1: np.ndarray, box2: np.ndarray) -> np.ndarray:
    """IoU matrix between box1 (N,4) and box2 (M,4), format x1y1x2y2."""
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    iw = np.minimum(np.expand_dims(box1[:, 2], 1), box2[:, 2]) - np.maximum(np.expand_dims(box1[:, 0], 1), box2[:, 0])
    ih = np.minimum(np.expand_dims(box1[:, 3], 1), box2[:, 3]) - np.maximum(np.expand_dims(box1[:, 1], 1), box2[:, 1])
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)
    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), 1) + area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)
    inter = iw * ih
    return inter / ua


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, x1y1x2y2=True):
    """
    IoU between two sets of boxes, returns (iou, inter_area, union_area).
    """
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    union_area = torch.clamp(union_area, min=1e-6)
    return inter_area / union_area, inter_area, union_area


def adjust_learning_rate(args, optimizer, epoch):
    """Reduce LR after 60 epochs (paper setting)."""
    lr_decay = getattr(args, "lr_dec", 0.1)
    if epoch == 60:
        for param_group in optimizer.param_groups:
            param_group["lr"] *= lr_decay
