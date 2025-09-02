import torch
import torch.nn as nn


def Reg_Loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Smooth L1 loss on (cx, cy, w, h)
    output/target: (B, 4)
    """
    sm_l1 = nn.SmoothL1Loss(reduction="mean")
    return (sm_l1(output[:, 0], target[:, 0]) +
            sm_l1(output[:, 1], target[:, 1]) +
            sm_l1(output[:, 2], target[:, 2]) +
            sm_l1(output[:, 3], target[:, 3]))


def GIoU_Loss(boxes1: torch.Tensor, boxes2: torch.Tensor, size: int) -> torch.Tensor:
    """
    Generalized IoU loss between predicted (cx,cy,w,h) and target (x1,y1,x2,y2) on absolute scale [0,size].
    boxes1: (B, 4) in cx,cy,w,h on absolute scale
    boxes2: (B, 4) in x1,y1,x2,y2 on absolute scale
    size:   int image side (e.g., 639)
    """
    bs = boxes1.size(0)
    # convert pred to x1y1x2y2 and clamp
    boxes1 = torch.cat([boxes1[:, :2] - boxes1[:, 2:] / 2,
                        boxes1[:, :2] + boxes1[:, 2:] / 2], dim=1)
    boxes1 = torch.clamp(boxes1, min=0, max=size)

    # Intersection
    max_xy = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    min_xy = torch.max(boxes1[:, :2], boxes2[:, :2])
    inter = torch.clamp(max_xy - min_xy, min=0)
    inter = inter[:, 0] * inter[:, 1]

    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter + 1e-7
    iou = inter / union

    # Smallest enclosing box
    enclose_lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    enclose_rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    enclose = torch.clamp(enclose_rb - enclose_lt, min=0)
    enclose_area = enclose[:, 0] * enclose[:, 1] + 1e-7

    giou = iou - (enclose_area - union) / enclose_area
    loss = (1.0 - giou).sum() / bs
    return loss
