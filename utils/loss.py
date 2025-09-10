import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

def Reg_Loss(output, target):
    sm_l1_loss = torch.nn.SmoothL1Loss(reduction='mean')
    
    loss_x1 = sm_l1_loss(output[:,0], target[:,0])
    loss_x2 = sm_l1_loss(output[:,1], target[:,1])
    loss_y1 = sm_l1_loss(output[:,2], target[:,2])
    loss_y2 = sm_l1_loss(output[:,3], target[:,3])

    return (loss_x1+loss_x2+loss_y1+loss_y2)


def GIoU_Loss(boxes1, boxes2, size):
    '''
    cal GIOU of two boxes or batch boxes
    '''

    # ===========cal IOU=============#
    # cal Intersection
    bs = boxes1.size(0)
    boxes1 = torch.cat([boxes1[:,:2]-(boxes1[:,2:]/2), boxes1[:,:2]+(boxes1[:,2:]/2)], dim=1)
    boxes1 = torch.clamp(boxes1, min=0, max=size)
    max_xy = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    min_xy = torch.max(boxes1[:, :2], boxes2[:, :2])

    inter = torch.clamp((max_xy - min_xy), min=0)
    inter = inter[:, 0] * inter[:, 1]
    boxes1Area = ((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]))
    boxes2Area = ((boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]))

    union_area = boxes1Area + boxes2Area - inter + 1e-7
    ious = inter / union_area

    # ===========cal enclose area for GIOU=============#
    enclose_left_up = torch.min(boxes1[:, :2], boxes2[:, :2])
    enclose_right_down = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    enclose = torch.clamp((enclose_right_down - enclose_left_up), min=0)
    enclose_area = enclose[:, 0] * enclose[:, 1] + 1e-7
    # cal GIOU
    gious = ious - 1.0 * (enclose_area - union_area) / enclose_area
    # GIOU Loss
    giou_loss = ((1-gious).sum())/bs
    return giou_loss

def bbox_xywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    # b: (...,4) (norm hoặc pixel) – phải cùng hệ với gt khi so sánh
    cx, cy, w, h = b[...,0], b[...,1], b[...,2], b[...,3]
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    return torch.stack([x1, y1, x2, y2], dim=-1)

def iou_xyxy(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # a,b: (B,4) cùng hệ toạ độ
    x1 = torch.max(a[:,0], b[:,0])
    y1 = torch.max(a[:,1], b[:,1])
    x2 = torch.min(a[:,2], b[:,2])
    y2 = torch.min(a[:,3], b[:,3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[:,2]-a[:,0]).clamp(min=0) * (a[:,3]-a[:,1]).clamp(min=0)
    area_b = (b[:,2]-b[:,0]).clamp(min=0) * (b[:,3]-b[:,1]).clamp(min=0)
    union = area_a + area_b - inter + eps
    return (inter / union).unsqueeze(-1)  # (B,1)

def quality_loss(qhat: torch.Tensor, iou_t: torch.Tensor, weight: float = 0.2) -> torch.Tensor:
    # qhat, iou_t: (B,1) in [0,1]
    return F.mse_loss(qhat, iou_t) * weight