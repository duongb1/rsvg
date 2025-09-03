import torch
import torch.nn as nn

def Reg_Loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Smooth L1 loss trên (cx, cy, w, h).
    output/target: (B, 4) giá trị normalized [0,1]
    """
    sm_l1 = nn.SmoothL1Loss(reduction="mean")
    return (sm_l1(output[:, 0], target[:, 0]) +
            sm_l1(output[:, 1], target[:, 1]) +
            sm_l1(output[:, 2], target[:, 2]) +
            sm_l1(output[:, 3], target[:, 3]))


def GIoU_Loss(boxes1: torch.Tensor, boxes2: torch.Tensor, size: int) -> torch.Tensor:
    """
    Generalized IoU loss.
    boxes1: (B,4) (cx,cy,w,h) trên absolute scale [0, size]
    boxes2: (B,4) (x1,y1,x2,y2) groundtruth absolute
    size:   int, image side (ví dụ 639)
    """
    bs = boxes1.size(0)

    # convert pred từ cxcywh -> xyxy
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


def EIoU_Loss_Compat(boxes1: torch.Tensor, boxes2: torch.Tensor, size: int) -> torch.Tensor:
    """
    Extended IoU (EIoU) loss — phiên bản "compat" cho training loop.
    Công thức: 1 - IoU + (center_dist/c^2) + (dw^2/cw^2) + (dh^2/ch^2)

    boxes1: (B,4) (cx,cy,w,h) absolute
    boxes2: (B,4) (x1,y1,x2,y2) absolute groundtruth
    size:   int image side
    """
    bs = boxes1.size(0)

    # convert cxcywh -> xyxy
    boxes1 = torch.cat([boxes1[:, :2] - boxes1[:, 2:] / 2,
                        boxes1[:, :2] + boxes1[:, 2:] / 2], dim=1)
    boxes1 = torch.clamp(boxes1, min=0, max=size)

    x1p, y1p, x2p, y2p = boxes1.unbind(-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(-1)

    # width/height
    wp = (x2p - x1p).clamp(min=0); hp = (y2p - y1p).clamp(min=0)
    wg = (x2g - x1g).clamp(min=0); hg = (y2g - y1g).clamp(min=0)

    # IoU
    xi1 = torch.max(x1p, x1g); yi1 = torch.max(y1p, y1g)
    xi2 = torch.min(x2p, x2g); yi2 = torch.min(y2p, y2g)
    wi = (xi2 - xi1).clamp(min=0); hi = (yi2 - yi1).clamp(min=0)
    inter = wi * hi
    union = (wp * hp + wg * hg - inter).clamp(min=1e-7)
    iou = inter / union

    # enclosing box
    xc1 = torch.min(x1p, x1g); yc1 = torch.min(y1p, y1g)
    xc2 = torch.max(x2p, x2g); yc2 = torch.max(y2p, y2g)
    cw = (xc2 - xc1).clamp(min=1e-7); ch = (yc2 - yc1).clamp(min=1e-7)

    # center distance
    cpx = 0.5 * (x1p + x2p); cpy = 0.5 * (y1p + y2p)
    cgx = 0.5 * (x1g + x2g); cgy = 0.5 * (y1g + y2g)
    rho2_center = (cpx - cgx) ** 2 + (cpy - cgy) ** 2

    # w/h distance
    rho2_w = (wp - wg) ** 2
    rho2_h = (hp - hg) ** 2

    loss = 1 - iou + rho2_center / (cw ** 2 + ch ** 2) + rho2_w / (cw ** 2) + rho2_h / (ch ** 2)
    return loss.mean()
