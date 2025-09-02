# utils/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

# ----------------------------
# Bbox conversions (xyxy <-> cxcywh)
# ----------------------------
def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def xyxy_to_cxcywh(box: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = box.unbind(-1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w  = (x2 - x1).clamp(min=0.0)
    h  = (y2 - y1).clamp(min=0.0)
    return torch.stack([cx, cy, w, h], dim=-1)

# ----------------------------
# Core geometry helpers
# ----------------------------
def _area_xyxy(b: torch.Tensor) -> torch.Tensor:
    w = (b[..., 2] - b[..., 0]).clamp(min=0.0)
    h = (b[..., 3] - b[..., 1]).clamp(min=0.0)
    return w * h

def iou_xyxy(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x1 = torch.max(pred[..., 0], tgt[..., 0])
    y1 = torch.max(pred[..., 1], tgt[..., 1])
    x2 = torch.min(pred[..., 2], tgt[..., 2])
    y2 = torch.min(pred[..., 3], tgt[..., 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    union = (_area_xyxy(pred) + _area_xyxy(tgt) - inter).clamp(min=eps)
    return inter / union

def giou_xyxy(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    iou = iou_xyxy(pred, tgt, eps)
    xc1 = torch.min(pred[..., 0], tgt[..., 0])
    yc1 = torch.min(pred[..., 1], tgt[..., 1])
    xc2 = torch.max(pred[..., 2], tgt[..., 2])
    yc2 = torch.max(pred[..., 3], tgt[..., 3])
    c_area = ((xc2 - xc1).clamp(min=0.0) * (yc2 - yc1).clamp(min=0.0)).clamp(min=eps)
    # recompute union (avoid redoing intersections again, but fine for clarity)
    x1 = torch.max(pred[..., 0], tgt[..., 0])
    y1 = torch.max(pred[..., 1], tgt[..., 1])
    x2 = torch.min(pred[..., 2], tgt[..., 2])
    y2 = torch.min(pred[..., 3], tgt[..., 3])
    inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
    union = (_area_xyxy(pred) + _area_xyxy(tgt) - inter).clamp(min=eps)
    return iou - (c_area - union) / c_area

def eiou_loss_xyxy(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    iou = iou_xyxy(pred, tgt, eps)
    wp = (pred[..., 2] - pred[..., 0]).clamp(min=0.0)
    hp = (pred[..., 3] - pred[..., 1]).clamp(min=0.0)
    wg = (tgt[..., 2] - tgt[..., 0]).clamp(min=0.0)
    hg = (tgt[..., 3] - tgt[..., 1]).clamp(min=0.0)

    xc1 = torch.min(pred[..., 0], tgt[..., 0])
    yc1 = torch.min(pred[..., 1], tgt[..., 1])
    xc2 = torch.max(pred[..., 2], tgt[..., 2])
    yc2 = torch.max(pred[..., 3], tgt[..., 3])
    cw = (xc2 - xc1).clamp(min=eps)
    ch = (yc2 - yc1).clamp(min=eps)

    cpx = 0.5 * (pred[..., 0] + pred[..., 2])
    cpy = 0.5 * (pred[..., 1] + pred[..., 3])
    cgx = 0.5 * (tgt[..., 0] + tgt[..., 2])
    cgy = 0.5 * (tgt[..., 1] + tgt[..., 3])

    rho2_center = (cpx - cgx) ** 2 + (cpy - cgy) ** 2
    diag2 = cw ** 2 + ch ** 2

    rho2_w = (wp - wg) ** 2
    rho2_h = (hp - hg) ** 2

    return 1.0 - iou + (rho2_center / (diag2 + eps)) + (rho2_w / (cw ** 2 + eps)) + (rho2_h / (ch ** 2 + eps))

# ----------------------------
# Your original APIs (kept)
# ----------------------------
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
    size:   int image side (e.g., 639). We clamp to [0, size] for robustness (giống bản cũ).
    """
    bs = boxes1.size(0)
    # convert pred to xyxy and clamp
    pred_xyxy = cxcywh_to_xyxy(boxes1)
    pred_xyxy = torch.clamp(pred_xyxy, min=0.0, max=float(size))
    tgt_xyxy  = torch.clamp(boxes2,   min=0.0, max=float(size))

    giou = giou_xyxy(pred_xyxy, tgt_xyxy)          # (B,)
    loss = (1.0 - giou).sum() / max(bs, 1)
    return loss

# ----------------------------
# Optional: extra compat helpers
# ----------------------------
def EIoU_Loss_Compat(boxes1: torch.Tensor, boxes2: torch.Tensor, size: int) -> torch.Tensor:
    """
    EIoU loss dùng cùng input kiểu GIoU_Loss ở trên (pred in cxcywh abs, tgt in xyxy abs).
    """
    bs = boxes1.size(0)
    pred_xyxy = torch.clamp(cxcywh_to_xyxy(boxes1), 0.0, float(size))
    tgt_xyxy  = torch.clamp(boxes2, 0.0, float(size))
    loss = eiou_loss_xyxy(pred_xyxy, tgt_xyxy).sum() / max(bs, 1)
    return loss

def CombinedLoss_Compat(
    boxes1: torch.Tensor,  # pred cxcywh (absolute)
    boxes2: torch.Tensor,  # gt   xyxy   (absolute)
    size: int,
    lam_giou: float = 1.0,
    lam_eiou: float = 1.0,
    beta: float = 1.0,
    reduce: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    """
    SmoothL1 (on xyxy) + lam_giou*(1-GIoU) + lam_eiou*EIoU
    vẫn dùng đúng kiểu tham số cũ (có 'size') để bạn thay thế nhẹ nhàng trong loop.

    Lưu ý: SmoothL1 tính trên xyxy. Nếu bạn muốn giữ SmoothL1 trên cxcywh như bản cũ,
    có thể cộng thêm Reg_Loss(output_cxcywh, target_cxcywh) vào ngoài hàm này.
    """
    pred_xyxy = torch.clamp(cxcywh_to_xyxy(boxes1), 0.0, float(size))
    tgt_xyxy  = torch.clamp(boxes2, 0.0, float(size))

    # Smooth-L1 over xyxy
    l1 = F.smooth_l1_loss(pred_xyxy, tgt_xyxy, reduction="none").sum(dim=-1)  # (B,)

    giou_term = 1.0 - giou_xyxy(pred_xyxy, tgt_xyxy)                          # (B,)
    eiou_term = eiou_loss_xyxy(pred_xyxy, tgt_xyxy)                            # (B,)

    total = l1 + lam_giou * giou_term + lam_eiou * eiou_term                  # (B,)
    if reduce == "mean":
        return total.mean()
    else:
        return total.sum()
