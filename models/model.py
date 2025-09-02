# models/model.py
import math
import os
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision import models
from torchvision.ops import generalized_box_iou

try:
    from transformers import AutoModel, AutoConfig
    _HAS_TRANSFORMERS = True
except Exception:
    _HAS_TRANSFORMERS = False


# -----------------------------
# Utils
# -----------------------------

def _masked_mean_pool(last_hidden_state: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pooling theo mask: (B, L, D) -> (B, D)"""
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    """box: (..., 4) with (cx, cy, w, h) in [0,1] -> (x1, y1, x2, y2) in [0,1]."""
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _normalize_xyxy_abs(xyxy: torch.Tensor, img_wh: torch.Tensor) -> torch.Tensor:
    """Normalize absolute xyxy (pixels) to [0,1] by (W,H)."""
    # img_wh: (B, 2) = (W, H)
    w = img_wh[:, 0].clamp(min=1e-6).unsqueeze(-1)
    h = img_wh[:, 1].clamp(min=1e-6).unsqueeze(-1)
    scale = torch.cat([w, h, w, h], dim=-1)
    return xyxy / scale


def _denormalize_xyxy(xyxy: torch.Tensor, img_wh: torch.Tensor) -> torch.Tensor:
    """Denormalize xyxy in [0,1] to absolute pixels by (W,H)."""
    w = img_wh[:, 0].unsqueeze(-1)
    h = img_wh[:, 1].unsqueeze(-1)
    scale = torch.cat([w, h, w, h], dim=-1)
    return xyxy * scale


# -----------------------------
# Positional Encoding (2D sine)
# -----------------------------

class SinePositionalEncoding2D(nn.Module):
    """
    Sinusoidal 2D positional encoding like DETR/ViT.
    Produces a (B, C, H, W) tensor to add onto features with the same shape.
    C must be even and divisible by 2.
    """
    def __init__(self, num_channels: int = 256, temperature: int = 10000):
        super().__init__()
        assert num_channels % 2 == 0, "pos channels must be even"
        self.num_channels = num_channels
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> produce (B, C, H, W)
        b, c, h, w = x.shape
        device = x.device
        y_embed = torch.linspace(0, 1, steps=h, device=device).unsqueeze(1).repeat(1, w)  # (H, W)
        x_embed = torch.linspace(0, 1, steps=w, device=device).unsqueeze(0).repeat(h, 1)  # (H, W)

        dim_t = torch.arange(c // 2, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (c // 2))

        pos_x = x_embed.unsqueeze(-1) / dim_t  # (H, W, C//2)
        pos_y = y_embed.unsqueeze(-1) / dim_t

        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-2)  # (H, W, 2, C//2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-2)

        pos_x = pos_x.flatten(-2)  # (H, W, C)
        pos_y = pos_y.flatten(-2)

        pos = torch.cat((pos_y, pos_x), dim=-1)  # (H, W, 2C)
        pos = pos[..., :c]  # keep C
        pos = pos.permute(2, 0, 1).unsqueeze(0).repeat(b, 1, 1, 1)  # (B, C, H, W)
        return pos


# -----------------------------
# Image Backbone (ResNet50 -> C5)
# -----------------------------

class ResNet50C5(nn.Module):
    """
    ResNet-50 backbone returning C5 (stride 32), optionally init from DETR weights.
    Followed by a 1x1 conv to 256 channels.
    """
    DETR_R50_URL = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"

    def __init__(self, out_channels: int = 256, init_from_detr: bool = True):
        super().__init__()
        # torchvision resnet50 without classifier
        backbone = models.resnet50(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4  # C5, 2048 channels
        self.proj = nn.Conv2d(2048, out_channels, kernel_size=1)

        if init_from_detr:
            self._init_from_detr()

    @torch.no_grad()
    def _init_from_detr(self):
        """
        Auto-download DETR R50 weights and map the backbone part into torchvision resnet50.
        """
        sd = load_state_dict_from_url(self.DETR_R50_URL, map_location="cpu")
        # DETR backbone keys look like "backbone.0.body.layer2.0.conv1.weight"
        # We strip the prefix "backbone.0.body." and load into our layers.
        mapped = {}
        prefix = "backbone.0.body."
        for k, v in sd.items():
            if k.startswith(prefix):
                new_k = k[len(prefix):]  # e.g. "layer2.0.conv1.weight"
                # Our module names match torchvision's resnet50 hierarchy under stem/layer{1..4}
                # Special-case root: conv1/bn1
                if new_k.startswith("conv1.") or new_k.startswith("bn1.") or new_k.startswith("relu.") or new_k.startswith("maxpool."):
                    mapped[f"stem.{new_k}"] = v
                else:
                    mapped[new_k] = v

        # Try to load into a temp model with matching names, then copy params
        temp = models.resnet50(weights=None)
        missing, unexpected = temp.load_state_dict(mapped, strict=False)
        # copy weights into ours
        self.stem[0].weight.copy_(temp.conv1.weight)
        self.stem[1].weight.copy_(temp.bn1.weight)
        self.stem[1].bias.copy_(temp.bn1.bias)
        self.stem[1].running_mean.copy_(temp.bn1.running_mean)
        self.stem[1].running_var.copy_(temp.bn1.running_var)

        # Copy blocks
        for i in range(1, 5):
            src = getattr(temp, f"layer{i}")
            dst = getattr(self, f"layer{i}")
            for (n_dst, m_dst), (n_src, m_src) in zip(dst.named_modules(), src.named_modules()):
                # copy only leaf modules with weights
                for attr in ("weight", "bias", "running_mean", "running_var"):
                    if hasattr(m_dst, attr) and hasattr(m_src, attr):
                        try:
                            getattr(m_dst, attr).copy_(getattr(m_src, attr))
                        except Exception:
                            pass  # shape mismatch safe-skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) -> (B, 256, H/32, W/32)
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)   # (B, 2048, H/32, W/32)
        x = self.proj(x)     # (B, 256,  H/32, W/32)
        return x


# -----------------------------
# LVFE: Language-guided Visual Feature Enhancement
# -----------------------------

class LVFE(nn.Module):
    """
    Multi-head attention where queries are visual tokens and keys/values are text tokens,
    iterated a few times, then expand pooled text and concat along channel to keep spatial info.
    """
    def __init__(self, v_dim=256, t_dim=768, heads=8, iters=3, qkv_dim=512, dropout=0.0):
        super().__init__()
        self.txt_proj = nn.Linear(t_dim, v_dim)
        self.q_proj   = nn.Linear(v_dim, qkv_dim)
        self.k_proj   = nn.Linear(v_dim, qkv_dim)
        self.v_proj   = nn.Linear(v_dim, qkv_dim)
        self.attn     = nn.MultiheadAttention(qkv_dim, heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(qkv_dim, v_dim)
        self.iters    = iters
        self.norm_q   = nn.LayerNorm(v_dim)
        self.norm_out = nn.LayerNorm(v_dim)

    def forward(self,
                Fv_2d: torch.Tensor,        # (B, C=256, H, W)
                Ft_seq: torch.Tensor,       # (B, L, t_dim)
                t_mask: Optional[torch.Tensor] = None  # (B, L) True for PAD (key_padding_mask)
                ) -> torch.Tensor:
        B, C, H, W = Fv_2d.shape
        Nv = H * W

        # Flatten visual tokens to (B, Nv, C)
        Fv = Fv_2d.flatten(2).transpose(1, 2)
        Fv = self.norm_q(Fv)

        # Project text to v_dim
        Ft = self.txt_proj(Ft_seq)

        # Text-guided attention (residual loop)
        for _ in range(self.iters):
            Q = self.q_proj(Fv)
            K = self.k_proj(Ft)
            V = self.v_proj(Ft)
            out, _ = self.attn(Q, K, V, key_padding_mask=t_mask)  # (B, Nv, qkv)
            Fv = Fv + self.out_proj(out)
            Fv = self.norm_out(Fv)

        # Expand pooled text (mean over valid tokens)
        if t_mask is None:
            Ft_mean = Ft.mean(dim=1, keepdim=True)  # (B,1,C)
        else:
            # invert mask for mean pooling: 1 for valid, 0 for pad
            inv = (~t_mask).float().unsqueeze(-1)
            Ft_mean = (Ft * inv).sum(dim=1, keepdim=True) / inv.sum(dim=1, keepdim=True).clamp(min=1e-6)

        Ft_expanded = Ft_mean.expand(B, Nv, C)
        Fvt = torch.cat([Fv, Ft_expanded], dim=-1)  # (B, Nv, 2C)

        # back to 2D
        Fvt_2d = Fvt.transpose(1, 2).reshape(B, 2*C, H, W)
        return Fvt_2d


# -----------------------------
# VLF Stacker + Learnable Query Token
# -----------------------------

class VLFStacker(nn.Module):
    """
    Stacked self-attention over concatenated multi-modal spatial tokens,
    with a learnable query token Tq that repeatedly attends the sequence to aggregate context.
    """
    def __init__(self, in_ch=512, hidden=512, heads=8, layers=3, dropout=0.0):
        super().__init__()
        self.proj = nn.Linear(in_ch, hidden)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden, nhead=heads,
                                       dim_feedforward=hidden * 4,
                                       dropout=dropout, batch_first=True)
            for _ in range(layers)
        ])
        self.tq = nn.Parameter(torch.randn(1, 1, hidden))
        self.blocks_q = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden, nhead=heads,
                                       dim_feedforward=hidden * 4,
                                       dropout=dropout, batch_first=True)
            for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, Fvt_2d: torch.Tensor) -> torch.Tensor:
        B, C, H, W = Fvt_2d.shape
        Nv = H * W
        X = Fvt_2d.flatten(2).transpose(1, 2)   # (B, Nv, C)
        X = self.proj(X)                         # (B, Nv, hidden)

        # stacked self-attn on X
        for blk in self.blocks:
            X = blk(X)

        # learnable token attends sequence repeatedly
        Tq = self.tq.expand(B, -1, -1)          # (B, 1, hidden)
        for blk in self.blocks_q:
            Y = torch.cat([Tq, X], dim=1)       # (B, 1+Nv, hidden)
            Y = blk(Y)
            Tq = Y[:, :1]                        # update Tq

        return self.norm(Tq.squeeze(1))          # (B, hidden)


# -----------------------------
# BBox Head
# -----------------------------

class BBoxHead(nn.Module):
    """
    Simple MLP -> 4 params (cx, cy, w, h) in [0,1] via sigmoid.
    """
    def __init__(self, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        x = self.net(q)
        return torch.sigmoid(x)  # normalized (cx, cy, w, h)


# -----------------------------
# Optional: EIoU loss
# -----------------------------

def _eiou_loss(pred_xyxy: torch.Tensor, tgt_xyxy: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    x1p, y1p, x2p, y2p = pred_xyxy.unbind(-1)
    x1g, y1g, x2g, y2g = tgt_xyxy.unbind(-1)

    wp = (x2p - x1p).clamp(min=0); hp = (y2p - y1p).clamp(min=0)
    wg = (x2g - x1g).clamp(min=0); hg = (y2g - y1g).clamp(min=0)

    xi1 = torch.max(x1p, x1g); yi1 = torch.max(y1p, y1g)
    xi2 = torch.min(x2p, x2g); yi2 = torch.min(y2p, y2g)
    wi = (xi2 - xi1).clamp(min=0); hi = (yi2 - yi1).clamp(min=0)
    inter = wi * hi
    union = (wp * hp + wg * hg - inter).clamp(min=eps)
    iou = inter / union

    xc1 = torch.min(x1p, x1g); yc1 = torch.min(y1p, y1g)
    xc2 = torch.max(x2p, x2g); yc2 = torch.max(y2p, y2g)
    cw = (xc2 - xc1).clamp(min=eps); ch = (yc2 - yc1).clamp(min=eps)

    cpx = 0.5 * (x1p + x2p); cpy = 0.5 * (y1p + y2p)
    cgx = 0.5 * (x1g + x2g); cgy = 0.5 * (y1g + y2g)

    rho2_center = (cpx - cgx) ** 2 + (cpy - cgy) ** 2
    rho2_w = (wp - wg) ** 2
    rho2_h = (hp - hg) ** 2

    return 1 - iou + rho2_center / (cw ** 2 + ch ** 2) + rho2_w / (cw ** 2) + rho2_h / (ch ** 2)


# -----------------------------
# The Full Model (MGVLF)
# -----------------------------

class MGVLF(nn.Module):
    """
    End-to-end text-guided visual grounding:
    images -> ResNet50(C5, DETR init)->256 -> LVFE -> VLFStacker -> BBoxHead.
    """

    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        init_backbone_from_detr: bool = True,
        v_dim: int = 256,               # visual channels after projection
        hidden: int = 512,              # transformer hidden
        heads: int = 8,
        layers: int = 3,
        lvfe_iters: int = 3,
        dropout: float = 0.0,
        use_positional_encoding: bool = True,
        freeze_text_encoder: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # --- Image backbone ---
        self.backbone = ResNet50C5(out_channels=v_dim, init_from_detr=init_backbone_from_detr)

        # --- Text encoder ---
        if not _HAS_TRANSFORMERS:
            raise ImportError("transformers is required for text encoder. Please install `transformers`.")
        cfg = AutoConfig.from_pretrained(text_model_name)
        self.text_hidden = cfg.hidden_size
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        # --- LVFE ---
        self.lvfe = LVFE(v_dim=v_dim, t_dim=self.text_hidden, heads=heads, iters=lvfe_iters,
                         qkv_dim=hidden, dropout=dropout)

        # --- Positional encoding ---
        self.use_posenc = use_positional_encoding
        if self.use_posenc:
            self.posenc = SinePositionalEncoding2D(num_channels=v_dim * 2)

        # --- VLF stacker & bbox head ---
        self.vlf = VLFStacker(in_ch=v_dim * 2, hidden=hidden, heads=heads, layers=layers, dropout=dropout)
        self.head = BBoxHead(hidden=hidden)

        # Freeze options
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    # ---------- Public API ----------

    @torch.no_grad()
    def infer(self,
              images: torch.Tensor,
              input_ids: torch.Tensor,
              attention_mask: torch.Tensor,
              image_sizes: Optional[List[Tuple[int, int]]] = None,
              ) -> Dict[str, torch.Tensor]:
        """
        Inference-only wrapper (no grad).
        """
        return self.forward(images, input_ids, attention_mask, image_sizes=image_sizes)

    def forward(self,
                images: torch.Tensor,                # (B,3,H,W)
                input_ids: torch.Tensor,             # (B,L)
                attention_mask: torch.Tensor,        # (B,L), 1 for valid, 0 for pad
                image_sizes: Optional[List[Tuple[int, int]]] = None,
                targets_xyxy: Optional[torch.Tensor] = None,  # (B,4) - can be normalized [0,1] or absolute
                loss_lambda: float = 1.0,
                compute_loss: bool = False,
                ) -> Dict[str, torch.Tensor]:
        """
        If compute_loss=True and targets provided, returns dict with losses in addition to predictions.
        - image_sizes: list of (H, W). If provided, we also return absolute xyxy in pixels.
        """
        B = images.size(0)

        # 1) Visual features
        Fv = self.backbone(images)  # (B, v_dim, H/32, W/32)

        # 2) Text features
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        Ft_seq = txt_out.last_hidden_state                        # (B, L, t_dim)
        # key_padding_mask expects True for PAD -> invert attention_mask
        t_mask = (attention_mask == 0)

        # 3) LVFE
        Fvt = self.lvfe(Fv, Ft_seq, t_mask=t_mask)                # (B, 2*v_dim, H/32, W/32)

        # 4) Add positional encoding
        if self.use_posenc:
            Fvt = Fvt + self.posenc(Fvt)

        # 5) VLF stacker
        q = self.vlf(Fvt)                                        # (B, hidden)

        # 6) Head -> normalized (cx, cy, w, h)
        pred_cxcywh = self.head(q)                                # (B, 4)
        pred_xyxy_norm = cxcywh_to_xyxy(pred_cxcywh).clamp(0.0, 1.0)  # (B, 4) in [0,1]

        out: Dict[str, torch.Tensor] = {
            "pred_cxcywh_norm": pred_cxcywh,
            "pred_xyxy_norm": pred_xyxy_norm,
        }

        # return absolute coords if image_sizes provided
        if image_sizes is not None:
            # image_sizes: list of (H, W)
            hw = torch.tensor([(w, h) for (h, w) in image_sizes], dtype=pred_xyxy_norm.dtype,
                              device=pred_xyxy_norm.device)
            pred_xyxy_abs = _denormalize_xyxy(pred_xyxy_norm, hw)  # (B, 4) pixels
            out["pred_xyxy_abs"] = pred_xyxy_abs

        # Optionally compute loss here (self-contained)
        if compute_loss and targets_xyxy is not None:
            if image_sizes is not None and targets_xyxy.max() > 2.0:  # looks like absolute pixels
                # normalize GT to [0,1]
                hw = torch.tensor([(w, h) for (h, w) in image_sizes], dtype=targets_xyxy.dtype,
                                  device=targets_xyxy.device)
                tgt_xyxy_norm = _normalize_xyxy_abs(targets_xyxy, hw)
            else:
                tgt_xyxy_norm = targets_xyxy

            # Smooth L1 on xyxy
            l1 = F.smooth_l1_loss(pred_xyxy_norm, tgt_xyxy_norm, reduction='none').sum(-1).mean()

            # GIoU
            giou = generalized_box_iou(pred_xyxy_norm, tgt_xyxy_norm)
            # generalized_box_iou returns pairwise (B,B); take diag
            giou_diag = giou.diag()
            giou_loss = (1.0 - giou_diag).mean()

            # EIoU
            eiou = _eiou_loss(pred_xyxy_norm, tgt_xyxy_norm).mean()

            total = l1 + loss_lambda * giou_loss + loss_lambda * eiou
            out["loss_l1"] = l1
            out["loss_giou"] = giou_loss
            out["loss_eiou"] = eiou
            out["loss_total"] = total

        return out

    # ---------- Optim Groups ----------

    def get_param_groups(self, lr_backbone=1e-5, lr_text=1e-5, lr_head=1e-4, weight_decay=1e-4):
        """
        Convenience helper to create 3 LR groups:
        - backbone (CNN)
        - text encoder (BERT)
        - fusion + head (LVFE, VLF, BBoxHead, posenc)
        """
        groups = [
            {"params": [p for p in self.backbone.parameters() if p.requires_grad],
             "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": [p for p in self.text_encoder.parameters() if p.requires_grad],
             "lr": lr_text, "weight_decay": weight_decay},
            {"params": [p for m in [self.lvfe, self.vlf, self.head] for p in m.parameters() if p.requires_grad] +
                       ([p for p in self.posenc.parameters()] if self.use_posenc else []),
             "lr": lr_head, "weight_decay": weight_decay},
        ]
        return groups
