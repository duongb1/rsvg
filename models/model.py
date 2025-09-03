import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.hub import load_state_dict_from_url
from transformers import AutoModel, AutoConfig


# =========================
# Positional Encoding (2D sine, DETR-style)
# =========================
class SinePositionalEncoding2D(nn.Module):
    """
    Tạo PE 2D kiểu DETR, trả về (B, N, C) để cộng vào token chuỗi.
    """
    def __init__(self, num_feats: int = 128, temperature: int = 10000, normalize: bool = True, scale: Optional[float] = None):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        mask: (B, H, W) boolean, True = pad. (giống DETR)
        return: (B, H, W, 2*num_feats) -> sẽ reshape thành (B, N, C)
        """
        assert mask is not None
        not_mask = ~mask  # False ở pad
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (B,H,W,C)
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=4).flatten(3)  # (B,H,W,2*C)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3)  # (B,H,W,4*num_feats)
        return pos  # (B,H,W,C_pe)


def build_sine_pos_for_feature(mask: torch.Tensor, c_model: int) -> torch.Tensor:
    """
    mask: (B, H, W) boolean
    c_model: d_model của encoder (256)
    return: (B, N, C) positional encodes
    """
    assert c_model % 2 == 0
    pe2d = SinePositionalEncoding2D(num_feats=c_model // 4)  # vì output là 4*num_feats
    pos2d = pe2d(mask)  # (B,H,W,C)
    B, H, W, C = pos2d.shape
    assert C == c_model, f"PE channel {C} must equal model dim {c_model}"
    return pos2d.view(B, H * W, C)  # (B,N,C)


# =========================
# ResNet50 backbone init from DETR (C5 -> 1x1 -> 256)
# =========================
class ResNet50FromDETR(nn.Module):
    DETR_R50_URL = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"

    def __init__(self, out_channels: int = 256, init_from_detr: bool = True):
        super().__init__()
        backbone = models.resnet50(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        self.proj = nn.Conv2d(2048, out_channels, kernel_size=1)

        if init_from_detr:
            self._init_from_detr()

    @torch.no_grad()
    def _init_from_detr(self):
        sd = load_state_dict_from_url(self.DETR_R50_URL, map_location="cpu")
        detrsd = sd.get("model", sd)

        # map weight backbone.0.body.* -> resnet50
        mapped = {}
        prefix = "backbone.0.body."
        for k, v in detrsd.items():
            if k.startswith(prefix):
                new_k = k[len(prefix):]  # layer1.0.conv1.weight ...
                mapped[new_k] = v

        temp = models.resnet50(weights=None)
        _ = temp.load_state_dict(mapped, strict=False)

        # copy stem
        self.stem[0].weight.copy_(temp.conv1.weight)
        self.stem[1].weight.copy_(temp.bn1.weight)
        self.stem[1].bias.copy_(temp.bn1.bias)
        self.stem[1].running_mean.copy_(temp.bn1.running_mean)
        self.stem[1].running_var.copy_(temp.bn1.running_var)

        # copy từng block của layer1..4
        for i in range(1, 5):
            src = getattr(temp, f"layer{i}")
            dst = getattr(self, f"layer{i}")
            for (n_dst, m_dst), (n_src, m_src) in zip(dst.named_modules(), src.named_modules()):
                for attr in ("weight", "bias", "running_mean", "running_var"):
                    if hasattr(m_dst, attr) and hasattr(m_src, attr):
                        try:
                            getattr(m_dst, attr).copy_(getattr(m_src, attr))
                        except Exception:
                            pass  # mismatch -> bỏ qua

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        return:
          feat: (B, 256, H/32, W/32)
          mask: (B, H/32, W/32) boolean (False = valid, True = pad) — ở đây giả sử ảnh đã letterbox, mask=False toàn bộ
        """
        B, _, H, W = x.shape
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)        # (B,2048,H/32,W/32)
        feat = self.proj(x)       # (B,256,H/32,W/32)
        mask = torch.zeros((B, feat.shape[-2], feat.shape[-1]), dtype=torch.bool, device=feat.device)
        return feat, mask


# =========================
# Visual Transformer Encoder (init from DETR)
# =========================
class VisualTransformer(nn.Module):
    DETR_R50_URL = "https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth"

    def __init__(self, dim=256, heads=8, layers=6, dropout=0.1, ff_dim: Optional[int] = None, init_from_detr: bool = True):
        super().__init__()
        if ff_dim is None:
            ff_dim = dim * 4  # chuẩn PyTorch
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        if init_from_detr:
            self._load_from_detr(dim)

    @torch.no_grad()
    def _load_from_detr(self, dim: int):
        sd = load_state_dict_from_url(self.DETR_R50_URL, map_location="cpu")
        detrsd = sd.get("model", sd)

        mysd = self.state_dict()
        mapped = {}
        # Map per layer
        for i in range(len(self.encoder.layers)):
            # self-attn QKV + out_proj
            for wb in ["in_proj_weight", "in_proj_bias", "out_proj.weight", "out_proj.bias"]:
                key_det = f"transformer.encoder.layers.{i}.self_attn.{wb}"
                key_me = f"encoder.layers.{i}.self_attn.{wb}"
                if key_det in detrsd and key_me in mysd and detrsd[key_det].shape == mysd[key_me].shape:
                    mapped[key_me] = detrsd[key_det]
            # FFN
            for wb in ["linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"]:
                key_det = f"transformer.encoder.layers.{i}.{wb}"
                key_me = f"encoder.layers.{i}.{wb}"
                if key_det in detrsd and key_me in mysd and detrsd[key_det].shape == mysd[key_me].shape:
                    mapped[key_me] = detrsd[key_det]
            # Norms
            for norm in ["norm1", "norm2"]:
                for wb in ["weight", "bias"]:
                    key_det = f"transformer.encoder.layers.{i}.{norm}.{wb}"
                    key_me = f"encoder.layers.{i}.{norm}.{wb}"
                    if key_det in detrsd and key_me in mysd and detrsd[key_det].shape == mysd[key_me].shape:
                        mapped[key_me] = detrsd[key_det]

        mysd.update(mapped)
        self.load_state_dict(mysd, strict=False)

    def forward(self, feat_2d: torch.Tensor, pos_tokens: torch.Tensor) -> torch.Tensor:
        """
        feat_2d: (B, 256, H, W)
        pos_tokens: (B, N, 256) — sine PE đã flatten
        return: (B, N, 256) encoded tokens
        """
        B, C, H, W = feat_2d.shape
        tokens = feat_2d.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens = tokens + pos_tokens                 # add PE
        out = self.encoder(tokens)                   # (B, N, C)
        return self.norm(out)


# =========================
# Language Transformer refine (proj 768→256 + 2 enc layers)
# =========================
class LanguageTransformer(nn.Module):
    def __init__(self, in_dim=768, out_dim=256, heads=8, layers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=heads, dim_feedforward=out_dim * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, Ft: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Ft: (B, L, in_dim)
        attn_mask: (B, L) with 1=valid, 0=pad
        """
        X = self.proj(Ft)                  # (B, L, 256)
        padmask = (attn_mask == 0)         # True = pad
        X = self.encoder(X, src_key_padding_mask=padmask)
        return self.norm(X)                # (B, L, 256)


# =========================
# LVFE (3 blocks): Q=visual, K/V=text
# =========================
class LVFEBlock(nn.Module):
    def __init__(self, v_dim=256, t_dim=256, qkv_dim=512, heads=8, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(v_dim, qkv_dim)
        self.k_proj = nn.Linear(t_dim, qkv_dim)
        self.v_proj = nn.Linear(t_dim, qkv_dim)
        self.attn = nn.MultiheadAttention(qkv_dim, heads, dropout=dropout, batch_first=True)
        self.out_proj = nn.Linear(qkv_dim, v_dim)
        self.norm = nn.LayerNorm(v_dim)

    def forward(self, Fv: torch.Tensor, Ft: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(Fv)               # (B, Nv, qkv)
        k = self.k_proj(Ft)               # (B, Lt, qkv)
        v = self.v_proj(Ft)               # (B, Lt, qkv)
        out, _ = self.attn(q, k, v)       # (B, Nv, qkv)
        return self.norm(Fv + self.out_proj(out))


class LVFE(nn.Module):
    def __init__(self, v_dim=256, t_dim=256, qkv_dim=512, heads=8, layers=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            LVFEBlock(v_dim, t_dim, qkv_dim, heads, dropout) for _ in range(layers)
        ])

    def forward(self, Fv_tokens: torch.Tensor, Ft_tokens: torch.Tensor) -> torch.Tensor:
        """
        Fv_tokens: (B, Nv, 256) — visual tokens
        Ft_tokens: (B, Lt, 256) — text tokens (refined)
        Return:
            Fvt: (B, Nv, 512) — concat enhanced visual with expanded text
        """
        Fv = Fv_tokens
        for blk in self.blocks:
            Fv = blk(Fv, Ft_tokens)       # (B,Nv,256)
        # Expand text (mean) -> concat
        Ft_mean = Ft_tokens.mean(dim=1, keepdim=True)  # (B,1,256)
        Ft_exp  = Ft_mean.expand(Fv.size(0), Fv.size(1), Ft_mean.size(-1))  # (B,Nv,256)
        return torch.cat([Fv, Ft_exp], dim=-1)  # (B,Nv,512)


# =========================
# VLF: 4 encoder layers + learnable query Tq (repeat 4)
# =========================
class VLFModule(nn.Module):
    def __init__(self, dim=512, heads=8, layers=4, query_repeats=4, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.Tq = nn.Parameter(torch.randn(1, 1, dim))
        self.q_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.q_norm = nn.LayerNorm(dim)
        self.query_repeats = query_repeats

    def forward(self, Fvt: torch.Tensor) -> torch.Tensor:
        """
        Fvt: (B, Nv, 512)
        return: (B, 512) fused vector
        """
        X = self.encoder(Fvt)         # (B,Nv,512)
        X = self.norm(X)
        B = X.size(0)
        Tq = self.Tq.expand(B, -1, -1)  # (B,1,512)
        for _ in range(self.query_repeats):
            new, _ = self.q_attn(Tq, X, X)
            Tq = self.q_norm(Tq + new)
        return Tq.squeeze(1)          # (B,512)


# =========================
# Prediction Head (MLP -> sigmoid 4-d)
# =========================
class PredictionHead(nn.Module):
    def __init__(self, in_dim=512, hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 4)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(x))  # (B,4) in [0,1] (cx,cy,w,h)


# =========================
# Full MGVLF model
# =========================
def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


class MGVLF(nn.Module):
    """
    VGRSS-style:
      - ResNet50FromDETR (C5 -> 256)
      - Visual Transformer (6L, init from DETR)
      - BERT + LanguageTransformer (proj/refine)
      - LVFE (3 blocks)
      - VLF (4 encoder layers + query repeats=4)
      - Head -> (cx,cy,w,h) in [0,1]
    """
    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        freeze_text_encoder: bool = False,
        freeze_backbone: bool = False,
        v_dim: int = 256,
        heads: int = 8,
        lvfe_layers: int = 3,
        vlf_layers: int = 4,
        vlf_query_repeats: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Visual
        self.backbone = ResNet50FromDETR(out_channels=v_dim, init_from_detr=True)
        self.vis_enc = VisualTransformer(dim=v_dim, heads=heads, layers=6, dropout=dropout, init_from_detr=True)

        # Text
        cfg = AutoConfig.from_pretrained(text_model_name)
        self.text_hidden = cfg.hidden_size
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.lang_tr = LanguageTransformer(in_dim=self.text_hidden, out_dim=v_dim, heads=heads, layers=2, dropout=dropout)

        # Fusion
        self.lvfe = LVFE(v_dim=v_dim, t_dim=v_dim, qkv_dim=512, heads=heads, layers=lvfe_layers, dropout=dropout)
        self.vlf = VLFModule(dim=2 * v_dim, heads=heads, layers=vlf_layers, query_repeats=vlf_query_repeats, dropout=dropout)
        self.head = PredictionHead(in_dim=2 * v_dim, hidden=2 * v_dim)

        # Freezing options
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            for p in self.vis_enc.parameters():
                p.requires_grad_(False)
        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,          # (B,3,H,W)
        input_ids: torch.Tensor,       # (B,L)
        attention_mask: torch.Tensor,  # (B,L) 1=valid, 0=pad
    ) -> Dict[str, torch.Tensor]:
        # 1) Visual backbone
        feat_2d, mask_2d = self.backbone(images)        # (B,256,H/32,W/32), (B,h,w)

        # 2) Positional encoding + visual encoder
        pos_tok = build_sine_pos_for_feature(mask_2d, c_model=feat_2d.size(1))  # (B,N,256)
        Fv_tokens = self.vis_enc(feat_2d, pos_tok)       # (B,N,256)

        # 3) Text branch
        txt_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        Ft_tokens = self.lang_tr(txt_out.last_hidden_state, attention_mask)  # (B,L,256)

        # 4) LVFE: text-guided visual enhancement + expand text + concat
        Fvt_tokens = self.lvfe(Fv_tokens, Ft_tokens)     # (B,N,512)

        # 5) VLF: 4-layer encoder + learnable Tq (repeat 4)
        fused = self.vlf(Fvt_tokens)                     # (B,512)

        # 6) Head -> bbox
        pred_cxcywh = self.head(fused)                   # (B,4) in [0,1]
        pred_xyxy = cxcywh_to_xyxy(pred_cxcywh).clamp(0.0, 1.0)

        return {
            "pred_cxcywh_norm": pred_cxcywh,
            "pred_xyxy_norm": pred_xyxy,
        }

    # ---- Param groups for optimizer (paper: head 1e-4; backbone/text 1e-5) ----
    def get_param_groups(self, lr_backbone=1e-5, lr_text=1e-5, lr_head=1e-4, weight_decay=1e-4):
        groups = [
            {"params": [p for p in list(self.backbone.parameters()) + list(self.vis_enc.parameters()) if p.requires_grad],
             "lr": lr_backbone, "weight_decay": weight_decay},
            {"params": [p for p in self.text_encoder.parameters() if p.requires_grad] + [p for p in self.lang_tr.parameters() if p.requires_grad],
             "lr": lr_text, "weight_decay": weight_decay},
            {"params": [p for m in [self.lvfe, self.vlf, self.head] for p in m.parameters() if p.requires_grad],
             "lr": lr_head, "weight_decay": weight_decay},
        ]
        return groups
