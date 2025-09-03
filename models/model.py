# models/model.py
import math
import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoModel, AutoConfig


# -----------------------------------------------------
# Utils
# -----------------------------------------------------
def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    """(cx,cy,w,h) in [0,1] -> (x1,y1,x2,y2) in [0,1]."""
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


# -----------------------------------------------------
# 2D Sine Positional Encoding
# -----------------------------------------------------
class SinePositionalEncoding2D(nn.Module):
    def __init__(self, dim=256, temperature=10000):
        super().__init__()
        self.dim = dim
        self.temperature = temperature

    def forward(self, B, H, W, device):
        """Return (B, H*W, dim)"""
        y_embed = torch.arange(H, device=device).unsqueeze(1).repeat(1, W)
        x_embed = torch.arange(W, device=device).unsqueeze(0).repeat(H, 1)

        dim_t = torch.arange(self.dim // 2, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.dim // 2))

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)

        pos = torch.cat((pos_y, pos_x), dim=-1)  # (H, W, dim)
        pos = pos.view(H * W, self.dim).unsqueeze(0).repeat(B, 1, 1)
        return pos


# -----------------------------------------------------
# Visual Backbone: ResNet50
# -----------------------------------------------------
class ResNet50(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        backbone = models.resnet50(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1,
            backbone.relu, backbone.maxpool
        )
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        self.proj = nn.Conv2d(2048, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)           # (B,2048,H/32,W/32)
        return self.proj(x)          # (B,256,H/32,W/32)


# -----------------------------------------------------
# Visual Transformer
# -----------------------------------------------------
class VisualTransformer(nn.Module):
    def __init__(self, dim=256, heads=8, layers=3, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, pos_enc):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)   # (B,N,C)
        tokens = tokens + pos_enc               # add 2D positional encoding
        out = self.encoder(tokens)
        return self.norm(out)                   # (B,N,C)


# -----------------------------------------------------
# Language Transformer refine
# -----------------------------------------------------
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

    def forward(self, Ft, attn_mask):
        X = self.proj(Ft)  # (B,L,256)
        padmask = (attn_mask == 0)
        X = self.encoder(X, src_key_padding_mask=padmask)
        return self.norm(X)  # (B,L,256)


# -----------------------------------------------------
# LVFE
# -----------------------------------------------------
class LVFEBlock(nn.Module):
    def __init__(self, v_dim=256, t_dim=256, qkv_dim=512, heads=8, dropout=0.1):
        super().__init__()
        self.q_proj = nn.Linear(v_dim, qkv_dim)
        self.k_proj = nn.Linear(t_dim, qkv_dim)
        self.v_proj = nn.Linear(t_dim, qkv_dim)
        self.attn = nn.MultiheadAttention(
            qkv_dim, heads, dropout=dropout, batch_first=True
        )
        self.out_proj = nn.Linear(qkv_dim, v_dim)
        self.norm = nn.LayerNorm(v_dim)

    def forward(self, Fv, Ft):
        q = self.q_proj(Fv)
        k = self.k_proj(Ft)
        v = self.v_proj(Ft)
        out, _ = self.attn(q, k, v)
        return self.norm(Fv + self.out_proj(out))


class LVFE(nn.Module):
    def __init__(self, v_dim=256, t_dim=256, qkv_dim=512, heads=8, layers=3, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            LVFEBlock(v_dim, t_dim, qkv_dim, heads, dropout) for _ in range(layers)
        ])

    def forward(self, Fv, Ft):
        for blk in self.blocks:
            Fv = blk(Fv, Ft)      # (B,Nv,256)
        # Expand Ft để concat
        B, Nv, C = Fv.shape
        Ft_exp = Ft.mean(1, keepdim=True).expand(B, Nv, C)
        return torch.cat([Fv, Ft_exp], dim=-1)  # (B,Nv,512)


# -----------------------------------------------------
# VLF Module
# -----------------------------------------------------
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
        self.q_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.q_norm = nn.LayerNorm(dim)
        self.query_repeats = query_repeats

    def forward(self, Fvt):
        X = self.encoder(Fvt)
        X = self.norm(X)
        B = X.size(0)
        Tq = self.Tq.expand(B, -1, -1)
        for _ in range(self.query_repeats):
            new, _ = self.q_attn(Tq, X, X)
            Tq = self.q_norm(Tq + new)
        return Tq.squeeze(1)


# -----------------------------------------------------
# Prediction Head
# -----------------------------------------------------
class PredictionHead(nn.Module):
    def __init__(self, in_dim=512, hidden=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4), nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)


# -----------------------------------------------------
# Full MGVLF
# -----------------------------------------------------
class MGVLF(nn.Module):
    def __init__(self, text_model="distilbert-base-uncased"):
        super().__init__()
        self.backbone = ResNet50()
        self.posenc = SinePositionalEncoding2D(dim=256)
        self.vis_tr = VisualTransformer()
        cfg = AutoConfig.from_pretrained(text_model)
        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.lang_tr = LanguageTransformer(in_dim=cfg.hidden_size)
        self.lvfe = LVFE()
        self.vlf = VLFModule()
        self.head = PredictionHead()

    def forward(self, images, input_ids, attention_mask):
        # Visual branch
        Fv = self.backbone(images)  # (B,256,H/32,W/32)
        B, C, H, W = Fv.shape
        pos = self.posenc(B, H, W, Fv.device)
        Fv_tokens = self.vis_tr(Fv, pos)  # (B,Nv,256)

        # Language branch
        out = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        Ft = self.lang_tr(out.last_hidden_state, attention_mask)  # (B,L,256)

        # LVFE
        Fvt = self.lvfe(Fv_tokens, Ft)   # (B,Nv,512)

        # VLF
        Tq = self.vlf(Fvt)               # (B,512)

        # Prediction
        return self.head(Tq)             # (B,4) normalized bbox
