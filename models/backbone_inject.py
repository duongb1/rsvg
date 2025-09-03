# models/backbone_inject.py
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class QABM(nn.Module):
    def __init__(self, c_vis_list=None, d_txt=768, use_direction=True, out_dim=256):
        super().__init__()
        self.use_direction = use_direction
        self.out_dim = out_dim

        if c_vis_list is not None:
            # Tự ép kênh nội bộ cho từng stage
            self.q_proj = nn.ModuleList([nn.Conv2d(c, out_dim, 1, bias=False) for c in c_vis_list])
            self.use_internal_proj = True
        else:
            # Giả định input đã ép về out_dim rồi
            self.q_proj = nn.Conv2d(out_dim, out_dim, 1, bias=False)
            self.use_internal_proj = False

        self.k_proj = nn.Linear(d_txt, out_dim, bias=False)
        self.v_proj = nn.Linear(d_txt, out_dim, bias=False)
        self.gamma_mlp = nn.Sequential(nn.Linear(d_txt, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        self.beta_mlp  = nn.Sequential(nn.Linear(d_txt, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        if use_direction:
            self.dir_mlp = nn.Sequential(nn.Linear(d_txt, out_dim // 4), nn.ReLU(), nn.Linear(out_dim // 4, 2))

    def _spatial_gate(self, idx, F, T, attn_mask):
        B, _, H, W = F.shape
        if self.use_internal_proj:
            Q = self.q_proj[idx](F).flatten(2).transpose(1, 2)  # ép kênh
        else:
            Q = self.q_proj(F).flatten(2).transpose(1, 2)       # assume đã 256
        K = self.k_proj(T)
        V = self.v_proj(T)
        attn_logits = torch.einsum("bnc,blc->bnl", Q, K) / (Q.size(-1) ** 0.5)
        if attn_mask is not None:
            mask = (1 - attn_mask).bool().unsqueeze(1)
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))
        attn = attn_logits.softmax(dim=-1)
        gate_vec = torch.einsum("bnl,blc->bnc", attn, V)
        gate = gate_vec.norm(dim=-1, keepdim=True)
        gate = gate / (gate.amax(dim=1, keepdim=True) + 1e-6)
        gate = gate.transpose(1, 2).reshape(B, 1, H, W)
        return gate

    def forward(self, feats, word_tokens, sent_vec, attn_mask=None):
        mod_feats, gates = [], []
        gamma = self.gamma_mlp(sent_vec)[:, :, None, None]
        beta  = self.beta_mlp(sent_vec)[:, :, None, None]
        for i, Fi in enumerate(feats):
            gi = self._spatial_gate(i, Fi, word_tokens, attn_mask)
            if self.use_direction:
                di = self._direction_prior(Fi, sent_vec)
                gi = gi * di
            Fi_mod = (gi * Fi) * (1 + gamma) + beta
            mod_feats.append(Fi_mod)
            gates.append(gi)
        return mod_feats, {"gates": gates}
