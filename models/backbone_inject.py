# models/backbone_inject.py
from typing import List, Tuple, Optional
import torch
import torch.nn as nn

class QABM(nn.Module):
    """
    Query-Aware Backbone Modulator (QABM)
    - Injects language early into the visual backbone (per scale).
    - Spatial gating (token-aggregated) + channel FiLM; optional directional prior.

    Inputs:
        feats: List[Tensor], length S, mỗi tensor có shape (B, C, H, W) từ backbone (ví dụ P2..P5)
        word_tokens: (B, L, Dt) text token embeddings
        sent_vec:    (B, Dt) sentence-level embedding
        attn_mask:   (B, L) attention mask (1 = valid, 0 = pad) (optional)

    Outputs:
        mod_feats: List[Tensor] đã điều chỉnh, cùng shape với feats
        aux: dict chứa "gates": List[(B,1,H,W)] cho từng stage
    """

    def __init__(self, c_vis_list=None, d_txt=768, use_direction=True, out_dim=256):
        super().__init__()
        self.use_direction = use_direction
        self.out_dim = out_dim

        if c_vis_list is not None:
            # Nếu backbone ra nhiều kênh (512, 1024, 2048) -> ép về out_dim (256)
            self.q_proj = nn.ModuleList([nn.Conv2d(c, out_dim, 1, bias=False) for c in c_vis_list])
            self.use_internal_proj = True
        else:
            # Nếu backbone đã ép ngoài về out_dim
            self.q_proj = nn.Conv2d(out_dim, out_dim, 1, bias=False)
            self.use_internal_proj = False

        self.k_proj = nn.Linear(d_txt, out_dim, bias=False)
        self.v_proj = nn.Linear(d_txt, out_dim, bias=False)

        self.gamma_mlp = nn.Sequential(
            nn.Linear(d_txt, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )
        self.beta_mlp = nn.Sequential(
            nn.Linear(d_txt, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim)
        )

        if use_direction:
            self.dir_mlp = nn.Sequential(
                nn.Linear(d_txt, out_dim // 4), nn.ReLU(), nn.Linear(out_dim // 4, 2)
            )

    @staticmethod
    def _coord_grid(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Sinh grid tọa độ chuẩn hóa [-1,1], shape (B,2,H,W)."""
        ys = torch.linspace(-1, 1, steps=H, device=device)
        xs = torch.linspace(-1, 1, steps=W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
        return grid  # (B,2,H,W)

    def _direction_prior(self, F: torch.Tensor, sent_vec: torch.Tensor) -> torch.Tensor:
        """Project sentence → vector 2D rồi chiếu vào grid để tạo mask định hướng."""
        B, C, H, W = F.shape
        vec2 = self.dir_mlp(sent_vec)  # (B,2)
        vec2 = torch.nn.functional.normalize(vec2, dim=-1)  # chuẩn hóa vector
        grid = self._coord_grid(B, H, W, F.device)  # (B,2,H,W)
        proj = (grid * vec2.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)  # (B,1,H,W)
        return torch.sigmoid(proj)

    def _spatial_gate(self, idx, F, T, attn_mask):
        """
        Spatial gate: học mặt nạ không gian dựa trên cross-attn giữa feature map và từ.
        F: (B,C,H,W), T: (B,L,Dt)
        """
        B, _, H, W = F.shape
        if self.use_internal_proj:
            Q = self.q_proj[idx](F).flatten(2).transpose(1, 2)  # (B,HW,C)
        else:
            Q = self.q_proj(F).flatten(2).transpose(1, 2)
        K = self.k_proj(T)
        V = self.v_proj(T)

        attn_logits = torch.einsum("bnc,blc->bnl", Q, K) / (Q.size(-1) ** 0.5)
        if attn_mask is not None:
            mask = (1 - attn_mask).bool().unsqueeze(1)  # (B,1,L)
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))
        attn = attn_logits.softmax(dim=-1)

        gate_vec = torch.einsum("bnl,blc->bnc", attn, V)  # (B,HW,C)
        gate = gate_vec.norm(dim=-1, keepdim=True)  # (B,HW,1)
        gate = gate / (gate.amax(dim=1, keepdim=True) + 1e-6)
        gate = gate.transpose(1, 2).reshape(B, 1, H, W)
        return gate

    def forward(self, feats: List[torch.Tensor], word_tokens: torch.Tensor,
                sent_vec: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
                ) -> Tuple[List[torch.Tensor], dict]:
        mod_feats, gates = [], []
        gamma = self.gamma_mlp(sent_vec)[:, :, None, None]
        beta = self.beta_mlp(sent_vec)[:, :, None, None]
        for i, Fi in enumerate(feats):
            gi = self._spatial_gate(i, Fi, word_tokens, attn_mask)
            if self.use_direction:
                di = self._direction_prior(Fi, sent_vec)
                gi = gi * di
            Fi_mod = (gi * Fi) * (1 + gamma) + beta
            mod_feats.append(Fi_mod)
            gates.append(gi)
        return mod_feats, {"gates": gates}
