# models/backbone_inject.py
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class QABM(nn.Module):
    """
    Query-Aware Backbone Modulator (QABM)
    - Injects language early into the visual backbone (per scale).
    - Spatial gating (token-aggregated) + channel FiLM; optional directional prior.
    Inputs:
        feats: List[Tensor] of length S; each is (B, C, H, W) from backbone stages (e.g., P2..P5)
        word_tokens: (B, L, Dt) text token embeddings
        sent_vec:    (B, Dt)   sentence-level embedding
        attn_mask:   (B, L)    1 for valid, 0 for pad  (optional)
    Returns:
        mod_feats:   List[Tensor] same shapes as feats, but language-conditioned
        aux:         dict with optional 'gates' (List[(B,1,H,W)]) for seeding decoder
    """
    def __init__(self, c_vis: int = 256, d_txt: int = 768, use_direction: bool = True):
        super().__init__()
        self.use_direction = use_direction
        self.q_proj = nn.Conv2d(c_vis, c_vis, kernel_size=1, bias=False)     # visual query
        self.k_proj = nn.Linear(d_txt, c_vis, bias=False)                    # text key
        self.v_proj = nn.Linear(d_txt, c_vis, bias=False)                    # text value (for gate strength)
        self.gamma_mlp = nn.Sequential(nn.Linear(d_txt, c_vis), nn.ReLU(), nn.Linear(c_vis, c_vis))
        self.beta_mlp  = nn.Sequential(nn.Linear(d_txt, c_vis), nn.ReLU(), nn.Linear(c_vis, c_vis))
        if use_direction:
            self.dir_mlp = nn.Sequential(nn.Linear(d_txt, c_vis//4), nn.ReLU(), nn.Linear(c_vis//4, 2))  # 2D dir

    @staticmethod
    def _coord_grid(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Returns normalized coords grid (B, 2, H, W) in [-1,1]."""
        ys = torch.linspace(-1, 1, steps=H, device=device)
        xs = torch.linspace(-1, 1, steps=W, device=device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # (B,2,H,W)
        return grid

    def _spatial_gate(self, F: torch.Tensor, T: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        F: (B,C,H,W), T: (B,L,Dt) -> gate g in (B,1,H,W)
        computes token-aggregated attention over spatial positions (no MHA to remain lightweight).
        """
        B, C, H, W = F.shape
        Q = self.q_proj(F).flatten(2).transpose(1, 2)             # (B, H*W, C)
        K = self.k_proj(T)                                        # (B, L, C)
        V = self.v_proj(T)                                        # (B, L, C)  (for gate magnitude)
        attn_logits = torch.einsum("bnc,blc->bnl", Q, K) / (C ** 0.5)  # (B, H*W, L)
        if attn_mask is not None:
            # mask: 1 valid, 0 pad -> set pad to -inf
            mask = (1 - attn_mask).bool().unsqueeze(1)            # (B,1,L)
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))
        attn = attn_logits.softmax(dim=-1)                        # (B, H*W, L)
        # aggregate value vectors -> scalar gate via norm
        gate_vec = torch.einsum("bnl,blc->bnc", attn, V)          # (B, H*W, C)
        gate = gate_vec.norm(dim=-1, keepdim=True)                # (B, H*W, 1)
        gate = gate / (gate.amax(dim=1, keepdim=True) + 1e-6)     # normalize per image
        gate = gate.transpose(1, 2).reshape(B, 1, H, W)           # (B,1,H,W)
        return gate

    def _direction_prior(self, F: torch.Tensor, sent_vec: torch.Tensor) -> torch.Tensor:
        """Project sentence to a 2D direction vector and align with coord grid -> (B,1,H,W) in (0,1)."""
        B, C, H, W = F.shape
        vec2 = self.dir_mlp(sent_vec)  # (B,2)
        vec2 = torch.nn.functional.normalize(vec2, dim=-1)
        grid = self._coord_grid(B, H, W, F.device)  # (B,2,H,W)
        proj = (grid * vec2.unsqueeze(-1).unsqueeze(-1)).sum(dim=1, keepdim=True)  # (B,1,H,W)
        return torch.sigmoid(proj)  # (B,1,H,W)

    def forward(
        self,
        feats: List[torch.Tensor],
        word_tokens: torch.Tensor,
        sent_vec: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], dict]:
        mod_feats, gates = [], []
        gamma = self.gamma_mlp(sent_vec)[:, :, None, None]  # (B,C,1,1)
        beta  = self.beta_mlp(sent_vec)[:, :, None, None]   # (B,C,1,1)
        for Fi in feats:
            gi = self._spatial_gate(Fi, word_tokens, attn_mask)     # (B,1,H,W)
            if self.use_direction:
                di = self._direction_prior(Fi, sent_vec)            # (B,1,H,W)
                gi = gi * di
            Fi_mod = (gi * Fi) * (1 + gamma) + beta                 # Fi_mod keeps (B,C,H,W)
            mod_feats.append(Fi_mod)
            gates.append(gi)
        return mod_feats, {"gates": gates}
