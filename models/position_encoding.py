"""
Various positional encodings for the transformer.
Refactor for MGVLF:
- PositionEmbeddingSine: 2D sine/cos encoding for images (NestedTensor).
- PositionEmbeddingLearned: learned absolute positions for 1D sequences (late fusion).
"""
from typing import Optional
import math
import torch
from torch import nn

from utils.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    Standard 2D sine/cos positional encoding, similar to DETR.
    Input:
        tensor_list: NestedTensor with fields:
            - tensors: (B, C, H, W)
            - mask:    (B, H, W) boolean, True = padded
    Output:
        pos: (B, C_pos, H, W) where C_pos = 2 * num_pos_feats
    """

    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000, normalize: bool = True, scale: Optional[float] = None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None, "Mask is required for PositionEmbeddingSine"

        # not_mask: 1 where valid, 0 where pad
        not_mask = (~mask).to(torch.float32)  # (B,H,W)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (B,H,W,F)
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B,2F,H,W)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Learned absolute positional embeddings for 1D sequences.
    Designed for inputs shaped (B, C, T) and returns (B, T, C).
    """

    def __init__(self, num_pos_feats: int = 256, max_positions: int = 1024):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.max_positions = max_positions
        self.embed = nn.Embedding(max_positions, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight, -0.02, 0.02)

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, T)
        returns: (B, T, C) positional embeddings aligned to T
        """
        b, c, t = x.shape
        if t > self.max_positions:
            # extend embedding on-the-fly if sequence is longer than preset
            self._extend(t)

        positions = torch.arange(t, device=x.device, dtype=torch.long)  # (T,)
        pos = self.embed(positions)                                     # (T, C)
        pos = pos.unsqueeze(0).expand(b, t, -1).contiguous()            # (B, T, C)
        return pos

    def _extend(self, new_max_len: int):
        old_weight = self.embed.weight.data
        self.max_positions = new_max_len
        self.embed = nn.Embedding(self.max_positions, self.num_pos_feats).to(old_weight.device)
        nn.init.uniform_(self.embed.weight, -0.02, 0.02)
        with torch.no_grad():
            self.embed.weight[: old_weight.size(0)].copy_(old_weight)


def build_position_encoding(args, position_embedding: str):
    """
    Factory:
      - 'sine'    -> PositionEmbeddingSine with num_pos_feats = hidden_dim // 2
      - 'learned' -> PositionEmbeddingLearned with num_pos_feats = hidden_dim
    """
    hidden_dim = getattr(args, "hidden_dim", 256)
    if position_embedding in ("sine", "v2"):
        num_pos_feats = hidden_dim // 2
        return PositionEmbeddingSine(num_pos_feats=num_pos_feats, normalize=True)
    elif position_embedding in ("learned", "v3"):
        return PositionEmbeddingLearned(num_pos_feats=hidden_dim)
    else:
        raise ValueError(f"Unknown position_embedding: {position_embedding}")
