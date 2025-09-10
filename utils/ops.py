# utils/ops.py
import torch

def batched_index_select(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    tensor:  (B, N, C)
    indices: (B, k) int64
    return:  (B, k, C)
    """
    B, N, C = tensor.shape
    arange = torch.arange(B, device=tensor.device).unsqueeze(-1)  # (B,1)
    return tensor[arange, indices, :]
