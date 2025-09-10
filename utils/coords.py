# utils/coords.py
import torch

def generate_coord(batch: int, height: int, width: int, device=None, dtype=torch.float32):
    device = device or torch.device("cpu")
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width,  device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xv_min = (xx * 2 - width) / width
    yv_min = (yy * 2 - height) / height
    xv_max = ((xx + 1) * 2 - width) / width
    yv_max = ((yy + 1) * 2 - height) / height
    xv_ctr = (xv_min + xv_max) / 2
    yv_ctr = (yv_min + yv_max) / 2
    hmap = torch.full((height, width), 1.0 / height, device=device, dtype=dtype)
    wmap = torch.full((height, width), 1.0 / width,  device=device, dtype=dtype)
    coord = torch.stack([xv_min, yv_min, xv_max, yv_max, xv_ctr, yv_ctr, hmap, wmap], dim=0)
    return coord.unsqueeze(0).repeat(batch, 1, 1, 1)  # (B,8,H,W)
