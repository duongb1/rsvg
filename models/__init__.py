"""
MGVLF Models Package
--------------------
Gói này chứa toàn bộ code định nghĩa mô hình:
- model.py: định nghĩa MGVLF (wrapper chính)
- CNN_MGVLF.py: định nghĩa CNN_MGVLF, VLFusion
- backbone.py: ResNet backbone + position encoding
- transformer.py: Encoder/Decoder transformer blocks
"""

from .model import MGVLF
from .CNN_MGVLF import build_CNN_MGVLF, build_VLFusion
from .backbone import build_backbone
from .transformer import build_vis_transformer, build_transformer, build_de

from .backbone_inject import QABM

__all__ = [
    "MGVLF",
    "build_CNN_MGVLF",
    "build_VLFusion",
    "build_backbone",
    "build_vis_transformer",
    "build_transformer",
    "build_de",
    "QABM",   # thêm vào đây
]
