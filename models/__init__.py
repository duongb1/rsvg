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

__all__ = [
    "MGVLF",
]
