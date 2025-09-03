"""
Models package
--------------

Chứa định nghĩa mô hình MGVLF theo paper VGRSS:
- model.py: ResNet50 backbone + LVFE + VLF + Prediction Head
"""

from .model import MGVLF

__all__ = [
    "MGVLF",
]
