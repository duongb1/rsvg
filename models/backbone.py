# Copyright (c) Facebook, Inc.
"""
Backbone modules.
Refactored for MGVLF: ResNet backbone with FrozenBN + multi-scale outputs + position embeddings.
"""
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from utils.misc import NestedTensor
from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d with fixed (frozen) affine and running stats.
    Copy from torchvision with an added eps before rsqrt to avoid nans on some variants.
    """

    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # ignore num_batches_tracked if present
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    """
    Wrap a torchvision ResNet and expose intermediate layers as a dict of NestedTensor.
    """

    def __init__(self, backbone: nn.Module, train_backbone: bool,
                 num_channels: int, return_interm_layers: bool):
        super().__init__()
        # Freeze all except layer2-4 when not training backbone
        for name, parameter in backbone.named_parameters():
            if not train_backbone or ('layer2' not in name and 'layer3' not in name and 'layer4' not in name):
                parameter.requires_grad_(False)

        if return_interm_layers:
            # expose resnet layers as 0..3 keys
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}

        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor) -> Dict[str, NestedTensor]:
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # upsample mask to feature spatial size
            mask = F.interpolate(m[None].float(), size=x.shape[-2:], mode="nearest").to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str, train_backbone: bool,
                 return_interm_layers: bool, dilation: bool):
        # e.g., name='resnet50'
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True,
            norm_layer=FrozenBatchNorm2d
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    """
    A thin module that applies backbone, then builds positional encodings for each returned feature.
    Returns:
        out: List[NestedTensor] aligned with scales (keys sorted)
        pos: List[Tensor] positional embeddings aligned with 'out'
    """

    def __init__(self, backbone: Backbone, position_embedding: nn.Module):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)  # dict name -> NestedTensor
        out: List[NestedTensor] = []
        pos_list: List[torch.Tensor] = []
        # keep the original ResNet order (0..3)
        for key in sorted(xs.keys(), key=lambda k: int(k)):
            x = xs[key]
            out.append(x)
            pos = self[1](x).to(x.tensors.dtype)
            pos_list.append(pos)
        return out, pos_list


def build_backbone(args):
    """
    Factory for backbone+positioning joiner.
    - Uses sine positional encoding for visual backbone.
    - return_interm_layers=True to expose multi-scale features (layer1..layer4).
    """
    # position encoding to accompany backbone outputs
    position_embedding = build_position_encoding(args, position_embedding='sine')

    # decide whether to train backbone parameters
    train_backbone = True  # default to True for fine-tuning
    # if you want to freeze most layers when LR<=0, uncomment:
    # train_backbone = getattr(args, "lr", 1e-4) > 0

    return_interm_layers = True  # we need multi-scale
    dilation = getattr(args, "dilation", False)

    backbone = Backbone(
        name=getattr(args, "backbone", "resnet50"),
        train_backbone=train_backbone,
        return_interm_layers=return_interm_layers,
        dilation=dilation
    )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
