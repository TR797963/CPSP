"""External ISNet runtime compatibility without editing the backbone source.

The bundled legacy DCNv2 extension targets an old PyTorch ABI.  This module
injects a state-dict-compatible DeformConvPack implementation backed by
``torchvision.ops.deform_conv2d`` before importing ISNet.  The mathematical
operator remains deformable convolution; this is not a plain-Conv2d fallback.
"""

from __future__ import annotations

import math
import sys
import types
from contextlib import contextmanager
from typing import Any, Dict, Iterator
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d


LEGACY_DCN_MODULE = "model.ISNet.DCNv2.DCN.modules.deform_conv"


class TorchvisionDeformConvPack(nn.Module):
    """State-dict-compatible replacement for the legacy DeformConvPack."""

    dcn_backend = "torchvision_deform_conv2d"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Any,
        stride: Any,
        padding: Any,
        dilation: Any = 1,
        groups: int = 1,
        deformable_groups: int = 1,
        im2col_step: int = 64,
        bias: bool = True,
        lr_mult: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError(f"in_channels={in_channels} must be divisible by groups={groups}")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels={out_channels} must be divisible by groups={groups}")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = int(groups)
        self.deformable_groups = int(deformable_groups)
        self.im2col_step = int(im2col_step)
        self.use_bias = bool(bias)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        if not self.use_bias:
            self.bias.requires_grad_(False)

        offset_channels = deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset = nn.Conv2d(
            in_channels,
            offset_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.conv_offset.lr_mult = lr_mult
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)
        init.zeros_(self.conv_offset.weight)
        init.zeros_(self.conv_offset.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        offsets = self.conv_offset(inputs)
        return deform_conv2d(
            inputs,
            offsets,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


def install_isnet_dcn_import_adapter() -> None:
    """Expose the compatible class under the legacy import path."""
    existing = sys.modules.get(LEGACY_DCN_MODULE)
    if existing is not None and getattr(existing, "_cpsp_adapter", False):
        return
    module = types.ModuleType(LEGACY_DCN_MODULE)
    module.DeformConvPack = TorchvisionDeformConvPack
    module.DeformConv = TorchvisionDeformConvPack
    module._cpsp_adapter = True
    sys.modules[LEGACY_DCN_MODULE] = module


@contextmanager
def _device_neutral_legacy_constructor() -> Iterator[None]:
    """Neutralize legacy constructor-time ``.cuda()`` calls only."""
    def identity_cuda(tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        return tensor

    with patch.object(torch.Tensor, "cuda", identity_cuda):
        yield


def build_isnet(mode: str = "test") -> nn.Module:
    install_isnet_dcn_import_adapter()
    from model.ISNet.model_ISNet import ISNet

    with _device_neutral_legacy_constructor():
        model = ISNet(mode=mode)
    return model


class FixedGradientMagnitude(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kernel_v = torch.tensor(
            [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        ).view(1, 1, 3, 3)
        kernel_h = torch.tensor(
            [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("weight_h", kernel_h)
        self.register_buffer("weight_v", kernel_v)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channel = inputs[:, :1]
        vertical = F.conv2d(channel, self.weight_v, padding=1)
        horizontal = F.conv2d(channel, self.weight_h, padding=1)
        return torch.sqrt(vertical.square() + horizontal.square() + 1e-6)


class ISNetStableLoss(nn.Module):
    """Numerically valid form of the public ISNet image/edge objective."""

    def __init__(self) -> None:
        super().__init__()
        from loss import SoftIoULoss

        self.softiou = SoftIoULoss()
        self.bce = nn.BCELoss()
        self.grad = FixedGradientMagnitude()

    def forward(self, predictions: Any, targets: torch.Tensor) -> torch.Tensor:
        edge_target = self.grad(targets).clamp_(0.0, 1.0)
        image_loss = self.softiou(predictions[0], targets)
        edge_loss = 10.0 * self.bce(predictions[1], edge_target)
        edge_loss = edge_loss + self.softiou(predictions[1], edge_target)
        return image_loss + edge_loss


def isnet_adapter_metadata(model: nn.Module) -> Dict[str, Any]:
    backends = sorted(
        {
            str(getattr(module, "dcn_backend"))
            for module in model.modules()
            if getattr(module, "dcn_backend", None) is not None
        }
    )
    return {
        "source_backbone_modified": False,
        "dcn_backends": backends,
        "operator_semantics": "deformable_convolution",
    }
