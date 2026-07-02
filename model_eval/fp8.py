# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weight-only FP8 quantization for the transformer blocks.

Stores every ``nn.Linear`` weight inside the transformer blocks as
``float8_e4m3fn`` with per-output-channel bf16 scales and dequantizes to
bf16 on the fly in ``forward``. This halves transformer weight memory
(~28 GB -> ~14 GB for the 14B model) using native torch only -- no
torchao/quanto/bitsandbytes dependency.

Only the blocks are quantized: the VAE, patch/text/time embeddings,
norms, and the output projection keep their original precision.
"""

import torch
from torch import nn
from torch.nn import functional as F

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max


class Fp8WeightLinear(nn.Module):
    """Drop-in ``nn.Linear`` replacement with fp8 weight storage.

    Quantization is symmetric per output channel: each row of the weight
    matrix is scaled into the e4m3 range and stored as ``float8_e4m3fn``
    alongside a bf16 scale. The forward pass dequantizes to bf16 and runs
    a standard ``F.linear``, so numerics differ from the bf16 model only
    by the weight-rounding error.
    """

    def __init__(self, linear: nn.Linear, device: torch.device | None = None):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        weight = linear.weight.detach()
        if device is not None:
            weight = weight.to(device)
        scale = weight.abs().amax(dim=1, keepdim=True).float() / FP8_MAX
        scale = scale.clamp(min=torch.finfo(torch.float32).tiny)
        quantized = (weight.float() / scale).clamp(-FP8_MAX, FP8_MAX)
        self.register_buffer("weight_fp8", quantized.to(FP8_DTYPE))
        self.register_buffer("weight_scale", scale.to(torch.bfloat16))
        if linear.bias is not None:
            bias = linear.bias.detach().clone()
            if device is not None:
                bias = bias.to(device)
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.weight_fp8.to(torch.bfloat16) * self.weight_scale
        return F.linear(hidden_states, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def _replace_linears(module: nn.Module, device: torch.device | None) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, Fp8WeightLinear(child, device))
            replaced += 1
        else:
            replaced += _replace_linears(child, device)
    return replaced


def quantize_transformer_blocks_fp8(transformer: nn.Module, device: torch.device | None = None) -> int:
    """Replace every ``nn.Linear`` inside ``transformer.blocks`` with ``Fp8WeightLinear``.

    Call AFTER loading the checkpoint (quantization freezes the loaded
    weights). When ``device`` is given and the transformer is still on CPU,
    each layer is quantized directly onto ``device``, so peak GPU memory
    stays at the fp8 footprint plus one bf16 layer -- the full bf16 model
    never touches the GPU. Returns the number of layers replaced.
    """
    blocks = getattr(transformer, "blocks", None)
    if blocks is None:
        raise ValueError("transformer has no .blocks ModuleList to quantize")
    total = 0
    for block in blocks:
        total += _replace_linears(block, device)
    if total == 0:
        raise ValueError("no nn.Linear layers found inside transformer.blocks")
    return total
