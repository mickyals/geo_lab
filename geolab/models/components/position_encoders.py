# fourier_features.py
import torch
import torch.nn as nn
from typing import Union, Optional
from torch import Tensor


class FourierFeatures(nn.Module):
    """
    Fourier feature mapping with optional trainable scale.
    Supports 'gaussian', 'positional', 'basic'.
    Automatically recomputes beta when scale is trainable.
    """

    def __init__(
        self,
        input_dimension: int,
        mapping_dimension: int,
        scale: Union[float, Tensor],
        type: str = "gaussian",
        trainable: bool = False,
    ) -> None:
        super().__init__()

        self.input_dimension = input_dimension
        self.mapping_dimension = mapping_dimension
        self.type = type

        scale_value = float(scale.item() if isinstance(scale, Tensor) else scale)
        self.scale = nn.Parameter(
            torch.tensor([scale_value], dtype=torch.float32),
            requires_grad=trainable,
        )

        # beta is only a buffer if scale is frozen
        if not trainable:
            beta = self._make_beta(scale_value)
            self.register_buffer("beta", beta)
        else:
            self.beta = None  # recompute each forward pass

    def _make_beta(self, scale_value: float) -> Tensor:
        half = self.mapping_dimension // 2

        if self.type == "gaussian":
            # Crucial: variance scaling
            beta = torch.randn(half, self.input_dimension) * scale_value

        elif self.type == "positional":
            j = torch.arange(half, dtype=torch.float32)
            freq = scale_value ** (j / half)
            beta = freq.view(-1, 1).expand(-1, self.input_dimension).contiguous()

        elif self.type == "basic":
            beta = torch.ones(half, self.input_dimension)

        else:
            raise ValueError(f"Unknown type '{self.type}'")

        return beta

    def _get_beta(self, x: Tensor) -> Tensor:
        # recompute if scale is trainable
        if self.beta is None:
            return self._make_beta(self.scale.item()).to(x.device, x.dtype)
        return self.beta.to(x.device, x.dtype)

    def forward(self, x: Tensor) -> Tensor:
        beta = self._get_beta(x)
        phase = 2 * torch.pi * (x @ beta.T)  # essential: projection matrix
        return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
















#############################################
##### TODO: Remove old code
#############################################



# import torch
# from torch import nn
#
# class FourierFeatures(nn.Module):
#     def __init__(self, input_dimension, mapping_dimension, scale, type, trainable=False):
#         super().__init__()
#
#         self.input_dimension = input_dimension
#         self.mapping_dimension = mapping_dimension
#         if isinstance(scale, torch.Tensor):
#             scale_value = scale.item()
#         else:
#             scale_value = float(scale)
#
#         self.scale = nn.Parameter(
#             torch.tensor([scale_value], dtype=torch.float32),
#             requires_grad=trainable
#         )
#
#         beta = self._define_beta(type, scale_value)
#         self.register_buffer('beta', beta)  # Register as buffer so it moves with model
#
#
#
#     def _define_beta(self, type, scale_value):
#         if type == 'gaussian':
#             # trying a fixed seed for pickling stability
#             #generator = torch.Generator()
#             #generator.manual_seed(12345)
#             beta = torch.randn(
#                 self.mapping_dimension // 2,
#                 self.input_dimension
#             ) * scale_value
#         elif type == 'positional':
#             j = torch.arange(self.mapping_dimension // 2, dtype=torch.float32)
#             beta = scale_value ** (j / (self.mapping_dimension // 2))
#             beta = beta.view(-1, 1).expand(-1, self.input_dimension).contiguous()
#         elif type == 'basic':
#             beta = torch.ones(self.mapping_dimension // 2, self.input_dimension)
#         else:
#             raise ValueError(f"Unknown type: {type}")
#         return beta
#
#     def forward(self, x):
#         x_transformed = 2 * torch.pi * (x.to(self.beta.dtype) @ self.beta.T)
#         return torch.cat([torch.sin(x_transformed), torch.cos(x_transformed)], dim=-1)
