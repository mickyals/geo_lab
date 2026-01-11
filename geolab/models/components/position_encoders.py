import torch
from typing import Optional, List
from torch import nn


class FourierFeatures(nn.Module):
    """Fourier feature encoding with selective dimension encoding."""

    def __init__(self, input_dimension, mapping_dimension, scale, type,
                 encode_dims: Optional[List[int]] = None, passthrough_dims=None, trainable=False):
        """
        Args:
            input_dimension: Total input dimensions
            mapping_dimension: Output dimension (must be even)
            scale: Frequency scale
            type: 'gaussian', 'positional', or 'basic'
            encode_dims: Which dimensions to encode (None = all).
                        Example: [0, 1] to encode only first two dims (lat/lon)
            trainable: Whether scale is trainable
        """
        super().__init__()

        self.input_dimension = input_dimension
        self.mapping_dimension = mapping_dimension
        self.encode_dims = encode_dims
        self.passthrough_dims = passthrough_dims

        # Number of dimensions to encode
        self.n_encode = len(self.encode_dims)

        if isinstance(scale, torch.Tensor):
            scale_value = scale.item()
        else:
            scale_value = float(scale)

        self.scale = nn.Parameter(
            torch.tensor([scale_value], dtype=torch.float32),
            requires_grad=trainable
        )

        # Beta only for encoded dimensions
        beta = self._define_beta(type, scale_value)
        self.register_buffer('beta', beta)

    def _define_beta(self, type, scale_value):
        """Create frequency matrix for ENCODED dimensions only."""
        if type == 'gaussian':
            beta = torch.randn(
                self.mapping_dimension // 2,
                self.n_encode,  # Only encode selected dims
                generator=torch.Generator().manual_seed(42)
            ) * scale_value
        elif type == 'positional':
            j = torch.arange(self.mapping_dimension // 2, dtype=torch.float32)
            beta = scale_value ** (j / (self.mapping_dimension // 2))
            beta = beta.view(-1, 1).expand(-1, self.n_encode).contiguous()
        elif type == 'basic':
            beta = torch.ones(self.mapping_dimension // 2, self.n_encode)
        else:
            raise ValueError(f"Unknown type: {type}")
        return beta

    def forward(self, x):
        """
        Args:
            x: (B, input_dimension) input tensor

        Returns:
            Encoded tensor: (B, mapping_dimension + len(passthrough_dims))
        """
        # Extract dimensions to encode
        x_encode = x[:, self.encode_dims]  # (B, n_encode)

        # Apply Fourier encoding
        x_transformed = 2 * torch.pi * (x_encode @ self.beta.T)
        encoded = torch.cat([torch.sin(x_transformed), torch.cos(x_transformed)], dim=-1)

        # Passthrough unencoded dimensions
        if self.passthrough_dims:
            x_passthrough = x[:, self.passthrough_dims]
            return torch.cat([encoded, x_passthrough], dim=-1)
        else:
            return encoded

    @property
    def output_dimension(self):
        """Total output dimension after encoding."""
        return self.mapping_dimension + len(self.passthrough_dims)