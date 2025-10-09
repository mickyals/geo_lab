import torch
from torch import nn

class FourierFeatures(nn.Module):
    def __init__(self, input_dimension, mapping_dimension, scale, type, trainable=False):
        super().__init__()

        self.input_dimension = input_dimension
        self.mapping_dimension = mapping_dimension
        self.scale = nn.Parameter(scale * torch.ones(1), requires_grad=trainable)
        beta = self._define_beta(type)
        self.register_buffer('beta', beta)  # Register as buffer so it moves with model



    def _define_beta(self, type):
        if type == 'gaussian':
            beta = torch.randn(self.mapping_dimension // 2, self.input_dimension) * self.scale
        elif type == 'positional':
            j = torch.arange(self.mapping_dimension // 2, dtype=torch.float32)
            beta = self.scale ** (j / (self.mapping_dimension // 2))
            beta = beta.view(-1, 1).expand(-1, self.input_dimension).contiguous()
        elif type == 'basic':
            beta = torch.ones(self.mapping_dimension // 2, self.input_dimension)
        else:
            raise ValueError(f"Unknown type: {type}")
        return beta

    def forward(self, x):
        x_transformed = 2 * torch.pi * (x @ self.beta.T)
        return torch.cat([torch.sin(x_transformed), torch.cos(x_transformed)], dim=-1)
