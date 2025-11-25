import torch
from torch import nn

class FourierFeatures(nn.Module):
    def __init__(self, input_dimension, mapping_dimension, scale, type, trainable=False):
        super().__init__()

        self.input_dimension = input_dimension
        self.mapping_dimension = mapping_dimension
        if isinstance(scale, torch.Tensor):
            scale_value = scale.item()
        else:
            scale_value = float(scale)
        
        self.scale = nn.Parameter(
            torch.tensor([scale_value], dtype=torch.float32), 
            requires_grad=trainable
        )
        
        beta = self._define_beta(type, scale_value)
        self.register_buffer('beta', beta)  # Register as buffer so it moves with model



    def _define_beta(self, type, scale_value):
        if type == 'gaussian':
            # trying a fixed seed for pickling stability
            generator = torch.Generator()
            generator.manual_seed(12345)
            beta = torch.randn(
                self.mapping_dimension // 2, 
                self.input_dimension,
                generator=generator
            ) * scale_value
        elif type == 'positional':
            j = torch.arange(self.mapping_dimension // 2, dtype=torch.float32)
            beta = scale_value ** (j / (self.mapping_dimension // 2))
            beta = beta.view(-1, 1).expand(-1, self.input_dimension).contiguous()
        elif type == 'basic':
            beta = torch.ones(self.mapping_dimension // 2, self.input_dimension) 
        else:
            raise ValueError(f"Unknown type: {type}")
        return beta

    def forward(self, x):
        x_transformed = 2 * torch.pi * (x.to(self.beta.dtype) @ self.beta.T)
        return torch.cat([torch.sin(x_transformed), torch.cos(x_transformed)], dim=-1)
