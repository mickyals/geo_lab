"""
ParamNet for Neural Implicit Flow.

Encodes temporal and vertical context into latent embeddings.
"""

import torch
import torch.nn as nn
from typing import List


class ParamNet(nn.Module):
    """Parameter network that encodes context into latent embeddings.
    
    Takes temporal and vertical coordinates (time, pressure_level) and encodes
    them into a latent representation z that will be used to generate ShapeNet weights.
    
    Args:
        input_dim: Dimension of input (typically 2: time, pressure_level)
        latent_dim: Dimension of output latent embedding z
        hidden_dims: List of hidden layer dimensions
        activation: Activation function ('relu', 'gelu', 'tanh', etc.)
        net_type: Network type ('mlp' or 'siren')
        omega_0: Frequency parameter for SIREN (only used if net_type='siren')
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: List[int],
        activation: str = 'tanh',
        net_type: str = 'mlp',
        omega_0: float = 30.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.net_type = net_type
        self.omega_0 = omega_0
        
        if net_type == 'mlp':
            self.net = self._build_mlp(input_dim, latent_dim, hidden_dims, activation)
        elif net_type == 'siren':
            self.net = self._build_siren(input_dim, latent_dim, hidden_dims, omega_0)
        else:
            raise ValueError(f"Unsupported net_type: {net_type}. Use 'mlp' or 'siren'.")
    
    def _build_mlp(self, input_dim, output_dim, hidden_dims, activation):
        """Build standard MLP with specified activation."""
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add activation except after last layer
            if i < len(dims) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'elu':
                    layers.append(nn.ELU())
                elif activation == 'leaky_relu':
                    layers.append(nn.LeakyReLU())
                else:
                    raise ValueError(f"Unsupported activation: {activation}")
        
        net = nn.Sequential(*layers)
        
        # Initialize weights
        for m in net.modules():
            if isinstance(m, nn.Linear):
                if activation in ['relu', 'leaky_relu', 'elu']:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        
        return net
    
    def _build_siren(self, input_dim, output_dim, hidden_dims, omega_0):
        """Build SIREN network with sine activations."""
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            linear = nn.Linear(dims[i], dims[i + 1])
            
            # SIREN initialization
            if i == 0:
                # First layer
                nn.init.uniform_(linear.weight, -1 / dims[i], 1 / dims[i])
            else:
                # Hidden layers
                nn.init.uniform_(
                    linear.weight,
                    -torch.sqrt(torch.tensor(6.0 / dims[i])) / omega_0,
                    torch.sqrt(torch.tensor(6.0 / dims[i])) / omega_0
                )
            nn.init.zeros_(linear.bias)
            
            layers.append(linear)
            
            # Add sine activation except after last layer
            if i < len(dims) - 2:
                layers.append(Sine(omega_0))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """Encode input to latent embedding.
        
        Args:
            x: Input tensor [batch_size, input_dim] containing [time, pressure_level]
            
        Returns:
            Latent embedding z of shape [batch_size, latent_dim]
        """
        return self.net(x)


class Sine(nn.Module):
    """Sine activation with frequency parameter."""
    
    def __init__(self, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
    
    def forward(self, x):
        return torch.sin(self.omega_0 * x)