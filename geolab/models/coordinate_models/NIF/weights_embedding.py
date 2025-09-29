"""
WeightsEmbeddingLayer for Neural Implicit Flow.

Maps latent embeddings to ShapeNet weights with proper initialization scaling.
"""

import torch
import torch.nn as nn
import math


class WeightsEmbeddingLayer(nn.Module):
    """Maps latent embedding z to ShapeNet weights and biases with proper scaling.
    
    This layer projects the latent vector z to the full set of weights and biases
    needed by the ShapeNet, applying appropriate scaling based on the activation
    function (Xavier/Glorot, He/Kaiming, or SIREN).
    
    Args:
        latent_dim: Dimension of input latent vector z
        shape_net_config: Dictionary with ShapeNet architecture details:
            - input_dim: ShapeNet input dimension
            - output_dim: ShapeNet output dimension
            - units: Hidden layer width
            - nlayers: Number of hidden layers
            - type: 'mlp' or 'siren'
            - activation: Activation function name (for MLP)
            - omega_0: Frequency parameter (for SIREN)
    """
    
    def __init__(self, latent_dim: int, shape_net_config: dict):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.config = shape_net_config
        
        # Calculate total number of parameters needed
        self.layer_shapes = self._compute_layer_shapes()
        self.total_params = sum(w.numel() + b.numel() 
                               for w, b in self.layer_shapes)
        
        # Linear projection from latent to all parameters
        self.projection = nn.Linear(latent_dim, self.total_params)
        
        # Initialize projection layer
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        
    def _compute_layer_shapes(self):
        """Compute weight and bias shapes for each ShapeNet layer."""
        input_dim = self.config['input_dim']
        hidden_dim = self.config['units']
        output_dim = self.config['output_dim']
        nlayers = self.config['nlayers']
        
        shapes = []
        
        # First layer: input_dim -> hidden_dim
        shapes.append((
            torch.Size([hidden_dim, input_dim]),  # weight
            torch.Size([hidden_dim])               # bias
        ))
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(nlayers):
            shapes.append((
                torch.Size([hidden_dim, hidden_dim]),
                torch.Size([hidden_dim])
            ))
        
        # Output layer: hidden_dim -> output_dim
        shapes.append((
            torch.Size([output_dim, hidden_dim]),
            torch.Size([output_dim])
        ))
        
        return shapes
    
    def _get_scaling_factors(self):
        """Compute scaling factors for each layer based on activation type."""
        net_type = self.config['type']
        scaling_factors = []
        
        for i, (w_shape, _) in enumerate(self.layer_shapes):
            fan_in = w_shape[1]
            fan_out = w_shape[0]
            
            if net_type == 'siren':
                if i == 0:
                    # First layer of SIREN
                    omega_0 = self.config.get('omega_0', 30.0)
                    scale = 1.0 / fan_in
                else:
                    # Hidden layers of SIREN
                    omega_0 = self.config.get('omega_0', 30.0)
                    scale = math.sqrt(6.0 / fan_in) / omega_0
            elif net_type == 'mlp':
                activation = self.config.get('activation', 'relu')
                if activation in ['relu', 'leaky_relu', 'elu']:
                    # He/Kaiming initialization
                    scale = math.sqrt(2.0 / fan_in)
                else:
                    # Xavier/Glorot initialization (for tanh, sigmoid, etc.)
                    scale = math.sqrt(6.0 / (fan_in + fan_out))
            else:
                # Default to Xavier
                scale = math.sqrt(6.0 / (fan_in + fan_out))
            
            scaling_factors.append(scale)
        
        return scaling_factors
    
    def forward(self, z):
        """Project latent z to scaled ShapeNet parameters.
        
        Args:
            z: Latent embedding tensor of shape [batch_size, latent_dim]
            
        Returns:
            List of tuples (weights, biases) for each ShapeNet layer.
            Each weight has shape [batch_size, fan_out, fan_in].
            Each bias has shape [batch_size, fan_out].
        """
        batch_size = z.shape[0]
        
        # Project to all parameters
        params = self.projection(z)  # [batch_size, total_params]
        
        # Get scaling factors
        scales = self._get_scaling_factors()
        
        # Split and reshape parameters
        layer_params = []
        offset = 0
        
        for (w_shape, b_shape), scale in zip(self.layer_shapes, scales):
            # Extract weights
            w_size = w_shape.numel()
            weights = params[:, offset:offset + w_size]
            weights = weights.view(batch_size, *w_shape)
            weights = weights * scale  # Apply scaling
            offset += w_size
            
            # Extract biases
            b_size = b_shape.numel()
            biases = params[:, offset:offset + b_size]
            biases = biases.view(batch_size, *b_shape)
            offset += b_size
            
            layer_params.append((weights, biases))
        
        return layer_params