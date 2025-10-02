"""
building the basal layer for any MLP with interchangeable activation functions and initializations
also residual base layer/block code for any MLP with interchangeable activation functions and initializations
basal layer for weight parametrization for parameternet to supply weights too
"""
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from geolab.models.components.core.initializations import get_initializer
from geolab.models.components.core.activations import get_activation



class BaseLayer(nn.Module):
    def __init__(self, in_features, out_features, activation,
                 initialization, is_last,
                 activation_kwargs, initialization_kwargs):

        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation = get_activation(activation, **(activation_kwargs if activation_kwargs else {}))
        self.initialization = get_initializer(initialization, **(initialization_kwargs if initialization_kwargs else {}))
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features)
        self.initialization(self.linear)

    def forward(self, x):
        if self.is_last:
            return self.linear(x)
        return self.activation(self.linear(x))



class ResidualBlock(nn.Module):
    """
following the resnet like block structure of Neural implicit flow and H-siren
    """
    def __init__(self, num_features, residual_weight,
                 activation, initialization,  initialization_kwargs, activation_kwargs):
        super().__init__()

        self.activation = get_activation(activation, **(activation_kwargs if activation_kwargs else {}))
        self.initialization = get_initializer(initialization, **(initialization_kwargs if initialization_kwargs else {}))
        self.num_features = num_features
        self.linear1 = nn.Linear(num_features, num_features)
        self.linear2 = nn.Linear(num_features, num_features)
        self.residual_weight = residual_weight
        self.initialization(self.linear1)
        self.initialization(self.linear2)


    def forward(self, x):

        identity =  x
        first = self.linear1(x)
        act1 = self.activation(first)
        second = self.linear2(act1)
        if self.residual_weight is not None:
            return ((1 - self.residual_weight) * self.activation(second)) + (self.residual_weight * identity) # typically 0.5f(x) + 0.5x to the next layer or block

        else:
            return self.activation(second) + identity


class BaseParametrizationLayer(nn.Module):
    """
    A parameterized layer that uses externally provided weights and biases.
    
    This layer is designed for scenarios where weights are dynamically generated,
    such as in hypernetworks or meta-learning. Instead of learning its own parameters,
    it accepts them as inputs during the forward pass.

    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features
        is_last (bool): Whether this is the final layer (no activation will be applied)
        activation (str): Name of the activation function to use
        initialization (str): Name of the weight initialization method
        initialization_kwargs (Dict[str, Any]): Additional arguments for the initializer
        activation_kwargs (Dict[str, Any]): Additional arguments for the activation function
    """
    
    def __init__(
        self,
        in_features,
        out_features,
        is_last,
        activation,
        activation_kwargs = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.is_last = is_last
        
        # Get activation function (None for the last layer)
        self.activation = (
            None 
            if is_last 
            else get_activation(activation, **(activation_kwargs or {}))
        )

    def forward(
        self, 
        x: torch.Tensor, 
        weights_and_biases: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass with externally provided weights and biases.
        
        Performs the operation: activation(W @ x + b)
        where W and b are provided as input parameters.

        Args:
            x: Input tensor of shape (batch_size, in_features)
            weights_and_biases: Tuple containing:
                - W: Weight matrix of shape (batch_size, out_features, in_features)
                - b: Bias vector of shape (batch_size, out_features, 1)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        W, b = weights_and_biases
        
        # Validate input shapes
        batch_size = x.size(0)
        assert W.shape == (batch_size, self.out_features, self.in_features), \
            f"Expected W shape {(batch_size, self.out_features, self.in_features)}, got {W.shape}"
        assert b.shape == (batch_size, self.out_features, 1), \
            f"Expected b shape {(batch_size, self.out_features, 1)}, got {b.shape}"
        
        # Reshape input for batch matrix multiplication: (batch_size, in_features) -> (batch_size, in_features, 1)
        x_reshaped = x.unsqueeze(-1)
        
        # Perform batched matrix multiplication: (b, o, i) @ (b, i, 1) -> (b, o, 1)
        out = torch.bmm(W, x_reshaped)
        
        # Add bias and remove the last dimension: (b, o, 1) -> (b, o)
        out = (out + b).squeeze(-1)
        
        # Apply activation if not the last layer
        if not self.is_last and self.activation is not None:
            out = self.activation(out)
            
        return out

    def extra_repr(self) -> str:
        """Extra representation string for the module."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"is_last={self.is_last}, "
            f"activation={self.activation.__class__.__name__ if self.activation else 'None'}"
        )
