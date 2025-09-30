
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from geolab.models.components.core.layers import ResidualBlock, BaseParametrizationLayer, BaseLayer
from geolab.models.components.core.embeddings import get_embedding
from geolab.models.components.core.initializations import get_initializer
from geolab.models.components.core.activations import get_activation



class NIF(nn.Module):
    def __init__(self):
        self.encoder = NIF_Encoder()
        self.reparameterizer = NIF_Reparameterizer()
        self.decoder = NIF_Decoder()

    def forward(self, x):

        encoder = self.encoder(x['valid_time'])
        weights = self.reparameterizer(encoder)
        decoder = self.decoder(weights, x['latitude'], x['longitude'], x['pressure_level'])
        return decoder, weights




class NIF_Encoder(nn.Module):
    """
    Neural Implicit Field Encoder - A configurable feedforward neural network.
    
    This module implements a standard feedforward neural network with configurable
    architecture and optional positional encoding. The network consists of an input layer,
    multiple hidden layers, and an output layer.
    
    Args:
        in_features (int): Number of input features
        n_layers (int): Total number of layers (including input and output layers)
        layer_width (int): Number of neurons in each hidden layer
        out_weights (int): Number of output neurons
        activation_name (str): Name of the activation function to use
        activation_kwargs (Dict[str, Any]): Additional arguments for the activation function
        initializer_name (str): Name of the weight initializer
        initializer_kwargs (Dict[str, Any]): Additional arguments for the weight initializer
        positional_encoding_name (Optional[str]): Type of positional encoding to apply
        positional_encoding_kwargs (Optional[Dict[str, Any]]): Arguments for positional encoding
    """
    
    def __init__(
        self,
        in_features: int,
        n_hidden_layers: int,
        layer_width: int,
        out_weights: int,
        activation_name: str,
        activation_kwargs: Dict[str, Any],
        initializer_name: str,
        initializer_kwargs: Dict[str, Any],
        positional_encoding_name: Optional[str] = None,
        positional_encoding_kwargs: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        # Store network architecture parameters
        self.in_features = in_features
        self.n_layers = n_hidden_layers
        self.layer_width = layer_width
        self.out_weights = out_weights
        
        # Store activation and initialization configurations
        self.activation_name = activation_name
        self.activation_kwargs = activation_kwargs or {}
        self.initializer_name = initializer_name
        self.initializer_kwargs = initializer_kwargs or {}
        
        # Set up positional encoding if specified
        self.positional_encoding = None
        if positional_encoding_name:
            self.positional_encoding = get_embedding(
                embedding_name=positional_encoding_name,
                in_features=in_features,
                **(positional_encoding_kwargs or {})
            )
            in_features = self.positional_encoding.out_features
        
        # Build the network
        self.net = self._build_network()
    
    def _build_network(self) -> nn.Sequential:
        """
        Construct the neural network architecture.
        
        Returns:
            nn.Sequential: The constructed neural network
        """
        layers = nn.Sequential()
        
        # Input layer
        input_features = self.positional_encoding.out_features if self.positional_encoding else self.in_features
        layers.add_module(
            "input_layer",
            self._create_layer(
                in_features=input_features,
                out_features=self.layer_width,
                is_first=True,
                is_last=False
            )
        )
        
        # Hidden layers
        for i in range(self.n_layers):
            layers.add_module(
                f"hidden_{i}",
                self._create_layer(
                    in_features=self.layer_width,
                    out_features=self.layer_width,
                    is_first=False,
                    is_last=False
                )
            )
        return layers
    
    def _create_layer(
        self,
        in_features: int,
        out_features: int,
        is_first: bool,
        is_last: bool
    ) -> BaseLayer:
        """
        Helper method to create a single network layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            is_first: Whether this is the first layer
            is_last: Whether this is the last layer
            
        Returns:
            BaseLayer: Configured neural network layer
        """
        return BaseLayer(
            in_features=in_features,
            out_features=out_features,
            activation=None if is_last else self.activation_name,
            activation_kwargs=None if is_last else self.activation_kwargs,
            initializer=self.initializer_name,
            initializer_kwargs={
                **self.initializer_kwargs,
                "in_features": in_features,
                "is_first": is_first
            },
            is_last=is_last
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, in_features)
            
        Returns:
            Output tensor of shape (batch_size, out_weights)
        """
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        return self.net(x)


class NIF_Reparameterizer(nn.Module):
    """
    A reparameterization module that can operate in either deterministic or stochastic mode.

    In deterministic mode, it acts as a simple linear transformation.
    In stochastic mode, it implements the reparameterization trick for variational inference,
    outputting a sample from a learned distribution.

    Args:
        hidden_width (int): Number of input features
        out_weights (int): Number of output features
        deterministic (bool): Whether to use deterministic mode
        initializer_name (str): Name of the weight initializer
        initializer_kwargs (Dict[str, Any]): Additional arguments for the weight initializer
    """

    def __init__(
            self,
            hidden_width: int,
            out_weights: int,
            deterministic: bool,
            initializer_name: str,
            initializer_kwargs: Dict[str, Any]
    ):
        super().__init__()
        self.hidden_width = hidden_width
        self.out_weights = out_weights
        self.deterministic = deterministic
        self.initializer_name = initializer_name
        self.initializer_kwargs = initializer_kwargs or {}

        self.mu_layer = self._create_output_layer()
        if not self.deterministic:
            self.logvar_layer = self._create_output_layer()

    def _create_output_layer(self) -> BaseLayer:
        """Helper method to create an output layer with the current configuration."""
        return BaseLayer(
            in_features=self.hidden_width,
            out_features=self.out_weights,
            activation=None,  # No activation for output layers
            initializer=self.initializer_name,
            initializer_kwargs={
                **self.initializer_kwargs,
                "in_features": self.hidden_width,
                "is_first": False
            },
            is_last=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the reparameterizer.

        In deterministic mode: returns a linear transformation of the input.
        In stochastic mode: returns a sample from N(mu, sigma^2) using the reparameterization trick.

        Args:
            x: Input tensor of shape (batch_size, hidden_width)

        Returns:
            Output tensor of shape (batch_size, out_weights)
        """
        if self.deterministic:
            return self.mu_layer(x)

        # In stochastic mode, output a sample from the learned distribution
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


class NIF_Decoder(nn.Module):
    pass





