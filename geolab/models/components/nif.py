
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from geolab.models.components.core.layers import ResidualBlock, BaseParametrizationLayer, BaseLayer
from geolab.models.components.core.embeddings import get_embedding
from geolab.models.components.core.initializations import get_initializer
from geolab.models.components.core.activations import get_activation


class NIF(nn.Module):
    def __init__(
            self,
            # Encoder parameters
            encoder_in_features: int,
            encoder_n_hidden_layers: int,
            encoder_layer_width: int,
            encoder_activation_name: str,
            encoder_activation_kwargs: Dict[str, Any],
            encoder_initializer_name: str,
            encoder_initializer_kwargs: Dict[str, Any],
            encoder_positional_encoding_name: Optional[str],
            encoder_positional_encoding_kwargs: Optional[Dict[str, Any]],

            # Decoder parameters
            decoder_in_features: int,
            decoder_hidden_width: int,
            decoder_n_hidden_layers: int,
            decoder_out_features: int,
            decoder_activation_name: str,
            decoder_activation_kwargs: Dict[str, Any],
            decoder_positional_encoding_name: Optional[str] = None,
            decoder_positional_encoding_kwargs: Optional[Dict[str, Any]] = None,

            # Reparameterizer parameters
            deterministic: bool = True,
            decoder_initializer_name: Optional[str] = None,
            decoder_initializer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        # Calculate total parameters needed for the decoder
        out_weights = (decoder_in_features * decoder_hidden_width) + \
                      (decoder_hidden_width ** 2 * decoder_n_hidden_layers) + \
                      (decoder_hidden_width * decoder_out_features)
        out_biases = decoder_hidden_width + \
                     (decoder_n_hidden_layers * decoder_hidden_width) + \
                     decoder_out_features
        out_weights_and_biases = out_weights + out_biases

        # Initialize components
        self.encoder = NIF_Encoder(
            in_features=encoder_in_features,
            n_hidden_layers=encoder_n_hidden_layers,
            layer_width=encoder_layer_width,
            activation_name=encoder_activation_name,
            activation_kwargs=encoder_activation_kwargs,
            initializer_name=encoder_initializer_name,
            initializer_kwargs=encoder_initializer_kwargs,
            positional_encoding_name=encoder_positional_encoding_name,
            positional_encoding_kwargs=encoder_positional_encoding_kwargs,
        )

        self.reparameterizer = NIF_Reparameterizer(
            hidden_width=encoder_layer_width,  # Assuming this matches encoder output
            out_weights=out_weights_and_biases,
            deterministic=deterministic,
            initializer_name=decoder_initializer_name or encoder_initializer_name,
            initializer_kwargs=decoder_initializer_kwargs or {}
        )

        self.decoder = NIF_Decoder(
            in_features=decoder_in_features,
            out_features=decoder_out_features,
            hidden_width=decoder_hidden_width,
            hidden_layers=decoder_n_hidden_layers,
            activation_name=decoder_activation_name,
            activation_kwargs=decoder_activation_kwargs,
            positional_encoding_name=decoder_positional_encoding_name,
            positional_encoding_kwargs=decoder_positional_encoding_kwargs or {}
        )

    def forward(self, x: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Process input through the network
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
    def __init__(self, in_features, out_features, hidden_width, hidden_layers,
                 activation_name, activation_kwargs, positional_encoding_name,
                 positional_encoding_kwargs):
        super().__init__()


        self.in_features = in_features
        self.out_features = out_features
        self.hidden_width = hidden_width
        self.n_layers = hidden_layers
        self.activation_name = activation_name
        self.activation_kwargs = activation_kwargs
        # Set up positional encoding if specified
        self.positional_encoding = None
        if positional_encoding_name:
            self.positional_encoding = get_embedding(
                embedding_name=positional_encoding_name,
                in_features=in_features,
                **(positional_encoding_kwargs or {})
            )
            in_features = self.positional_encoding.out_features
        self.input_weights = (in_features * hidden_width)
        self.hidden_weights = (hidden_width * hidden_width) * hidden_layers
        self.output_weights = (hidden_width * out_features)
        self.total_weights = self.input_weights + self.hidden_weights + self.output_weights
        self.input_biases = hidden_width
        self.hidden_biases = hidden_width * hidden_layers
        self.output_biases = out_features
        self.total_biases = self.input_biases + self.hidden_biases + self.output_biases

        self.net = self._build_network()

    def _build_network(self):
        layers = nn.ModuleList()

        # Input layer
        input_features = self.positional_encoding.out_features if self.positional_encoding else self.in_features
        layers.add_module(
            "input_layer",
            BaseParametrizationLayer(
                in_features=input_features,
                out_features=self.hidden_width,
                is_last=False,
                activation=self.activation_name,
                activation_kwargs= {**self.activation_kwargs, "in_features": input_features, "is_first": True}
            )
        )

        for i in range(self.n_layers):
            layers.add_module(
                f"hidden_layer {i+1}",
                BaseParametrizationLayer(
                    in_features=self.hidden_width,
                    out_features=self.hidden_width,
                    is_last=False,
                    activation_name=self.activation_name,
                    activation_kwargs= {**self.initializer_kwargs, "in_features": self.hidden_width, "is_first": False}
                )
            )

        layers.add_module('output_layer',
                          BaseParametrizationLayer(
                            in_features=self.hidden_width,
                            out_features=self.out_features,
                            is_last=True,
                            activation_name=self.activation_name,
                            activation_kwargs= {**self.initializer_kwargs, "in_features": self.hidden_width, "is_first": False}
                          )
                        )

        return layers

    def forward(self, x, weights_and_biases):
        all_weights = weights_and_biases[:, :self.total_weights]
        all_biases = weights_and_biases[:, self.total_weights:]

        assert all_weights.shape[1] == self.total_weights
        assert all_biases.shape[1] == self.total_biases

        # Split weights and biases
        input_weights = all_weights[:, :self.input_weights]
        input_biases = all_biases[:, :self.input_biases]

        hidden_weights = all_weights[:, self.input_weights:self.input_weights + self.hidden_weights]
        hidden_biases = all_biases[:, self.input_biases:self.input_biases + self.hidden_biases]

        output_weights = all_weights[:, self.input_weights + self.hidden_weights:]
        output_biases = all_biases[:, self.input_biases + self.hidden_biases:]

        # Reshape weights and biases for each layer
        # Input layer
        input_weights = input_weights.view(-1, self.hidden_width, self.in_features)  # [batch, hidden, in]
        input_biases = input_biases.unsqueeze(-1)  # [batch, hidden, 1]

        # Hidden layers
        hidden_weights = hidden_weights.view(-1, self.n_layers, self.hidden_width,
                                             self.hidden_width)  # [batch, n_layers, hidden, hidden]
        hidden_biases = hidden_biases.view(-1, self.n_layers, self.hidden_width)  # [batch, n_layers, hidden]

        # Output layer
        output_weights = output_weights.view(-1, self.out_features, self.hidden_width)  # [batch, out, hidden]
        output_biases = output_biases.unsqueeze(-1)  # [batch, out, 1]

        # Process input through the network
        # Input layer
        x = self.net[0](x, (input_weights, input_biases))

        # Hidden layers
        for i in range(self.n_layers):
            h_w = hidden_weights[:, i]  # [batch, hidden, hidden]
            h_b = hidden_biases[:, i].unsqueeze(-1)  # [batch, hidden, 1]
            x = self.net[i + 1](x, (h_w, h_b))  # +1 because input layer is at index 0

        # Output layer
        x = self.net[-1](x, (output_weights, output_biases))

        return x











