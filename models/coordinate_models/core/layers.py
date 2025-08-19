# Base layer types: Linear, FourierLayer, SIRENLayer
import torch
import torch.nn as nn
from ..core.initializations import get_initializer
from ..core.activations import get_activation

LAYERS = {}

def register_layer(name, description=""):
    """
    Decorator to register a layer.

    Args:
        name (str): The name of the layer.
        description (str, optional): A description of the layer. Defaults to "".

    Raises:
        ValueError: If a layer with the same name already exists.

    Returns:
        The layer class.
    """
    name_upper = name.upper()

    def decorator(cls):
        """
        Decorator function to register an embedding class.

        Args:
            cls (type): The embedding class.

        Raises:
            ValueError: If an embedding with the same name already exists.

        Returns:
            The embedding class.
        """
        if name_upper in LAYERS:
            raise ValueError(f"Embedding with name {name_upper} already exists.")
        LAYERS[name_upper] = {"cls": cls, "description": description}
        return cls

    return decorator


@register_layer("BASE", "Base mlp layer allowing for custom initializer and activation")
class BaseLayer(nn.Module):
    """
    Base layer class for neural networks.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        initializer (str): Initializer for the linear layer.
        activation (str): Activation function for the linear layer.
        is_last (bool, optional): Whether it is the last layer in the network. Defaults to False.
        initializer_kwargs (dict, optional): Keyword arguments for the initializer. Defaults to None.
        activation_kwargs (dict, optional): Keyword arguments for the activation function. Defaults to None.
    """
    def __init__(self, in_features, out_features, initializer, activation, is_last=False, initializer_kwargs=None, activation_kwargs=None):
        """
        Initialize the BaseLayer class.
        """
        super().__init__()
        # Initialize the input and output features
        self.in_features = in_features
        self.out_features = out_features
        # Initialize the linear layer
        self.linear = nn.Linear(in_features, out_features)
        # Initialize the activation function
        self.activation = get_activation(activation, **(activation_kwargs if activation_kwargs is not None else {}))
        # Initialize the weight initialization method
        self.initializer = get_initializer(initializer,  **(initializer_kwargs if initializer_kwargs is not None else {}))
        # Initialize whether it is the last layer in the network
        self.is_last = is_last
        # Apply the weight initialization to the linear layer
        self.initializer(self.linear)

    def forward(self, x):
        """
        Forward pass of the BaseLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the linear layer
            and activation function (if it is the last layer).
        """
        # If the layer is the last layer, just apply the linear layer
        if self.is_last:
            return self.linear(x)
        # Otherwise, apply the linear layer and the activation function
        return self.activation(self.linear(x))


@register_layer("RESIDUAL", "Residual block allowing for custom initializer and activation")
class ResidualBaseBlock(nn.Module):
    """
    Residual block for the neural network.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        initializer (str): Initializer for the linear layers.
        activation (str): Activation function for the linear layers.
        residual_weight (float, optional): Weight for the residual connection. Defaults to None, which means no weighting.
        initializer_kwargs (dict, optional): Keyword arguments for the initializer. Defaults to None.
        activation_kwargs (dict, optional): Keyword arguments for the activation function. Defaults to None.
    """
    def __init__(self, in_features, out_features, initializer, activation, residual_weight=None, initializer_kwargs=None, activation_kwargs=None):
        """
        Initialize the ResidualBaseBlock class.
        """
        super().__init__()
        # Activation function for the linear layers
        self.activation = get_activation(activation, **activation_kwargs)
        # Weight for the residual connection
        self.residual_weight = residual_weight

        # Linear layers
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

        # Only create a projection if needed
        if in_features != out_features:
            # Residual layer
            self.residual = nn.Linear(in_features, out_features)
        else:
            # Identity layer for the residual connection
            self.residual = nn.Identity()

        # Initialize the linear layers
        init_fn = get_initializer(initializer, **initializer_kwargs)
        init_fn(self.linear1)
        init_fn(self.linear2)
        if not isinstance(self.residual, nn.Identity):
            init_fn(self.residual)

    def forward(self, x):
        """
        Forward pass of the ResidualBaseBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the residual block.
        """
        # Apply the first linear layer and activation function
        main_path = self.activation(self.linear1(x))

        # Apply the second linear layer and activation function
        main_path = self.activation(self.linear2(main_path))

        # Apply the residual layer (either identity or linear)
        res = self.residual(x)

        # Add or weight the main path and residual path
        if self.residual_weight is None:
            # Add the main path and residual path
            return main_path + res
        else:
            # Weight the main path and residual path
            return (1 - self.residual_weight) * main_path + self.residual_weight * res


