# Weight initialization strategies (standard, Xavier, SIREN-specific, etc.)

import torch
import torch.nn.init as init
import math

INITIALIZERS = {}


def register_initializer(name):
    def decorator(cls):
        if name in INITIALIZERS:
            raise ValueError(f"Initializer with name {name} already exists.")
        INITIALIZERS[name] = cls
        return cls
    return decorator


def get_initializer(name, **kwargs):
    name = name.upper()
    if name not in INITIALIZERS:
        raise ValueError(f"Initializer with name {name} does not exist.")
    return INITIALIZERS[name](**kwargs)




#===============================================================================
# SIREN-specific initializers
#===============================================================================

@register_initializer("SIREN")
class SirenInit:
    def __init__(self, in_features, is_first=False, omega=30):
        """
        Initializes the SirenInit class.

        Args:
            in_features (int): Number of input features.
            is_first (bool, optional): Indicates whether the layer is the first layer. Defaults to False.
            omega (float, optional): Frequency parameter. Defaults to 30.
        """
        self.is_first = is_first  # Indicates whether the layer is the first layer
        self.in_features = in_features  # Number of input features
        self.omega = omega  # Frequency parameter

    def __call__(self, linear_layer):
        """
        Initialize the weights of a linear layer with SIREN-specific strategy.

        Args:
            linear_layer (torch.nn.Linear): The linear layer to be initialized.

        Returns:
            None
        """
        with torch.no_grad():
            if self.is_first:
                # If the layer is the first layer, initialize weights uniformly
                # with range [-1 / in_features, 1 / in_features]
                bound = 1 / self.in_features
            else:
                # If the layer is not the first layer, initialize weights with
                # a range proportional to sqrt(6 / in_features) multiplied
                # by the frequency parameter
                bound = math.sqrt(6 / self.in_features)/self.omega

            # Initialize the weights of the linear layer with the computed bound
            linear_layer.weight.uniform_(-bound, bound)

#===============================================================================
# FINER-specific initializers
#===============================================================================

@register_initializer("FINER")
class FinerInit:
    """
    Initializer for the Finer model.

    Args:
        in_features (int): Number of input features.
        is_first (bool, optional): Indicates whether the layer is the first layer. Defaults to False.
        omega (float, optional): Frequency parameter. Defaults to 30.
        k (int, optional): Scaling parameter. Defaults to 1.
    """
    def __init__(self, in_features, is_first=False, omega=30, k = 1):
        """
        Initializes the FinerInit class.

        Args:
            in_features (int): Number of input features.
            is_first (bool, optional): Indicates whether the layer is the first layer. Defaults to False.
            omega (float, optional): Frequency parameter. Defaults to 30.
            k (int, optional): Scaling parameter. Defaults to 1.
        """
        self.is_first = is_first
        self.in_features = in_features
        self.omega = omega
        self.k = k

    def __call__(self, linear_layer):
        """
        Applies the initializer to the given linear layer.

        Args:
            linear_layer (torch.nn.Linear): The linear layer to initialize.

        The initializer initializes the weights and biases of the linear layer.
        The weights are initialized with values sampled from a uniform distribution
        with bounds determined by the `is_first` and `in_features` attributes.
        The biases are initialized with values sampled from a uniform distribution
        with bounds of `-k` and `k`.

        Note:
            The `is_first` attribute indicates whether the layer is the first layer
            in the model. If it is the first layer, the bounds for the weights are
            determined by the reciprocal of `in_features`. Otherwise, the bounds
            are determined by the square root of 6 divided by `in_features` multiplied
            by `omega`.
        """
        with torch.no_grad():
            # Determine the bounds for the weights
            if self.is_first:
                bound = 1 / self.in_features
            else:
                bound = math.sqrt(6 / self.in_features)/self.omega

            # Initialize the weights
            linear_layer.weight.uniform_(-bound, bound)

            # Initialize the biases
            if linear_layer.bias is not None:
                linear_layer.bias.uniform_(-self.k, self.k)


@register_initializer("XAVIER_UNIFORM_FINER")
class XavierUniformFinerInit:
    """
    Initializer that applies the Xavier uniform initialization with a finer scale.

    The Xavier uniform initialization initializes the weights of a linear layer with
    values sampled from a uniform distribution. The bounds for the weights are
    determined by the square root of 6 divided by the number of input features
    multiplied by the `gain` parameter. The biases are initialized with values
    sampled from a uniform distribution with bounds of `-k` and `k`.

    Attributes:
        gain (float): The gain parameter of the Xavier initialization.
        k (float): The range of the uniform distribution for the biases.
    """
    def __init__(self, gain=1.0, k=1):
        """
        Initializes the XavierUniformFinerInit class.

        Args:
            gain (float, optional): The gain parameter of the Xavier initialization.
                Defaults to 1.0.
            k (float, optional): The range of the uniform distribution for the biases.
                Defaults to 1.0.
        """
        self.gain = gain
        self.k = k

    def __call__(self, linear_layer):
        """
        Applies the Xavier uniform initialization with a finer scale to the given linear layer.

        Args:
            linear_layer (torch.nn.Linear): The linear layer to initialize.

        The initializer initializes the weights and biases of the linear layer.
        The weights are initialized with values sampled from a uniform distribution
        with bounds determined by the `gain` attribute. The biases are initialized
        with values sampled from a uniform distribution with bounds of `-k` and `k`.
        """
        with torch.no_grad():
            # Initialize the weights with the Xavier uniform initialization
            init.xavier_uniform_(linear_layer.weight, gain=self.gain)

            # Initialize the biases with the uniform distribution
            if linear_layer.bias is not None:
                linear_layer.bias.uniform_(-self.k, self.k)

@register_initializer("XAVIER_FINER_NORM")
class XavierNormFinerInit:
    """
    Initializer that applies the Xavier normal initialization with a finer scale.

    The Xavier normal initialization initializes the weights of a linear layer with
    values sampled from a normal distribution. The mean and standard deviation of
    the normal distribution are both 0. The bounds for the weights are determined by
    the square root of 2 divided by the number of input features multiplied by the
    `gain` parameter. The biases are initialized with values sampled from a uniform
    distribution with bounds of `-k` and `k`.

    Attributes:
        gain (float): The gain parameter of the Xavier initialization.
        k (float): The range of the uniform distribution for the biases.
    """
    def __init__(self, gain=1.0, k=1):
        """
        Initializes the XavierNormFinerInit class.

        Args:
            gain (float, optional): The gain parameter of the Xavier initialization.
                Defaults to 1.0.
            k (float, optional): The range of the uniform distribution for the biases.
                Defaults to 1.0.
        """
        self.gain = gain
        self.k = k

    def __call__(self, linear_layer):
        """
        Initialize the weights and biases of the `linear_layer` with the Xavier normal
        initialization with a finer scale.

        The Xavier normal initialization initializes the weights of a linear layer with
        values sampled from a normal distribution. The mean and standard deviation of
        the normal distribution are both 0. The bounds for the weights are determined by
        the square root of 2 divided by the number of input features multiplied by the
        `gain` parameter. The biases are initialized with values sampled from a uniform
        distribution with bounds of `-k` and `k`.

        Args:
            linear_layer (torch.nn.Linear): The linear layer to be initialized.
        """
        with torch.no_grad():
            # Initialize the weights with the Xavier normal initialization
            init.xavier_normal_(linear_layer.weight, gain=self.gain)

            # Initialize the biases with the uniform distribution
            if linear_layer.bias is not None:
                linear_layer.bias.uniform_(-self.k, self.k)


#===============================================================================
# Standard initializers
#===============================================================================

@register_initializer("XAVIER_UNIFORM")
class XavierUniformInit:
    """
    Class for initializing the weights of a linear layer with the Xavier uniform
    initialization.

    Attributes:
        gain (float): The gain parameter of the Xavier initialization.
    """
    def __init__(self, gain=1.0):
        """
        Initializes the XavierUniformInit class with the gain parameter.

        Args:
            gain (float, optional): The gain parameter of the Xavier initialization.
                Defaults to 1.0.
        """
        # Gain parameter of the Xavier initialization
        self.gain = gain

    def __call__(self, linear_layer):
        with torch.no_grad():
            init.xavier_uniform_(linear_layer.weight, gain=self.gain)

@register_initializer("XAVIER_NORMAL")
class XavierNormalInit:
    """
    Class for initializing the weights of a linear layer with the Xavier normal
    initialization.

    Attributes:
        gain (float): The gain parameter of the Xavier initialization.
    """
    def __init__(self, gain=1.0):
        """
        Initializes the XavierNormalInit class with the gain parameter.

        Args:
            gain (float, optional): The gain parameter of the Xavier initialization.
                Defaults to 1.0.
        """
        # Gain parameter of the Xavier normal initialization
        self.gain = gain

    def __call__(self, linear_layer):
        """
        Initializes the weights of the `linear_layer` with the Xavier normal
        initialization.

        Args:
            linear_layer (torch.nn.Linear): The linear layer to be initialized.
        """
        with torch.no_grad():
            # Initialize the weights with the Xavier normal initialization
            init.xavier_normal_(linear_layer.weight, gain=self.gain)
            # The `gain` parameter scales the standard deviation of the normal
            # distribution. The default value of 1.0 corresponds to the standard
            # deviation of the normal distribution being equal to the square root
            # of 2 / number_of_input_features. Increasing the gain parameter
            # increases the standard deviation of the normal distribution, which
            # in turn allows the weights to be initialized with a larger range
            # of values.

@register_initializer("NORMAL")
class NormalInit:
    """
    Class for initializing the weights of a linear layer with a normal
    distribution.

    Attributes:
        mean (float): The mean of the normal distribution.
        std (float): The standard deviation of the normal distribution.
    """
    def __init__(self, mean=0.0, std=0.1):
        """
        Initializes the NormalInit class with the mean and standard deviation of
        the normal distribution.

        Args:
            mean (float, optional): The mean of the normal distribution. Defaults
                to 0.0.
            std (float, optional): The standard deviation of the normal
                distribution. Defaults to 0.1.
        """
        # Mean of the normal distribution
        self.mean = mean
        # Standard deviation of the normal distribution
        self.std = std

    def __call__(self, linear_layer):
        """
        Initialize the weights of the `linear_layer` with a normal distribution.

        Args:
            linear_layer (torch.nn.Linear): The linear layer to be initialized.
        """
        with torch.no_grad():
            # Initialize the weights with a normal distribution
            # The mean and standard deviation of the normal distribution are
            # specified by the `mean` and `std` attributes of the class.
            linear_layer.weight.normal_(self.mean, self.std)
            # The `mean` parameter specifies the mean of the normal distribution,
            # and the `std` parameter specifies the standard deviation of the
            # normal distribution. The default values are 0.0 and 0.1 respectively.

@register_initializer("UNIFORM")
class UniformInit:
    def __init__(self, a=-0.1, b=0.1):
        """
        Initializes the UniformInit class with the lower and upper bounds of the
        uniform distribution.

        Args:
            a (float, optional): The lower bound of the uniform distribution.
                Defaults to -0.1.
            b (float, optional): The upper bound of the uniform distribution.
                Defaults to 0.1.
        """
        # Lower bound of the uniform distribution
        self.a = a
        # Upper bound of the uniform distribution
        self.b = b
        """
        The lower and upper bounds of the uniform distribution.
        """

    def __call__(self, linear_layer):
        """
        Initialize the weights of the `linear_layer` with a uniform distribution.

        Args:
            linear_layer (torch.nn.Linear): The linear layer to be initialized.
        """
        with torch.no_grad():
            # Initialize the weights with a uniform distribution
            # The bounds of the uniform distribution are specified by the `a`
            # and `b` attributes of the class.
            linear_layer.weight.uniform_(self.a, self.b)
            """
            Initialize the weights with a uniform distribution.

            The bounds of the uniform distribution are specified by the `a`
            and `b` attributes of the class. The default values are -0.1 and 0.1
            respectively.
            """

@register_initializer("IDENTITY")
class IdentityInit:
    def __call__(self, linear_layer):
        """
        Initialize the weights of the `linear_layer` as an identity matrix.

        Args:
            linear_layer (torch.nn.Linear): The linear layer to be initialized.

        Raises:
            ValueError: If the input shape is not square.
        """
        # Identity initialization is only valid for square matrices.
        if linear_layer.weight.shape[0] != linear_layer.weight.shape[1]:
            raise ValueError("Identity initialization is only valid for square matrices.")

        with torch.no_grad():
            # Set the weights to an identity matrix.
            linear_layer.weight.copy_(torch.eye(linear_layer.weight.shape[0]))

@register_initializer("ORTHOGONAL")
class OrthogonalInit:
    """
    Initializes the weights of a linear layer with an orthogonal matrix.

    Attributes:
        gain (float): The gain factor to be applied to the orthogonal matrix.
            Defaults to 1.0.
    """
    def __init__(self, gain=1.0):
        """
        Initializes the OrthogonalInit object.

        Args:
            gain (float, optional): The gain factor to be applied to the orthogonal matrix.
                Defaults to 1.0.
        """
        # Gain factor to be applied to the orthogonal matrix.
        self.gain = gain

    def __call__(self, linear_layer):
        """
        Initialize the weights of the `linear_layer` with an orthogonal matrix.

        Args:
            linear_layer (torch.nn.Linear): The linear layer to be initialized.
        """
        with torch.no_grad():
            # Initialize the weights with an orthogonal matrix using the torch.nn.init.orthogonal_
            # function. The gain parameter is used to scale the orthogonal matrix.
            init.orthogonal_(linear_layer.weight, gain=self.gain)