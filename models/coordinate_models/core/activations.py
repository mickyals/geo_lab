# Custom activations: sine, Gaussian, ReLU, etc.
# Can also include scaled/periodic activations for SIREN
# consideration for using basic activations from torch.nn



import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = {}



def register_activation(name, description = ""):
    """
    Register an activation function.

    Args:
        name (str): The name of the activation function.
        activation (callable): The activation function.

    Raises:
        ValueError: If an activation function with the same name already exists.

    Returns:
        The activation function.

    """
    name = name.upper()
    def decorator(cls):
        """
        Decorator to register an activation function.

        Args:
            cls (type): The activation function class.

        Raises:
            ValueError: If an activation function with the same name already exists.

        Returns:
            The activation function class.
        """
        if name in ACTIVATIONS:
            raise ValueError(f"Activation with name {name} already exists.")
        ACTIVATIONS[name] = {"cls": cls, "description": description}
        return cls
    return decorator



def get_activation(name, **kwargs):
    """
    Get an activation function based on its name.

    Args:
        name (str): The name of the activation function.
        **kwargs: Additional keyword arguments to pass to the activation function.

    Returns:
        nn.Module: The activation function.

    Raises:
        ValueError: If the activation function with the given name does not exist.
    """
    name = name.upper()
    # Check if the activation function exists
    if name not in ACTIVATIONS:
        raise ValueError(f"Activation with name {name} does not exist.")

    # Return the activation function
    return ACTIVATIONS[name]['cls'](**kwargs)

def generate_alpha(x):
    """
    Calculate the absolute value of the input tensor plus 1.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Tensor with the absolute value of x plus 1.

    Note:
        This function is used to generate a scaling factor for activations.
    """
    # Calculate the absolute value of the input tensor and add 1.
    # This is done to ensure that the activation function is scaled properly.
    with torch.no_grad():
        return torch.abs(x) + 1


#===================================================================
# Sine activation
#===================================================================

@register_activation("SINE", description="Sine activation")
class sine_activation(nn.Module):
    """
    Sine activation module.

    Applies a sine activation function to the input tensor.

    Attributes:
        omega (float): The frequency parameter of the sine function.
    """
    def __init__(self, omega=30):
        """
        Initializes the SineActivation module.

        Args:
            omega (float, optional): The frequency parameter of the sine function.
                Defaults to 30.
        """
        super().__init__()
        self.omega = omega
        """
        The frequency parameter of the sine function.
        """

    def forward(self, x):
        """
        Apply the sine activation function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with the sine activation applied.
        """
        # Apply the sine activation function to the input tensor.
        # The frequency parameter is self.omega.
        return torch.sin(self.omega * x)

#===================================================================
# Finer activation
#===================================================================
@register_activation("FINER", description="Finer activation")
class sine_finer(nn.Module):
    """
    Sine activation module with a finer scale.

    Applies a sine activation function to the input tensor with a finer scale.

    Attributes:
        omega (float): The frequency parameter of the sine function.
    """
    def __init__(self, omega=30):
        """
        Initializes the SineFiner module.

        Args:
            omega (float, optional): The frequency parameter of the sine function.
                Defaults to 30.
        """
        super().__init__()
        self.omega = omega
        """
        The frequency parameter of the sine function.
        """

    def forward(self, x):
        """
        Apply the sine activation function to the input tensor with a finer scale.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with the sine activation applied.
        """
        # Apply the sine activation function to the input tensor.
        # The frequency parameter is self.omega, and the scale parameter is
        # calculated using the generate_alpha function.
        alpha = generate_alpha(x)
        return torch.sin(self.omega * alpha * x)

#===================================================================
# Gaussian activation
#===================================================================
@register_activation("GAUSSIAN", description="Gaussian activation")
class gaussian_activation(nn.Module):
    """
    Gaussian activation module.

    Applies a Gaussian activation function to the input tensor.

    Attributes:
        sigma (float): The standard deviation parameter of the Gaussian function.
    """
    def __init__(self, sigma=10):
        """
        Initializes the GaussianActivation module.

        Args:
            sigma (float, optional): The standard deviation parameter of the Gaussian function.
                Defaults to 10.
        """
        super().__init__()
        self.sigma = sigma
        """
        The standard deviation parameter of the Gaussian function.
        """

    def forward(self, x):
        return torch.exp(-(self.sigma * x)**2)

#===================================================================
# Finer Gaussian activation
#===================================================================
@register_activation("GAUSSIAN_FINER", description="Finer Gaussian activation")
class gaussian_finer(nn.Module):
    """
    GaussianFiner activation module.

    Applies a Gaussian activation function to the input tensor with a finer scale.

    Attributes:
        sigma (float): The standard deviation parameter of the Gaussian function.
        omega (float): The frequency parameter of the Gaussian function.
    """
    def __init__(self, sigma=10, omega=30):
        """
        Initializes the GaussianFiner module.

        Args:
            sigma (float, optional): The standard deviation parameter of the Gaussian function.
                Defaults to 10.
            omega (float, optional): The frequency parameter of the Gaussian function.
                Defaults to 30.
        """
        super().__init__()

        # Standard deviation parameter of the Gaussian function
        self.sigma = sigma

        # Frequency parameter of the Gaussian function
        self.omega = omega

    def forward(self, x):
        """
        Apply the GaussianFiner activation function to the input tensor with a finer scale.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with the GaussianFiner activation applied.
        """
        # Calculate the finer scaled sine activation using the generate_alpha function.
        alpha = generate_alpha(x)
        finer = torch.sin(self.omega * alpha * x)

        # Calculate the scaler for the standard deviation parameter using the frequency parameter.
        scaler = self.sigma/self.omega

        # Apply the GaussianFiner activation function to the finer scaled sine activation.
        return torch.exp(-(scaler * finer)**2)

#===================================================================
# Wire activation
#===================================================================
@register_activation("WIRE", description="Wire activation")
class wire_activation(nn.Module):
    """
    Wire activation module.

    Applies the wire activation function to the input tensor.

    Attributes:
        omega_0 (float): The frequency parameter of the wire activation function.
        sigma_0 (float): The standard deviation parameter of the wire activation function.
    """

    def __init__(self, omega_0=20, sigma_0=10):
        """
        Initializes the WireActivation module.

        Args:
            omega_0 (float, optional): The frequency parameter of the wire activation function.
                Defaults to 20.
            sigma_0 (float, optional): The standard deviation parameter of the wire activation function.
                Defaults to 10.
        """
        super().__init__()

        # Frequency parameter of the wire activation function
        self.omega_0 = omega_0

        # Standard deviation parameter of the wire activation function
        self.sigma_0 = sigma_0

    def forward(self, x):
        """
        Apply the wire activation function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with the wire activation applied.
        """
        # Calculate the complex exponential component of the wire activation function.
        complex_exp = torch.exp(1j * self.omega_0 * x)

        # Calculate the real exponential component of the wire activation function.
        # The absolute value of x is taken to ensure that the activation function is scaled properly.
        real_exp = torch.exp(-torch.abs(self.sigma_0 * x)**2)

        # Apply the wire activation function to the input tensor.
        return complex_exp * real_exp

#===================================================================
# Finer Wire activation
#===================================================================
@register_activation("WIRE_FINER", description="Finer Wire activation")
class wire_finer(nn.Module):
    """
    WireFiner activation module.

    Applies the wire activation function to the input tensor with a finer scale.

    Attributes:
        omega_0 (float): The frequency parameter of the wire activation function.
        sigma_0 (float): The standard deviation parameter of the wire activation function.
        omega_finer (float): The frequency parameter of the finer scale wire activation function.
    """

    def __init__(self, omega_0=20, sigma_0=10, omega_finer=5):
        """
        Initializes the WireFiner module.

        Args:
            omega_0 (float, optional): The frequency parameter of the wire activation function.
                Defaults to 20.
            sigma_0 (float, optional): The standard deviation parameter of the wire activation function.
                Defaults to 10.
            omega_finer (float, optional): The frequency parameter of the finer scale wire activation function.
                Defaults to 5.
        """
        super().__init__()

        # Frequency parameter of the wire activation function
        self.omega_0 = omega_0

        # Standard deviation parameter of the wire activation function
        self.sigma_0 = sigma_0

        # Frequency parameter of the finer scale wire activation function
        self.omega_finer = omega_finer

    def forward(self, x):
        """
        Apply the wire activation function to the input tensor with a finer scale.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with the wire activation applied.
        """
        # Calculate the finer scaled value of the input tensor.
        alpha = generate_alpha(x)
        z = alpha * x

        # Apply the sine activation function to the finer scaled value.
        y = torch.sin(self.omega_finer * z)

        # Calculate the scalers for the frequency and standard deviation parameters.
        scaler_omega = self.omega_0 / self.omega_finer
        scaler_sigma = self.sigma_0 / self.omega_finer

        # Calculate the complex exponential component of the wire activation function.
        complex_exp = torch.exp(1j * scaler_omega * y)

        # Calculate the real exponential component of the wire activation function.
        # The absolute value of y is taken to ensure that the activation function is scaled properly.
        real_exp = torch.exp(-(scaler_sigma * torch.abs(y))**2)

        # Apply the wire activation function to the input tensor.
        return complex_exp * real_exp

#===================================================================
# Hyperbolic Sine activation
#===================================================================
@register_activation("HOSC", description="Hyperbolic Sine activation")
class hosc_activation(nn.Module):
    """
    Hyperbolic Sine activation module.

    Applies a hyperbolic sine activation function to the input tensor.

    Attributes:
        beta (float): The scaling parameter of the hyperbolic sine function.
    """

    def __init__(self, beta=10):
        """
        Initializes the HOSCActivation module.

        Args:
            beta (float, optional): The scaling parameter of the hyperbolic sine function.
                Defaults to 10.
        """
        super().__init__()
        self.beta = beta
        """
        The scaling parameter of the hyperbolic sine function.
        """

    def forward(self, x):
        """
        Apply the hyperbolic sine activation function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with the hyperbolic sine activation applied.
        """
        # Calculate the hyperbolic sine activation function using the scaling parameter and the input tensor.
        # The hyperbolic sine function is calculated using the torch.sin function and then passed through the
        # torch.tanh function to obtain the final activation value.
        return torch.tanh(self.beta * torch.sin(x))

#===================================================================
# Finer Hyperbolic Sine activation
#===================================================================
@register_activation("HOSC_FINER", description="Finer Hyperbolic Sine activation")
class hosc_finer(nn.Module):
    """
    Hyperbolic sine activation module with a finer scale.

    Applies a hyperbolic sine activation function to the input tensor with a finer scale.

    Attributes:
        beta (float): The scaling parameter of the hyperbolic sine function.
        omega (float): The frequency parameter of the hyperbolic sine function.
    """

    def __init__(self, beta=10, omega=30):
        """
        Initializes the HOSCFiner module.

        Args:
            beta (float, optional): The scaling parameter of the hyperbolic sine function.
                Defaults to 10.
            omega (float, optional): The frequency parameter of the hyperbolic sine function.
                Defaults to 30.
        """
        super().__init__()
        self.beta = beta
        """
        The scaling parameter of the hyperbolic sine function.
        """
        self.omega = omega
        """
        The frequency parameter of the hyperbolic sine function.
        """

    def forward(self, x):
        """
        Apply the finer scaled hyperbolic sine activation function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with the finer scaled hyperbolic sine activation applied.

        Note:
            The finer scaled hyperbolic sine activation function is calculated using the following formula:
                tanh(beta_scaler * sin(omega * alpha * x))
            where alpha is calculated using the generate_alpha function, beta_scaler is calculated using the
            scaling parameter and the frequency parameter of the hyperbolic sine function, and omega is the frequency
            parameter of the hyperbolic sine function.
        """
        # Calculate the scaling factor for the beta parameter using the frequency parameter of the hyperbolic sine
        # function.
        beta_scaler = self.beta / self.omega

        # Calculate the finer scaled alpha value using the generate_alpha function and the input tensor.
        alpha = generate_alpha(x)

        # Calculate the finer scaled hyperbolic sine activation using the scaling factor, frequency parameter,
        # alpha value, and input tensor.
        return torch.tanh(beta_scaler * torch.sin(self.omega * alpha * x))

#===================================================================
# Sinc activation
#===================================================================
@register_activation("SINC", description="Sinc activation")
class sinc_activation(nn.Module):
    """
    Sinc activation module.

    Applies a sinc activation function to the input tensor.

    Attributes:
        omega (float): The frequency parameter of the sinc function.
    """

    def __init__(self, omega=30):
        """
        Initializes the SincActivation module.

        Args:
            omega (float, optional): The frequency parameter of the sinc function.
                Defaults to 30.
        """
        super().__init__()

        # The frequency parameter of the sinc function.
        self.omega = omega

    def forward(self, x):
        """
        Apply the sinc activation function to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The tensor with the sinc activation applied.

        Note:
            The sinc activation function is calculated using the following formula:
                sin(omega * x) / (omega * x)
            where omega is the frequency parameter of the sinc function.
        """
        # Apply the sinc activation function to the input tensor.
        # The frequency parameter is self.omega.
        return torch.sinc(self.omega * x)