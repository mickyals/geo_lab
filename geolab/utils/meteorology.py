import numpy as np
import torch


def omega_to_w(omega, pressure_level, temperature):
    """
    Compute the vertical component of velocity (w in m/s) from the vertical velocity (omega in Pa/s^-1) and input coordinates.

    Parameters
    ----------
    omega : float
        Angular velocity in rad/s.
    pressure_level : float
        Pressure level in hPa.
    temperature : float
        Temperature in Kelvin.

    Returns
    -------
    w : float
        Vertical velocity in m/s.
    """
    # Gas constant in J/(kg*K)
    gas_constant = 287.058
    # Gravitational acceleration in m/s^2
    gravitational_acceleration = 9.80665

    # Convert pressure level from hPa to Pa
    pressure_level = pressure_level * 100

    # Compute density
    rho = pressure_level / (gas_constant * temperature)

    # Compute vertical velocity
    w = -omega / (rho * gravitational_acceleration)

    return w



def compute_troposphere_gradients(inputs_tensor, model_outputs):
    """
    Compute the gradients of the troposphere model outputs with respect to the input coordinates.

    Parameters
    ----------
    inputs_tensor : torch.Tensor
        Stacked input tensor with shape [batch, 4] where columns are:
        [longitude, latitude, pressure_level, time]
    model_outputs : dict
        Dictionary containing the model outputs: u, v, w, and z.

    Returns
    -------
    grads : dict
        Dictionary containing the gradients of the model outputs with respect to the input coordinates.
    """
    # Get model outputs
    u_pred = model_outputs["u"]
    v_pred = model_outputs["v"]
    w_pred = model_outputs["w"]
    z_pred = model_outputs["z"]

    grads = {}

    # Compute gradients with respect to the full input tensor
    # Then extract the relevant column for each coordinate
    
    # Gradient of u with respect to all inputs
    grad_u = torch.autograd.grad(
        u_pred.sum(),
        inputs_tensor,
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    
    if grad_u is not None:
        grads["u_x"] = grad_u[:, 0]  # longitude (column 0)
        grads["u_y"] = grad_u[:, 1]  # latitude (column 1)
        grads["u_p"] = grad_u[:, 2]  # pressure (column 2)
        grads["u_t"] = grad_u[:, 3]  # time (column 3)
    else:
        # Fallback if gradient is None
        grads["u_x"] = torch.zeros_like(u_pred)
        grads["u_y"] = torch.zeros_like(u_pred)
        grads["u_p"] = torch.zeros_like(u_pred)
        grads["u_t"] = torch.zeros_like(u_pred)

    # Gradient of v with respect to all inputs
    grad_v = torch.autograd.grad(
        v_pred.sum(),
        inputs_tensor,
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    
    if grad_v is not None:
        grads["v_x"] = grad_v[:, 0]
        grads["v_y"] = grad_v[:, 1]
        grads["v_p"] = grad_v[:, 2]
        grads["v_t"] = grad_v[:, 3]
    else:
        grads["v_x"] = torch.zeros_like(v_pred)
        grads["v_y"] = torch.zeros_like(v_pred)
        grads["v_p"] = torch.zeros_like(v_pred)
        grads["v_t"] = torch.zeros_like(v_pred)

    # Gradient of w with respect to pressure
    grad_w = torch.autograd.grad(
        w_pred.sum(),
        inputs_tensor,
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    
    if grad_w is not None:
        grads["w_p"] = grad_w[:, 2]  # pressure (column 2)
    else:
        grads["w_p"] = torch.zeros_like(w_pred)

    # Gradient of z with respect to longitude and latitude
    grad_z = torch.autograd.grad(
        z_pred.sum(),
        inputs_tensor,
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]
    
    if grad_z is not None:
        grads["z_x"] = grad_z[:, 0]  # longitude (column 0)
        grads["z_y"] = grad_z[:, 1]  # latitude (column 1)
    else:
        grads["z_x"] = torch.zeros_like(z_pred)
        grads["z_y"] = torch.zeros_like(z_pred)

    return grads

def coriolis_force(latitude, earth_radius=6371222.9, central_latitude=0):
    """
    Calculate the Coriolis force given the latitude.

    Parameters
    ----------
    latitude : float or torch.Tensor
        Latitude in degrees.
    earth_radius : float, optional
        Radius of the Earth in meters. Defaults to 6371222.9.
    central_latitude : float, optional
        Latitude of the central meridian in degrees. Defaults to 0.

    Returns
    -------
    f_0 : float or torch.Tensor
        Coriolis force at the central meridian.
    f : float or torch.Tensor
        Coriolis force at the given latitude.
    beta : float or torch.Tensor
        Beta parameter in the Coriolis force equation.
        https://en.wikipedia.org/wiki/Beta_plane
    """
    # Initialize the output dictionary
    coriolis_force = {}

    # Angular velocity of the Earth in radians per second
    omega = 7.2921e-5

    # Convert latitude to radians - coriolis force is only defined for latitudes between -90 and 90 but is symmetric about the equator
    latitude_rad = torch.abs(latitude) * np.pi / 180

    # f normalized: 0 at equator, 1 at poles
    # f = 2 * omega * sin(latitude_rad)
    f = 2 * omega * torch.sin(latitude_rad)
    coriolis_force["f"] = f / (2 * omega)

    # f_0 at central latitude (magnitude, 0–1)
    # f_0 = 2 * omega * sin(central_latitude_rad)
    central_latitude_rad = torch.abs(torch.tensor(central_latitude, device=latitude.device)) * (torch.pi / 180)
    f_0 = 2 * omega * torch.sin(central_latitude_rad)
    coriolis_force["f_0"] = f_0 / (2 * omega)

    # beta normalized by its max at equator (cos(0) = 1)
    # beta = (2 * omega / earth_radius) * cos(central_latitude_rad)
    beta = (2 * omega / earth_radius) * torch.cos(central_latitude_rad)
    beta_max = 2 * omega / earth_radius
    coriolis_force["beta"] = beta / beta_max

    return coriolis_force
