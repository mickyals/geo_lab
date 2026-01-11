import numpy as np
from typing import Dict
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

def compute_troposphere_gradients(
        inputs_tensor: torch.Tensor,
        model_outputs: Dict[str, torch.Tensor],
        coord_labels: Dict[str, int]  # NEW parameter
        ) -> Dict[str, torch.Tensor]:
    """
    Compute the gradients of the troposphere model outputs with respect to the input coordinates.

    Parameters
    ----------
    inputs_tensor : torch.Tensor
        Stacked input tensor with shape [batch, 4] where columns are in ANY order
    model_outputs : dict
        Dictionary containing the model outputs: u, v, w, and z.
    coord_labels : dict
        Mapping of coordinate names to column indices

    Returns
    -------
    grads : dict
        Dictionary containing the gradients of the model outputs with respect to the input coordinates.
    """
    # Extract indices
    lon_idx = coord_labels['longitude']
    lat_idx = coord_labels['latitude']
    p_idx = coord_labels['pressure_level']
    t_idx = coord_labels['valid_time']

    # Get model outputs
    u_pred = model_outputs["u"]
    v_pred = model_outputs["v"]
    w_pred = model_outputs["w"]
    z_pred = model_outputs["z"]

    grads = {}

    # Compute gradients with respect to the full input tensor
    grad_u = torch.autograd.grad(
        u_pred,
        inputs_tensor,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]

    if grad_u is not None:
        grads["u_x"] = grad_u[:, lon_idx]  # Use label-based indexing
        grads["u_y"] = grad_u[:, lat_idx]
        grads["u_p"] = grad_u[:, p_idx]
        grads["u_t"] = grad_u[:, t_idx]
    else:
        grads["u_x"] = torch.zeros_like(u_pred)
        grads["u_y"] = torch.zeros_like(u_pred)
        grads["u_p"] = torch.zeros_like(u_pred)
        grads["u_t"] = torch.zeros_like(u_pred)

    # Gradient of v with respect to all inputs
    grad_v = torch.autograd.grad(
        v_pred,
        inputs_tensor,
        grad_outputs=torch.ones_like(v_pred),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]

    if grad_v is not None:
        grads["v_x"] = grad_v[:, lon_idx]
        grads["v_y"] = grad_v[:, lat_idx]
        grads["v_p"] = grad_v[:, p_idx]
        grads["v_t"] = grad_v[:, t_idx]
    else:
        grads["v_x"] = torch.zeros_like(v_pred)
        grads["v_y"] = torch.zeros_like(v_pred)
        grads["v_p"] = torch.zeros_like(v_pred)
        grads["v_t"] = torch.zeros_like(v_pred)

    # Gradient of w with respect to pressure
    grad_w = torch.autograd.grad(
        w_pred,
        inputs_tensor,
        grad_outputs=torch.ones_like(w_pred),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]

    if grad_w is not None:
        grads["w_p"] = grad_w[:, p_idx]
    else:
        grads["w_p"] = torch.zeros_like(w_pred)

    # Gradient of z with respect to longitude and latitude
    grad_z = torch.autograd.grad(
        z_pred,
        inputs_tensor,
        grad_outputs=torch.ones_like(z_pred),
        create_graph=True,
        retain_graph=True,
        allow_unused=True
    )[0]

    if grad_z is not None:
        grads["z_x"] = grad_z[:, lon_idx]
        grads["z_y"] = grad_z[:, lat_idx]
    else:
        grads["z_x"] = torch.zeros_like(z_pred)
        grads["z_y"] = torch.zeros_like(z_pred)

    return grads


def coriolis_force(latitude, earth_radius=6371222.9, central_latitude=0, return_full=False):
    """
    Calculate the Coriolis force given the latitude.

    Optimized version: reduces tensor allocations and unnecessary normalizations.
    Only computes f_0 and beta when needed for regional/beta-plane approximations.

    Parameters
    ----------
    latitude : torch.Tensor
        Latitude in degrees.
    earth_radius : float, optional
        Radius of the Earth in meters. Defaults to 6371222.9.
    central_latitude : float, optional
        Latitude of the central meridian in degrees for beta-plane approximation.
        Defaults to 0.
    return_full : bool, optional
        If True, also compute f_0 and beta for beta-plane approximations.
        If False (default), only return f for better performance.

    Returns
    -------
    dict
        Dictionary with:
        - "f": Coriolis parameter (always included)
        - "f_0": Coriolis parameter at central latitude (if return_full=True)
        - "beta": Beta parameter for beta-plane (if return_full=True)

    Examples
    --------
    >>> # Global simulation (fast)
    >>> coriolis = coriolis_force(lat_tensor)
    >>> f = coriolis["f"]

    >>> # Regional model with beta-plane
    >>> coriolis = coriolis_force(lat_tensor, central_latitude=45.0, return_full=True)
    >>> f, f_0, beta = coriolis["f"], coriolis["f_0"], coriolis["beta"]
    """
    # Constants - create once per call, reuse
    omega = 7.2921e-5  # Just use float, torch will handle dtype/device from latitude

    # Convert latitude to radians
    latitude_rad = torch.abs(latitude) * (torch.pi / 180.0)

    # f = 2 * omega * sin(latitude_rad) - return PHYSICAL value, not normalized
    f = 2 * omega * torch.sin(latitude_rad)

    if not return_full:
        return {"f": f}  # Fast path for global simulations

    # Beta-plane approximation parameters (for regional models)
    central_lat_rad = abs(central_latitude) * (torch.pi / 180.0)
    f_0 = 2 * omega * torch.sin(torch.tensor(central_lat_rad,
                                             dtype=latitude.dtype,
                                             device=latitude.device))

    beta = (2 * omega / earth_radius) * torch.cos(torch.tensor(central_lat_rad,
                                                               dtype=latitude.dtype,
                                                               device=latitude.device))

    return {
        "f": f,  # Physical values, not normalized
        "f_0": f_0,
        "beta": beta
    }