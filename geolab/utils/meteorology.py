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



def compute_troposphere_gradients(inputs, model_outputs):
    """
    Compute the gradients of the troposphere model outputs with respect to the input coordinates.

    Parameters
    ----------
    inputs : dict
        Dictionary containing the input coordinates: longitude, latitude, pressure, and time.
    model_outputs : dict
        Dictionary containing the model outputs: u, v, w, and z.

    Returns
    -------
    grads : dict
        Dictionary containing the gradients of the model outputs with respect to the input coordinates.
    """

    xlongitude, ylatitude, pressure, time = inputs["longitude"], inputs["latitude"], inputs["pressure_level"], inputs["time"]
    u_pred, v_pred, w_pred, z_pred = model_outputs["u"], model_outputs["v"], model_outputs["w"], model_outputs["z"]

    grads = {}

    # Compute the gradient of u with respect to time
    grads["u_t"] = torch.autograd.grad(
        u_pred,
        time,
        grad_outputs=torch.ones_like(u_pred),
        retain_graph=True,
        create_graph=True
    ) # 1 of 11

    # Compute the gradient of v with respect to time
    grads["v_t"] = torch.autograd.grad(
        v_pred,
        time,
        grad_outputs=torch.ones_like(v_pred),
        retain_graph=True,
        create_graph=True
    ) # 2 of 11

    # Compute the gradient of u with respect to longitude
    grads["u_x"] = torch.autograd.grad(
        u_pred,
        xlongitude,
        grad_outputs=torch.ones_like(u_pred),
        retain_graph=True,
        create_graph=True
    ) # 3 of 11

    # Compute the gradient of v with respect to longitude
    grads["v_x"] = torch.autograd.grad(
        v_pred,
        xlongitude,
        grad_outputs=torch.ones_like(v_pred),
        retain_graph=True,
        create_graph=True
    ) # 4 of 11

    # Compute the gradient of u with respect to latitude
    grads["u_y"] = torch.autograd.grad(
        u_pred,
        ylatitude,
        grad_outputs=torch.ones_like(u_pred),
        retain_graph=True,
        create_graph=True
    ) # 5 of 11

    # Compute the gradient of v with respect to latitude
    grads["v_y"] = torch.autograd.grad(
        v_pred,
        ylatitude,
        grad_outputs=torch.ones_like(v_pred),
        retain_graph=True,
        create_graph=True
    ) # 6 of 11

    # Compute the gradient of u with respect to pressure
    grads["u_p"] = torch.autograd.grad(
        u_pred,
        pressure,
        grad_outputs=torch.ones_like(u_pred),
        retain_graph=True,
        create_graph=True
    ) # 7 of 11

    # Compute the gradient of v with respect to pressure
    grads["v_p"] = torch.autograd.grad(
        v_pred,
        pressure,
        grad_outputs=torch.ones_like(v_pred),
        retain_graph=True,
        create_graph=True
    ) # 8 of 11

    # Compute the gradient of w with respect to pressure
    grads["w_p"] = torch.autograd.grad(
        w_pred,
        pressure,
        grad_outputs=torch.ones_like(w_pred),
        retain_graph=True,
        create_graph=True
    ) # 9 of 11

    # Compute the gradient of z with respect to longitude
    grads["z_x"] = torch.autograd.grad(
        z_pred,
        xlongitude,
        grad_outputs=torch.ones_like(z_pred),
        retain_graph=True,
        create_graph=True
    ) # 10 of 11

    # Compute the gradient of z with respect to latitude
    grads["z_y"] = torch.autograd.grad(
        z_pred,
        ylatitude,
        grad_outputs=torch.ones_like(z_pred),
        retain_graph=True,
        create_graph=True
    ) # 11 of 11

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
    central_latitude_rad = torch.abs(central_latitude) * np.pi / 180
    f_0 = 2 * omega * torch.sin(central_latitude_rad)
    coriolis_force["f_0"] = f_0 / (2 * omega)

    # beta normalized by its max at equator (cos(0) = 1)
    # beta = (2 * omega / earth_radius) * cos(central_latitude_rad)
    beta = (2 * omega / earth_radius) * torch.cos(central_latitude_rad)
    beta_max = 2 * omega / earth_radius
    coriolis_force["beta"] = beta / beta_max

    return coriolis_force

