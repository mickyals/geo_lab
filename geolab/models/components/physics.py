from geolab.utils.meteorology import coriolis_force, compute_troposphere_gradients
import torch


def troposphere_pde_residual(inputs_tensor, outputs, statistics, mass_balance=True):
    """
    Compute the residuals of the troposphere PDEs.
    Args:
        inputs_tensor (torch.Tensor): Stacked input tensor with shape [batch, 4] where columns are:
                                      [longitude, latitude, pressure_level, time]
        outputs (dict): Dictionary containing the model outputs: u, v, w, and z (geopotential).
        mass_balance (bool, optional): Whether to include the mass balance term in the residuals. Defaults to True.
    Returns:
        tuple: Tuple of the residuals of the longitudinal, latitudinal, and mass balance terms.
    """
    u, v, w, z = outputs["u"], outputs["v"], outputs["w"], outputs["z"]
    
    # Physical constants
    omega = 7.2921e-5  # Earth's angular velocity in rad/s
    R_earth = 6371222.9  # Earth radius in meters
    
    # Get scaling factors for inputs
    lon_scale = 2.0 / (statistics["longitude"][1] - statistics["longitude"][0])
    lat_scale = 2.0 / (statistics["latitude"][1] - statistics["latitude"][0])
    p_scale = 2.0 / (statistics["pressure_level"][1] - statistics["pressure_level"][0])
    t_scale = 1.0 if statistics["valid_time"][0]==statistics["valid_time"][1] else 2.0 / (statistics["valid_time"][1] - statistics["valid_time"][0])
    
    # Get scaling factors for outputs
    u_scale = 2.0 / (statistics["u"][1] - statistics["u"][0])
    v_scale = 2.0 / (statistics["v"][1] - statistics["v"][0])
    w_scale = 2.0 / (statistics["w"][1] - statistics["w"][0])
    z_scale = 2.0 / (statistics["z"][1] - statistics["z"][0])


    # Apply scaling to ensure all terms are dimensionless and of similar magnitude
    
    # Apply scaling to ensure all terms are dimensionless and of similar magnitude
    scale_factors = {
        'u': u_scale, 'v': v_scale, 'w': w_scale, 'z': z_scale,
        'x': lon_scale, 'y': lat_scale, 'p': p_scale, 't': t_scale
    }
    
    latitude = inputs_tensor[:, 1]
    grads = compute_troposphere_gradients(inputs_tensor, outputs)
    
    # Get PHYSICAL (not normalized) Coriolis parameter
    coriolis_params = coriolis_force(latitude)
    f = coriolis_params["f"] * 2 * omega  # Denormalize: f = 2*Ω*sin(lat)
    
    # Convert longitude/latitude gradients to physical spatial gradients
    latitude_rad = torch.abs(latitude) * torch.pi / 180.0
    
    # Metric factors for spherical coordinates
    # dx = R * cos(lat) * dlon (in radians)
    # dy = R * dlat (in radians)
    dx_dlon = R_earth * torch.cos(latitude_rad) * (torch.pi / 180.0)  # meters per degree lon
    dy_dlat = R_earth * (torch.pi / 180.0)  # meters per degree lat
    
    # Navier-Stokes equations in pressure coordinates (neglecting friction)
    # du/dt + u*du/dx + v*du/dy + w*du/dp - f*v = -dΦ/dx
    navier_stokes_longitudinal = (
        grads["u_t"] * (scale_factors["u"]/scale_factors["t"]) 
        + u * grads["u_x"] * (scale_factors["u"]/scale_factors["x"]) / dx_dlon
        + v * grads["u_y"] * (scale_factors["u"]/scale_factors["y"]) / dy_dlat
        + w * grads["u_p"] * (scale_factors["u"]/scale_factors["p"])
        - f * v 
        + grads["z_x"] * (scale_factors["z"]/scale_factors["x"]) / dx_dlon
    )
    
    # dv/dt + u*dv/dx + v*dv/dy + w*dv/dp + f*u = -dΦ/dy
    navier_stokes_latitudinal = (
        grads["v_t"] * (scale_factors["v"]/scale_factors["t"]) 
        + u * grads["v_x"] * (scale_factors["v"]/scale_factors["x"]) / dx_dlon
        + v * grads["v_y"] * (scale_factors["v"]/scale_factors["y"]) / dy_dlat
        + w * grads["v_p"] * (scale_factors["v"]/scale_factors["p"]) 
        + f * u 
        + grads["z_y"] * (scale_factors["z"]/scale_factors["y"]) / dy_dlat
    )
    
    if mass_balance:
        # Mass continuity: du/dx + dv/dy + dw/dp = 0
        mass_continuity = (
            grads["u_x"] * (scale_factors["u"]/scale_factors["x"]) / dx_dlon
            + grads["v_y"] * (scale_factors["v"]/scale_factors["y"]) / dy_dlat
            + grads["w_p"] * (scale_factors["w"]/scale_factors["p"])
        )
        return navier_stokes_longitudinal, navier_stokes_latitudinal, mass_continuity
    
    return navier_stokes_longitudinal, navier_stokes_latitudinal, torch.ones_like(navier_stokes_longitudinal)