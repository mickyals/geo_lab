from geolab.utils.meteorology import coriolis_force, compute_troposphere_gradients
from typing import Dict, Tuple
import torch


def troposphere_pde_residual( inputs_tensor: torch.Tensor,
                              outputs: Dict[str, torch.Tensor],
                              statistics: Dict,
                              coord_labels: Dict[str, int],
                              var_labels: Dict[str, int], # Optional, for validation
                              mass_balance: bool = True
                            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the residuals of the troposphere PDEs.
    """
    # Extract coordinates by name, not by hardcoded index
    lon_idx = coord_labels['longitude']
    lat_idx = coord_labels['latitude']
    p_idx = coord_labels['pressure_level']
    t_idx = coord_labels['valid_time']

    # Extract outputs
    u, v, w, z = outputs["u"], outputs["v"], outputs["w"], outputs["z"]

    # Physical constants
    omega = 7.2921e-5
    R_earth = 6371222.9

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

    scale_factors = {
        'u': u_scale, 'v': v_scale, 'w': w_scale, 'z': z_scale,
        'x': lon_scale, 'y': lat_scale, 'p': p_scale, 't': t_scale
    }

    # FIX: Use label-based indexing instead of hardcoded [1]
    latitude = inputs_tensor[:, lat_idx]  # <-- CHANGE THIS LINE

    # Pass coord_labels to gradient computation
    grads = compute_troposphere_gradients(inputs_tensor, outputs, coord_labels)

    # Get PHYSICAL (not normalized) Coriolis parameter
    coriolis_params = coriolis_force(latitude)
    f = coriolis_params["f"] * 2 * omega

    # Convert longitude/latitude gradients to physical spatial gradients
    latitude_rad = torch.abs(latitude) * torch.pi / 180.0

    dx_dlon = R_earth * torch.cos(latitude_rad) * (torch.pi / 180.0)
    dy_dlat = R_earth * (torch.pi / 180.0)

    # Navier-Stokes equations (rest unchanged)
    navier_stokes_longitudinal = (
        grads["u_t"] * (scale_factors["u"]/scale_factors["t"])
        + u * grads["u_x"] * (scale_factors["u"]/scale_factors["x"]) / dx_dlon
        + v * grads["u_y"] * (scale_factors["u"]/scale_factors["y"]) / dy_dlat
        + w * grads["u_p"] * (scale_factors["u"]/scale_factors["p"])
        - f * v
        + grads["z_x"] * (scale_factors["z"]/scale_factors["x"]) / dx_dlon
    )

    navier_stokes_latitudinal = (
        grads["v_t"] * (scale_factors["v"]/scale_factors["t"])
        + u * grads["v_x"] * (scale_factors["v"]/scale_factors["x"]) / dx_dlon
        + v * grads["v_y"] * (scale_factors["v"]/scale_factors["y"]) / dy_dlat
        + w * grads["v_p"] * (scale_factors["v"]/scale_factors["p"])
        + f * u
        + grads["z_y"] * (scale_factors["z"]/scale_factors["y"]) / dy_dlat
    )

    if mass_balance:
        mass_continuity = (
            grads["u_x"] * (scale_factors["u"]/scale_factors["x"]) / dx_dlon
            + grads["v_y"] * (scale_factors["v"]/scale_factors["y"]) / dy_dlat
            + grads["w_p"] * (scale_factors["w"]/scale_factors["p"])
        )
        return navier_stokes_longitudinal, navier_stokes_latitudinal, mass_continuity

    return navier_stokes_longitudinal, navier_stokes_latitudinal, torch.ones_like(navier_stokes_longitudinal)