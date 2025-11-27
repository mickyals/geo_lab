from geolab.utils.meteorology import coriolis_force, compute_troposphere_gradients


def troposphere_pde_residual(inputs_tensor, outputs, statistics, mass_balance=True):
    """
    Compute the residuals of the troposphere PDEs.

    Args:
        inputs_tensor (torch.Tensor): Stacked input tensor with shape [batch, 4] where columns are:
                                      [longitude, latitude, pressure_level, time]
        outputs (dict): Dictionary containing the model outputs: u, v, w, and z.
        mass_balance (bool, optional): Whether to include the mass balance term in the residuals. Defaults to True.

    Returns:
        tuple: Tuple of the residuals of the longitudinal, latitudinal, and mass balance terms.
    """
    u, v, w, z = outputs["u"], outputs["v"], outputs["w"], outputs["z"]

    # Get scaling factors for inputs following https://arxiv.org/abs/2403.19923v2 for affine transformations
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
    scale_factors = {
        'u': u_scale, 'v': v_scale, 'w': w_scale, 'z': z_scale,
        'x': lon_scale, 'y': lat_scale, 'p': p_scale, 't': t_scale
    }

    latitude = inputs_tensor[:, 1]
    grads = compute_troposphere_gradients(inputs_tensor, outputs)


    coriolis_params = coriolis_force(latitude)

    navier_stokes_longitudinal = grads["u_t"] * (scale_factors["u"]/scale_factors["t"]) \
                                 + u * grads["u_x"] * (scale_factors["u"]/scale_factors["x"]) \
                                 + v * grads["u_y"] * (scale_factors["u"]/scale_factors["y"]) \
                                 + w * grads["u_p"] * (scale_factors["u"]/scale_factors["p"])\
                                 - (coriolis_params["beta"] * latitude) * v \
                                 + grads["z_x"] * (scale_factors["z"]/scale_factors["x"])

    navier_stokes_latitudinal = grads["v_t"] * (scale_factors["v"]/scale_factors["t"]) \
                                + u * grads["v_x"] * (scale_factors["v"]/scale_factors["x"]) \
                                + v * grads["v_y"] * (scale_factors["v"]/scale_factors["y"]) \
                                + w * grads["v_p"] * (scale_factors["v"]/scale_factors["p"]) \
                                + (coriolis_params["beta"] * latitude) * u \
                                + grads["z_y"] * (scale_factors["z"]/scale_factors["y"])

    # the f_0 term is not added for a global case BUT it is added for a regional case
    # (coriolis_params["f_0"] + coriolis_params["beta"] * latitude)

    if mass_balance:
        mass_continuity = grads["u_x"] * (scale_factors["u"]/scale_factors["x"]) \
                          + grads["v_y"] * (scale_factors["v"]/scale_factors["y"])\
                          + grads["w_p"] * (scale_factors["w"]/scale_factors["p"]) #mass continuity equation of the atmosphere

        return navier_stokes_longitudinal, navier_stokes_latitudinal, mass_continuity

    return navier_stokes_longitudinal, navier_stokes_latitudinal