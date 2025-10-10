from geolab.utils.meteorology import coriolis_force, compute_troposphere_gradients


def troposphere_pde_residual(inputs_tensor, outputs, mass_balance=True):
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

    # Extract coordinates from inputs_tensor for Coriolis calculation
    latitude = inputs_tensor[:, 1]  # latitude is column 1

    grads = compute_troposphere_gradients(inputs_tensor, outputs)
    coriolis_params = coriolis_force(latitude)

    navier_stokes_longitudinal = grads["u_t"] \
                                 + u * grads["u_x"] \
                                 + v * grads["u_y"] \
                                 + w * grads["u_p"] \
                                 - (coriolis_params["beta"] * latitude) * v \
                                 + grads["z_x"]

    navier_stokes_latitudinal = grads["v_t"] \
                                + u * grads["v_x"] \
                                + v * grads["v_y"] \
                                + w * grads["v_p"] \
                                + (coriolis_params["beta"] * latitude) * u \
                                + grads["z_y"]

    # the f_0 term is not added for a global case BUT it is added for a regional case
    # (coriolis_params["f_0"] + coriolis_params["beta"] * latitude)

    if mass_balance:
        mass_continuity = grads["u_x"] + grads["v_y"] + grads["w_p"]  # mass continuity equation of the atmosphere

        return navier_stokes_longitudinal, navier_stokes_latitudinal, mass_continuity

    return navier_stokes_longitudinal, navier_stokes_latitudinal