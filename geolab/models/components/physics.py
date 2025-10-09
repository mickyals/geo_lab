from geolab.utils.meteorology import coriolis_force, compute_troposphere_gradients


def troposphere_pde_residual(inputs, outputs, mass_balance = True):
    """
    Compute the residuals of the troposphere PDEs.

    Args:
        inputs (dict): Dictionary containing the input coordinates: longitude, latitude, pressure, and time.
        outputs (dict): Dictionary containing the model outputs: u, v, w, and z.
        mass_balance (bool, optional): Whether to include the mass balance term in the residuals. Defaults to True.

    Returns:
        tuple: Tuple of the residuals of the longitudinal, latitudinal, and mass balance terms.
    """
    u, v, w, z = outputs["u"], outputs["v"], outputs["w"], outputs["z"]

    grads = compute_troposphere_gradients(inputs, outputs)
    coriolis_params = coriolis_force(inputs["latitude"])

    navier_stokes_longitudinal = grads["u_t"] \
                                 + u * grads["u_x"] \
                                 + v * grads["u_y"] \
                                 + w * grads["u_p"] \
                                 - (coriolis_params["beta"] * inputs["latitude"]) * v \
                                 + grads["z_x"]

    navier_stokes_latitudinal = grads["v_t"] \
                                + u * grads["v_x"] \
                                + v * grads["v_y"] \
                                + w * grads["v_p"] \
                                + (coriolis_params["beta"] * inputs["latitude"]) * u \
                                + grads["z_y"]

    # the f_0 term is not added for a global case BUT it is added for a regional case
    # (coriolis_params["f_0"] + coriolis_params["beta"] * inputs['latitude'])

    if mass_balance:
        mass_continuity = grads["u_x"] + w * grads["v_y"] + grads["z_p"] # mass continuity equation of the atmosphere

        return navier_stokes_longitudinal, navier_stokes_latitudinal, mass_continuity

    return navier_stokes_longitudinal, navier_stokes_latitudinal