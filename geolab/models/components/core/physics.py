from geolab.utils.meteorology import coriolis_force, compute_troposphere_gradients


def troposphere_pde_residual(inputs, outputs, mass_balance = True):

    u, v, w, z = outputs["u"], outputs["v"], outputs["w"], outputs["z"]

    grads = compute_troposphere_gradients(inputs, outputs)
    coriolis_params = coriolis_force(inputs["latitude"])

    navier_stokes_longitudinal = grads["u_t"] \
                                 + u * grads["u_x"] \
                                 + v * grads["u_y"] \
                                 + w * grads["u_p"] \
                                 - (coriolis_params["f_0"] + coriolis_params["beta"] * inputs["latitude"]) * v \
                                 + grads["z_x"]

    navier_stokes_latitudinal = grads["v_t"] \
                                + u * grads["v_x"] \
                                + v * grads["v_y"] \
                                + w * grads["v_p"] \
                                + (coriolis_params["f_0"] + coriolis_params["beta"] * inputs["latitude"]) * u \
                                + grads["z_y"]

    if mass_balance:
        mass_continuity = grads["u_x"] + w * grads["v_y"] + grads["z_p"]

        return navier_stokes_longitudinal, navier_stokes_latitudinal, mass_continuity

    return navier_stokes_longitudinal, navier_stokes_latitudinal