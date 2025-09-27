import numpy as np


def omega_to_w(omega, pressure_level, temperature):

    gas_constant = 287.058 # J/(kg*K)
    gravitational_acceleration = 9.80665 # m/s^2

    pressure_level = pressure_level * 100 # Convert hPa to Pa

    rho = pressure_level / (gas_constant * temperature)

    w = -omega / (rho * gravitational_acceleration)

    return w

