from typing import Callable

import numpy as np
import xarray as xr
from geolab.data.components.coordinate_data.mesh import ERA5MultiData
from torch.utils.data import Dataset, DataLoader


class TroposphereDataset(Dataset):
    def __init__(self,
                 root_dir,
                 read_data_fn,
                 solution_vars,
                 indices,
                 prc_collocation_points,
                 use_lhs,
                 dynamic,
                 ):
        era5_data = ERA5MultiData(root_dir, read_data_fn, solution_vars)
        training_points = era5_data.get_collocation_points(prc_collocation_points, use_lhs, dynamic=dynamic)
        real_size = len(training_points['real'][1][solution_vars[0]])
        virtual_size = len(training_points['virtual']['valid_time']) if dynamic else 0

        # Use provided indices or default to all
        if indices is None:
            data_length = real_size + virtual_size
            indices = np.arange(data_length)

        self.solution_vars = solution_vars
        self.lower_bounds = era5_data.get_lower_bounds()
        self.upper_bounds = era5_data.get_upper_bounds()
        self.dataset_coords = era5_data.dataset_coords

        # Store only selected data
        self.input_data = {k: v[indices] for k, v in input_data.items()}
        self.output_data = {k: v[indices] for k, v in output_data.items()}
        self.idx = indices



    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        longitude = self.input_data['longitude'][idx]
        latitude = self.input_data['latitude'][idx]
        pressure_level = self.input_data['pressure_level'][idx]
        valid_time = self.input_data['valid_time'][idx]

        u = self.output_data['u'][idx] # east west wind - moving across degrees of longitude
        v  = self.output_data['v'][idx] # north south wind - moving across degrees of latitude
        w = self.output_data['w'][idx]  # this is the omega - vertical velocity NOT w velocity of wind
        z = self.output_data['z'][idx] # geopotential z = g * h

        x = {
            "longitude": self._wrap_longitude(longitude,
                                              self.lower_bounds["coords"]["longitude"],
                                              self.upper_bounds["coords"]["longitude"]),

            "latitude": self._normalise(self.lower_bounds["coords"]["latitude"],
                                        self.upper_bounds["coords"]["latitude"],
                                        latitude),
            "pressure_level": self._normalise(self.lower_bounds["coords"]["pressure_level"],
                                              self.upper_bounds["coords"]["pressure_level"],
                                              pressure_level),
            "time": self.time_delta_normalised(self.lower_bounds["coords"]["valid_time"],
                                               self.upper_bounds["coords"]["valid_time"],
                                               valid_time),
        }

        # normalised outputs
        y = {
            "u": self._normalise(self.lower_bounds["solution"]["u"], self.upper_bounds["solution"]["u"], u),
            "v": self._normalise(self.lower_bounds["solution"]["v"], self.upper_bounds["solution"]["v"], v),
            "w": self._normalise(self.lower_bounds["solution"]["w"], self.upper_bounds["solution"]["w"], w),
            "z": self._normalise(self.lower_bounds["solution"]["z"], self.upper_bounds["solution"]["z"], z),
        }

        return x, y






    def _normalise(self, data_min, data_max, data):
        normalised = 2 * ((data - data_min)/(data_max - data_min)) - 1
        return normalised

    def time_delta_normalised(self, t_min, t_max, time):
        t0, t1 = t_min, t_max
        dt = (time - t0).astype('timedelta64[ns]').astype(np.float64)
        tot = (t1 - t0).astype('timedelta64[ns]').astype(np.float64)
        return dt/tot

    def _wrap_longitude(self, longitude, lon_min, lon_max):
        longitude = (((longitude + 180) % 360) - 180)
        longitude_norm = 2 * ((longitude - lon_min) / (lon_max - lon_min)) - 1
        return longitude_norm