import torch
from torch.utils.data import Dataset
import dask.array as darray
import numpy as np
from utils import io
from data.processors import processors
from data.datamodules import sampler

class ERA5Troposphere3D(Dataset):
    def __init__(self,
                 filepath,
                 batch_size,
                 train_pc=0.8,
                 val_pc=0.1,
                 test_pc=0.1,
                 num_epochs=100):
        assert train_pc + val_pc + test_pc == 1, "train_pc + val_pc + test_pc must be 1"



        data = io.load_data(filepath)
        data = processors.wrap_longitude(data)
        dims = io.get_dimensions(data)

        # dimensions
        n_lat = dims["latitude"]
        n_lon = dims["longitude"]
        n_time = dims["valid_time"]
        n_pressure_level = dims["pressure_level"]
        n_samples = n_lat * n_lon * n_time * n_pressure_level

        _, lat_lon_norm_coords = processors.extract_lat_lon_coordinates(data)
        time_coords = processors.extract_time_coordinates(data, "valid_time")
        _,pressure_coords = processors.extract_other_coordinates(data, ["pressure_level"])
        _,variable_data = processors.extract_variables(data, ["t"]) # shape Lat x Lon x Time x PressureLevel


        space_through_pressure = darray.tile(lat_lon_norm_coords, (n_pressure_level,1))
        levels_through_space = darray.repeat(pressure_coords, lat_lon_norm_coords.shape[0], axis=0)

        spatial_coords = darray.hstack([space_through_pressure, levels_through_space])

        spatial_coords_through_time = darray.tile(spatial_coords, (n_time,1))

        times = darray.repeat(darray.arange(n_time), n_pressure_level * n_lat * n_lon)

        spatiotemporal_coords = darray.hstack([spatial_coords_through_time, times[:, None]])
        # shape n_samples x 4 -> idx[0] = lat, idx[1] = lon, idx[2] = pressure, idx[3] = time

        self.spatiotemporal_coords = spatiotemporal_coords
        self.variable_data = variable_data
        self.n_samples = n_samples
        self.spatial_points = n_lat * n_lon * n_pressure_level
        self.N_sensors = n_lat * n_lon
        self.batch_size = batch_size
        self.sampler = sampler.sampler()
        self.train_pc = int(train_pc * n_samples)





    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):

        print(type(index))
        if not isinstance(index, np.ndarray):
            index = np.array(index)

        spatiotemporal_coords = darray.take(self.spatiotemporal_coords, index, axis=0)
        variable_data = darray.take(self.variable_data,index, axis=0)

        lat_lon_batches = spatiotemporal_coords[:,0:2]
        pressure_batches = spatiotemporal_coords[:,2]
        time_batches = spatiotemporal_coords[:,3]

        #print(spatiotemporal_batches.shape)

        return lat_lon_batches.compute(), pressure_batches.compute(), time_batches.compute(), variable_data.compute()










