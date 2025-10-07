"""
class for era5 data object
    - dimensions
        - dimension_names
        - dimension_shape
    - coordinates
        - coordinate_names
        - coordinate_points (name:str, value:flaot, dtype:str
        - num_points (variable_name:str, value:int)
    - data_variables
        - variable_names
        - variables (variable_name:str, value:array, dtype:str)
        - num_points (variable_name:str, value:int)
    - attributes

class for building the dataset
"""
import numpy as np
from pyDOE import lhs
from torch.utils.data import Dataset
from geolab.utils.meteorology import omega_to_w, compute_troposphere_gradients, coriolis_force

class ERA5MultiData:
    def __init__(self, data_dir, read_data_fn, variables):

        with read_data_fn(data_dir) as ds:
            # get dimensions
            dimensions = dict(ds.dims)

            # get the coordinates
            coordinates = {ds.coords[i]: (ds.coords.values, len(ds.coords.values), str(ds.coords.values.dtype))
                           for i in ds.coords}

            # get the variables
            data_variables = {ds.data_vars[var].name: (ds.data_vars[var].data, ds.data_vars[var].size, str(ds.coords.values.dtype), ds.data_vars[var].shape ) for var in variables}

            # get the metadata
            attributes = ds.attrs

        self.dimensions = dimensions
        self.coordinates = coordinates
        self.variables = data_variables
        self.attributes = attributes




    def subset_data(self, data_items, coords_items,
                       time_idx_range, pressure_idx_range, latitude_idx_range, longitude_idx_range):

        t = self._make_slice(time_idx_range)
        p = self._make_slice(pressure_idx_range)
        la = self._make_slice(latitude_idx_range)
        lo = self._make_slice(longitude_idx_range)

        data = {}
        for k, v in data_items.items():
            sliced = v[t, p, la, lo]
            data[k] = (sliced, sliced.size, sliced.shape)

        coordinates = {k: [v[i], v[i].shape, str(v[i].dtype), v[i].size]for i, (k, v) in enumerate(coords_items.items())}

        return coordinates, data


    def generate_virtual_samples(self, num_samples, coord_stats_dict):

        coord_names = list(coord_stats_dict.keys())
        lower_bounds = np.array([coord_stats_dict[name][0] for name in coord_names])
        upper_bounds = np.array([coord_stats_dict[name][1] for name in coord_names])

        if lhs:
            points = lhs(len(coord_names), samples=num_samples)

        else:
           points = np.random.uniform(lower_bounds, upper_bounds, (num_samples, len(coord_names)))

        virtual_points = {name: points[:, i] for i, name in enumerate(coord_names)}

        return virtual_points

    @staticmethod
    def _make_slice(idx_range):
        if not idx_range:
            return slice(None)
        return slice(idx_range[0], idx_range[1])

    def _get_stats(self, item):
        return {k: [v.min(), v.max(), v.mean(), v.std(), ] for k, v in item.items()}


    def _get_size(self, item):
        return {k: v.size for k, v in item.items()}

    def normalise(self, data_dict, data_stats_dict, method='minmax'):

        if method == 'minmax':
            return {k: (v - data_stats_dict[k][0]) / (data_stats_dict[k][1] - data_stats_dict[k][0]) for k, v in data_dict.items()}
        elif method == 'standard':
            return {k: (v - data_stats_dict[k][2]) / data_stats_dict[k][3] for k, v in data_dict.items()}
        elif method == 'uniform':
            return {k: 2 * (v - data_stats_dict[k][0]) / (data_stats_dict[k][1] - data_stats_dict[k][0]) - 1 for k, v in data_dict.items()}
        else:
            raise NotImplementedError (f"Method {method} not implemented")

    def denormalise(self, data_dict, data_stats_dict, method='minmax'):
        if method == 'minmax':
            return {k: data_stats_dict[k][0] + (v * (data_stats_dict[k][1] - data_stats_dict[k][0])) for k, v in data_dict.items()}
        elif method == 'standard':
            return {k: data_stats_dict[k][2] + (v * data_stats_dict[k][3]) for k, v in data_dict.items()}
        elif method == 'uniform':
            return {k: data_stats_dict[k][0] + ((v + 1) / 2) * (data_stats_dict[k][1] - data_stats_dict[k][0]) for k, v in data_dict.items()}
        else:
            raise NotImplementedError (f"Method {method} not implemented")

    def generate_mesh(self, coords_dict, indexing='ij'):

        coords_array = [coords_dict[coord].values for coord in coords_dict]
        mesh = np.meshgrid(*coords_array, indexing=indexing)
        coords = {}
        for i, coord in enumerate(coords_dict):
            arr = mesh[i]
            coords[coord] = [arr.ravel(), arr.shape, str(arr.dtype), arr.size]

        return coords




class ERA5MultiDataset(Dataset):
    def __init__(self, indices, coordinate_names, coordinates,
                 variable_names, variables, stats, physics_loss = False):
        self.indices = indices
        self.coordinate_names = coordinate_names
        self.coordinates = coordinates
        self.variable_names = variable_names
        self.variables = variables
        self.stats = stats

        """
        the data, indices etc are provided as preprocessed amounts with
        stats so that you can denormalise the data should it be required.
        
        adding the physics loss functions and the mse loss for every entry                                                                                                                                     
        """

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pass

    def add_physics_loss(self):
