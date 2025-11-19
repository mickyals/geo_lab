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
import torch
from torch.utils.data import Dataset

class ERA5MultiData:
    def __init__(self, data_dir, read_data_fn, variables):

        with read_data_fn(data_dir) as ds:
            # get dimensions
            dimensions = dict(ds.sizes)

            # List of coordinates to keep
            coords_to_keep = ['valid_time', 'pressure_level', 'latitude', 'longitude']

            # Filter coordinates
            coordinates = {
                i: (ds.coords[i].data.astype(np.float32),
                    ds.coords[i].data.size,
                    str(ds.coords[i].values.dtype),
                    ds.coords[i].shape)
                for i in ds.coords if i in coords_to_keep
            }

            # get the variables
            data_variables = {ds.data_vars[var].name: (ds.data_vars[var].astype(np.float32), ds.data_vars[var].size, str(ds.data_vars[var].dtype), ds.data_vars[var].shape ) for var in variables}

            # get the metadata
            attributes = ds.attrs

        self.dimensions = dimensions
        self.coordinates = coordinates
        self.variables = data_variables
        self.attributes = attributes




    def subset_data(self,time_idx_range, pressure_idx_range, latitude_idx_range, longitude_idx_range):

        t = self._make_slice(time_idx_range)
        p = self._make_slice(pressure_idx_range)
        la = self._make_slice(latitude_idx_range)
        lo = self._make_slice(longitude_idx_range)

        data = {}
        for k, v in self.variables.items():
            data_array = v[0].values  # Get the actual numpy array
            sliced = data_array[t, p, la, lo]
            data[k] = (sliced, sliced.size, sliced.shape, str(sliced.dtype))

        coordinates = {}
        for k, v in self.coordinates.items():
            data_array = v[0]
            # Convert to lowercase and split for case-insensitive matching
            parts = k.lower().split('_')
            # Apply the appropriate slice based on the coordinate name parts
            if 'time' in parts:
                sliced = data_array[t]
            elif 'level' in parts or 'pressure' in parts:
                sliced = data_array[p]
            elif 'lat' in parts or 'latitude' in parts:
                sliced = data_array[la]
            elif 'lon' in parts or 'longitude' in parts:
                sliced = data_array[lo]
            else:
                sliced = data_array  # Default to full data if we can't determine the dimension

            coordinates[k] = (sliced, sliced.size, sliced.shape, str(sliced.dtype))

        return coordinates, data


    def generate_virtual_samples(self, num_samples, coord_stats_dict, use_lhs=True):

        coord_names = list(coord_stats_dict.keys())
        lower_bounds = np.array([coord_stats_dict[name][0] for name in coord_names])
        upper_bounds = np.array([coord_stats_dict[name][1] for name in coord_names])

        if use_lhs:
            points = lhs(len(coord_names), samples=num_samples)
            points = lower_bounds + (upper_bounds - lower_bounds) * points

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
        return {k: [v[0].min(), v[0].max(), v[0].mean(), v[0].std() ] for k, v in item.items()}


    def _get_size(self, item):
        return {k: v[0].size for k, v in item.items()}


    def generate_mesh(self, coords_dict, indexing='ij'):
        """Generate a mesh grid from coordinate arrays.
        
        Args:
            coords_dict: Dictionary of coordinate arrays in the format (data, size, shape, dtype)
            indexing: 'ij' for matrix indexing, 'xy' for Cartesian indexing
            
        Returns:
            Dictionary with the same structure as input, containing mesh grid coordinates
        """
        # Extract just the coordinate data (first element of each tuple)
        coord_names = list(coords_dict.keys())
        coord_arrays = [coords_dict[name][0] for name in coord_names]
        
        # Create mesh grid
        mesh_arrays = np.meshgrid(*coord_arrays, indexing=indexing)
        
        # Create output dictionary with the same structure as input
        result = {}
        for i, name in enumerate(coord_names):
            arr = mesh_arrays[i]
            result[name] = (
                arr.ravel(),          # data
                arr.size,     # size
                arr.shape,    # shape
                str(arr.dtype)  # dtype
            )
            
        return result

    def run(self, time_idx_range, pressure_idx_range, latitude_idx_range, longitude_idx_range, indexing,
            num_samples=10000, include_virtual=False, use_lhs=True, **kwargs):
        """Run the data processing pipeline.
        
        Args:
            time_idx_range: Range of time indices to include
            pressure_idx_range: Range of pressure level indices to include
            latitude_idx_range: Range of latitude indices to include
            longitude_idx_range: Range of longitude indices to include
            num_samples: Number of virtual samples to generate if include_virtual is True
            include_virtual: Whether to include virtual samples
            use_lhs: Whether to use Latin Hypercube Sampling for virtual points
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing processed data and statistics
        """
        subset_coords, subset_data = self.subset_data(time_idx_range, pressure_idx_range, 
                                                   latitude_idx_range, longitude_idx_range)
        # Get statistics for coordinates and data
        subset_coords_stats = self._get_stats(subset_coords)
        subset_data_stats = self._get_stats(subset_data)
        
        # Combine all statistics into a single flat dictionary
        statistics = {**subset_coords_stats, **subset_data_stats}
        
        first_var = next(iter(subset_data))
        real_count = subset_data[first_var][1]
        real_coordinates = self.generate_mesh(subset_coords, indexing=indexing)
        real_classifier = np.ones(real_count, dtype=bool)

        if include_virtual:
            virtual_coordinates = self.generate_virtual_samples(
                num_samples=num_samples,
                coord_stats_dict=subset_coords_stats,
                use_lhs=use_lhs
            )
            virtual_count = virtual_coordinates[list(virtual_coordinates.keys())[0]].size
            virtual_data = {var: np.zeros(virtual_count)
                         for var in self.variables}
            virtual_classifier = np.zeros(virtual_count, dtype=bool)

            combined_coords = {
                    k: np.concatenate([real_coordinates[k][0], virtual_coordinates[k]])  # [0] to get data from (data, size, shape, dtype)
                    for k in real_coordinates
                }

            combined_data = {
                    var: np.concatenate([
                        subset_data[var][0].flatten(),  # First element of tuple is the data array
                        virtual_data[var]   # Zeros for virtual data
                    ])
                    for var in self.variables
                }

            combined_classifier = np.concatenate([real_classifier, virtual_classifier])


            # Combine real and virtual data
            combined_data = {
                'data': {**combined_coords, **combined_data, 'classification': combined_classifier},
                'count': [real_count + virtual_count, real_count, virtual_count]
            }
            
            return combined_data, statistics

        # Return real data only (first element of the subset_data tuple is the data array)
        coordinates = {k: v[0] for k, v in real_coordinates.items()}
        variable_data = {var: subset_data[var][0].flatten() for var in self.variables}
        return {
            'data': {**coordinates, **variable_data, 'classification': real_classifier,},
            'count': [real_count]
        }, statistics



class ERA5MultiDataset(Dataset):
    def __init__(self,
                 data,
                 statistics,
                 indices,
                 include_virtual,
                 variables,
                 pi_scale):
        self.data = data[indices]
        self.statistics = statistics
        self.idx = indices
        self.include_virtual = include_virtual
        self.variables = variables
        self.pi_scale = pi_scale

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        # --------------------
        # Extract raw coordinates
        # --------------------
        lon_raw = self.data['longitude'][idx]
        lat_raw = self.data['latitude'][idx]
        pressure = self.data['pressure_level'][idx]
        time_raw = self.data['valid_time'][idx]


        # --------------------
        # Normalize coordinates
        # --------------------
        # Time to [-1, 1]
        t_min, t_max = self.statistics['valid_time'][0], self.statistics['valid_time'][1]
        time_norm = 2 * self.time_delta_normalised(t_min, t_max, time_raw) - 1  # [-1, 1]

        # Latitude to [-1, 1] then scale [-pi/2, pi/2]
        lat_min, lat_max = self.statistics['latitude'][0], self.statistics['latitude'][1]
        lat_norm = self.normalise(lat_raw, lat_min, lat_max)


        # Longitude to [-1, 1] then scale [-pi, pi]
        lon_min, lon_max = self.statistics['longitude'][0], self.statistics['longitude'][1]
        lon_norm = self.normalise(lon_raw, lon_min, lon_max)
        if self.pi_scale:
            lon_norm = lon_norm * torch.pi
            lat_norm = lat_norm * (torch.pi / 2)

        coords = {
            "longitude": torch.tensor(lon_norm, dtype=torch.float32),
            "latitude": torch.tensor(lat_norm, dtype=torch.float32),
            "pressure_level": torch.tensor(pressure, dtype=torch.float32),
            "time": torch.tensor(time_norm, dtype=torch.float32)
        }

        # --------------------
        # Normalize variable data
        # --------------------
        vars_data = {}

        for var in self.variables:
            var_min, var_max = self.statistics[var][0], self.statistics[var][1]
            value = self.data[var][idx]
            value_norm = self.normalise(value, var_min, var_max)
            vars_data[var] = torch.tensor(value_norm, dtype=torch.float32)

        # --------------------
        # Classification
        # --------------------
        classification = torch.tensor(self.data['classification'][idx])

        return {
            "coords": coords,
            "variables": vars_data,
            "classification": classification
        }

    def normalise(self, x, x_min, x_max):

        if x_min == x_max:
            return 1.0
        return 2.0 * (x - x_min) / (x_max - x_min) - 1.0

    def time_delta_normalised(self, t_min, t_max, time):
        t0, t1 = t_min, t_max
        if t0 == t1:
            return 1.0
        dt = (time - t0)
        tot = (t1 - t0)
        t = dt / tot
        return t
