from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


class ERA5MultiData:

    def __init__(self,
                 data_dir: Union[str, Path],
                 variables: List[str],
                 time_idx_range: Optional[Tuple[int, int]] = None,
                 pressure_idx_range: Optional[Tuple[int, int]] = None,
                 latitude_idx_range: Optional[Tuple[int, int]] = None,
                 longitude_idx_range: Optional[Tuple[int, int]] = None,
                 preload: bool = True,
                 ):

        self.data_dir = Path(data_dir)
        self.variables = variables
        self.time_idx_range = time_idx_range
        self.pressure_idx_range = pressure_idx_range
        self.latitude_idx_range = latitude_idx_range
        self.longitude_idx_range = longitude_idx_range
        self.preload = preload

        # Will be populated in load()
        self.ds = None  # xarray Dataset
        self.data_arrays = {}  # Dict[var_name, np.ndarray]
        self.coordinates = {}  # Dict[coord_name, np.ndarray]
        self.coord_labels = {}  # Dict[coord_name, int]
        self.variable_labels = {}  # Dict[var_name, int]
        self.coord_sizes = None  # np.ndarray of coordinate dimensions

        # Load data
        self.load()

    def load(self):
        print(f"Loading ERA5 data from {self.data_dir}")

        # Open dataset
        if self.data_dir.is_file():
            ds = xr.open_dataset(self.data_dir)
        else:
            # Load multiple files if directory
            ds = xr.open_mfdataset(
                str(self.data_dir / "*.nc"),
                combine='by_coords',
                parallel=True,
            )

        # Select only requested variables
        ds = ds[self.variables]

        # Apply slicing
        ds = self._apply_slicing(ds)

        # Build coordinate labels (mapping_name -> index)
        for idx, coord in enumerate(ds.coords):
            self.coord_labels[coord] = idx

        # Build variable labels (mapping_name -> index)
        for idx, var in enumerate(ds.data_vars):
            self.variable_labels[var] = idx

        # Store coordinate sizes
        self.coord_sizes = np.array([ds.coords.dims[coord] for coord in ds.coords])

        self._build_coordinate_arrays(ds)

        if self.preload:
            print("Preloading data into memory...")
            self._build_target_arrays(ds)
            ds.close()
        else:
            self.ds = ds

        print(f"Successfully loaded {len(self.variables)}:")
        for var in self.variables:
            if self.preload:
                shape = self.data_arrays[var].shape
            else:
                shape = self.ds[var].shape
            print(f"   {var}: {shape}")

    @staticmethod
    def _make_slice(idx_range: Optional[Union[Tuple[int, int], List[int]]]) -> slice:

        if not idx_range:  # Handles None, empty list, empty tuple
            return slice(None)
        return slice(idx_range[0], idx_range[1])

    def _apply_slicing(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply index-based slicing to dataset."""
        slice_dict = {}

        if self.time_idx_range is not None:
            slice_dict['valid_time'] = self._make_slice(self.time_idx_range)

        if self.pressure_idx_range is not None:
            slice_dict['pressure_level'] = self._make_slice(self.pressure_idx_range)

        if self.latitude_idx_range is not None:
            slice_dict['latitude'] = self._make_slice(self.latitude_idx_range)

        if self.longitude_idx_range is not None:
            slice_dict['longitude'] = self._make_slice(self.longitude_idx_range)

        if slice_dict:
            print(f"Applying slicing: {slice_dict}")
            ds = ds.isel(slice_dict)
        else:
            print("No slicing applied - using all indices")

        return ds


    def _build_coordinate_arrays(self, ds):
        """ Extract coordinates as numpy arrays"""

        # Time data
        time_data = ds['valid_time'].values  # numpy array of datetime64
        self.coordinates['valid_time'] = time_data.astype('datetime64[s]').astype(np.float32)


        # Pressure Level data
        pressure_data = ds['pressure_level'].values
        self.coordinates['pressure_level'] = pressure_data.astype(np.float32)

        # Latitude
        latitude_data = ds['latitude'].values
        self.coordinates['latitude'] = latitude_data.astype(np.float32)

        # Longitude
        longitude_data = ds['longitude'].values
        longitude_data = ((longitude_data + 180) % 360) - 180
        self.coordinates['longitude'] = longitude_data.astype(np.float32)

        for name, data in self.coordinates.items():
            print(f"  {name}: shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")

    def _build_target_arrays(self, ds):
        """Load variable data into numpy arrays."""
        for var in self.variables:
            print(f"  Loading {var}...")
            data = ds[var].values.astype(np.float32)
            self.data_arrays[var] = data
            print(f"    Memory: {data.nbytes / 1e9:.2f} GB")

    def get_variable(self, var_name: str, indices: Optional[List[int]] = None):
        """Get variable data as numpy array."""
        if var_name not in self.data_arrays:
            raise ValueError(f"Variable {var_name} not found in dataset.")
        return self.data_arrays[var_name][indices]

    def get_coords_at_index(self,
                            time_idx: Union[int, np.ndarray],
                            pressure_idx: Union[int, np.ndarray],
                            latitude_idx: Union[int, np.ndarray],
                            longitude_idx: Union[int, np.ndarray]):
        """Get coordinate values at specific indices."""
        return np.stack([self.coordinates['valid_time'][time_idx],
                         self.coordinates['pressure_level'][pressure_idx],
                         self.coordinates['latitude'][latitude_idx],
                         self.coordinates['longitude'][longitude_idx]], axis=-1)

    def get_values_at_index(self,
                            var_name: str,
                            time_idx: Union[int, np.ndarray],
                            pressure_idx: Union[int, np.ndarray],
                            latitude_idx: Union[int, np.ndarray],
                            longitude_idx: Union[int, np.ndarray]):
        """ get variable values qat given indices"""
        data = self.data_arrays[var_name] if self.preload else self.ds[var_name].values
        return data[time_idx, pressure_idx, latitude_idx, longitude_idx]

    @property
    def num_times(self) -> int:
        return len(self.coordinates['valid_time'])

    @property
    def num_pressure_levels(self) -> int:
        return len(self.coordinates['pressure_level'])

    @property
    def num_latitudes(self) -> int:
        return len(self.coordinates['latitude'])

    @property
    def num_longitudes(self) -> int:
        return len(self.coordinates['longitude'])

    @property
    def latitudes(self) -> np.ndarray:
        return self.coordinates['latitude']

    @property
    def longitudes(self) -> np.ndarray:
        return self.coordinates['longitude']

    @property
    def pressure_levels(self) -> np.ndarray:
        return self.coordinates['pressure_level']

    @property
    def times(self) -> np.ndarray:
        return self.coordinates['valid_time']

    @property
    def input_dim(self) -> int:
        return len(self.coord_labels)

    @property
    def output_dim(self):
        return len(self.variable_labels)


class ERA5MultiDataset(Dataset):
    """PyTorch Dataset for ERA5 data.

    Returns raw coordinate-value pairs. Normalization is handled by the datamodule.
    """

    def __init__(
            self,
            data: ERA5MultiData,
            indices: np.ndarray,
            statistics: Dict,
            variables: List[str],
    ):
        """Initialize dataset.

        Args:
            data: ERA5MultiData instance
            indices: Array of shape (N, 4) with [time_idx, pressure_idx, lat_idx, lon_idx]
            statistics: Dict of statistics (not used here, kept for compatibility)
            variables: List of variable names
        """
        self.data = data
        self.indices = indices
        self.variables = variables

    def __len__(self) -> int:
        """Number of samples in dataset."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Index into self.indices

        Returns:
            Dict with keys:
                - 'coords': Tensor of shape (4,) with [time, pressure, lat, lon] (raw values)
                - 'values': Tensor of shape (num_vars,) with atmospheric variable values (raw)
        """
        # Get 4D index
        time_idx, pressure_idx, lat_idx, lon_idx = self.indices[idx]

        # Get raw coordinate values
        coords = self.data.get_coords_at_index(
            time_idx, pressure_idx, lat_idx, lon_idx
        )  # Shape: (4,)

        # Get raw variable values
        values = []
        for var in self.variables:
            val = self.data.get_values_at_index(
                var, time_idx, pressure_idx, lat_idx, lon_idx
            )
            values.append(val)

        values = np.array(values, dtype=np.float32)  # Shape: (num_vars,)

        # Convert to tensors
        coords_tensor = torch.from_numpy(coords).float()
        values_tensor = torch.from_numpy(values).float()

        return {
            'coords': coords_tensor,
            'values': values_tensor,
        }