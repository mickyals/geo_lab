from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


class ERA5MultiData:
    """A class for handling and processing multi-variable ERA5 climate reanalysis data.
    
    This class provides functionality to load, slice, and access ERA5 climate data stored in NetCDF format.
    It supports both lazy loading and preloading of data into memory for faster access.
    
    Attributes:
        data_dir (Path): Directory containing the ERA5 NetCDF files.
        variables (List[str]): List of variable names to load from the dataset.
        time_idx_range (Optional[Tuple[int, int]]): Start and end indices for time dimension.
        pressure_idx_range (Optional[Tuple[int, int]]): Start and end indices for pressure level dimension.
        latitude_idx_range (Optional[Tuple[int, int]]): Start and end indices for latitude dimension.
        longitude_idx_range (Optional[Tuple[int, int]]): Start and end indices for longitude dimension.
        preload (bool): If True, loads all data into memory. If False, uses lazy loading.
        ds (xr.Dataset): The xarray Dataset containing the loaded data.
        data_arrays (Dict[str, np.ndarray]): Dictionary mapping variable names to their data arrays.
        coordinates (Dict[str, np.ndarray]): Dictionary mapping coordinate names to their values.
        coord_labels (Dict[str, int]): Mapping of coordinate names to their dimension indices.
        variable_labels (Dict[str, int]): Mapping of variable names to their indices.
        coord_sizes (np.ndarray): Array of coordinate dimension sizes.
    """

    def __init__(self,
                 data_dir: Union[str, Path],
                 variables: List[str],
                 time_idx_range: Optional[Tuple[int, int]] = None,
                 pressure_idx_range: Optional[Tuple[int, int]] = None,
                 latitude_idx_range: Optional[Tuple[int, int]] = None,
                 longitude_idx_range: Optional[Tuple[int, int]] = None,
                 preload: bool = True):
        """Initialize the ERA5MultiData instance and load the specified data.
        
        Args:
            data_dir: Path to the directory containing ERA5 NetCDF files or a single NetCDF file.
            variables: List of variable names to load from the dataset.
            time_idx_range: Optional (start, end) indices for time dimension.
            pressure_idx_range: Optional (start, end) indices for pressure level dimension.
            latitude_idx_range: Optional (start, end) indices for latitude dimension.
            longitude_idx_range: Optional (start, end) indices for longitude dimension.
            preload: If True, loads all data into memory. If False, uses lazy loading.
        """
        self.data_dir = Path(data_dir)
        self.variables = variables
        self.time_idx_range = time_idx_range
        self.pressure_idx_range = pressure_idx_range
        self.latitude_idx_range = latitude_idx_range
        self.longitude_idx_range = longitude_idx_range
        self.preload = preload

        # Initialize data structures that will be populated in load()
        self.ds = None  # Will hold the xarray Dataset
        self.data_arrays = {}  # Maps variable names to their numpy arrays
        self.coordinates = {}  # Maps coordinate names to their numpy arrays
        self.coord_labels = {}  # Maps coordinate names to their dimension indices
        self.variable_labels = {}  # Maps variable names to their indices
        self.coord_sizes = None  # Will hold sizes of each coordinate dimension

        # Load the data
        self.load()

    def load(self) -> None:
        """Load and prepare the ERA5 dataset based on initialization parameters.
        
        This method handles both single file and multiple file loading, applies the specified
        variable selection and slicing, and either preloads data into memory or keeps it on disk.
        """
        print(f"Loading ERA5 data from {self.data_dir}")

        # Open dataset - handle both single file and directory of files
        if self.data_dir.is_file():
            ds = xr.open_dataset(self.data_dir)
        else:
            # Use xarray's multi-file dataset functionality to combine multiple NetCDF files
            ds = xr.open_mfdataset(
                str(self.data_dir / "*.nc"),
                combine='by_coords',  # Combine along coordinates
                parallel=True,        # Enable parallel loading for better performance
            )

        # Select only the variables that were requested
        ds = ds[self.variables]

        # Apply any specified slicing to the dimensions
        ds = self._apply_slicing(ds)

        # Build mapping from coordinate names to their dimension indices
        for idx, coord in enumerate(ds.sizes.keys()):
            self.coord_labels[coord] = idx

        # Build mapping from variable names to their indices
        for idx, var in enumerate(ds.keys()):
            self.variable_labels[var] = idx

        # Store the size of each coordinate dimension
        self.coord_sizes = np.array([value for _, value in ds.sizes.items()])

        # Extract coordinate values into numpy arrays
        self._build_coordinate_arrays(ds)

        # Handle data loading based on preload setting
        if self.preload:
            print("Preloading data into memory...")
            self._build_target_arrays(ds)
            ds.close()  # Close the dataset after preloading
        else:
            self.ds = ds  # Keep the dataset open for lazy loading

        # Print summary of loaded data
        print(f"Successfully loaded {len(self.variables)} variables:")
        for var in self.variables:
            if self.preload:
                shape = self.data_arrays[var].shape
            else:
                shape = self.ds[var].shape
            print(f"   {var}: {shape}")

    @staticmethod
    def _make_slice(idx_range: Optional[Union[Tuple[int, int], List[int]]]) -> slice:
        """Create a slice object from a range tuple or list.
        
        Args:
            idx_range: Optional tuple or list of [start, end] indices.
                      If None, empty, or falsy, returns a slice(None) which selects all elements.
                      
        Returns:
            slice: A slice object representing the specified range.
        """
        if not idx_range:  # Handles None, empty list, empty tuple
            return slice(None)
        return slice(idx_range[0], idx_range[1])

    def _apply_slicing(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply index-based slicing to the dataset based on initialization parameters.
        
        This method creates a slice dictionary for each dimension that has a specified
        index range and applies it to the dataset using xarray's isel method.
        
        Args:
            ds: The xarray Dataset to slice.
            
        Returns:
            xr.Dataset: The sliced dataset.
        """
        slice_dict = {}

        # Create slice objects for each dimension if a range was specified
        if self.time_idx_range is not None:
            slice_dict['valid_time'] = self._make_slice(self.time_idx_range)

        if self.pressure_idx_range is not None:
            slice_dict['pressure_level'] = self._make_slice(self.pressure_idx_range)

        if self.latitude_idx_range is not None:
            slice_dict['latitude'] = self._make_slice(self.latitude_idx_range)

        if self.longitude_idx_range is not None:
            slice_dict['longitude'] = self._make_slice(self.longitude_idx_range)

        # Apply the slicing if any dimensions were specified
        if slice_dict:
            print(f"Applying slicing: {slice_dict}")
            ds = ds.isel(slice_dict)  # Apply the slicing using xarray's isel
        else:
            print("No slicing applied - using all indices")

        return ds

    def _build_coordinate_arrays(self, ds: xr.Dataset) -> None:
        """Extract coordinate values from the dataset and store them as numpy arrays.
        
        This method processes the time, pressure level, latitude, and longitude coordinates,
        performs any necessary transformations, and stores them in the coordinates dictionary.
        
        Args:
            ds: The xarray Dataset containing the coordinate data.
        """
        # Process time coordinate - convert to seconds since epoch
        time_data = ds['valid_time'].values  # numpy array of datetime64
        self.coordinates['valid_time'] = time_data.astype('datetime64[s]').astype(np.float32)

        # Process pressure levels (hPa or Pa)
        pressure_data = ds['pressure_level'].values
        self.coordinates['pressure_level'] = pressure_data.astype(np.float32)

        # Process latitude (degrees north, -90 to 90)
        latitude_data = ds['latitude'].values
        self.coordinates['latitude'] = latitude_data.astype(np.float32)

        # Process longitude (degrees east, -180 to 180)
        # Convert from 0-360 to -180 to 180 if necessary
        longitude_data = ds['longitude'].values
        longitude_data = ((longitude_data + 180) % 360) - 180  # Convert to -180 to 180 range
        self.coordinates['longitude'] = longitude_data.astype(np.float32)

        # Print summary of coordinate ranges
        print("Coordinate ranges:")
        for name, data in self.coordinates.items():
            print(f"  {name}: shape={data.shape}, range=[{data.min():.2f}, {data.max():.2f}]")

    def _build_target_arrays(self, ds: xr.Dataset) -> None:
        """Load variable data from the dataset into memory as numpy arrays.
        
        This method is called when preload=True to load the actual data values
        for each variable into memory for faster access during training/inference.
        
        Args:
            ds: The xarray Dataset containing the variable data.
        """
        for var in self.variables:
            print(f"  Loading {var}...")
            # Extract data and ensure it's in float32 format to save memory
            data = ds[var].values.astype(np.float32)
            self.data_arrays[var] = data
            # Print memory usage information
            print(f"    Memory usage: {data.nbytes / 1e9:.2f} GB")

    def get_variable(self, var_name: str, indices: Optional[List[int]] = None) -> np.ndarray:
        """Retrieve data for a specific variable, optionally at specified indices.
        
        Args:
            var_name: Name of the variable to retrieve.
            indices: Optional list of indices to select specific elements from the variable.
                    If None, returns the entire array.
                    
        Returns:
            np.ndarray: The requested variable data as a numpy array.
            
        Raises:
            ValueError: If the specified variable is not found in the dataset.
        """
        if var_name not in self.data_arrays:
            raise ValueError(f"Variable {var_name} not found in dataset.")
            
        if indices is not None:
            return self.data_arrays[var_name][indices]
        return self.data_arrays[var_name]

    def get_coords_at_index(self,
                            time_idx: Union[int, np.ndarray],
                            pressure_idx: Union[int, np.ndarray],
                            latitude_idx: Union[int, np.ndarray],
                            longitude_idx: Union[int, np.ndarray]) -> np.ndarray:
        """Get the coordinate values at the specified dimensional indices.
        
        Args:
            time_idx: Index or indices for the time dimension.
            pressure_idx: Index or indices for the pressure level dimension.
            latitude_idx: Index or indices for the latitude dimension.
            longitude_idx: Index or indices for the longitude dimension.
            
        Returns:
            np.ndarray: Stacked array of coordinate values with shape (..., 4) where the last dimension
                      contains [time, pressure, latitude, longitude] values.
        """
        return np.stack([
            self.coordinates['valid_time'][time_idx],
            self.coordinates['pressure_level'][pressure_idx],
            self.coordinates['latitude'][latitude_idx],
            self.coordinates['longitude'][longitude_idx]
        ], axis=-1)

    def get_values_at_index(self,
                            var_name: str,
                            time_idx: Union[int, np.ndarray],
                            pressure_idx: Union[int, np.ndarray],
                            latitude_idx: Union[int, np.ndarray],
                            longitude_idx: Union[int, np.ndarray]) -> np.ndarray:
        """Retrieve variable values at the specified dimensional indices.
        
        Args:
            var_name: Name of the variable to retrieve values for.
            time_idx: Index or indices for the time dimension.
            pressure_idx: Index or indices for the pressure level dimension.
            latitude_idx: Index or indices for the latitude dimension.
            longitude_idx: Index or indices for the longitude dimension.
            
        Returns:
            np.ndarray: The requested values as a numpy array. The shape will match the
                      broadcasted shape of the input indices.
                      
        Note:
            If preload=False, this will trigger disk reads for each access.
            For better performance with repeated access, use preload=True.
        """
        # Get data from memory if preloaded, otherwise read from disk
        data = self.data_arrays[var_name] if self.preload else self.ds[var_name].values
        # Use numpy's advanced indexing to get the requested values
        return data[time_idx, pressure_idx, latitude_idx, longitude_idx]

    @property
    def num_times(self) -> int:
        """int: Number of time steps in the dataset."""
        return len(self.coordinates['valid_time'])

    @property
    def num_pressure_levels(self) -> int:
        """int: Number of pressure levels in the dataset."""
        return len(self.coordinates['pressure_level'])

    @property
    def num_latitudes(self) -> int:
        """int: Number of latitude points in the dataset."""
        return len(self.coordinates['latitude'])

    @property
    def num_longitudes(self) -> int:
        """int: Number of longitude points in the dataset."""
        return len(self.coordinates['longitude'])

    @property
    def latitudes(self) -> np.ndarray:
        """np.ndarray: Array of latitude values in degrees north."""
        return self.coordinates['latitude']

    @property
    def longitudes(self) -> np.ndarray:
        """np.ndarray: Array of longitude values in degrees east (-180 to 180)."""
        return self.coordinates['longitude']

    @property
    def pressure_levels(self) -> np.ndarray:
        """np.ndarray: Array of pressure level values in hPa or Pa."""
        return self.coordinates['pressure_level']

    @property
    def times(self) -> np.ndarray:
        """np.ndarray: Array of time values as seconds since epoch."""
        return self.coordinates['valid_time']

    @property
    def input_dim(self) -> int:
        """int: Dimensionality of the input space (number of coordinate dimensions)."""
        return len(self.coord_labels)

    @property
    def output_dim(self) -> int:
        """int: Dimensionality of the output space (number of variables)."""
        return len(self.variable_labels)

    @property
    def get_var_labels(self):
        return [self.variable_labels.keys()]

    @property
    def var_order(self) -> List[str]:
        """List[str]: Variable names sorted by their index."""
        return sorted(self.variable_labels.keys(),
                      key=lambda k: self.variable_labels[k])

    @property
    def coord_order(self) -> List[str]:
        """List[str]: Coordinate names sorted by their index."""
        return sorted(self.coord_labels.keys(),
                      key=lambda k: self.coord_labels[k])


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