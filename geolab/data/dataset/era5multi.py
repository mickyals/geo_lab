from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import json
import torch
from torch.utils.data import Dataset
import xarray as xr


def convert_netcdf_to_npy(
        source_dir: str,
        dest_dir: str,
        variables: list[str]
):
    """
    convert_netcdf_to_npy(data_dir, "./era5_npy", ["u", "v", "t", "z", "w"])
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)

    print(f"Opening NetCDFs from {source_path}...")
    if source_path.is_file():
        ds = xr.open_dataset(source_path)
    else:
        ds = xr.open_mfdataset(str(source_path / "*.nc"), combine='by_coords')

    # Helper function to convert numpy scalars to python natives for JSON
    def get_stats(arr):
        return {
            "minimum": float(arr.min()),
            "maximum": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std())
        }

    # 1. Process Coordinates
    print("Processing coordinates...")
    coords = {}
    coord_stats = {}

    # Time
    coords['valid_time'] = ds['valid_time'].values.astype('datetime64[s]').astype(np.float32)
    coord_stats['valid_time'] = get_stats(coords['valid_time'])

    # Pressure
    coords['pressure_level'] = ds['pressure_level'].values.astype(np.float32)
    coord_stats['pressure_level'] = get_stats(coords['pressure_level'])

    # Latitude
    coords['latitude'] = ds['latitude'].values.astype(np.float32)
    coord_stats['latitude'] = get_stats(coords['latitude'])

    # Longitude
    lon_data = ds['longitude'].values
    lon_data = ((lon_data + 180) % 360) - 180
    coords['longitude'] = lon_data.astype(np.float32)
    coord_stats['longitude'] = get_stats(coords['longitude'])

    # Save Coordinates
    np.savez(dest_path / "coords.npz", **coords)

    # 2. Process Variables
    shape_info = {}
    var_stats = {}
    for var in variables:
        print(f"Processing variable: {var}...")
        if var not in ds:
            continue

        data = ds[var].values.astype(np.float32)
        var_stats[var] = get_stats(data)

        np.save(dest_path / f"{var}.npy", data)
        # Convert shape tuple to list of ints for JSON safety
        shape_info[var] = [int(s) for s in data.shape]

    # 3. Save Metadata
    metadata = {
        "coord_labels": {k: i for i, k in enumerate(["valid_time", "pressure_level", "latitude", "longitude"])},
        "variable_labels": {k: i for i, k in enumerate(variables)},
        "coord_stats": coord_stats,
        "var_stats": var_stats,
        "shapes": shape_info
    }

    with open(dest_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Conversion complete. Data saved to {dest_path}")


class ERA5MultiData:

    def __init__(self,
                 data_dir: Union[str, Path],
                 variables: List[str],
                 time_idx_range: Optional[Tuple[int, int]] = None,
                 pressure_idx_range: Optional[Tuple[int, int]] = None,
                 latitude_idx_range: Optional[Tuple[int, int]] = None,
                 longitude_idx_range: Optional[Tuple[int, int]] = None,
                 preload: bool = True
                 ):
        self.data_dir = Path(data_dir)
        self.variables = variables
        self.preload = preload

        # Load Metadata and Stats
        with open(self.data_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.coord_labels = metadata['coord_labels']
        self.variable_labels = metadata['variable_labels']
        self.coord_stats = metadata['coord_stats']
        self.var_stats = metadata['var_stats']

        time_slice = slice(*time_idx_range) if time_idx_range else slice(None)
        pressure_slice = slice(*pressure_idx_range) if pressure_idx_range else slice(None)
        latitude_slice = slice(*latitude_idx_range) if latitude_idx_range else slice(None)
        longitude_slice = slice(*longitude_idx_range) if longitude_idx_range else slice(None)

        with np.load(self.data_dir / "coords.npz") as loader:
            self.coordinates = {
                'valid_time': loader['valid_time'][time_slice],
                'pressure_level': loader['pressure_level'][pressure_slice],
                'latitude': loader['latitude'][latitude_slice],
                'longitude': loader['longitude'][longitude_slice]
            }
        # Initialize Data Arrays (Memory Mapped or Preloaded)
        self.data_arrays = {}
        for var in self.variables:
            arr = np.load(self.data_dir / f"{var}.npy", mmap_mode='r')
            # Apply slicing to the memmap
            sliced_arr = arr[time_slice, pressure_slice, latitude_slice, longitude_slice]
            self.data_arrays[var] = sliced_arr.copy() if preload else sliced_arr

    def get_coords_at_index(self, t_idx, p_idx, lat_idx, lon_idx) -> np.ndarray:
        return np.stack([
            self.coordinates['valid_time'][t_idx],
            self.coordinates['pressure_level'][p_idx],
            self.coordinates['latitude'][lat_idx],
            self.coordinates['longitude'][lon_idx]
        ], axis=-1)

    def get_values_at_index(self, var_name, t_idx, p_idx, lat_idx, lon_idx) -> np.ndarray:
        return self.data_arrays[var_name][t_idx, p_idx, lat_idx, lon_idx]

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
        """np.ndarray: Array of latitude values in degrees."""
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
        return list(self.variable_labels.keys())

    @property
    def var_order(self) -> List[str]:
        """List[str]: Variable names sorted by their index."""
        return list(self.variable_labels.keys())

    @property
    def time_idx(self) -> int:
        return self.coord_labels['valid_time']

    @property
    def pressure_idx(self) -> int:
        return self.coord_labels['pressure_level']

    @property
    def latitude_idx(self) -> int:
        return self.coord_labels['latitude']

    @property
    def longitude_idx(self) -> int:
        return self.coord_labels['longitude']

    @property
    def coord_order(self) -> List[str]:
        """List[str]: Coordinate names sorted by their index."""
        return list(self.coord_labels.keys())

    @property
    def num_points(self):
        return len(self.coordinates['valid_time']) * len(self.coordinates['pressure_level']) * len(
            self.coordinates['latitude']) * len(self.coordinates['longitude'])

    @property
    def coord_statistics(self):
        return self.coord_stats

    @property
    def variable_statistics(self):
        return self.var_stats


class ERA5MultiDataset(Dataset):
    def __init__(self, data: ERA5MultiData, indices: np.ndarray, variables: List[str], **kwargs):
        self.data = data
        self.indices = indices
        self.variables = variables

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        t, p, lat, lon = self.indices[idx]
        coords = self.data.get_coords_at_index(t, p, lat, lon)
        values = np.array([self.data.get_values_at_index(v, t, p, lat, lon) for v in self.variables])

        return {
            'coords': torch.from_numpy(coords).float(),
            'values': torch.from_numpy(values).float()
        }