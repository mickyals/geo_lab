# ERA5MultiData Class Documentation

The `ERA5MultiData` class provides a convenient interface for working with ERA5 weather data, which is stored in a structured grid format. This class offers methods to extract 2D surfaces, 3D volumes, and generate collocation points for physics-informed machine learning applications.

## Class Overview

```python
class ERA5MultiData:
    """A class for handling and processing ERA5 weather data on a structured grid.
    
    This class provides methods to extract and manipulate weather data from ERA5 datasets,
    including extracting 2D surfaces, 3D volumes, and generating collocation points for
    physics-informed machine learning applications.
    """
    
    def __init__(self, root_dir: str, read_data_fn: callable, solution_vars: list[str]):
        # Initialization code...
```

## Initialization

### `__init__(root_dir, read_data_fn, solution_vars)`

Initialize an ERA5MultiData instance.

**Parameters:**
- `root_dir` (str): Path to the root directory containing the ERA5 data files.
- `read_data_fn` (callable): Function that takes a root directory and returns an xarray Dataset.
- `solution_vars` (list[str]): List of variable names to include in the solution (e.g., ['z', 't', 'u', 'v', 'w']).

**Available Variables:**

#### Coordinate Variables:
- `valid_time`: Time dimension (24 time steps)
- `pressure_level`: Vertical pressure levels (17 levels from 200hPa to 850hPa)
- `latitude`: Latitude coordinates (-90° to 90°)
- `longitude`: Longitude coordinates (0° to 359.75°)

#### Solution Variables:
- `z`: Geopotential (m²/s²)
- `t`: Temperature (K)
- `u`: Eastward wind component (m/s)
- `v`: Northward wind component (m/s)
- `w`: Vertical velocity (Pa/s)

**Initialized Attributes:**

- `lower_bounds`: Dictionary containing minimum values for both coordinates and solution variables
  ```python
  {
      'coords': {
          'valid_time': float,    # Minimum time value
          'pressure_level': float,  # Minimum pressure level (hPa)
          'latitude': float,      # Minimum latitude (-90°)
          'longitude': float      # Minimum longitude (0°)
      },
      'solution': {
          'z': float,  # Minimum geopotential (m²/s²)
          't': float,  # Minimum temperature (K)
          'u': float,  # Minimum eastward wind (m/s)
          'v': float,  # Minimum northward wind (m/s)
          'w': float   # Minimum vertical velocity (Pa/s)
      }
  }
  ```

- `upper_bounds`: Dictionary containing maximum values for both coordinates and solution variables
  ```python
  {
      'coords': {
          'valid_time': float,    # Maximum time value
          'pressure_level': float,  # Maximum pressure level (hPa)
          'latitude': float,      # Maximum latitude (90°)
          'longitude': float      # Maximum longitude (359.75°)
      },
      'solution': {
          'z': float,  # Maximum geopotential (m²/s²)
          't': float,  # Maximum temperature (K)
          'u': float,  # Maximum eastward wind (m/s)
          'v': float,  # Maximum northward wind (m/s)
          'w': float   # Maximum vertical velocity (Pa/s)
      }
  }
  ```

- `solution_mean`: Dictionary of mean values for each solution variable
  ```python
  {
      'z': float,  # Mean geopotential (m²/s²)
      't': float,  # Mean temperature (K)
      'u': float,  # Mean eastward wind (m/s)
      'v': float,  # Mean northward wind (m/s)
      'w': float   # Mean vertical velocity (Pa/s)
  }
  ```

- `solution_std`: Dictionary of standard deviations for each solution variable
  ```python
  {
      'z': float,  # Standard deviation of geopotential (m²/s²)
      't': float,  # Standard deviation of temperature (K)
      'u': float,  # Standard deviation of eastward wind (m/s)
      'v': float,  # Standard deviation of northward wind (m/s)
      'w': float   # Standard deviation of vertical velocity (Pa/s)
  }
  ```

**Example Usage:**
```python
# Accessing coordinate bounds
min_lat = era5_data.lower_bounds['coords']['latitude']
max_lon = era5_data.upper_bounds['coords']['longitude']

# Accessing solution bounds
min_temp = era5_data.lower_bounds['solution']['t']
max_wind = era5_data.upper_bounds['solution']['u']

# Accessing statistics
temp_mean = era5_data.solution_mean['t']
temp_std = era5_data.solution_std['t']
```

## Core Methods

### Surface Extraction Methods

#### `get_pressure_surface(valid_time_idx=0, pressure_level_idx=0, solutions=True)`
Extract a 2D horizontal slice of the data at the specified pressure level and time index.

**Parameters:**
- `valid_time_idx` (int): Index of the time step to extract (default: 0)
- `pressure_level_idx` (int): Index of the pressure level to extract (default: 0)
- `solutions` (bool): Whether to include solution variables (default: True)

**Returns:**
- If `solutions=True`: Tuple of (coords_dict, solutions_dict)
- If `solutions=False`: coords_dict

#### `get_longitude_surface(valid_time_idx=0, longitude_idx=0, solutions=True)`
Extract a 2D vertical slice (latitude-pressure) at the specified longitude and time index.

**Parameters:**
- `valid_time_idx` (int): Index of the time step to extract (default: 0)
- `longitude_idx` (int): Index of the longitude to extract (default: 0)
- `solutions` (bool): Whether to include solution variables (default: True)

#### `get_latitude_surface(valid_time_idx=0, latitude_idx=0, solutions=True)`
Extract a 2D vertical slice (longitude-pressure) at the specified latitude and time index.

**Parameters:**
- `valid_time_idx` (int): Index of the time step to extract (default: 0)
- `latitude_idx` (int): Index of the latitude to extract (default: 0)
- `solutions` (bool): Whether to include solution variables (default: True)

### Volume Extraction Methods

#### `get_inner_volume(solutions=True)`
Extract the inner volume of the 4D data, excluding boundary points.

**Parameters:**
- `solutions` (bool): Whether to include solution variables (default: True)

**Returns:**
- If `solutions=True`: Tuple of (coords_dict, solutions_dict)
- If `solutions=False`: coords_dict

#### `get_initial_surface(solutions=True)`
Extract the initial time step surface data (first time step), including both the surface (bottom) and top of atmosphere.

**Parameters:**
- `solutions` (bool): Whether to include solution variables (default: True)

### Collocation Points Generation

#### `get_collocation_points(num_samples, use_lhs=True)`
Generate collocation points within the domain bounds using Latin Hypercube Sampling or uniform random sampling.

**Parameters:**
- `num_samples` (int): Number of collocation points to generate
- `use_lhs` (bool): Whether to use Latin Hypercube Sampling (True) or uniform random sampling (False)

**Returns:**
- Dictionary where keys are coordinate names and values are 1D arrays of sampled points.

## Usage Example

```python
# Initialize the ERA5 data handler
root_dir = "path/to/era5/data.nc"
solution_vars = ['z', 't', 'u', 'v', 'w']
era5_data = ERA5MultiData(root_dir, xr.open_dataset, solution_vars)

# Get a pressure surface (2D slice)
coords, solutions = era5_data.get_pressure_surface(
    valid_time_idx=0,
    pressure_level_idx=5,  # e.g., 700hPa level
    solutions=True
)

# Get collocation points for PINN training
points = era5_data.get_collocation_points(
    num_samples=10000,
    use_lhs=True  # Use Latin Hypercube Sampling
)

# Access the sampled points
valid_times = points['valid_time']
pressure_levels = points['pressure_level']
latitudes = points['latitude']
longitudes = points['longitude']
```

## Data Structure

The ERA5 data is expected to have the following dimensions and coordinates:
- `valid_time`: Time dimension (24 time steps)
- `pressure_level`: Vertical levels (17 pressure levels from 200hPa to 850hPa)
- `latitude`: Latitude coordinates (-90° to 90°)
- `longitude`: Longitude coordinates (0° to 359.75°)

## Dependencies

- numpy
- xarray
- pyDOE (for Latin Hypercube Sampling)

## Notes

- All methods that return coordinate dictionaries ensure that the arrays are properly aligned, meaning the i-th element of each array corresponds to the same point in space-time.
- The class includes assertions to verify the consistency of array lengths between coordinates and solution variables.
- The collocation points generation supports both Latin Hypercube Sampling (LHS) and uniform random sampling, with LHS being the default for better space-filling properties.
