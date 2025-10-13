# ERA5Multi Dataset Documentation

## Overview
The ERA5Multi module provides tools for working with ERA5 reanalysis data, offering functionality for data loading, preprocessing, and dataset creation for machine learning applications.

## Main Classes

### 1. ERA5MultiData

#### Purpose
Handles loading, subsetting, and preprocessing of ERA5 reanalysis data.

#### Key Features
- Loads ERA5 data from netCDF files
- Subsets data based on time, pressure levels, latitude, and longitude
- Generates virtual samples for data augmentation
- Creates coordinate meshes for spatial-temporal analysis

#### Core Methods

##### `__init__(data_dir, read_data_fn, variables)`
- **Parameters**:
  - `data_dir`: Directory containing ERA5 data files
  - `read_data_fn`: Function to read the data (e.g., `xarray.open_dataset`)
  - `variables`: List of variable names to load

##### `subset_data(time_idx_range, pressure_idx_range, latitude_idx_range, longitude_idx_range)`
- **Parameters**:
  - `time_idx_range`: Range of time indices to include
  - `pressure_idx_range`: Range of pressure level indices
  - `latitude_idx_range`: Range of latitude indices
  - `longitude_idx_range`: Range of longitude indices
- **Returns**: Tuple of (coordinates, data) dictionaries

##### `generate_virtual_samples(num_samples, coord_stats_dict, use_lhs=True)`
- **Parameters**:
  - `num_samples`: Number of virtual samples to generate
  - `coord_stats_dict`: Dictionary with coordinate statistics
  - `use_lhs`: Whether to use Latin Hypercube Sampling
- **Returns**: Dictionary of virtual coordinate points

##### `generate_mesh(coords_dict, indexing='ij')`
- **Parameters**:
  - `coords_dict`: Dictionary of coordinate arrays
  - `indexing`: 'ij' for matrix indexing, 'xy' for Cartesian
- **Returns**: Dictionary with mesh grid coordinates

### 2. ERA5MultiDataset

#### Purpose
PyTorch Dataset class for ERA5 data, suitable for training machine learning models.

#### Key Features
- Handles data normalization
- Supports both real and virtual samples
- Provides efficient data loading

#### Core Methods

##### `__getitem__(idx)`
- **Parameters**: `idx` - Index of the sample to retrieve
- **Returns**: Dictionary containing:
  - `'coords'`: Dictionary of normalized and scaled coordinate tensors with `requires_grad=True`
    - `'longitude'`: Scaled to `[-π, π]`
    - `'latitude'`: Scaled to `[-π/2, π/2]`
    - `'pressure_level'`: Normalized to `[-1, 1]`
    - `'time'`: Normalized to `[-1, 1]`
  - `'variables'`: Dictionary of normalized variable tensors in `[-1, 1]` range
  - `'classification'`: Tensor indicating real (1) or virtual (0) sample

##### `normalise(x, x_min, x_max)`
- **Parameters**:
  - `x`: Data to normalize
  - `x_min`: Minimum value for normalization
  - `x_max`: Maximum value for normalization
- **Returns**: Normalized data in range `[-1, 1]`

##### `time_delta_normalised(t_min, t_max, time)`
- **Parameters**:
  - `t_min`: Minimum time value
  - `t_max`: Maximum time value
  - `time`: Time value to normalize
- **Returns**: Time normalized to `[0, 1]` range

## Data Structure

### Input Data Format
- Expected format: netCDF files with dimensions (time, level, latitude, longitude)
- Required coordinates: time, level, latitude, longitude
- Variables: Any number of atmospheric variables (e.g., temperature, humidity)

### Output Structure

#### From `ERA5MultiData`
```python
{
    'data': {
        'longitude': (data_array, size, shape, dtype),
        'latitude': (data_array, size, shape, dtype),
        'pressure_level': (data_array, size, shape, dtype),
        'valid_time': (data_array, size, shape, dtype),
        'temperature': (data_array, size, shape, dtype),  # example variable
        'classification': (bool_array, size)  # True for real, False for virtual samples
    },
    'count': [total_samples, real_samples, virtual_samples]
}
```

#### From `ERA5MultiDataset.__getitem__`
```python
{
    'coords': {
        'longitude': torch.Tensor,  # shape: [1], requires_grad=True, range: [-π, π]
        'latitude': torch.Tensor,   # shape: [1], requires_grad=True, range: [-π/2, π/2]
        'pressure_level': torch.Tensor,  # shape: [1], requires_grad=True, range: [-1, 1]
        'time': torch.Tensor        # shape: [1], requires_grad=True, range: [-1, 1]
    },
    'variables': {
        'temperature': torch.Tensor,  # shape: [1], range: [-1, 1]
        # ... other variables
    },
    'classification': torch.Tensor  # shape: [1], 1 for real, 0 for virtual
}
```

## Usage Example

```python
import xarray as xr
from geolab.data.dataset.era5multi import ERA5MultiData, ERA5MultiDataset

# Initialize data processor
data_processor = ERA5MultiData(
    data_dir='path/to/era5/data',
    read_data_fn=xr.open_dataset,
    variables=['temperature', 'humidity']
)

# Get subset of data
coords, data = data_processor.subset_data(
    time_idx_range=(0, 100),
    pressure_idx_range=(0, 10),
    latitude_idx_range=(0, 50),
    longitude_idx_range=(0, 100)
)

# Generate virtual samples
virtual_points = data_processor.generate_virtual_samples(
    num_samples=10000,
    coord_stats_dict={k: (v[0].min(), v[0].max()) for k, v in coords.items()}
)

# Create PyTorch dataset
dataset = ERA5MultiDataset(
    data=data,
    statistics=statistics,
    indices=indices,
    include_virtual=True,
    variables=['temperature', 'humidity'],
    scale=True
)
```

## Notes
- All coordinates and data are automatically converted to float64 for numerical stability
- Virtual samples are generated using Latin Hypercube Sampling by default for better coverage
- Normalization and Scaling:
  - Time: Normalized to `[0, 1]` then scaled to `[-1, 1]`
  - Latitude: Normalized to `[-1, 1]` then scaled to `[-π/2, π/2]`
  - Longitude: Normalized to `[-1, 1]` then scaled to `[-π, π]`
  - Pressure and other variables: Normalized to `[-1, 1]`
- Coordinate tensors have `requires_grad=True` for gradient computation
- The dataset handles both real and virtual samples, with classification labels (1 for real, 0 for virtual)
