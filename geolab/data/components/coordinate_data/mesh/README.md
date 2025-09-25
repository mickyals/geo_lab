# Mesh Package

## Overview
Core functionality for handling spatial and spatio-temporal mesh data, providing boundary access and sampling capabilities for geospatial applications.

## Core Features

### Boundary Access
Access points on all domain boundaries:
- `on_north_boundary()`: Points at maximum Y boundary
- `on_south_boundary()`: Points at minimum Y boundary  
- `on_east_boundary()`: Points at maximum X boundary
- `on_west_boundary()`: Points at minimum X boundary
- `on_upper_boundary()`: Points at maximum Z boundary (3D)
- `on_lower_boundary()`: Points at minimum Z boundary (3D)
- `on_initial_boundary()`: Points at initial time (t=0)
- `collection_points(use_lhs=False)`: Get points from the mesh, with two sampling strategies:
  - `use_lhs=False` (default): Returns points from the original mesh grid, excluding boundary points
  - `use_lhs=True`: Generates points using Latin Hypercube Sampling within the domain bounds, providing better coverage in high-dimensional spaces

### Point Sampling
```python
# Sample points using Latin Hypercube Sampling
points, time_values, _ = mesh.collection_points(
    N_f=1000,  # Number of points
    use_lhs=True
)

# Get non-boundary points from mesh
points, time_values, solutions = mesh.collection_points(
    N_f=1000,
    use_lhs=False,
    solution_names=['var1', 'var2']
)
```

### Initialization
```python
from geolab.data.components.coordinate_data.mesh import XarrayMesh

# Initialize with xarray dataset
mesh = XarrayMesh(
    read_data_fn=your_loader_function,
    spatial_dims=['x', 'y'],
    time_dim='time',  # optional
    solution_vars=['var1', 'var2']  # optional
)
```

## Integration
Part of the GeoLab framework's data processing pipeline, working with:
- Spatial domain definitions
- Data loading utilities
- Model training interfaces

Each boundary method returns a tuple of (spatial_coords, time_values, solution_values) for the specified boundary.
