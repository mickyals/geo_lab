# Domains Package

## Overview
The domains package provides core functionality for managing spatial and temporal domains in geospatial applications. It handles coordinate transformations, mesh generation, and provides utilities for working with spatial data in both structured and unstructured formats.

## Core Components

### GeoSpatioTemporalDomain
Manages spatial and temporal domains with support for mesh generation and coordinate transformations.

#### Initialization
```python
from geolab.data.components.coordinate_data.domains.space import GeoSpatioTemporalDomain
import numpy as np

# For a 2D spatial domain
x = np.linspace(0, 10, 100)  # x-coordinates
y = np.linspace(0, 20, 200)  # y-coordinates

# Initialize domain
domain = GeoSpatioTemporalDomain(
    spatial_domain=[x, y],  # List of 1D coordinate arrays for each spatial dimension
    shape=[100, 200],       # Shape of the spatial domain (points in each dimension)
    temporal_domain=None,   # Optional 1D array of time values for temporal domain
    indexing='ij',          # 'ij' for matrix indexing or 'xy' for Cartesian
    sparse=False,           # Set to True for sparse mesh representation
    dtype=np.float64        # Data type for coordinate arrays
)
```

#### Key Methods
- `generate_mesh()`: Generates a mesh grid from the spatial domain
- `load_mesh`: Property that lazily loads the mesh when first accessed
- `spatial_bounds`: Access the spatial coordinate arrays
- `temporal_bounds`: Access the temporal coordinate array (if any)

### Key Features
- **Spatial Domain Management**: Handle 2D or 3D spatial domains
- **Temporal Support**: Optional time dimension for spatio-temporal data
- **Flexible Indexing**: Support for both matrix ('ij') and Cartesian ('xy') indexing
- **Memory Efficient**: Optional sparse mesh representation for large domains
- **Type Safety**: Configurable data type for coordinate arrays

## Usage Example
```python
import numpy as np
from geolab.data.components.coordinate_data.domains.space import GeoSpatialDomain

# Create coordinate arrays
x = np.linspace(-10, 10, 100)
y = np.linspace(-5, 5, 100)

# Initialize domain with time dimension
time = np.linspace(0, 10, 50)  # 50 time steps

domain = GeoSpatialDomain(
    spatial_domain=[x, y],
    shape=[100, 100],
    temporal_domain=time
)

# Generate the mesh (lazy-loaded on first access)
mesh = domain.mesh  # Returns [X, Y, T] for 2D + time

# Or generate explicitly
X, Y, T = domain.generate_mesh()
```

## Assumptions
1. Spatial coordinates are provided as a list of 1D numpy arrays
2. The order of spatial dimensions follows (x, y, z) for 3D or (x, y) for 2D
3. Time is treated as a separate, optional dimension
4. All coordinate arrays must have consistent shapes
5. For temporal data, time values must be provided as a 1D numpy array
