"""
Xarray-based mesh handling module for geospatial data processing.

This module provides a flexible framework for working with spatial and spatio-temporal
mesh data using xarray datasets. It handles structured grids with support for various
boundary conditions, data sampling strategies, and coordinate systems.

Core Functionality:
- Mesh representation and manipulation for xarray datasets
- Boundary condition handling for spatial dimensions (north, south, east, west, lower, upper)
- Support for both spatial and temporal dimensions
- Data sampling methods including Latin Hypercube Sampling (LHS)
- Seamless integration with xarray for geospatial data handling

Key Class:
    XarrayMesh: Implementation for working with xarray datasets

Assumptions:
1. Spatial dimensions follow the convention: (x, y, z) for 3D or (x, y) for 2D
2. Time is always the last dimension if present
3. Mesh data is assumed to be structured (gridded) but can be non-uniform
4. Solution variables are stored as numpy arrays with dimensions matching the mesh
5. For temporal data, time is assumed to be the last dimension in all arrays
6. Boundary methods return points in a consistent format: (spatial_coords, time_values, solution_values)

Example Usage:
    ```python
    # Initialize a mesh from an xarray dataset
    mesh = XarrayMesh(
        root_dir='path/to/data.nc',
        read_data_fn=xr.open_dataset,
        spatial_dims=['longitude', 'latitude'],
        time_dim='time',
        solution_vars=['temperature', 'pressure']
    )
    
    # Get points at the northern boundary
    coords, time, solutions = mesh.on_north_boundary(['temperature'])
    ```
"""

from typing import Callable, List, Optional, Dict, Tuple

import numpy as np
import pyDOE as lhs
import warnings
warnings.filterwarnings('ignore')

from geolab.data.components.coordinate_data.domains.space import GeoSpatialDomain



class XarrayMesh():
    """
    Mesh class for handling xarray datasets with spatial and temporal dimensions.
    
    T
    
    Args:
        root_dir: Path to the dataset file
        read_data_fn: Function to read the dataset file
        spatial_dims: List of dimension names for spatial coordinates
        time_dim: Name of the time dimension (if any)
        solution_vars: List of variable names to include in the solution domain
    """
    
    def __init__(
        self,
        root_dir: str,
        read_data_fn: Callable,
        spatial_dims: Optional[List[str]] = None,
        time_dim: Optional[str] = None,
        solution_vars: Optional[List[str]] = None
    ):
        """
        Initialize the Xarray mesh with the given dataset and dimensions.
        
        Args:
            root_dir: Path to the dataset file or URL
            read_data_fn: Function to read the dataset (e.g., xr.open_dataset)
            spatial_dims: List of dimension names for spatial coordinates
            time_dim: Optional name of the time dimension
            solution_vars: Optional list of variable names to include in the solution domain
        """
        # Load dataset using provided function
        dataset = read_data_fn(root_dir)
        
        # Determine spatial dimensions if not provided
        if spatial_dims is None:
            spatial_dims = [
                dim for dim in dataset.dims 
                if dim != time_dim  # Exclude time dimension if specified
            ]
        
        # Get time dimension if it exists in the dataset
        if time_dim is not None and time_dim not in dataset.dims:
            time_dim = None
        
        # Extract spatial domain coordinates
        spatial_domain = [dataset.coords[dim].values for dim in spatial_dims]
        
        # Handle temporal domain if it exists
        temporal_domain = None
        if time_dim is not None:
            temporal_domain = dataset.coords[time_dim].values

        spatiotemporal_dims = spatial_dims + [time_dim] if time_dim is not None else spatial_dims

        # Determine shape including time if it exists
        shape = [len(arr) for arr in spatial_domain]
        if time_dim is not None:
            shape.append(len(temporal_domain))
        
        # Extract solution variables and ensure they have the same shape as the mesh
        if solution_vars is None:
            solution_domain = {var: dataset[var].values for var in dataset.data_vars}
        else:
            solution_domain = {var: dataset[var].values for var in solution_vars}
        
        # Calculate spatial bounds
        lb_space = {dim: arr.min() for dim, arr in zip(spatial_dims, spatial_domain)}
        ub_space = {dim: arr.max() for dim, arr in zip(spatial_dims, spatial_domain)}
        
        # Calculate temporal bounds if time dimension exists
        lb_time = {time_dim: temporal_domain.min()} if time_dim is not None else None
        ub_time = {time_dim: temporal_domain.max()} if time_dim is not None else None
        
        # Calculate solution bounds
        lb_solution = {key: np.nanmin(arr) for key, arr in solution_domain.items()}
        ub_solution = {key: np.nanmax(arr) for key, arr in solution_domain.items()}
        

        
        # Initialize base class attributes
        self.solution_domain = solution_domain
        self.dataset_coords = {name: len(coord) for name, coord in dataset.coords.items()}
        self.lower_bounds = {
            'spatial': lb_space,
            'time': lb_time,
            'solution': lb_solution
        }
        self.upper_bounds = {
            'spatial': ub_space,
            'time': ub_time,
            'solution': ub_solution
        }
        
        # Initialize the spatial domain handler
        self.mesh = GeoSpatialDomain(
            spatial_domain=spatial_domain,
            shape=shape,
            temporal_domain=temporal_domain,
            spatiotemporal_dims=spatiotemporal_dims,
        )
        self.dims_keys = spatiotemporal_dims