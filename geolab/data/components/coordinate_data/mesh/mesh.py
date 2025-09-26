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

import warnings
warnings.filterwarnings('ignore')


class ERA5MultiData():
    def __init__(self, root_dir, read_data_fn, spatial_vars, time_var, solution_vars):
        """
        Initialize an ERA5MultiData instance.

        Parameters
        ----------
        root_dir : str
            Root directory of the data
        read_data_fn : callable
            Function to read in the data
        spatial_vars : list
            List of spatial variables
        time_var : list
            List of time variables
        solution_vars : list
            List of solution variables
        """
        # Initialize base class attributes

        dataset = read_data_fn(root_dir)

        # Convert dataset coords to a dict with NumPy arrays
        coords_domain = {var: arr.values for var, arr in dataset.coords.items()}

        # Select spatial and temporal variables directly using set intersection
        spatial_keys = set(spatial_vars) & coords_domain.keys()
        temporal_keys = set(time_var) & coords_domain.keys()

        spatial_domain = {k: coords_domain[k] for k in spatial_keys}
        temporal_domain = {k: coords_domain[k] for k in temporal_keys}

        # Solution variables
        solution_keys = set(solution_vars) & dataset.data_vars.keys()
        solution_domain = {k: dataset[k].values for k in solution_keys}
        # For spatial variables
        lb_space = {var: np.min(arr) for var, arr in spatial_domain.items()}
        ub_space = {var: np.max(arr) for var, arr in spatial_domain.items()}

        # For temporal variables
        lb_time = {var: np.min(arr) for var, arr in temporal_domain.items()}
        ub_time = {var: np.max(arr) for var, arr in temporal_domain.items()}

        # For solution variables
        lb_solution = {var: np.min(arr) for var, arr in solution_domain.items()}
        ub_solution = {var: np.max(arr) for var, arr in solution_domain.items()}


        # Initialize attributes
        self.solution_domain = solution_domain
        self.shape = dataset[solution_vars[0]].values.shape

        self.dataset_coords = {var: len(arr) for var, arr in dataset.coords.items()} #the order and shape of dims in solution domain

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

        self.spatial_domain = spatial_domain
        self.temporal_domain = temporal_domain

    def collection_points(self, num_points, use_lhs=True):
        """
        Sample points from the dataset's domain.
        
        Parameters
        ----------
        num_points : int
            Number of points to sample
        use_lhs : bool, optional
            Whether to use Latin Hypercube Sampling (default: True). 
            If False, uses uniform random sampling.
            
        Returns
        -------
        dict
            Dictionary containing sampled points for all dimensions (spatial + temporal)
            with variable names as keys.
        """
        spatial_keys = list(self.spatial_domain.keys())
        time_keys = list(self.temporal_domain.keys())

        spatial_dim = len(spatial_keys)
        total_dim = spatial_dim + len(time_keys)

        # Get lower and upper bounds as arrays in the same order
        lb = np.array([self.lower_bounds['spatial'][k] for k in spatial_keys] +
                      [self.lower_bounds['time'][k] for k in time_keys])
        ub = np.array([self.upper_bounds['spatial'][k] for k in spatial_keys] +
                      [self.upper_bounds['time'][k] for k in time_keys])

        if use_lhs:
            # Latin Hypercube Sampling
            from pyDOE import lhs as lhs_sample  # Import here to avoid circular imports
            # Generate samples in [0, 1]^total_dim
            samples = lhs_sample(total_dim, samples=num_points)
            # Scale samples to the actual ranges
            scaled_samples = lb + (ub - lb) * samples
            
            # Combine spatial and temporal components into a single dictionary
            spatiotemporal_domain = {}
            for i, var in enumerate(spatial_keys + time_keys):
                spatiotemporal_domain[var] = scaled_samples[:, i]
        else:
            # Flatten the mesh (all combinations of spatial + time coordinates)
            spatial_grid = np.meshgrid(*[self.spatial_domain[k] for k in spatial_keys],
                                     indexing='ij')
            time_grid = np.meshgrid(*[self.temporal_domain[k] for k in time_keys],
                                  indexing='ij') if time_keys else []
            
            # Create a single dictionary with all variables
            spatiotemporal_domain = {}
            for var, grid in zip(spatial_keys, spatial_grid):
                spatiotemporal_domain[var] = grid.flatten()
            
            for var, grid in zip(time_keys, time_grid):
                spatiotemporal_domain[var] = grid.flatten()

        return spatiotemporal_domain

    def on_initial_boundary(self, solutions=True):
        pass
