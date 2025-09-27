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
from pyDOE import lhs
import warnings
warnings.filterwarnings('ignore')


class ERA5MultiData:
    """A class for handling and processing ERA5 weather data on a structured grid.
    
    This class provides methods to extract and manipulate weather data from ERA5 datasets,
    including extracting 2D surfaces, 3D volumes, and generating collocation points for
    physics-informed machine learning applications.
    
    Attributes
    ----------
    root_dir : str
        Path to the root directory containing the ERA5 data files.
    read_data_fn : callable
        Function to read the data from the root directory.
    solution_vars : list of str
        List of variable names to include in the solution.
    lower_bounds : dict
        Dictionary containing the minimum values for each coordinate.
    upper_bounds : dict
        Dictionary containing the maximum values for each coordinate.
    """
    
    def __init__(self, root_dir: str, read_data_fn: callable, solution_vars: list[str]):
        """Initialize an ERA5MultiData instance.

        Parameters
        ----------
        root_dir : str
            Path to the root directory containing the ERA5 data files.
        read_data_fn : callable
            Function that takes a root directory and returns an xarray Dataset.
        solution_vars : list of str
            List of variable names to include in the solution (e.g., ['z', 't', 'u', 'v', 'w']).
            Note: 'w' (vertical velocity) will be automatically converted from Pa/s to m/s.
        """

        
        # Initialize base class attributes

        with read_data_fn(root_dir) as dataset:

            coords = [coord for coord in dataset.coords]

            # Get coordinate bounds
            lb_coords = {var: np.min(arr.values) for var, arr in dataset.coords.items()}
            ub_coords = {var: np.max(arr.values) for var, arr in dataset.coords.items()}
            lb_coords["longitude"] = (((lb_coords["longitude"] + 180) % 360) - 180)
            ub_coords["longitude"] = (((ub_coords["longitude"] + 180) % 360) - 180)

            shape = dataset.solution_vars[0].shape
            num_points = np.prod(shape)

            # Initialize dictionaries for solution statistics
            lb_solution = {}
            ub_solution = {}
            solution_mean = {}
            solution_std = {}

            for var in solution_vars:
                var_data = dataset[var].values

                # (Optional) compute stats here if you’re still tracking bounds/mean/std
                lb_solution[var] = np.min(var_data)
                ub_solution[var] = np.max(var_data)
                solution_mean[var] = np.mean(var_data)
                solution_std[var] = np.std(var_data)

        # Store instance variables
        self.read_data_fn = read_data_fn
        self.root_dir = root_dir
        self.solution_vars = solution_vars
        self._dataset = dataset  # Store the modified dataset
        self.dataset_coords = coords
        self.shape = shape
        self.num_points = num_points

        # Set bounds
        self.lower_bounds = {
            'coords': lb_coords,
            'solution': lb_solution
        }
        self.upper_bounds = {
            'coords': ub_coords,
            'solution': ub_solution
        }

        # Store statistics
        self.solution_mean = solution_mean
        self.solution_std = solution_std


    def get_pressure_surface(self, valid_time_idx: int = 0, pressure_level_idx: int = 0, solutions: bool = True) -> tuple:
        """Extract a 2D surface of data at a specific pressure level and time.
        
        This method extracts a 2D horizontal slice of the data at the specified
        pressure level and time index, optionally including the solution variables.
        
        Parameters
        ----------
        valid_time_idx : int, optional
            Index of the time step to extract, by default 0
        pressure_level_idx : int, optional
            Index of the pressure level to extract, by default 0
        solutions : bool, optional
            Whether to include solution variables, by default True
            
        Returns
        -------
        tuple
            If solutions is True, returns a tuple of (coords_dict, solutions_dict)
            If solutions is False, returns coords_dict
            
            coords_dict: dict
                Dictionary with coordinate arrays (valid_time, latitude, longitude)
            solutions_dict: dict
                Dictionary with solution variable arrays at the specified surface
        """

        with self.read_data_fn(self.root_dir) as ds:
            get_pressure_surface_ds = ds.isel(valid_time=valid_time_idx, pressure_level=pressure_level_idx)
            
            # Create meshgrid for all coordinate dimensions
            coord_arrays = [get_pressure_surface_ds[coord].values for coord in get_pressure_surface_ds.coords]
            mesh = np.meshgrid(*coord_arrays, indexing='ij')
            
            # Create dictionary with raveled coordinate arrays
            pressure_surface_coords = {
                coord: mesh[i].ravel() 
                for i, coord in enumerate(get_pressure_surface_ds.coords)
            }
            
            if solutions:
                pressure_surface_solutions = {}
                for var in self.solution_vars:
                    pressure_surface_solutions[var] = get_pressure_surface_ds[var].values.ravel()

                assert len(pressure_surface_coords['valid_time']) == len(pressure_surface_solutions[self.solution_vars[0]]) \
                == len(pressure_surface_coords['latitude']) == len(pressure_surface_coords['longitude'])

                
                return pressure_surface_coords, pressure_surface_solutions
            
            return pressure_surface_coords


    def get_longitude_surface(self, valid_time_idx: int = 0, longitude_idx: int = 0, solutions: bool = True) -> tuple:
        """Extract a 2D surface of data at a specific longitude and time.
        
        This method extracts a 2D vertical slice (latitude-pressure) of the data at the
        specified longitude and time index, optionally including the solution variables.
        
        Parameters
        ----------
        valid_time_idx : int, optional
            Index of the time step to extract, by default 0
        longitude_idx : int, optional
            Index of the longitude to extract, by default 0
        solutions : bool, optional
            Whether to include solution variables, by default True
            
        Returns
        -------
        tuple
            If solutions is True, returns a tuple of (coords_dict, solutions_dict)
            If solutions is False, returns coords_dict
        """
        with self.read_data_fn(self.root_dir) as ds:
            get_longitude_surface_ds = ds.isel(valid_time=valid_time_idx, longitude=longitude_idx)

            # Create meshgrid for all coordinate dimensions
            coord_arrays = [get_longitude_surface_ds[coord].values for coord in get_longitude_surface_ds.coords]
            mesh = np.meshgrid(*coord_arrays, indexing='ij')

            # Create dictionary with raveled coordinate arrays
            longitude_surface_coords = {
                coord: mesh[i].ravel()
                for i, coord in enumerate(get_longitude_surface_ds.coords)
            }

            if solutions:
                longitude_surface_solutions = {}
                for var in self.solution_vars:
                    longitude_surface_solutions[var] = get_longitude_surface_ds[var].values.ravel()

                assert len(longitude_surface_coords['valid_time']) == len(longitude_surface_solutions[self.solution_vars[0]]) \
                == len(longitude_surface_coords['latitude']) == len(longitude_surface_coords['longitude'])


                return longitude_surface_coords, longitude_surface_solutions

            return longitude_surface_coords

    def get_latitude_surface(self, valid_time_idx: int = 0, latitude_idx: int = 0, solutions: bool = True) -> tuple:
        """Extract a 2D surface of data at a specific latitude and time.
        
        This method extracts a 2D vertical slice (longitude-pressure) of the data at the
        specified latitude and time index, optionally including the solution variables.
        
        Parameters
        ----------
        valid_time_idx : int, optional
            Index of the time step to extract, by default 0
        latitude_idx : int, optional
            Index of the latitude to extract, by default 0
        solutions : bool, optional
            Whether to include solution variables, by default True
            
        Returns
        -------
        tuple
            If solutions is True, returns a tuple of (coords_dict, solutions_dict)
            If solutions is False, returns coords_dict
        """

        with self.read_data_fn(self.root_dir) as ds:
            get_latitude_surface_ds = ds.isel(valid_time=valid_time_idx, latitude=latitude_idx)

            # Create meshgrid for all coordinate dimensions
            coord_arrays = [get_latitude_surface_ds[coord].values for coord in get_latitude_surface_ds.coords]
            mesh = np.meshgrid(*coord_arrays, indexing='ij')

            # Create dictionary with raveled coordinate arrays
            latitude_surface_coords = {
                coord: mesh[i].ravel()
                for i, coord in enumerate(get_latitude_surface_ds.coords)
            }

            if solutions:
                latitude_surface_solutions = {}
                for var in self.solution_vars:
                    latitude_surface_solutions[var] = get_latitude_surface_ds[var].values.ravel()

                assert len(latitude_surface_coords['valid_time']) == len(latitude_surface_solutions[self.solution_vars[0]]) \
                == len(latitude_surface_coords['latitude']) == len(latitude_surface_coords['longitude'])


                return latitude_surface_coords, latitude_surface_solutions

            return latitude_surface_coords


    def get_inner_volume(self, solutions: bool = True) -> tuple:
        """Extract the inner volume of the 4D data, excluding boundary points.
        
        This method extracts all points that are not on the domain boundaries,
        effectively creating a volume that excludes the outermost layers in all dimensions.
        
        Parameters
        ----------
        solutions : bool, optional
            Whether to include solution variables, by default True
            
        Returns
        -------
        tuple
            If solutions is True, returns a tuple of (coords_dict, solutions_dict)
            If solutions is False, returns coords_dict
        """

        with self.read_data_fn(self.root_dir) as ds:
            inner_volume_ds = ds.isel(pressure_level=slice(1, -1), latitude=slice(1, -1), longitude=slice(1, -1))

            # Create meshgrid for all coordinate dimensions
            coord_arrays = [inner_volume_ds[coord].values for coord in inner_volume_ds.coords]
            mesh = np.meshgrid(*coord_arrays, indexing='ij')

            # Create dictionary with raveled coordinate arrays
            inner_volume_coords = {
                coord: mesh[i].ravel()
                for i, coord in enumerate(inner_volume_ds.coords)
            }

            if solutions:
                inner_volume_solutions = {}
                for var in self.solution_vars:
                    inner_volume_solutions[var] = inner_volume_ds[var].values.ravel()

                assert len(inner_volume_solutions[self.solution_vars[0]]) \
                == len(inner_volume_coords['latitude']) == len(inner_volume_coords['longitude']) \
                       == len(inner_volume_coords['pressure_level']) == len(inner_volume_coords['valid_time'])


                return inner_volume_coords, inner_volume_solutions

            return inner_volume_coords

    def get_initial_surface(self, solutions: bool = True) -> tuple:
        """Extract the initial time step surface data (first time step).
        
        This method extracts data for the first time step, including both the surface
        (bottom) and top of atmosphere (top pressure level) for the initial conditions.
        
        Parameters
        ----------
        solutions : bool, optional
            Whether to include solution variables, by default True
            
        Returns
        -------
        tuple
            If solutions is True, returns a tuple of (coords_dict, solutions_dict)
            If solutions is False, returns coords_dict
        """

        with self.read_data_fn(self.root_dir) as ds:
            base = ds.isel(valid_time=0, pressure_level=0)
            top = ds.isel(valid_time=0, pressure_level=-1)

            base_coords_arrays = [base[coord].values for coord in base.coords]
            top_coords_arrays = [top[coord].values for coord in top.coords]

            base_mesh = np.meshgrid(*base_coords_arrays, indexing='ij')
            top_mesh = np.meshgrid(*top_coords_arrays, indexing='ij')

            base_coords = [base_mesh[i].ravel() for i in base_mesh]
            top_coords = [top_mesh[i].ravel() for i in top_mesh]

            initial_surface_coords = {
                coord: base_coords[i] + top_coords[i]
                for i, coord in enumerate(base.coords)
            }

            if solutions:
                initial_surface_solutions = {}
                for var in self.solution_vars:
                    initial_surface_solutions[var] = base[var].values.ravel() + top[var].values.ravel()

                assert len(initial_surface_solutions[self.solution_vars[0]]) \
                    == len(initial_surface_coords['latitude']) == len(initial_surface_coords['longitude']) \
                    == len(initial_surface_coords['pressure_level']) == len(initial_surface_coords['valid_time'])

                                                            
                
                return initial_surface_coords, initial_surface_solutions

            return initial_surface_coords

    def get_collocation_points(self, num_samples: int, use_lhs: bool = True) -> Dict[str, np.ndarray]:
        """Generate collocation points within the domain bounds using Latin Hypercube Sampling.
        
        Parameters
        ----------
        num_samples : int
            Number of collocation points to generate
        use_lhs : bool, optional
            Whether to use Latin Hypercube Sampling (True) or uniform random sampling (False).
            Default is True.
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary where keys are coordinate names and values are 1D arrays of sampled points.
            The arrays are aligned such that the i-th element of each array corresponds to the same point.
        """
        # Get lower and upper bounds for all coordinates
        coord_names = ['valid_time', 'pressure_level', 'latitude', 'longitude']
        lb = np.array([self.lower_bounds['coords'][dim] for dim in coord_names])
        ub = np.array([self.upper_bounds['coords'][dim] for dim in coord_names])
        
        if use_lhs:
            # Generate points using Latin Hypercube Sampling in [0,1]^d
            points_01 = lhs(len(coord_names), samples=num_samples)
            # Scale to [lb, ub]
            points = lb + (ub - lb) * points_01
        else:
            # Simple random sampling
            points = np.random.uniform(lb, ub, size=(num_samples, len(coord_names)))
        
        # Convert to dictionary with coordinate names as keys
        collocation_points = {name: points[:, i] for i, name in enumerate(coord_names)}
        
        return collocation_points


    def get_full_data(self):
        """Extract all data points from the dataset as flattened arrays.
        
        This method loads the entire dataset and returns all coordinate points and solution
        variables as flattened 1D arrays. This is useful when you need to work with the
        complete dataset in memory, such as for training machine learning models or
        performing full-domain analysis.
        
        Returns
        -------
        tuple
            A tuple containing two dictionaries:
            - coords: Dictionary where keys are coordinate names (e.g., 'valid_time', 'pressure_level',
              'latitude', 'longitude') and values are 1D numpy arrays of coordinate values.
            - solutions: Dictionary where keys are variable names (as specified in solution_vars)
              and values are 1D numpy arrays of the corresponding variable values.
              
        Example
        -------
        >>> era5 = ERA5MultiData(root_dir, xr.open_dataset, ['z', 't', 'u', 'v', 'w'])
        >>> coords, solutions = era5.get_full_data()
        >>> # Access coordinate arrays
        >>> times = coords['valid_time']
        >>> lats = coords['latitude']
        >>> # Access solution variables
        >>> temperatures = solutions['t']
        >>> u_wind = solutions['u']
        
        Notes
        -----
        - The returned arrays are flattened (raveled) to 1D, with all points in the grid.
        - The order of points is consistent between coordinate and solution arrays.
        - This method loads the entire dataset into memory, which may be large for high-resolution
          or long time series data.
        - The method uses a context manager to ensure proper file handling.
        """
        with self.read_data_fn(self.root_dir) as ds:
            # Create meshgrid of all coordinate points
            coords_array = [ds[coord].values for coord in ds.coords]
            mesh = np.meshgrid(*coords_array, indexing='ij')
            
            # Create dictionary of flattened coordinate arrays
            coords = {coord: mesh[i].ravel() for i, coord in enumerate(ds.coords)}
            
            # Create dictionary of flattened solution variables
            solutions = {var: ds[var].values.ravel() for var in self.solution_vars}

            
            return coords, solutions
