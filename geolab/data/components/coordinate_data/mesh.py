"""
Mesh handling module for geospatial data processing.

This module provides a flexible framework for working with spatial and spatio-temporal
mesh data, with support for various boundary conditions, data sampling strategies,
and coordinate systems. It is designed to handle both structured and unstructured
grids, with special support for geospatial data through integration with xarray.

Core Functionality:
- Mesh representation and manipulation through the MeshBase class
- Boundary condition handling for all spatial dimensions (north, south, east, west, lower, upper)
- Support for both spatial and temporal dimensions
- Data sampling methods including Latin Hypercube Sampling (LHS)
- Integration with xarray for geospatial data handling

Key Classes:
    MeshBase: Abstract base class defining the interface for all mesh types
    XarrayMesh: Implementation for working with xarray datasets
    GeoSpatialDomain: Handles spatial domain representation (imported from space.py)

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

from typing import Callable, List, Optional, Dict, Tuple, Any, Union

import numpy as np
import pyDOE as lhs
import torch

from geolab.data.components.coordinate_data.space import GeoSpatialDomain


class MeshBase:
    """
    Base class for handling spatial and temporal mesh data with boundary conditions.
    
    This class provides a foundation for working with spatial and spatio-temporal
    data, including methods for accessing boundary points, generating sampling points,
    and handling both spatial and temporal dimensions of the mesh.
    
    The class is designed to be extended by specific mesh implementations (e.g., Xarray)
    that handle different data formats and storage backends.
    
    Attributes:
        mesh (Optional[GeoSpatialDomain]): The underlying spatial domain handler that manages
            the spatial grid and coordinate system.
        solution_domain (Dict[str, np.ndarray]): Dictionary mapping variable names to their
            corresponding solution arrays.
        lower_bounds (Dict[str, Dict]): Lower bounds for different domains:
            - 'spatial': Dict[str, float] - Minimum values for each spatial dimension
            - 'time': Optional[Dict[str, float]] - Minimum time value (if temporal data exists)
            - 'solution': Dict[str, float] - Minimum values for each solution variable
        upper_bounds (Dict[str, Dict]): Upper bounds for different domains, with the same
            structure as lower_bounds but containing maximum values.
    """

    def __init__(self):
        """
        Initialize the base mesh with default attributes.
        
        Sets up the basic data structures for storing mesh and solution data,
        and initializes bounds for spatial, temporal, and solution domains.
        """
        # Initialize the spatial domain handler (to be set by child classes)
        self.mesh: Optional[GeoSpatialDomain] = None
        
        # Dictionary to store solution variables (e.g., temperature, pressure)
        # Key: variable name (str), Value: numpy array containing the data
        self.solution_domain: Dict[str, np.ndarray] = {}
        
        # Store minimum bounds for spatial, temporal, and solution data
        self.lower_bounds: Dict[str, Optional[Dict]] = {
            'spatial': {},      # Will contain min values for each spatial dimension
            'time': None,       # Will contain min time value if temporal data exists
            'solution': {}      # Will contain min values for each solution variable
        }
        
        # Store maximum bounds (same structure as lower_bounds)
        self.upper_bounds: Dict[str, Optional[Dict]] = {
            'spatial': {},      # Will contain max values for each spatial dimension
            'time': None,       # Will contain max time value if temporal data exists
            'solution': {}      # Will contain max values for each solution variable
        }

    def on_initial_boundary(
        self, 
        solution_vars: List[str], 
        is_time: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Extract data points at the initial time (t0) along all boundaries.
        
        This method is particularly useful for setting initial conditions in
        time-dependent simulations where you need the state at t=0.

        Parameters
        ----------
        solution_vars : List[str]
            List of variable names to extract from the solution domain.
        is_time : bool, optional
            Whether the mesh includes a temporal component, by default False.

        Returns
        -------
        Tuple containing:
            np.ndarray: Spatial coordinates of points at t0, shape (n_points, n_dims).
            Optional[np.ndarray]: Time values (if is_time=True), shape (n_points, 1).
            Dict[str, np.ndarray]: Solution values for each variable at t0.
            
        Notes
        -----
        - For temporal meshes (is_time=True), returns the first time slice (t=0).
        - For static meshes, returns all points with time_domain=None.
        """
        # Stack the spatial mesh dimensions into a single array
        # Exclude the last dimension (time) if this is a temporal mesh
        mesh_arrays = self.mesh.load_mesh
        
        if not is_time:
            # For static mesh: stack all spatial dimensions
            spatial_mesh = np.stack(mesh_arrays, axis=-1)
        else:
            # For temporal mesh: stack all dimensions including time
            spatial_mesh = np.stack(mesh_arrays, axis=-1)
        
        # Handle temporal data if specified
        if is_time:
            # Get spatial coordinates at first time step (t=0)
            # shape: (n_points, n_spatial_dims)
            spatial_domain = np.squeeze(spatial_mesh[:, 0:1, :], axis=1)
            
            # Get time values (first time step)
            time_domain = mesh_arrays[-1][:, 0]  # First time slice
            
            # Extract solution variables at t=0
            solution_domain = {
                name: self.solution_domain[name][:, 0:1]  # First time slice
                for name in solution_vars
                if name in self.solution_domain
            }
        else:
            # For static mesh, use all points
            spatial_domain = spatial_mesh
            time_domain = None
            
            # Extract all solution variables
            solution_domain = {
                name: self.solution_domain[name] 
                for name in solution_vars 
                if name in self.solution_domain
            }

        return spatial_domain, time_domain, solution_domain

    def on_north_boundary(self, solution_names: list) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Return points at the northern spatial boundary (max Y).

        This method returns the points at the northern boundary of the spatial domain,
        along with the corresponding time and solution values.

        Parameters
        ----------
        solution_names : List[str]
            List of variable names to extract from the solution domain.

        Returns
        -------
        Tuple containing:
            np.ndarray: Spatial coordinates of points at the northern boundary, shape (n_points, n_dims).
            Optional[np.ndarray]: Time values (if mesh has a temporal component), shape (n_points, 1).
            Dict[str, np.ndarray]: Solution values for each variable at the northern boundary.
        """

        # Stack the spatial mesh dimensions into a single array
        spatial_mesh = np.stack(self.mesh.load_mesh[:-1], axis=-1)

        # Get spatial coordinates at the northern boundary (max Y)
        # shape: (n_points, n_spatial_dims)
        spatial_domain = spatial_mesh[:, -1:, :]  # max along Y-axis (north)
        spatial_domain = np.squeeze(spatial_domain, axis=1)

        # Handle temporal data if specified
        time_domain = self.mesh.load_mesh[-1] if self.mesh.temporal_bounds else None

        # Extract solution variables at the northern boundary
        solution_domain = {
            name: self.solution_domain[name][:, -1:]  # max along Y-axis (north)
            for name in solution_names
            if name in self.solution_domain
        }

        return spatial_domain, time_domain, solution_domain

    def on_south_boundary(self, solution_names: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Return points at the southern spatial boundary (min Y).

        This method returns the points at the southern boundary of the spatial domain,
        along with the corresponding time and solution values.

        Parameters
        ----------
        solution_names : List[str]
            List of variable names to extract from the solution domain.

        Returns
        -------
        Tuple containing:
            np.ndarray: Spatial coordinates of points at the southern boundary, shape (n_points, n_dims).
            Optional[np.ndarray]: Time values (if mesh has a temporal component), shape (n_points, 1).
            Dict[str, np.ndarray]: Solution values for each variable at the southern boundary.

        Notes
        -----
        The southern boundary is defined as the minimum along the Y-axis.
        """
        # Get points at the southern boundary (min along Y-axis)
        spatial_mesh = np.stack(self.mesh.load_mesh[:-1], axis=-1)
        spatial_domain = spatial_mesh[:, :1, :]  # min along Y-axis (south)
        spatial_domain = np.squeeze(spatial_domain, axis=1)

        # Get time values if mesh has a temporal component
        time_domain = None
        if self.mesh.temporal_bounds:
            time_domain = self.mesh.load_mesh[-1]

        # Extract solution values at the southern boundary
        solution_domain = {}
        for name in solution_names:
            if name in self.solution_domain:
                solution_domain[name] = self.solution_domain[name][:, :1]  # min along Y-axis (south)

        return spatial_domain, time_domain, solution_domain

    def on_east_boundary(self, solution_names: list) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Return points at the eastern spatial boundary (max X).

        This method returns the points at the eastern boundary of the spatial domain,
        along with the corresponding time and solution values.

        Parameters
        ----------
        solution_names : List[str]
            List of variable names to extract from the solution domain.

        Returns
        -------
        Tuple containing:
            np.ndarray: Spatial coordinates of points at the eastern boundary, shape (n_points, n_dims).
            Optional[np.ndarray]: Time values (if mesh has a temporal component), shape (n_points, 1).
            Dict[str, np.ndarray]: Solution values for each variable at the eastern boundary.
        """
        # Get spatial coordinates at the eastern boundary (max X)
        spatial_mesh = np.stack(self.mesh.load_mesh[:-1], axis=-1)
        spatial_domain = spatial_mesh[-1:, :, :]  # max along X-axis (east)
        spatial_domain = np.squeeze(spatial_domain, axis=0)

        # Handle temporal data if specified
        time_domain = self.mesh.load_mesh[-1] if self.mesh.temporal_bounds else None

        # Extract solution variables at the eastern boundary
        solution_domain = {
            name: self.solution_domain[name][-1:, :]  # max along X-axis (east)
            for name in solution_names
            if name in self.solution_domain
        }

        return spatial_domain, time_domain, solution_domain

    def on_west_boundary(self, solution_names: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Return points at the western spatial boundary (min X).

        This method returns the points at the western boundary of the spatial domain,
        along with the corresponding time and solution values.

        Parameters
        ----------
        solution_names : List[str]
            List of variable names to extract from the solution domain.

        Returns
        -------
        Tuple containing:
            np.ndarray: Spatial coordinates of points at the western boundary, shape (n_points, n_dims).
            Optional[np.ndarray]: Time values (if mesh has a temporal component), shape (n_points, 1).
            Dict[str, np.ndarray]: Solution values for each variable at the western boundary.
        """
        # Get spatial coordinates at the western boundary (min X)
        spatial_mesh = np.stack(self.mesh.load_mesh[:-1], axis=-1)
        spatial_domain = spatial_mesh[:1, :, :]  # min along X-axis (west)
        spatial_domain = np.squeeze(spatial_domain, axis=0)

        # Handle temporal data if specified
        time_domain = self.mesh.load_mesh[-1] if self.mesh.temporal_bounds else None

        # Extract solution variables at the western boundary
        solution_domain = {
            name: self.solution_domain[name][:1, :]  # min along X-axis (west)
            for name in solution_names
            if name in self.solution_domain
        }

        return spatial_domain, time_domain, solution_domain

    def on_lower_boundary(self, solution_names: list) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Return points at the lower boundary in the vertical dimension (min Z).

        This method returns the points at the lower boundary of the spatial domain,
        along with the corresponding time and solution values.

        Parameters
        ----------
        solution_names : List[str]
            List of variable names to extract from the solution domain.

        Returns
        -------
        Tuple containing:
            np.ndarray: Spatial coordinates of points at the lower boundary, shape (n_points, n_dims).
            Optional[np.ndarray]: Time values (if mesh has a temporal component), shape (n_points, 1).
            Dict[str, np.ndarray]: Solution values for each variable at the lower boundary.
        """
        # Get spatial coordinates at the lower boundary (min Z)
        spatial_mesh = np.stack(self.mesh.load_mesh[:-1], axis=-1)
        spatial_domain = spatial_mesh[:, :, :1]  # min along Z-axis (lower)
        spatial_domain = np.squeeze(spatial_domain, axis=2)

        # Handle temporal data if specified
        time_domain = self.mesh.load_mesh[-1] if self.mesh.temporal_bounds else None

        # Extract solution variables at the lower boundary
        solution_domain = {
            name: self.solution_domain[name][:, :, :1]  # min along Z-axis (lower)
            for name in solution_names
            if name in self.solution_domain
        }

        return spatial_domain, time_domain, solution_domain

    def on_upper_boundary(self, solution_names: list) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
        """
        Return points at the upper boundary in the vertical dimension (max Z).

        This method returns the points at the upper boundary of the spatial domain,
        along with the corresponding time and solution values.

        Parameters
        ----------
        solution_names : List[str]
            List of variable names to extract from the solution domain.

        Returns
        -------
        Tuple containing:
            np.ndarray: Spatial coordinates of points at the upper boundary, shape (n_points, n_dims).
            Optional[np.ndarray]: Time values (if mesh has a temporal component), shape (n_points, 1).
            Dict[str, np.ndarray]: Solution values for each variable at the upper boundary.
        """
        # Get spatial coordinates at the upper boundary (max Z)
        spatial_mesh = np.stack(self.mesh.load_mesh[:-1], axis=-1)
        spatial_domain = spatial_mesh[:, :, -1:]  # max along Z-axis (upper)
        spatial_domain = np.squeeze(spatial_domain, axis=2)

        # Handle temporal data if specified
        time_domain = self.mesh.load_mesh[-1] if self.mesh.temporal_bounds else None

        # Extract solution variables at the upper boundary
        solution_domain = {
            name: self.solution_domain[name][:, :, -1:]  # max along Z-axis (upper)
            for name in solution_names
            if name in self.solution_domain
        }

        return spatial_domain, time_domain, solution_domain

    def collection_points(
            self,
            N_f: int,
            use_lhs: bool = True,
            solution_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """
        Generate a collection of points for data collection.

        Parameters
        ----------
        N_f : int
            Number of points to collect.
        use_lhs : bool
            Whether to use Latin Hypercube Sampling or all points from mesh.
        solution_names : list of str, optional
            Names of solution variables to return if use_lhs=False.

        Returns
        -------
        spatial_domain : np.ndarray
            Array of spatial points (N_f, spatial_dim)
        time_domain : np.ndarray or None
            Array of time points (N_f, 1) if mesh has a temporal dimension; else None
        solution_domain : dict[str, np.ndarray] or None
            Dictionary of solution values at the points if use_lhs=False; else None
        """

        # Get spatial and time dimensions
        spatial_dims = list(self.lower_bounds['spatial'].keys())
        time_dim = list(self.lower_bounds['time'].keys())[0] if self.lower_bounds['time'] else None

        if use_lhs:
            # Construct arrays for LHS based on bounds dictionaries
            # Lower and upper bounds for spatial dimensions
            spatial_lb = np.array([self.lower_bounds['spatial'][dim] for dim in spatial_dims])
            spatial_ub = np.array([self.upper_bounds['spatial'][dim] for dim in spatial_dims])

            # Lower and upper bounds for time dimension
            if time_dim:
                time_lb = self.lower_bounds['time'][time_dim]
                time_ub = self.upper_bounds['time'][time_dim]

                # Generate LHS points
                f = lhs(len(spatial_dims) + 1, N_f)  # +1 for time
                spatial_domain = spatial_lb + (spatial_ub - spatial_lb) * f[:, :-1]
                time_domain = time_lb + (time_ub - time_lb) * f[:, -1][:, None]
            else:
                # Generate LHS points
                f = lhs(len(spatial_dims), N_f)
                spatial_domain = spatial_lb + (spatial_ub - spatial_lb) * f
                time_domain = None

            solution_domain = None  # collocation points, no solution needed
        else:
            # Use all points from flattened mesh
            spatial_domain, time_domain, solution_domain_full = self.flatten_mesh(solution_names)
            solution_domain = solution_domain_full if solution_names else None

        return spatial_domain, time_domain, solution_domain

    def flatten_mesh(self, solution_names: Optional[List[str]] = None) -> Tuple[
        np.ndarray, Optional[np.ndarray], Optional[Dict[str, np.ndarray]]]:
        """
        Flatten the mesh data for training.

        Parameters
        ----------
        solution_names : List[str], optional
            Names of the solution variables to return. If None, no solution values are returned.

        Returns
        -------
        spatial_domain : np.ndarray
            Flattened spatial points of shape (N, spatial_dim)
        time_domain : np.ndarray or None
            Flattened time points of shape (N, 1) if mesh has temporal component; else None
        solution_domain : dict[str, np.ndarray] or None
            Flattened solution values at each point, if solution_names is provided
        """
        mesh_arrays = self.mesh.load_mesh  # List of arrays: spatial dims (+ optional time)
        spatial_arrays = mesh_arrays[:-1] if self.mesh.temporal_bounds else mesh_arrays
        spatial_domain = np.column_stack([arr.flatten() for arr in spatial_arrays])

        if self.mesh.temporal_bounds:
            time_domain = mesh_arrays[-1].flatten()[:, None]
        else:
            time_domain = None

        solution_domain = None
        if solution_names is not None:
            solution_domain = {
                name: self.solution_domain[name].flatten()[:, None] for name in solution_names
            }

        return spatial_domain, time_domain, solution_domain


class XarrayMesh(MeshBase):
    """
    Mesh class for handling xarray datasets with spatial and temporal dimensions.
    
    This class extends MeshBase to work specifically with xarray datasets, providing
    methods to load and process spatial and temporal data from various file formats.
    
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
        """Initialize the Xarray mesh with the given dataset and dimensions."""
        super().__init__()
        
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
        
        # Extract solution variables
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
        
        # Determine shape including time if it exists
        shape = [len(arr) for arr in spatial_domain]
        if time_dim is not None:
            shape.append(len(temporal_domain))
        
        # Initialize base class attributes
        self.solution_domain = solution_domain
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
            temporal_domain=temporal_domain
        )