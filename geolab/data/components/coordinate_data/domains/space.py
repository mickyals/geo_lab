"""
Spatial domain handling for geospatial data processing.

This module provides the GeoSpatialDomain class for managing spatial and temporal
domains in geospatial applications. It handles coordinate transformations,
mesh generation, and provides utilities for working with spatial data in both
structured and unstructured formats.

Core Functionality:
- Spatial domain representation and management
- Coordinate system handling
- Mesh generation and manipulation
- Support for both spatial and temporal dimensions
- Integration with numpy for numerical operations

Key Classes:
    GeoSpatialDomain: Manages spatial domains and coordinate transformations

Assumptions:
1. Spatial coordinates are provided as a list of 1D numpy arrays
2. The order of spatial dimensions follows (x, y, z) for 3D or (x, y) for 2D
3. Time is treated as a separate, optional dimension
4. All coordinate arrays must have consistent shapes
5. For temporal data, time values must be provided as a 1D numpy array

Example Usage:
    ```python
    # Create a 2D spatial domain
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 20, 200)
    spatial_domain = [x, y]
    
    # Create a GeoSpatialDomain instance
    domain = GeoSpatialDomain(
        spatial_domain=spatial_domain,
        shape=[100, 200],
        temporal_domain=None
    )
    
    # Generate the mesh grid
    mesh = domain.generate_mesh()
    or
    mesh = domain.load_mesh # lazy loading
    ```
"""

from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np


class GeoSpatialDomain:
    """
    A class to represent and manipulate spatial domains with optional temporal component.
    
    This class provides functionality to work with spatial data, including mesh generation,
    coordinate transformations, and domain management. It supports both spatial and
    spatio-temporal data through a unified interface.
    
    Attributes:
        spatial_bounds (List[np.ndarray]): List of coordinate arrays for each spatial dimension
        temporal_bounds (Optional[np.ndarray]): Optional array of time values
        shape (List[int]): Shape of the spatial domain (excluding time)
        indexing (str): Indexing convention ('ij' for matrix, 'xy' for Cartesian)
        sparse (bool): Whether to use sparse representation for large meshes
        dtype: Data type of the coordinate arrays
        _mesh (Optional[List[np.ndarray]]): Cached mesh grid
    """
    def __init__(
            self,
            spatial_domain: List[np.ndarray],
            shape: List[int],
            temporal_domain: Optional[List[np.ndarray]] = None,
            spatiotemporal_dims: Optional[List[str]] = None,
            indexing: str = 'ij',
            sparse: bool = False,
            dtype: type = np.float64
    ):
        """Initialize a GeoSpatialDomain.

        Args:
            spatial_bounds: List of (min, max) tuples for each spatial dimension
            temporal_bounds: Optional (min, max) tuple for time dimension
            shape: Number of points in each dimension (including time if present)
            indexing: 'ij' for matrix indexing or 'xy' for Cartesian indexing
            sparse: If True, return sparse meshgrid to save memory
            dtype: Data type of the coordinate arrays
        """
        self.spatial_domain = spatial_domain
        self.temporal_domain = temporal_domain if temporal_domain is not None else None
        self.spatiotemporal_dims = spatiotemporal_dims if spatiotemporal_dims is not None else None
        self.shape = shape
        self.indexing = indexing
        self.sparse = sparse
        self.dtype = dtype
        self._mesh = None

    def generate_mesh(self) -> Dict[str, np.ndarray]:
        """Generate coordinate mesh.

        Returns:
            List of coordinate arrays, one for each dimension.
            For 3D spatial + time: [X, Y, Z, T]
            For 3D spatial: [X, Y, Z]
            For 2D spatial: [X, Y]
        """
        # Generate spatial coordinates
        axes = [spatial_coords for spatial_coords in self.spatial_domain
            ] # axes is a list of numpy array, each array is length n in self.shape
        names = self.spatiotemporal_dims


        # Add time dimension if needed
        if self.temporal_domain is not None:
            axes.append(self.temporal_domain) #


        # Generate mesh
        mesh = np.meshgrid(*axes, indexing=self.indexing, sparse=self.sparse) # the shape of self._mesh is the same as self.shape

        self._mesh = {name: arr for name, arr in zip(names, mesh)}
        return self._mesh

    @property
    def load_mesh(self) -> List[np.ndarray]:
        """Lazy-loading mesh property."""
        if self._mesh is None:
            self.generate_mesh()
        return self._mesh

    def __len__(self) -> int:
        """Total number of points in the mesh."""
        return int(np.prod(self.shape))

    def __getitem__(self, index: Union[int, slice, Tuple]) -> np.ndarray:
        """Get point(s) from the mesh.

        Args:
            index: Integer, slice, or tuple of indices/slices

        Returns:
            Array of points with shape (..., n_dims)
        """
        mesh = self.load_mesh  # This will generate the mesh if it doesn't exist
        if isinstance(index, (int, slice)):
            return np.column_stack([dim.flat[index] for dim in mesh])
        elif isinstance(index, tuple):
            return np.column_stack([dim[index] for dim in mesh])
        else:
            raise TypeError("Index must be int, slice, or tuple")