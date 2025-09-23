from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np


class GeoSpatialDomain:
    def __init__(
            self,
            spatial_domain: List[np.ndarray],
            shape: List[int],
            temporal_domain: Optional[List[np.ndarray]] = None,
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
        self.spatial_bounds = spatial_domain
        self.temporal_bounds = temporal_domain if temporal_domain else None
        self.shape = shape
        self.indexing = indexing
        self.sparse = sparse
        self.dtype = dtype
        self._mesh = None

    def generate_mesh(self) -> List[np.ndarray]:
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


        # Add time dimension if needed
        if self.temporal_bounds is not None:
            axes.append(self.time_domain) #

        # Generate mesh
        self._mesh = np.meshgrid(*axes, indexing=self.indexing, sparse=self.sparse) # the shape of self._mesh is the same as self.shape
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
        mesh = self.mesh
        if isinstance(index, (int, slice)):
            return np.column_stack([dim.flat[index] for dim in mesh])
        elif isinstance(index, tuple):
            return np.column_stack([dim[index] for dim in mesh])
        else:
            raise TypeError("Index must be int, slice, or tuple")