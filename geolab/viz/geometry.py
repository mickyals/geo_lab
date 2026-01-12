"""Geometry generation for atmospheric visualization and sampling."""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from pyDOE3 import lhs


class GeometryGenerator:
    """Generate coordinate grids and random samples for atmospheric fields.

    All methods return tensors in shape (N, input_dim) ordered by coord_labels.
    All coordinates are in physical units (not normalized).
    """

    def __init__(self,
                 coord_domain: Dict[str, Tuple[float, float]],
                 coord_labels: Dict[str, int],
                 resolution: Optional[Dict[str, float]] = None):
        """
        Args:
            coord_domain: Physical ranges for each coordinate.
                         Example: {'longitude': (-180, 180), 'latitude': (-90, 90), ...}
            coord_labels: Mapping of coordinate names to dimension indices.
                         Example: {'longitude': 0, 'latitude': 1, ...}
            resolution: Step size for each coordinate when creating grids.
                       Example: {'longitude': 2.0, 'latitude': 2.0, ...}
                       Optional - required for line(), plane(), volume() but not point()
        """
        self.domain = coord_domain
        self.labels = coord_labels
        self.resolution = resolution if resolution is not None else {}

        # Derive coordinate order from labels
        self.coord_order = sorted(coord_labels.keys(), key=lambda k: coord_labels[k])
        self.input_dim = len(coord_labels)

        # Validate consistency
        if set(coord_domain.keys()) != set(coord_labels.keys()):
            raise ValueError(
                f"Mismatch between domain and labels. "
                f"Domain: {set(coord_domain.keys())}, Labels: {set(coord_labels.keys())}"
            )

    def point(self, **coords) -> torch.Tensor:
        """Generate a single point in space.

        Args:
            **coords: Keyword args specifying coordinate values.
                     Example: longitude=0, latitude=45, pressure_level=500, valid_time=0.5

        Returns:
            (1, input_dim) tensor with coordinates in coord_order

        Example:
            >>> point(longitude=0, latitude=45, pressure_level=500, valid_time=0.5)
            tensor([[0., 45., 500., 0.5]])
        """
        # Validate all coordinates provided
        if set(coords.keys()) != set(self.coord_order):
            raise ValueError(
                f"Must provide exactly one value for each coordinate. "
                f"Expected: {self.coord_order}, Got: {list(coords.keys())}"
            )

        # Validate values are scalars
        for name, value in coords.items():
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"Coordinate '{name}' must be a scalar, got {type(value)}"
                )

            # Validate within domain
            min_val, max_val = self.domain[name]
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"Coordinate '{name}' value {value} outside domain [{min_val}, {max_val}]"
                )

        # Build tensor in coord_order
        point_coords = torch.tensor(
            [[coords[name] for name in self.coord_order]],
            dtype=torch.float32
        )

        return point_coords

    def line(self, axis: str, **fixed_coords) -> torch.Tensor:
        """Generate 1D line along one coordinate axis.

        Args:
            axis: Coordinate name to vary (e.g., 'valid_time')
            **fixed_coords: Fixed values for other coordinates

        Returns:
            (N, input_dim) tensor where N = number of points along axis

        Example:
            >>> line(axis='valid_time', 
            ...      longitude=0, latitude=45, pressure_level=500)
            # Returns time series at fixed spatial location
        """
        # Validate axis
        if axis not in self.coord_order:
            raise ValueError(f"Axis '{axis}' not in coordinates: {self.coord_order}")

        if axis not in self.resolution:
            raise ValueError(f"Resolution not specified for axis '{axis}'")

        # Validate fixed coords
        expected_fixed = set(self.coord_order) - {axis}
        if set(fixed_coords.keys()) != expected_fixed:
            raise ValueError(
                f"Must fix all coordinates except '{axis}'. "
                f"Expected: {expected_fixed}, Got: {set(fixed_coords.keys())}"
            )

        # Generate line along axis
        min_val, max_val = self.domain[axis]
        step = self.resolution[axis]
        axis_values = torch.arange(min_val, max_val + step, step, dtype=torch.float32)
        axis_values = axis_values[axis_values <= max_val]

        # Build full coordinate tensor
        n_points = len(axis_values)
        coords = torch.zeros(n_points, self.input_dim, dtype=torch.float32)

        for i, name in enumerate(self.coord_order):
            if name == axis:
                coords[:, i] = axis_values
            else:
                coords[:, i] = fixed_coords[name]

        return coords

    def plane(self, axes: List[str], **fixed_coords) -> torch.Tensor:
        """Generate 2D plane by varying two coordinate axes.

        Args:
            axes: Two coordinate names to vary (e.g., ['longitude', 'latitude'])
            **fixed_coords: Fixed values for other coordinates

        Returns:
            (N, input_dim) tensor where N = n_axis0 * n_axis1

        Example:
            >>> plane(axes=['longitude', 'latitude'],
            ...       pressure_level=500, valid_time=0.5)
            # Returns horizontal slice at 500 hPa
        """
        # Validate axes
        if len(axes) != 2:
            raise ValueError(f"Must specify exactly 2 axes, got {len(axes)}")

        for axis in axes:
            if axis not in self.coord_order:
                raise ValueError(f"Axis '{axis}' not in coordinates: {self.coord_order}")
            if axis not in self.resolution:
                raise ValueError(f"Resolution not specified for axis '{axis}'")

        # Validate fixed coords
        expected_fixed = set(self.coord_order) - set(axes)
        if set(fixed_coords.keys()) != expected_fixed:
            raise ValueError(
                f"Must fix all coordinates except {axes}. "
                f"Expected: {expected_fixed}, Got: {set(fixed_coords.keys())}"
            )

        # Generate 1D arrays for each axis
        axis_arrays = []
        for axis in axes:
            min_val, max_val = self.domain[axis]
            step = self.resolution[axis]
            arr = torch.arange(min_val, max_val + step, step, dtype=torch.float32)
            arr = arr[arr <= max_val]
            axis_arrays.append(arr)

        # Create meshgrid
        grid = torch.meshgrid(*axis_arrays, indexing='ij')
        n_points = grid[0].numel()

        # Build full coordinate tensor
        coords = torch.zeros(n_points, self.input_dim, dtype=torch.float32)

        for i, name in enumerate(self.coord_order):
            if name in axes:
                axis_idx = axes.index(name)
                coords[:, i] = grid[axis_idx].flatten()
            else:
                coords[:, i] = fixed_coords[name]

        return coords

    def volume(self, axes: List[str], **fixed_coords) -> torch.Tensor:
        """Generate 3D spatial volume (time should be fixed).

        Args:
            axes: Three spatial coordinate names (e.g., ['longitude', 'latitude', 'pressure_level'])
            **fixed_coords: Fixed values for other coordinates (typically time)

        Returns:
            (N, input_dim) tensor where N = n_axis0 * n_axis1 * n_axis2

        Example:
            >>> volume(axes=['longitude', 'latitude', 'pressure_level'],
            ...        valid_time=0.5)
            # Returns 3D spatial volume at fixed time
        """
        # Validate axes
        if len(axes) != 3:
            raise ValueError(f"Must specify exactly 3 axes, got {len(axes)}")

        for axis in axes:
            if axis not in self.coord_order:
                raise ValueError(f"Axis '{axis}' not in coordinates: {self.coord_order}")
            if axis not in self.resolution:
                raise ValueError(f"Resolution not specified for axis '{axis}'")

        # Validate fixed coords
        expected_fixed = set(self.coord_order) - set(axes)
        if set(fixed_coords.keys()) != expected_fixed:
            raise ValueError(
                f"Must fix all coordinates except {axes}. "
                f"Expected: {expected_fixed}, Got: {set(fixed_coords.keys())}"
            )

        # Generate 1D arrays for each axis
        axis_arrays = []
        for axis in axes:
            min_val, max_val = self.domain[axis]
            step = self.resolution[axis]
            arr = torch.arange(min_val, max_val + step, step, dtype=torch.float32)
            arr = arr[arr <= max_val]
            axis_arrays.append(arr)

        # Create meshgrid
        grid = torch.meshgrid(*axis_arrays, indexing='ij')
        n_points = grid[0].numel()

        # Build full coordinate tensor
        coords = torch.zeros(n_points, self.input_dim, dtype=torch.float32)

        for i, name in enumerate(self.coord_order):
            if name in axes:
                axis_idx = axes.index(name)
                coords[:, i] = grid[axis_idx].flatten()
            else:
                coords[:, i] = fixed_coords[name]

        return coords

    def random_spatial_samples(self,
                               n_samples: int,
                               fixed_coords: Dict[str, float],
                               sampling_method: str = 'uniform',
                               seed: Optional[int] = None) -> torch.Tensor:
        """Generate random samples in spatial domain (time fixed).

        Args:
            n_samples: Number of random points to generate
            fixed_coords: Coordinates to keep fixed (e.g., {'valid_time': 0.5})
            sampling_method: 'uniform' or 'gaussian' (centered at domain midpoint)
            seed: Random seed for reproducibility

        Returns:
            (n_samples, input_dim) tensor with random spatial coordinates

        Example:
            >>> random_spatial_samples(n_samples=1000,
            ...                        fixed_coords={'valid_time': 0.5},
            ...                        sampling_method='uniform')
            # Returns 1000 random points in 3D space at t=0.5
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Determine which coords to vary
        varying_coords = [c for c in self.coord_order if c not in fixed_coords]

        # Generate random samples in [0, 1]
        if sampling_method == 'uniform':
            samples_normalized = torch.rand(n_samples, len(varying_coords))
        elif sampling_method == 'gaussian':
            # ~95% of samples in [-0.1, 1.1] (clipped to [0, 1])
            samples_normalized = torch.randn(n_samples, len(varying_coords)) * 0.3 + 0.5
            samples_normalized = torch.clamp(samples_normalized, 0.0, 1.0)
        elif sampling_method == 'lhs':
            # Latin Hypercube Sampling for better space coverage
            samples_normalized = lhs(n=len(varying_coords), samples=n_samples)
            samples_normalized = torch.from_numpy(samples_normalized).float()

        else:
            raise ValueError(
                f"Unknown sampling method: {sampling_method}. "
                f"Choose from: 'uniform', 'normal', 'lhs'"
            )

        # Convert to real coordinates
        samples_real = self._convert_to_real_coord_values(
            samples_normalized, varying_coords
        )

        # Build full coordinate tensor
        coords = torch.zeros(n_samples, self.input_dim, dtype=torch.float32)

        varying_idx = 0
        for i, name in enumerate(self.coord_order):
            if name in varying_coords:
                coords[:, i] = samples_real[:, varying_idx]
                varying_idx += 1
            else:
                coords[:, i] = fixed_coords[name]

        return coords

    def random_spatiotemporal_samples(self,
                                      n_samples: int,
                                      use_lhs: bool = True,
                                      seed: Optional[int] = None) -> torch.Tensor:
        """Generate random samples in full 4D spatiotemporal domain.

        Args:
            n_samples: Number of random points to generate
            use_lhs: If True, use Latin Hypercube Sampling (better space coverage)
                    If False, use uniform random sampling
            seed: Random seed for reproducibility

        Returns:
            (n_samples, input_dim) tensor with random 4D coordinates

        Example:
            >>> random_spatiotemporal_samples(n_samples=10000, use_lhs=True)
            # Returns 10000 LHS-sampled points in 4D space
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate samples in [0, 1]^input_dim
        if use_lhs:
            samples_normalized = lhs(n=self.input_dim, samples=n_samples)
            samples_normalized = torch.from_numpy(samples_normalized).float()
        else:
            samples_normalized = torch.rand(n_samples, self.input_dim)

        # Convert to real coordinates
        samples_real = self._convert_to_real_coord_values(
            samples_normalized, self.coord_order
        )

        return samples_real

    def _convert_to_real_coord_values(self,
                                      samples_normalized: torch.Tensor,
                                      coord_names: List[str]) -> torch.Tensor:
        """Convert normalized [0, 1] samples to real coordinate values.

        Args:
            samples_normalized: (N, D) tensor with values in [0, 1]
            coord_names: List of coordinate names corresponding to columns

        Returns:
            (N, D) tensor with values in real coordinate ranges
        """
        samples_real = torch.zeros_like(samples_normalized)

        for i, name in enumerate(coord_names):
            min_val, max_val = self.domain[name]
            samples_real[:, i] = samples_normalized[:, i] * (max_val - min_val) + min_val

        return samples_real

    def compute_num_points(self, axes: List[str]) -> int:
        """Compute number of grid points for given axes with current resolution.

        Args:
            axes: List of coordinate names to vary

        Returns:
            Total number of grid points (product of points along each axis)

        Example:
            >>> compute_num_points(['valid_time'])  # Line
            100
            >>> compute_num_points(['longitude', 'latitude'])  # Plane
            16380
            >>> compute_num_points(['longitude', 'latitude', 'pressure_level'])  # Volume
            147420
        """
        if self.resolution is None or len(self.resolution) == 0:
            raise ValueError("Resolution not set, cannot compute num_points")

        # Validate axes
        for axis in axes:
            if axis not in self.coord_order:
                raise ValueError(f"Axis '{axis}' not in coordinates: {self.coord_order}")
            if axis not in self.resolution:
                raise ValueError(f"Resolution not specified for axis '{axis}'")

        # Compute number of points along each axis
        n_points = 1
        for axis in axes:
            min_val, max_val = self.domain[axis]
            step = self.resolution[axis]
            n = len(torch.arange(min_val, max_val + step, step))
            n_points *= n

        return n_points

    def compute_space_size(self, axes: List[str]) -> Dict[str, int]:
        """Compute grid dimensions for given axes with current resolution.

        Args:
            axes: List of coordinate names to vary

        Returns:
            Dict mapping coordinate names to number of grid points

        Example:
            >>> compute_space_size(['longitude', 'latitude'])
            {'longitude': 180, 'latitude': 91}
        """
        if self.resolution is None or len(self.resolution) == 0:
            raise ValueError("Resolution not set, cannot compute space_size")

        # Validate axes
        for axis in axes:
            if axis not in self.coord_order:
                raise ValueError(f"Axis '{axis}' not in coordinates: {self.coord_order}")
            if axis not in self.resolution:
                raise ValueError(f"Resolution not specified for axis '{axis}'")

        size = {}
        for axis in axes:
            min_val, max_val = self.domain[axis]
            step = self.resolution[axis]
            n = len(torch.arange(min_val, max_val + step, step))
            size[axis] = n

        return size

    @property
    def space_size(self) -> Dict[str, int]:
        """Grid dimensions for ALL coordinates with current resolution.

        Returns:
            Dict mapping coordinate names to number of grid points (or None if no resolution)
        """
        if self.resolution is None or len(self.resolution) == 0:
            raise ValueError("Resolution not set, cannot compute space_size")

        size = {}
        for coord in self.coord_order:
            if coord in self.resolution:
                min_val, max_val = self.domain[coord]
                step = self.resolution[coord]
                n = len(torch.arange(min_val, max_val + step, step))
                size[coord] = n
            else:
                size[coord] = None  # Not gridded

        return size

    @property
    def total_grid_points(self) -> int:
        """Total number of points if gridding ALL coordinates with current resolution.

        Returns:
            Product of grid sizes across all coordinates with resolution specified
        """
        if self.resolution is None or len(self.resolution) == 0:
            raise ValueError("Resolution not set, cannot compute total_grid_points")

        axes_with_resolution = [c for c in self.coord_order if c in self.resolution]
        return self.compute_num_points(axes_with_resolution)

    def __repr__(self) -> str:
        """String representation of geometry generator."""
        return (
            f"GeometryGenerator(\n"
            f"  domain={self.domain},\n"
            f"  coord_order={self.coord_order},\n"
            f"  resolution={self.resolution}\n"
            f")"
        )