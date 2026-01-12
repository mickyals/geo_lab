"""Data aggregation for visualization."""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


class DataAggregator:
    """Aggregate data along coordinate dimensions.

    All methods assume data and coords are aligned (same ordering).
    """

    def __init__(self, coord_labels: Dict[str, int]):
        """
        Args:
            coord_labels: Mapping of coordinate names to dimension indices
                         Example: {'longitude': 0, 'latitude': 1, ...}
        """
        self.coord_labels = coord_labels
        self.coord_order = sorted(coord_labels.keys(), key=lambda k: coord_labels[k])

    def reduce_axis(self,
                    data: torch.Tensor,
                    coords: torch.Tensor,
                    axis: str,
                    method: str = 'mean') -> Tuple[torch.Tensor, torch.Tensor]:
        """Reduce along one coordinate axis.

        Args:
            data: (N, num_vars) data values
            coords: (N, 4) coordinates
            axis: Coordinate name to reduce over (e.g., 'longitude')
            method: Aggregation method - 'mean', 'std', 'min', 'max', 'median'

        Returns:
            (aggregated_data, aggregated_coords) where the specified axis is removed

        Example:
            >>> # Average over longitude
            >>> data_agg, coords_agg = aggregator.reduce_axis(
            ...     data, coords, axis='longitude', method='mean'
            ... )
        """
        if axis not in self.coord_labels:
            raise ValueError(f"Unknown axis: {axis}. Available: {list(self.coord_labels.keys())}")

        axis_idx = self.coord_labels[axis]

        # Get unique combinations of other coordinates
        other_dims = [i for i in range(coords.shape[1]) if i != axis_idx]
        coords_other = coords[:, other_dims]

        # Find unique coordinate combinations
        unique_coords, inverse_indices = self._unique_rows(coords_other)

        # Aggregate data
        n_unique = unique_coords.shape[0]
        num_vars = data.shape[1] if data.ndim > 1 else 1

        if data.ndim == 1:
            aggregated = torch.zeros(n_unique)
        else:
            aggregated = torch.zeros(n_unique, num_vars)

        for i in range(n_unique):
            mask = inverse_indices == i
            values = data[mask]

            if method == 'mean':
                aggregated[i] = values.mean(dim=0)
            elif method == 'std':
                aggregated[i] = values.std(dim=0)
            elif method == 'min':
                aggregated[i] = values.min(dim=0)[0]
            elif method == 'max':
                aggregated[i] = values.max(dim=0)[0]
            elif method == 'median':
                aggregated[i] = values.median(dim=0)[0]
            else:
                raise ValueError(f"Unknown method: {method}")

        return aggregated, unique_coords

    def zonal_mean(self,
                   data: torch.Tensor,
                   coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Average over longitude (zonal mean).

        Args:
            data: (N, num_vars) data values
            coords: (N, 4) coordinates

        Returns:
            (zonal_data, zonal_coords) with longitude dimension removed

        Example:
            >>> zonal_data, zonal_coords = aggregator.zonal_mean(data, coords)
            >>> # zonal_coords now has shape (M, 3) - no longitude
        """
        return self.reduce_axis(data, coords, axis='longitude', method='mean')

    def temporal_mean(self,
                      data: torch.Tensor,
                      coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Average over time.

        Args:
            data: (N, num_vars) data values
            coords: (N, 4) coordinates

        Returns:
            (time_avg_data, spatial_coords) with time dimension removed
        """
        return self.reduce_axis(data, coords, axis='valid_time', method='mean')

    def spatial_bin(self,
                    data: torch.Tensor,
                    coords: torch.Tensor,
                    bins: Dict[str, int],
                    method: str = 'mean') -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Bin data spatially for heatmaps.

        Args:
            data: (N,) or (N, num_vars) values to aggregate
            coords: (N, 4) coordinates
            bins: Dict specifying number of bins per dimension
                 Example: {'longitude': 72, 'latitude': 36}
            method: Aggregation method - 'mean', 'sum', 'count'

        Returns:
            (binned_data, bin_edges) where:
                binned_data: Gridded array
                bin_edges: Dict mapping coord names to edge arrays

        Example:
            >>> binned, edges = aggregator.spatial_bin(
            ...     data, coords,
            ...     bins={'longitude': 72, 'latitude': 36},
            ...     method='mean'
            ... )
        """
        # Convert to numpy for histogram operations
        coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
        data_np = data.cpu().numpy() if torch.is_tensor(data) else data

        if data_np.ndim == 1:
            # Single variable
            return self._spatial_bin_single(data_np, coords_np, bins, method)
        else:
            # Multiple variables - bin each separately
            binned_vars = []
            for i in range(data_np.shape[1]):
                binned, edges = self._spatial_bin_single(
                    data_np[:, i], coords_np, bins, method
                )
                binned_vars.append(binned)

            # Stack along new dimension
            binned_data = np.stack(binned_vars, axis=-1)
            return binned_data, edges

    def _spatial_bin_single(self,
                            data: np.ndarray,
                            coords: np.ndarray,
                            bins: Dict[str, int],
                            method: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Bin single variable spatially (internal helper)."""
        # Currently only support 2D binning (lon-lat)
        if set(bins.keys()) == {'longitude', 'latitude'}:
            lon_idx = self.coord_labels['longitude']
            lat_idx = self.coord_labels['latitude']

            lons = coords[:, lon_idx]
            lats = coords[:, lat_idx]

            # Create histogram
            H, lon_edges, lat_edges = np.histogram2d(
                lons, lats,
                bins=[bins['longitude'], bins['latitude']],
                weights=data
            )

            if method == 'mean':
                # Normalize by counts
                counts, _, _ = np.histogram2d(
                    lons, lats,
                    bins=[bins['longitude'], bins['latitude']]
                )
                H = np.divide(H, counts, where=counts > 0, out=np.zeros_like(H))
            elif method == 'sum':
                pass  # Already summed
            elif method == 'count':
                H, _, _ = np.histogram2d(lons, lats, bins=[bins['longitude'], bins['latitude']])

            edges = {'longitude': lon_edges, 'latitude': lat_edges}
            return H, edges

        else:
            raise NotImplementedError(
                f"Only lon-lat binning supported currently. Got: {list(bins.keys())}"
            )

    def temporal_evolution(self,
                           data: torch.Tensor,
                           coords: torch.Tensor,
                           spatial_agg: str = 'mean') -> Tuple[torch.Tensor, torch.Tensor]:
        """Create time series by aggregating over space.

        Args:
            data: (N, num_vars) data values
            coords: (N, 4) coordinates
            spatial_agg: How to aggregate spatial dimensions - 'mean', 'std', etc.

        Returns:
            (time_series, time_coords) where:
                time_series: (T, num_vars) values at each time step
                time_coords: (T,) time values

        Example:
            >>> # Get global mean time series
            >>> time_series, times = aggregator.temporal_evolution(
            ...     data, coords, spatial_agg='mean'
            ... )
        """
        time_idx = self.coord_labels['valid_time']

        # Get unique time values
        unique_times = torch.unique(coords[:, time_idx], sorted=True)
        n_times = len(unique_times)
        num_vars = data.shape[1] if data.ndim > 1 else 1

        if data.ndim == 1:
            time_series = torch.zeros(n_times)
        else:
            time_series = torch.zeros(n_times, num_vars)

        # Aggregate at each time step
        for i, t in enumerate(unique_times):
            mask = coords[:, time_idx] == t
            values = data[mask]

            if spatial_agg == 'mean':
                time_series[i] = values.mean(dim=0)
            elif spatial_agg == 'std':
                time_series[i] = values.std(dim=0)
            elif spatial_agg == 'min':
                time_series[i] = values.min(dim=0)[0]
            elif spatial_agg == 'max':
                time_series[i] = values.max(dim=0)[0]

        return time_series, unique_times

    def _unique_rows(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find unique rows in a 2D tensor.

        Returns:
            (unique_rows, inverse_indices)
        """
        # Convert to numpy for easier unique operation
        arr = tensor.cpu().numpy()

        # Use numpy's unique with return_inverse
        unique_arr, inverse = np.unique(arr, axis=0, return_inverse=True)

        # Convert back to torch
        unique_tensor = torch.from_numpy(unique_arr).to(tensor.device)
        inverse_tensor = torch.from_numpy(inverse).to(tensor.device)

        return unique_tensor, inverse_tensor