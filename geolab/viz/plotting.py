"""Plotting utilities for atmospheric data."""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from typing import Dict, List, Optional, Tuple
from matplotlib.figure import Figure
import colorcet as cc


class Plotter:
    """Create matplotlib/cartopy figures for atmospheric data."""

    def __init__(self, coord_labels: Dict[str, int], var_labels: Dict[str, str]):
        """
        Args:
            coord_labels: Mapping of coordinate names to indices
            var_labels: Mapping of variable names to human-readable labels
                       Example: {'u': 'Zonal Wind (m/s)', 'v': 'Meridional Wind (m/s)'}
        """
        self.coord_labels = coord_labels
        self.var_labels = var_labels

        # Default colormaps
        self.default_cmaps = {
            'u': 'RdBu_r',
            'v': 'RdBu_r',
            'w': 'RdBu_r',
            't': 'cet_CET_R1',
            'z': 'cet_rainbow',
            'default': 'viridis'
        }

    # def plot_2d_field(self,
    #                   data: np.ndarray,
    #                   coords: np.ndarray,
    #                   var_name: str,
    #                   projection: str = 'cartopy',
    #                   **kwargs) -> Figure:
    #     """Plot 2D field (horizontal slice or cross-section).

    #     Args:
    #         data: 2D array to plot
    #         coords: Coordinate array (used for extent/axes)
    #         var_name: Variable name
    #         projection: 'cartopy' for horizontal, 'lat_pressure' or 'lon_pressure' for vertical
    #         **kwargs: Additional styling options:
    #             - title: str
    #             - vmin, vmax: float
    #             - cmap: str
    #             - levels: int or list

    #     Returns:
    #         matplotlib Figure

    #     Example:
    #         >>> fig = plotter.plot_2d_field(
    #         ...     pred_2d, coords, 'u',
    #         ...     projection='cartopy',
    #         ...     title='Zonal Wind at 500 hPa'
    #         ... )
    #     """
    #     if projection == 'cartopy':
    #         return self._plot_horizontal(data, var_name, **kwargs)
    #     elif projection in ['lat_pressure', 'meridional']:
    #         return self._plot_meridional(data, var_name, **kwargs)
    #     elif projection in ['lon_pressure', 'zonal']:
    #         return self._plot_zonal_section(data, var_name, **kwargs)
    #     else:
    #         raise ValueError(f"Unknown projection: {projection}")

    # def _plot_horizontal(self, data, var_name, **kwargs):
    #     """Plot horizontal slice using cartopy."""
    #     title = kwargs.get('title', f'{var_name.upper()} Horizontal Slice')
    #     cmap = kwargs.get('cmap', self.default_cmaps.get(var_name, 'viridis'))
    #     vmin = kwargs.get('vmin', None)
    #     vmax = kwargs.get('vmax', None)
    #     levels = kwargs.get('levels', 20)

    #     fig = plt.figure(figsize=(14, 8))
    #     ax = plt.axes(projection=ccrs.PlateCarree())

    #     # Add features
    #     ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    #     ax.add_feature(cfeature.COASTLINE, linewidth=0.6)

    #     extent = [-180, 180, -90, 90]

    #     # Plot data
    #     im = ax.contourf(
    #         data.T,  # Transpose for proper orientation
    #         levels=levels,
    #         cmap=cmap,
    #         vmin=vmin,
    #         vmax=vmax,
    #         extent=extent,
    #         transform=ccrs.PlateCarree()
    #     )

    #     ax.set_title(title, fontsize=14)

    #     # Gridlines
    #     gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    #     gl.top_labels = False
    #     gl.right_labels = False

    #     # Colorbar
    #     cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    #     cbar.set_label(self.var_labels.get(var_name, var_name), fontsize=12)

    #     plt.tight_layout()
    #     return fig

    # def _plot_meridional(self, data, var_name, **kwargs):
    #     """Plot meridional cross-section (lat-pressure)."""
    #     title = kwargs.get('title', f'{var_name.upper()} Meridional Section')
    #     cmap = kwargs.get('cmap', self.default_cmaps.get(var_name, 'viridis'))
    #     vmin = kwargs.get('vmin', None)
    #     vmax = kwargs.get('vmax', None)
    #     levels = kwargs.get('levels', 20)
    #     lats = kwargs.get('lats', np.arange(-90, 91, 2))
    #     pressures = kwargs.get('pressures', np.arange(100, 1001, 50))

    #     fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    #     # Create meshgrid
    #     lat_grid, pressure_grid = np.meshgrid(lats, pressures, indexing='xy')

    #     # Plot
    #     im = ax.contourf(lat_grid, pressure_grid, data.T,
    #                      levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)

    #     ax.invert_yaxis()  # Pressure increases downward
    #     ax.set_xlabel('Latitude (°)', fontsize=12)
    #     ax.set_ylabel('Pressure (hPa)', fontsize=12)
    #     ax.set_title(title, fontsize=14)
    #     ax.grid(True, alpha=0.3)

    #     cbar = plt.colorbar(im, ax=ax)
    #     cbar.set_label(self.var_labels.get(var_name, var_name), fontsize=12)

    #     plt.tight_layout()
    #     return fig

    # def _plot_zonal_section(self, data, var_name, **kwargs):
    #     """Plot zonal cross-section (lon-pressure)."""
    #     title = kwargs.get('title', f'{var_name.upper()} Zonal Section')
    #     cmap = kwargs.get('cmap', self.default_cmaps.get(var_name, 'viridis'))
    #     vmin = kwargs.get('vmin', None)
    #     vmax = kwargs.get('vmax', None)
    #     levels = kwargs.get('levels', 20)
    #     lons = kwargs.get('lons', np.arange(-180, 181, 2))
    #     pressures = kwargs.get('pressures', np.arange(100, 1001, 50))

    #     fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    #     # Create meshgrid
    #     lon_grid, pressure_grid = np.meshgrid(lons, pressures, indexing='xy')

    #     # Plot
    #     im = ax.contourf(lon_grid, pressure_grid, data.T,
    #                      levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)

    #     ax.invert_yaxis()
    #     ax.set_xlabel('Longitude (°)', fontsize=12)
    #     ax.set_ylabel('Pressure (hPa)', fontsize=12)
    #     ax.set_title(title, fontsize=14)
    #     ax.grid(True, alpha=0.3)

    #     cbar = plt.colorbar(im, ax=ax)
    #     cbar.set_label(self.var_labels.get(var_name, var_name), fontsize=12)

    #     plt.tight_layout()
    #     return fig

    def plot_2d_field(self,
                      data: np.ndarray,
                      coords: np.ndarray,
                      var_name: str,
                      projection: str = 'cartopy',
                      **kwargs) -> Figure:
        """Plot 2D field (horizontal slice or cross-section)."""
        # Pass 'coords' to the specific handlers
        if projection == 'cartopy':
            return self._plot_horizontal(data, coords, var_name, **kwargs)
        elif projection in ['lat_pressure', 'meridional']:
            return self._plot_meridional(data, coords, var_name, **kwargs)
        elif projection in ['lon_pressure', 'zonal']:
            return self._plot_zonal_section(data, coords, var_name, **kwargs)
        else:
            raise ValueError(f"Unknown projection: {projection}")

    def _plot_horizontal(self, data, coords, var_name, **kwargs):
        """Plot horizontal slice with dynamic extent."""
        title = kwargs.get('title', f'{var_name.upper()} Horizontal Slice')
        cmap = kwargs.get('cmap', self.default_cmaps.get(var_name, 'viridis'))
        vmin = kwargs.get('vmin', data.min())
        vmax = kwargs.get('vmax', data.max())
        
        # BUG FIX: Calculate extent from coordinates instead of hardcoding
        # Determine which columns are lon/lat. If coords is (N, 2), assume [lon, lat]
        lon_idx = self.coord_labels.get('longitude', 0) if coords.shape[1] > 1 else 0
        lat_idx = self.coord_labels.get('latitude', 1) if coords.shape[1] > 1 else 1
        
        lons = coords[:, lon_idx]
        lats = coords[:, lat_idx]
        extent = [lons.min(), lons.max(), lats.min(), lats.max()]

        fig = plt.figure(figsize=(14, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        
        # Set map extent
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Plot data. Remove .T if visualizer reshapes to (n_lat, n_lon)
        # Using imshow with extent and origin='lower' is often more robust for gridded data
        im = ax.imshow(
            data, 
            extent=extent,
            transform=ccrs.PlateCarree(),
            cmap=cmap, vmin=vmin, vmax=vmax,
            origin='lower', aspect='auto'
        )

        ax.set_title(title, fontsize=14)
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
        return fig

    def _plot_meridional(self, data, coords, var_name, **kwargs):
        """Plot meridional cross-section with dynamic axes."""
        # BUG FIX: Extract actual coordinate values
        lat_idx = self.coord_labels.get('latitude', 0)
        pres_idx = self.coord_labels.get('pressure_level', 1)
        
        lats = np.unique(coords[:, lat_idx])
        pressures = np.unique(coords[:, pres_idx])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        # Ensure data is (n_press, n_lat) for meshgrid matching
        im = ax.contourf(lats, pressures, data, levels=20, 
                         cmap=kwargs.get('cmap', self.default_cmaps.get(var_name, 'viridis')))
        
        ax.invert_yaxis()
        ax.set_xlabel('Latitude (°)')
        ax.set_ylabel('Pressure (hPa)')
        plt.colorbar(im, ax=ax, label=self.var_labels.get(var_name, var_name))
        return fig

    def plot_comparison(self, data1, data2, coords, var_name, **kwargs):
        """Side-by-side with dynamic extent."""
        # BUG FIX: Use calculated extent for all panels
        lon_idx = self.coord_labels.get('longitude', 0)
        lat_idx = self.coord_labels.get('latitude', 1)
        lons, lats = coords[:, lon_idx], coords[:, lat_idx]
        extent = [lons.min(), lons.max(), lats.min(), lats.max()]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, d in enumerate([data1, data2, data1-data2]):
            im = axes[i].imshow(d, extent=extent, origin='lower', aspect='auto',
                                cmap='RdBu_r' if i==2 else 'viridis')
            plt.colorbar(im, ax=axes[i])
        return fig

    def plot_1d_profile(self,
                        data: np.ndarray,
                        coord_values: np.ndarray,
                        var_name: str,
                        coord_name: str = 'time',
                        **kwargs) -> Figure:
        """Plot 1D profile (time series, vertical profile, etc.).

        Args:
            data: 1D array of values
            coord_values: 1D array of coordinate values
            var_name: Variable name
            coord_name: Coordinate being plotted against
            **kwargs: title, xlabel, ylabel, etc.

        Returns:
            matplotlib Figure
        """
        title = kwargs.get('title', f'{var_name.upper()} Profile')
        xlabel = kwargs.get('xlabel', coord_name)
        ylabel = kwargs.get('ylabel', self.var_labels.get(var_name, var_name))

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(coord_values, data, 'b-o', markersize=4)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

        # Invert y-axis if plotting pressure
        if coord_name == 'pressure' or 'pressure' in coord_name.lower():
            ax.invert_yaxis()

        plt.tight_layout()
        return fig

    def plot_scatter(self,
                     coords: np.ndarray,
                     values: Optional[np.ndarray] = None,
                     projection: str = '3d',
                     **kwargs) -> Figure:
        """Plot scatter of points in space.

        Args:
            coords: (N, 3) or (N, 4) coordinate array
            values: Optional (N,) values for coloring points
            projection: '3d', 'lat_lon', 'lat_pressure', or 'lon_pressure'
            **kwargs: title, color, cmap, etc.

        Returns:
            matplotlib Figure
        """
        if projection == '3d':
            return self._plot_scatter_3d(coords, values, **kwargs)
        else:
            return self._plot_scatter_2d(coords, values, projection, **kwargs)

    def _plot_scatter_3d(self, coords, values, **kwargs):
        """3D scatter plot."""
        from mpl_toolkits.mplot3d import Axes3D

        title = kwargs.get('title', '3D Scatter Plot')
        cmap = kwargs.get('cmap', 'viridis')

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Assume coords are [lon, lat, pressure, ...]
        lon_idx = self.coord_labels.get('longitude', 0)
        lat_idx = self.coord_labels.get('latitude', 1)
        press_idx = self.coord_labels.get('pressure_level', 2)

        scatter = ax.scatter(
            coords[:, lon_idx],
            coords[:, lat_idx],
            coords[:, press_idx],
            c=values if values is not None else coords[:, press_idx],
            cmap=cmap,
            alpha=0.6,
            s=20
        )

        ax.invert_zaxis()
        ax.set_xlabel('Longitude (°)', fontsize=12)
        ax.set_ylabel('Latitude (°)', fontsize=12)
        ax.set_zlabel('Pressure (hPa)', fontsize=12)
        ax.set_title(title, fontsize=14)

        plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
        plt.tight_layout()
        return fig

    def _plot_scatter_2d(self, coords, values, projection, **kwargs):
        """2D scatter plot."""
        title = kwargs.get('title', f'{projection} Scatter')
        cmap = kwargs.get('cmap', 'viridis')

        fig, ax = plt.subplots(figsize=(10, 6))

        # Initialize default values
        x = None
        y = None
        xlabel, ylabel = 'X', 'Y'

        # Select coordinates based on projection
        if projection == 'lat_lon':
            if 'longitude' in self.coord_labels and 'latitude' in self.coord_labels:
                x = coords[:, self.coord_labels['longitude']]
                y = coords[:, self.coord_labels['latitude']]
                xlabel, ylabel = 'Longitude (°)', 'Latitude (°)'
        elif projection == 'lat_pressure' and 'latitude' in self.coord_labels and 'pressure_level' in self.coord_labels:
            x = coords[:, self.coord_labels['latitude']]
            y = coords[:, self.coord_labels['pressure_level']]
            xlabel, ylabel = 'Latitude (°)', 'Pressure (hPa)'
            ax.invert_yaxis()
        elif projection == 'lon_pressure' and 'longitude' in self.coord_labels and 'pressure_level' in self.coord_labels:
            x = coords[:, self.coord_labels['longitude']]
            y = coords[:, self.coord_labels['pressure_level']]
            xlabel, ylabel = 'Longitude (°)', 'Pressure (hPa)'
            ax.invert_yaxis()
        else:
            # Fallback to first two dimensions if projection is not recognized
            if coords.shape[1] >= 2:
                x = coords[:, 0]
                y = coords[:, 1]
                xlabel, ylabel = 'X', 'Y'
            else:
                raise ValueError(f"Invalid projection or coordinate dimensions for scatter plot: {projection}")

        if x is None or y is None:
            raise ValueError(f"Could not determine coordinates for projection: {projection}")

        scatter = ax.scatter(x, y, c=values, cmap=cmap, alpha=0.5, s=10)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3)

        if values is not None:
            plt.colorbar(scatter, ax=ax)

        plt.tight_layout()
        return fig

    def plot_error_heatmap(self,
                           errors: np.ndarray,
                           coords: np.ndarray,
                           var_names: List[str],
                           **kwargs) -> Figure:
        """Plot spatial distribution of errors for multiple variables.

        Args:
            errors: (N, num_vars) error values
            coords: (N, 4) coordinates
            var_names: List of variable names
            **kwargs: bins, etc.

        Returns:
            matplotlib Figure
        """
        bins = kwargs.get('bins', {'longitude': 72, 'latitude': 36})

        n_vars = len(var_names)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        lon_idx = self.coord_labels['longitude']
        lat_idx = self.coord_labels['latitude']

        lons = coords[:, lon_idx]
        lats = coords[:, lat_idx]

        for i, (var_name, ax) in enumerate(zip(var_names, axes)):
            if i >= n_vars:
                ax.axis('off')
                continue

            # Bin errors
            H, xedges, yedges = np.histogram2d(
                lons, lats,
                bins=[bins['longitude'], bins['latitude']],
                weights=errors[:, i]
            )

            # Normalize by counts
            counts, _, _ = np.histogram2d(lons, lats,
                                          bins=[bins['longitude'], bins['latitude']])
            H = np.divide(H, counts, where=counts > 0, out=np.zeros_like(H))

            # Plot
            im = ax.imshow(H.T, origin='lower',
                           extent=[-180, 180, -90, 90],
                           cmap='YlOrRd', aspect='auto')
            ax.set_title(f'{var_name.upper()} MSE', fontsize=12)
            ax.set_xlabel('Longitude (°)', fontsize=10)
            ax.set_ylabel('Latitude (°)', fontsize=10)
            ax.grid(True, alpha=0.3)
            plt.colorbar(im, ax=ax, label='MSE')

        # Hide unused subplots
        for i in range(n_vars, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        return fig

    def plot_comparison(self,
                        data1: np.ndarray,
                        data2: np.ndarray,
                        coords: np.ndarray,
                        var_name: str,
                        labels: Tuple[str, str] = ('Model', 'Ground Truth'),
                        **kwargs) -> Figure:
        """Side-by-side comparison plot.

        Args:
            data1: First dataset (e.g., model predictions)
            data2: Second dataset (e.g., ground truth)
            coords: Coordinates
            var_name: Variable name
            labels: Labels for each dataset
            **kwargs: vmin, vmax, etc.

        Returns:
            matplotlib Figure with 3 panels (data1, data2, difference)
        """
        vmin = kwargs.get('vmin', min(data1.min(), data2.min()))
        vmax = kwargs.get('vmax', max(data1.max(), data2.max()))
        cmap = kwargs.get('cmap', self.default_cmaps.get(var_name, 'viridis'))

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: data1
        im1 = axes[0].imshow(data1.T, origin='lower',
                             extent=[-180, 180, -90, 90],
                             cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title(f'{labels[0]}: {var_name.upper()}', fontsize=12)
        axes[0].set_xlabel('Longitude (°)')
        axes[0].set_ylabel('Latitude (°)')
        plt.colorbar(im1, ax=axes[0])

        # Panel 2: data2
        im2 = axes[1].imshow(data2.T, origin='lower',
                             extent=[-180, 180, -90, 90],
                             cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title(f'{labels[1]}: {var_name.upper()}', fontsize=12)
        axes[1].set_xlabel('Longitude (°)')
        axes[1].set_ylabel('Latitude (°)')
        plt.colorbar(im2, ax=axes[1])

        # Panel 3: Difference
        diff = data1 - data2
        im3 = axes[2].imshow(diff.T, origin='lower',
                             extent=[-180, 180, -90, 90],
                             cmap='RdBu_r',
                             vmin=-np.abs(diff).max(),
                             vmax=np.abs(diff).max())
        axes[2].set_title(f'Difference ({labels[0]} - {labels[1]})', fontsize=12)
        axes[2].set_xlabel('Longitude (°)')
        axes[2].set_ylabel('Latitude (°)')
        plt.colorbar(im3, ax=axes[2], label='Difference')

        plt.tight_layout()
        return fig
