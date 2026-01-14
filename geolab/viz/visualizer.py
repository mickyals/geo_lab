"""Visualization functions for atmospheric model predictions.

Generic visualization that composes geometry, aggregation, and plotting.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Literal
from matplotlib.figure import Figure

from geolab.viz import GeometryGenerator, ModelInference, DataAggregator, Plotter


def plot_field(
        pl_module,
        geometry_spec: Dict,
        var_names: List[str],
        aggregation_spec: Optional[Dict] = None,
        plot_type: str = '2d_field',
        data_source: Literal['model', 'ground_truth', 'comparison'] = 'model',
        batch: Optional[Dict] = None,
        projection: str = 'cartopy'
) -> Dict[str, Optional[Figure]]:
    """
    Generic field plotting function.

    Args:
        pl_module: Lightning module
        geometry_spec: Geometry specification dict:
            {'type': 'plane', 'axes': ['longitude', 'latitude'],
             'pressure_level': 500, 'valid_time': 0.0,
             'resolution': {'longitude': 2.0, 'latitude': 2.0}}

            or

            {'type': 'volume', 'axes': ['longitude', 'latitude', 'pressure_level'],
             'valid_time': 0.0,
             'resolution': {'longitude': 2.0, 'latitude': 2.0, 'pressure_level': 50.0}}

        var_names: Variables to plot
        aggregation_spec: Optional aggregation:
            {'type': 'zonal_mean'} or
            {'type': 'temporal_mean'} or
            {'type': 'reduce_axis', 'axis': 'longitude', 'method': 'mean'}

        plot_type: '2d_field', '1d_profile', 'scatter'
        data_source: 'model' (predictions), 'ground_truth' (from batch),
                     'comparison' (side-by-side)
        batch: Required if data_source is 'ground_truth' or 'comparison'
        projection: 'cartopy', 'meridional', 'lat_pressure', etc.

    Returns:
        Dict mapping var_name to figure

    Examples:
        # Horizontal slice
        plot_field(
            pl_module,
            geometry_spec={
                'type': 'plane',
                'axes': ['longitude', 'latitude'],
                'pressure_level': 500,
                'valid_time': 0.0,
                'resolution': {'longitude': 2.0, 'latitude': 2.0}
            },
            var_names=['u', 'v'],
            projection='cartopy'
        )

        # Meridional slice
        plot_field(
            pl_module,
            geometry_spec={
                'type': 'plane',
                'axes': ['latitude', 'pressure_level'],
                'longitude': 0,
                'valid_time': 0.0,
                'resolution': {'latitude': 2.0, 'pressure_level': 50.0}
            },
            var_names=['w'],
            projection='meridional'
        )

        # Zonal mean
        plot_field(
            pl_module,
            geometry_spec={
                'type': 'plane',
                'axes': ['longitude', 'latitude'],
                'pressure_level': 500,
                'valid_time': 0.0,
                'resolution': {'longitude': 2.0, 'latitude': 2.0}
            },
            var_names=['u', 'v'],
            aggregation_spec={'type': 'zonal_mean'},
            plot_type='1d_profile'
        )
    """
    figures = {}

    try:
        # Step 1: Generate coordinates from geometry spec
        coords = _generate_coordinates(pl_module, geometry_spec)

        # Step 2: Get model predictions and/or ground truth
        if data_source in ['model', 'comparison']:
            inference = ModelInference(pl_module, pl_module.datamodule)
            model_preds = inference.predict(coords, denormalize_output=True, batch_size=5000)

        if data_source in ['ground_truth', 'comparison']:
            if batch is None:
                raise ValueError("batch required for ground_truth or comparison")
            # Extract ground truth from batch matching coords
            ground_truth = _extract_ground_truth(batch, coords, pl_module)

        # Step 3: Apply aggregation if specified
        if aggregation_spec:
            aggregator = DataAggregator(coord_labels=pl_module.datamodule.data.coord_labels)

            if data_source in ['model', 'comparison']:
                model_preds, coords = _apply_aggregation(
                    model_preds, coords, aggregation_spec, aggregator
                )

            if data_source in ['ground_truth', 'comparison']:
                ground_truth, coords = _apply_aggregation(
                    ground_truth, coords, aggregation_spec, aggregator
                )

        # Step 4: Plot each variable
        plotter = Plotter(
            coord_labels=pl_module.datamodule.data.coord_labels,
            var_labels={var: var.upper() for var in var_names}
        )

        for var in var_names:
            var_idx = pl_module.datamodule.data.var_order.index(var)

            if data_source == 'model':
                fig = _plot_variable(
                    model_preds[:, var_idx], coords, var,
                    plot_type, projection, plotter, geometry_spec
                )

            elif data_source == 'ground_truth':
                fig = _plot_variable(
                    ground_truth[:, var_idx], coords, var,
                    plot_type, projection, plotter, geometry_spec
                )

            elif data_source == 'comparison':
                fig = plotter.plot_comparison(
                    data1=model_preds[:, var_idx].cpu().numpy(),
                    data2=ground_truth[:, var_idx].cpu().numpy(),
                    coords=coords.cpu().numpy(),
                    var_name=var,
                    labels=('Model', 'Ground Truth')
                )

            figures[var] = fig

    except Exception as e:
        print(f"Error in plot_field: {e}")
        import traceback
        traceback.print_exc()

    return figures


def plot_scatter(
        self,
        pl_module,
        batch: Dict[str, torch.Tensor],
        split_name: str,
        projection: str = 'lat_lon',
        color_by: Optional[str] = None
) -> Optional[Figure]:
    """
    Plot scatter of data points in coordinate space.

    Args:
        pl_module: Lightning module
        batch: Batch with 'coords' and optionally 'values'
        split_name: 'train', 'val', or 'test'
        projection: 'lat_lon', 'lat_pressure', 'lon_pressure', '3d'
        color_by: Optional variable name to color points by

    Returns:
        Figure showing spatial distribution of points
    """
    try:
        # Ensure tensors are on CPU for plotting
        coords = batch['coords'].cpu().numpy()

        # Get color values if requested
        values = None
        if color_by and color_by in pl_module.datamodule.data.var_order:
            var_idx = pl_module.datamodule.data.var_order.index(color_by)
            values = batch['values'][:, var_idx].cpu().numpy()

        plotter = Plotter(
            coord_labels=pl_module.datamodule.data.coord_labels,
            var_labels={color_by: color_by.upper()} if color_by else {}
        )

        fig = plotter.plot_scatter(
            coords=coords,
            values=values,
            projection=projection,
            title=f'{split_name.capitalize()} Data Distribution'
        )

        return fig

    except Exception as e:
        print(f"Error in plot_scatter: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_error_heatmap(
        pl_module,
        batch: Dict[str, torch.Tensor],
        var_names: List[str]
) -> Optional[Figure]:
    """
    Plot error distribution heatmap.

    Args:
        pl_module: Lightning module
        batch: Batch with coords and values
        var_names: Variables to plot

    Returns:
        Figure with error heatmaps
    """
    try:
        # Ensure all tensors are on the same device as the model
        device = next(pl_module.parameters()).device
        coords = batch['coords'].to(device)
        targets = batch['values'].to(device)

        inference = ModelInference(pl_module, pl_module.datamodule)
        preds = inference.predict(coords, denormalize_output=True, batch_size=5000)

        # Ensure preds is on the same device as targets
        preds = preds.to(device)

        errors = (preds - targets).abs()

        # Move to CPU for plotting
        errors_np = errors.cpu().numpy()
        coords_np = coords.cpu().numpy()

        plotter = Plotter(
            coord_labels=pl_module.datamodule.data.coord_labels,
            var_labels={var: var.upper() for var in var_names}
        )

        fig = plotter.plot_error_heatmap(
            errors=errors_np,
            coords=coords_np,
            var_names=var_names
        )

        return fig

    except Exception as e:
        print(f"Error in plot_error_heatmap: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_physics_residuals(
        pl_module,
        batch: Dict[str, torch.Tensor],
        var_names: List[str]
) -> Optional[Figure]:
    """Plot physics residuals for PINN model."""
    if not pl_module.train_pinn:
        return None

    try:
        coords = batch['coords'][:1000]

        inference = ModelInference(pl_module, pl_module.datamodule)
        results = inference.predict_with_physics(coords, batch_size=500)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        residuals = [
            ('Mass Continuity', results['mass_continuity']),
            ('NS Longitude', results['ns_longitude']),
            ('NS Latitude', results['ns_latitude'])
        ]

        for ax, (name, residual) in zip(axes, residuals):
            res_np = residual.cpu().numpy()
            ax.hist(res_np, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Residual', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(name, fontsize=12)
            ax.grid(True, alpha=0.3)

            mean_res = res_np.mean()
            ax.axvline(mean_res, color='r', linestyle='--',
                       label=f'Mean: {mean_res:.2e}')
            ax.legend()

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"Error in plot_physics_residuals: {e}")
        import traceback
        traceback.print_exc()
        return None


# Helper functions

def _generate_coordinates(pl_module, geometry_spec: Dict) -> torch.Tensor:
    """Generate coordinates from geometry specification."""
    geom_type = geometry_spec['type']
    resolution = geometry_spec.get('resolution', {'longitude': 2.0, 'latitude': 2.0})

    geometry = GeometryGenerator(
        coord_domain=pl_module.datamodule.coordinate_ranges,
        coord_labels=pl_module.datamodule.data.coord_labels,
        resolution=resolution
    )

    if geom_type == 'plane':
        axes = geometry_spec['axes']
        fixed_coords = {k: v for k, v in geometry_spec.items()
                        if k not in ['type', 'axes', 'resolution']}
        return geometry.plane(axes=axes, **fixed_coords)

    elif geom_type == 'volume':
        axes = geometry_spec['axes']
        fixed_coords = {k: v for k, v in geometry_spec.items()
                        if k not in ['type', 'axes', 'resolution']}
        return geometry.volume(axes=axes, **fixed_coords)

    elif geom_type == 'line':
        axis = geometry_spec['axis']
        fixed_coords = {k: v for k, v in geometry_spec.items()
                        if k not in ['type', 'axis', 'resolution']}
        return geometry.line(axis=axis, **fixed_coords)

    else:
        raise ValueError(f"Unknown geometry type: {geom_type}")


def _apply_aggregation(data, coords, aggregation_spec, aggregator):
    """Apply aggregation to data and coords."""
    agg_type = aggregation_spec['type']

    if agg_type == 'zonal_mean':
        return aggregator.zonal_mean(data, coords)

    elif agg_type == 'temporal_mean':
        return aggregator.temporal_mean(data, coords)

    elif agg_type == 'reduce_axis':
        axis = aggregation_spec['axis']
        method = aggregation_spec.get('method', 'mean')
        return aggregator.reduce_axis(data, coords, axis, method)

    elif agg_type == 'temporal_evolution':
        spatial_agg = aggregation_spec.get('spatial_agg', 'mean')
        return_indices = aggregation_spec.get('return_indices', True)
        return aggregator.temporal_evolution(data, coords, spatial_agg, return_indices)


def _plot_variable(data, coords, var_name, plot_type, projection, plotter, geometry_spec):
    """Plot a single variable."""
    # Ensure data is on CPU and convert to numpy
    if torch.is_tensor(data):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = np.asarray(data)
        
    if torch.is_tensor(coords):
        coords_np = coords.detach().cpu().numpy()
    else:
        coords_np = coords

    if plot_type == '2d_field':
        # Reshape if needed
        if len(data_np.shape) == 1:
            # Try to reshape to 2D if we have coordinate information
            if 'axes' in geometry_spec and len(geometry_spec['axes']) == 2:
                # Get unique coordinates for each axis
                axis1 = geometry_spec['axes'][0]
                axis2 = geometry_spec['axes'][1]
                unique1 = np.unique(coords_np[:, 0])
                unique2 = np.unique(coords_np[:, 1])
                
                if len(unique1) * len(unique2) == len(data_np):
                    # Reshape to 2D grid
                    data_np = data_np.reshape(len(unique2), len(unique1))
                else:
                    # If we can't reshape to a grid, use scatter plot instead
                    return plotter.plot_scatter(
                        coords=coords_np,
                        values=data_np,
                        projection=projection,
                        title=f"{var_name.upper()}"
                    )

        return plotter.plot_2d_field(
            data=data_np,
            coords=coords_np,
            var_name=var_name,
            projection=projection
        )

    elif plot_type == '1d_profile':
        # Extract appropriate coordinate for x-axis
        coord_values = coords_np[:, 0]  # Simplification

        return plotter.plot_1d_profile(
            data=data_np,
            coord_values=coord_values,
            var_name=var_name
        )

    elif plot_type == 'scatter':
        values = data_np if data_np.ndim == 1 else data_np
        return plotter.plot_scatter(
            coords=coords_np,
            values=values,
            projection=projection
        )

    else:
        raise ValueError(f"Unknown plot type: {plot_type}")