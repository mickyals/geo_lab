"""
Visualization utilities for atmospheric model outputs.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_error_heatmap(
    model,
    batch: Dict,
    var_names: List[str],
) -> Optional[plt.Figure]:
    """
    Plot spatial distribution of errors on lat-lon grid.
    
    Args:
        model: Lightning module
        batch: Validation batch
        var_names: List of variable names
    
    Returns:
        matplotlib Figure or None
    """
    # Extract data from batch
    coords = _extract_coords_from_batch(batch)
    targets = _extract_targets_from_batch(batch)
    
    # Get predictions
    preds = model(coords)
    
    # Compute errors
    errors = (preds - targets).pow(2)
    
    # Extract lat/lon (denormalize if needed for plotting)
    lats = coords[:, 1].cpu().numpy()
    lons = coords[:, 0].cpu().numpy()
    
    # Create figure with 2x3 grid for 5 variables
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (var_name, ax) in enumerate(zip(var_names, axes)):
        if i >= len(var_names):
            ax.axis('off')
            continue
            
        # Bin errors spatially
        H, xedges, yedges = np.histogram2d(
            lons, lats,
            bins=[380, 190],  # 5° bins
            weights=errors[:, i].cpu().numpy()
        )
        
        # Normalize counts to get mean error per bin
        counts, _, _ = np.histogram2d(lons, lats, bins=[380, 190])
        H = np.divide(H, counts, where=counts > 0, out=np.zeros_like(H))
        
        # Plot heatmap
        im = ax.imshow(
            H.T, origin='lower',
            extent=[-180, 180, -90, 90],
            cmap='YlOrRd', aspect='auto'
        )
        ax.set_title(f'{var_name.upper()} MSE Distribution')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, label='MSE')
    
    # Hide last subplot if we have 5 variables
    if len(var_names) == 5:
        axes[5].axis('off')
    
    plt.tight_layout()
    return fig


def plot_physics_residuals(
    model,
    batch: Dict,
    var_names: List[str]
) -> Optional[plt.Figure]:
    """
    Plot spatial maps of physics residuals (PINN only).
    
    Args:
        model: Lightning module
        batch: Validation batch
        var_names: List of variable names
    
    Returns:
        matplotlib Figure or None
    """
    from geolab.models.components import troposphere_pde_residual
    
    # Extract coordinates and enable gradients
    coords = _extract_coords_from_batch(batch)
    coords = coords.detach().requires_grad_(True)  # Detach first, then enable gradients
    
    # Enable gradient computation even though model is in eval mode
    with torch.enable_grad():
        # Forward pass with gradients
        preds = model(coords)
        
        # Create model outputs dict
        model_outputs_dict = {var_names[i]: preds[:, i] for i in range(len(var_names))}
        
        # Compute physics residuals
        ns_longitude, ns_latitude, mass_cont = troposphere_pde_residual(
            coords, model_outputs_dict
        )
    
    # Extract coordinates (detached for plotting)
    lats = coords[:, 1].detach().cpu().numpy()
    lons = coords[:, 0].detach().cpu().numpy()
    
    # Create residual maps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    residuals = {
        'NS Longitude': ns_longitude.abs().detach().cpu().numpy(),
        'NS Latitude': ns_latitude.abs().detach().cpu().numpy(),
        'Mass Continuity': mass_cont.abs().detach().cpu().numpy()
    }
    
    for (name, residual), ax in zip(residuals.items(), axes):
        # Bin residuals spatially
        H, xedges, yedges = np.histogram2d(
            lons, lats, bins=[72, 36],
            weights=residual
        )
        
        # Normalize by counts
        counts, _, _ = np.histogram2d(lons, lats, bins=[72, 36])
        H = np.divide(H, counts, where=counts > 0, out=np.zeros_like(H))
        
        im = ax.imshow(
            H.T, origin='lower',
            extent=[-180, 180, -90, 90],
            cmap='Reds', aspect='auto'
        )
        ax.set_title(f'{name} Residual')
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, label='|Residual|')
    
    plt.tight_layout()
    return fig


def plot_horizontal_slices(
    model,
    batch: Dict,
    pressure: int,
    var_names: List[str],
    grid_resolution: Optional[dict] = None
) -> Dict[str, plt.Figure]:
    """
    Plot 2D maps of model predictions at a fixed pressure level.
    
    Args:
        model: Lightning module
        batch: Validation batch (used only for timestep reference)
        pressure: Pressure level in hPa
        var_names: List of variable names
        grid_resolution: Grid resolution in degrees
    
    Returns:
        Dictionary mapping variable names to figures
    """
    if grid_resolution is None:
        grid_resolution = {'longitude': 2, 'latitude': 2}
    
    # Extract normalization parameters from model
    statistics = getattr(model, 'statistics', None)
    pi_scale = getattr(model, 'pi_scale', False)
    
    if statistics is None:
        raise ValueError(
            "Model must have 'statistics' attribute for proper coordinate normalization. "
            "Pass statistics dict to model during initialization."
        )
    
    # Create dense regular grid at this pressure level
    grid_coords, norm_grid_coords, n_lats, n_lons = _create_horizontal_grid(
        pressure, 
        timestep=0,  # Middle of normalized time range [0, 1]
        resolution=grid_resolution,
        device=model.device,
        statistics=statistics,
        pi_scale=pi_scale
    )
    
    # Get model predictions on dense grid
    preds = model(norm_grid_coords)
    
    # Create figures for each variable
    figures = {}
    
    for i, var_name in enumerate(var_names):
        pred_field = preds[:, i].reshape(n_lons, n_lats).cpu().numpy()
        
        fig = _create_horizontal_plot(
            pred_field, var_name, pressure
        )
        figures[var_name] = fig
    
    return figures


def plot_meridional_slices(
    model,
    batch: Dict,
    longitude: int,
    var_names: List[str],
    pressure_levels: List[int],
    grid_resolution: Optional[dict] = None
) -> Dict[str, plt.Figure]:
    """
    Plot vertical cross-sections along a meridian (lat-pressure slice).
    
    Args:
        model: Lightning module
        batch: Validation batch (used only for timestep reference)
        longitude: Fixed longitude for the slice
        var_names: List of variable names
        pressure_levels: List of pressure levels
        grid_resolution: Grid resolution in degrees
    
    Returns:
        Dictionary mapping variable names to figures
    """
    if grid_resolution is None:
        grid_resolution = {'longitude': 2, 'latitude': 2}
    
    # Extract normalization parameters from model
    statistics = getattr(model, 'statistics', None)
    pi_scale = getattr(model, 'pi_scale', False)
    
    if statistics is None:
        raise ValueError(
            "Model must have 'statistics' attribute for proper coordinate normalization."
        )
    
    # Create lat-pressure grid at fixed longitude
    grid_coords, norm_grid_coords, n_lats = _create_meridional_grid(
        longitude,
        pressure_levels,
        timestep=0,
        resolution=grid_resolution,
        device=model.device,
        statistics=statistics,
        pi_scale=pi_scale
    )
    
    # Get model predictions
    preds = model(norm_grid_coords)
    
    # Create figures for each variable
    figures = {}
    n_pressure = len(pressure_levels)
    
    for i, var_name in enumerate(var_names):
        pred_field = preds[:, i].reshape(n_lats, n_pressure).cpu().numpy()
        
        fig = _create_meridional_plot(
            pred_field, var_name, longitude, pressure_levels, grid_resolution
        )
        figures[var_name] = fig
    
    return figures


def plot_zonal_mean(
    model,
    batch: Dict,
    var_names: List[str],
    pressure_levels: List[int],
    grid_resolution: Optional[dict] = None
) -> Dict[str, plt.Figure]:
    """
    Plot zonal mean (longitude-averaged) profiles.
    
    Args:
        model: Lightning module
        batch: Validation batch (used only for timestep reference)
        var_names: List of variable names
        pressure_levels: List of pressure levels
        grid_resolution: Grid resolution in degrees
    
    Returns:
        Dictionary mapping variable names to figures
    """
    if grid_resolution is None:
        grid_resolution = {'longitude': 2, 'latitude': 2}

    # Extract normalization parameters from model
    statistics = getattr(model, 'statistics', None)
    pi_scale = getattr(model, 'pi_scale', False)
    
    if statistics is None:
        raise ValueError(
            "Model must have 'statistics' attribute for proper coordinate normalization."
        )

    # Create full lat-lon-pressure grid
    grid_coords, norm_grid_coords, n_lats, n_lons, n_pressure = _create_full_grid(
        pressure_levels,
        timestep=0,
        resolution=grid_resolution,
        device=model.device,
        statistics=statistics,
        pi_scale=pi_scale
    )
    
    # Get model predictions
    preds = model(norm_grid_coords)
    
    figures = {}
    
    for i, var_name in enumerate(var_names):
        pred_field = preds[:, i].reshape(n_lons, n_lats, n_pressure).cpu().numpy()
        
        # Zonal mean (average over longitude dimension)
        pred_zonal = pred_field.mean(axis=0)  # [n_lats, n_pressure]
        
        fig = _create_zonal_mean_plot(
            pred_zonal, var_name, pressure_levels, grid_resolution
        )
        figures[var_name] = fig
    
    return figures


# ============================================================================
# Helper Functions
# ============================================================================

def _extract_coords_from_batch(batch: Dict) -> torch.Tensor:
    """Extract coordinates from batch dictionary."""
    coords = batch['coords']
    coord_list = [
        coords['longitude'],
        coords['latitude'],
        coords['pressure_level'],
        coords['time']
    ]
    return torch.stack(coord_list, dim=1).float()


def _extract_targets_from_batch(batch: Dict) -> torch.Tensor:
    """Extract target variables from batch dictionary."""
    variables = batch['variables']
    return torch.stack(list(variables.values()), dim=1).float()


def _create_horizontal_grid(
    pressure: int,
    timestep: float = 0,
    resolution: Optional[dict] = None,
    device: torch.device = None,
    statistics: dict = None,
    pi_scale: bool = False
) -> torch.Tensor:
    """
    Create a regular lat-lon grid at fixed pressure and time.
    
    Args:
        pressure: Pressure level in hPa
        timestep: Time value in [0, 1] range 
        resolution: Grid resolution in degrees
        device: Device to create tensor on
        statistics: Statistics dict from training data
        pi_scale: Whether to scale lat/lon by pi (must match training)
    
    Returns:
        Tuple of (grid, norm_grid, n_lats, n_lons)
    """
    if resolution is None:
        resolution = {'longitude': 2, 'latitude': 2}
    
    if statistics is None:
        raise ValueError("Must provide statistics dict for coordinate normalization!")
    
    lons = torch.arange(-180, 180, resolution['longitude'], dtype=torch.float32)
    lats = torch.arange(-90, 90+resolution['latitude'], resolution['latitude'], dtype=torch.float32)

    n_lats = lats.numel()
    n_lons = lons.numel()
    
    lon_grid, lat_grid = torch.meshgrid(lons, lats, indexing='ij')
    
    n_points = lon_grid.numel()
    grid = torch.zeros(n_points, 4)
    grid[:, 0] = lon_grid.flatten()
    grid[:, 1] = lat_grid.flatten()
    grid[:, 2] = pressure
    grid[:, 3] = timestep

    # ============================================
    # MATCH TRAINING NORMALIZATION EXACTLY
    # ============================================
    norm_grid = torch.zeros(n_points, 4)
    
    # Longitude: normalize to [-1, 1], then optionally scale by π
    lon_min = statistics['longitude'][0]
    lon_max = statistics['longitude'][1]
    norm_grid[:, 0] = 2.0 * (grid[:, 0] - lon_min) / (lon_max - lon_min) - 1.0
    if pi_scale:
        norm_grid[:, 0] = norm_grid[:, 0] * torch.pi
    
    # Latitude: normalize to [-1, 1], then optionally scale by π/2
    lat_min = statistics['latitude'][0]
    lat_max = statistics['latitude'][1]
    norm_grid[:, 1] = 2.0 * (grid[:, 1] - lat_min) / (lat_max - lat_min) - 1.0
    if pi_scale:
        norm_grid[:, 1] = norm_grid[:, 1] * (torch.pi / 2)
    
    # Pressure: DO NOT NORMALIZE (keep raw value like in training)
    norm_grid[:, 2] = grid[:, 2]
    
    # Time: already in [0, 1] range, use as-is
    norm_grid[:, 3] = timestep
    
    if device is not None:
        norm_grid = norm_grid.to(device)
    
    return grid, norm_grid, n_lats, n_lons


def _create_meridional_grid(
    longitude: int,
    pressure_levels: List[int],
    timestep: float = 0,
    resolution: Optional[dict] = None,
    device: torch.device = None,
    statistics: dict = None,
    pi_scale: bool = False
) -> torch.Tensor:
    """
    Create a lat-pressure grid at fixed longitude.
    
    Args:
        longitude: Fixed longitude
        pressure_levels: List of pressure levels
        timestep: Time value in [0, 1] range
        resolution: Latitude resolution in degrees
        device: Device to create tensor on
        statistics: Statistics dict from training data
        pi_scale: Whether to scale lat/lon by pi
    
    Returns:
        Tuple of (grid, norm_grid, n_lats)
    """
    if resolution is None:
        resolution = {'longitude': 2, 'latitude': 2}
    
    if statistics is None:
        raise ValueError("Must provide statistics dict for coordinate normalization!")
    
    lats = torch.arange(-90, 90+resolution['latitude'], resolution['latitude'], dtype=torch.float32)
    n_lats = lats.numel()
    pressures = torch.tensor(pressure_levels, dtype=torch.float32)
    
    lat_grid, pressure_grid = torch.meshgrid(lats, pressures, indexing='ij')
    
    n_points = lat_grid.numel()
    grid = torch.zeros(n_points, 4)
    grid[:, 0] = longitude
    grid[:, 1] = lat_grid.flatten()
    grid[:, 2] = pressure_grid.flatten()
    grid[:, 3] = timestep

    # ============================================
    # MATCH TRAINING NORMALIZATION EXACTLY
    # ============================================
    norm_grid = torch.zeros(n_points, 4)
    
    # Longitude: normalize to [-1, 1], then optionally scale by π
    lon_min = statistics['longitude'][0]
    lon_max = statistics['longitude'][1]
    norm_grid[:, 0] = 2.0 * (grid[:, 0] - lon_min) / (lon_max - lon_min) - 1.0
    if pi_scale:
        norm_grid[:, 0] = norm_grid[:, 0] * torch.pi
    
    # Latitude: normalize to [-1, 1], then optionally scale by π/2
    lat_min = statistics['latitude'][0]
    lat_max = statistics['latitude'][1]
    norm_grid[:, 1] = 2.0 * (grid[:, 1] - lat_min) / (lat_max - lat_min) - 1.0
    if pi_scale:
        norm_grid[:, 1] = norm_grid[:, 1] * (torch.pi / 2)
    
    # Pressure: DO NOT NORMALIZE (keep raw value)
    norm_grid[:, 2] = grid[:, 2]
    
    # Time: already in [0, 1] range
    norm_grid[:, 3] = timestep
    
    if device is not None:
        norm_grid = norm_grid.to(device)
    
    return grid, norm_grid, n_lats


def _create_full_grid(
    pressure_levels: List[int],
    timestep: float = 0,
    resolution: Optional[dict] = None,
    device: torch.device = None,
    statistics: dict = None,
    pi_scale: bool = False
) -> torch.Tensor:
    """
    Create a full lat-lon-pressure grid.
    
    Args:
        pressure_levels: List of pressure levels
        timestep: Time value in [0, 1] range
        resolution: Grid resolution in degrees
        device: Device to create tensor on
        statistics: Statistics dict from training data
        pi_scale: Whether to scale lat/lon by pi
    
    Returns:
        Tuple of (grid, norm_grid, n_lats, n_lons, n_pressures)
    """
    if resolution is None:
        resolution = {'longitude': 2, 'latitude': 2}
    
    if statistics is None:
        raise ValueError("Must provide statistics dict for coordinate normalization!")
    
    lons = torch.arange(-180, 180, resolution['longitude'], dtype=torch.float32)
    lats = torch.arange(-90, 90+resolution['latitude'], resolution['latitude'], dtype=torch.float32)
    pressures = torch.tensor(pressure_levels, dtype=torch.float32)

    n_lons = lons.numel()
    n_lats = lats.numel()
    n_pressures = pressures.numel()
    
    lon_grid, lat_grid, pressure_grid = torch.meshgrid(lons, lats, pressures, indexing='ij')
    
    n_points = lon_grid.numel()
    grid = torch.zeros(n_points, 4)
    grid[:, 0] = lon_grid.flatten()
    grid[:, 1] = lat_grid.flatten()
    grid[:, 2] = pressure_grid.flatten()
    grid[:, 3] = timestep

    # ============================================
    # MATCH TRAINING NORMALIZATION EXACTLY
    # ============================================
    norm_grid = torch.zeros(n_points, 4)
    
    # Longitude: normalize to [-1, 1], then optionally scale by π
    lon_min = statistics['longitude'][0]
    lon_max = statistics['longitude'][1]
    norm_grid[:, 0] = 2.0 * (grid[:, 0] - lon_min) / (lon_max - lon_min) - 1.0
    if pi_scale:
        norm_grid[:, 0] = norm_grid[:, 0] * torch.pi
    
    # Latitude: normalize to [-1, 1], then optionally scale by π/2
    lat_min = statistics['latitude'][0]
    lat_max = statistics['latitude'][1]
    norm_grid[:, 1] = 2.0 * (grid[:, 1] - lat_min) / (lat_max - lat_min) - 1.0
    if pi_scale:
        norm_grid[:, 1] = norm_grid[:, 1] * (torch.pi / 2)
    
    # Pressure: DO NOT NORMALIZE (keep raw value)
    norm_grid[:, 2] = grid[:, 2]
    
    # Time: already in [0, 1] range
    norm_grid[:, 3] = timestep
    
    if device is not None:
        norm_grid = norm_grid.to(device)
    
    return grid, norm_grid, n_lats, n_lons, n_pressures


def _create_horizontal_plot(
    pred_field: np.ndarray,
    var_name: str,
    pressure: int
) -> plt.Figure:
    """Create a single-panel plot showing model prediction."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    extent = [-180, 180, -90, 90]
    
    # Determine colormap based on variable
    if var_name in ['u', 'v']:
        cmap = 'RdBu_r'
        vmax = np.abs(pred_field).max()
        vmin = -vmax
    else:
        cmap = 'viridis'
        vmin, vmax = None, None
    
    im = ax.imshow(
        pred_field.T, origin='lower', extent=extent,
        cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax
    )
    ax.set_title(f'{var_name.upper()} @ {pressure} hPa')
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label=_get_var_label(var_name))
    
    plt.tight_layout()
    return fig


def _create_meridional_plot(
    pred_field: np.ndarray,
    var_name: str,
    longitude: int,
    pressure_levels: List[int],
    resolution: Optional[dict] = None
) -> plt.Figure:
    """Create meridional cross-section plot."""
    if resolution is None:
        resolution = {'longitude': 2, 'latitude': 2}
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    lats = np.arange(-90, 90+resolution['latitude'], resolution['latitude'])
    pressures = np.array(pressure_levels)
    
    # Determine colormap
    if var_name in ['u', 'v']:
        cmap = 'RdBu_r'
        vmax = np.abs(pred_field).max()
        vmin = -vmax
    else:
        cmap = 'viridis'
        vmin, vmax = None, None
    
    # Contour plot
    im = ax.contourf(lats, pressures, pred_field.T, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'{var_name.upper()} Meridional Section @ {longitude}°E')
    ax.set_xlabel('Latitude (°)')
    ax.set_ylabel('Pressure (hPa)')
    ax.invert_yaxis()  # Pressure increases downward
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label=_get_var_label(var_name))
    
    plt.tight_layout()
    return fig


def _create_zonal_mean_plot(
    pred_zonal: np.ndarray,
    var_name: str,
    pressure_levels: List[int],
    resolution: Optional[dict] = None
) -> plt.Figure:
    """Create zonal mean plot."""
    if resolution is None:
        resolution = {'longitude': 2, 'latitude': 2}
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    lats = np.arange(-90, 90+resolution['latitude'], resolution['latitude'])
    pressures = np.array(pressure_levels)
    
    # Determine colormap
    if var_name in ['u', 'v']:
        cmap = 'RdBu_r'
        vmax = np.abs(pred_zonal).max()
        vmin = -vmax
    else:
        cmap = 'viridis'
        vmin, vmax = None, None
    
    # Contour plot
    im = ax.contourf(lats, pressures, pred_zonal.T, levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(f'{var_name.upper()} Zonal Mean')
    ax.set_xlabel('Latitude (°)')
    ax.set_ylabel('Pressure (hPa)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label=_get_var_label(var_name))
    
    plt.tight_layout()
    return fig


def _get_var_label(var_name: str) -> str:
    """Get appropriate label for variable."""
    labels = {
        't': 'Temperature (K)',
        'w': 'Vertical Velocity (Pa/s)',
        'u': 'Zonal Wind (m/s)',
        'z': 'Geopotential Height (m)',
        'v': 'Meridional Wind (m/s)'
    }
    return labels.get(var_name, var_name)


    