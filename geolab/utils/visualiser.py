"""
Visualization utilities for atmospheric model outputs.
"""
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from typing import Dict, List, Optional
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import colorcet as cc

# Ground truth value ranges from ERA5 data
GROUND_TRUTH_RANGES = {
    "t": {
        "horizontal": {  # For horizontal slices at specific pressure levels
            200: {"vmin": 193.58, "vmax": 231.82},
            500: {"vmin": 223.57, "vmax": 275.73},
            850: {"vmin": 225.91, "vmax": 305.72}
        },
        "vertical": {  # For meridional and zonal plots (all pressure levels)
            "vmin": 195.82,  # Min across all levels
            "vmax": 293.45   # Max across all levels
        }
    },
    "u": {
        "horizontal": {
            200: {"vmin": -92.99, "vmax": 92.99},
            500: {"vmin": -61.90, "vmax": 61.90},
            850: {"vmin": -51.97, "vmax": 51.97}
        },
        "vertical": {
            "vmin": -68.21,  # Use the most extreme symmetric range
            "vmax": 68.21
        }
    },
    "v": {
        "horizontal": {
            200: {"vmin": -63.03, "vmax": 63.03},
            500: {"vmin": -52.41, "vmax": 52.41},
            850: {"vmin": -54.49, "vmax": 54.49}
        },
        "vertical": {
            "vmin": -56.76,
            "vmax": 56.76
        }
    },
    "z": {
        "horizontal": {
            200: {"vmin": 102408.81, "vmax": 122770.81},
            500: {"vmin": 45832.16, "vmax": 58609.67},
            850: {"vmin": 6847.90, "vmax": 16639.25}
        },
        "vertical": {
            "vmin": 8399.23,
            "vmax": 122101.88
        }
    },
    "w": {
        "horizontal": {
            200: {"vmin": -0.26, "vmax": 0.20},
            500: {"vmin": -0.77, "vmax": 0.45},
            850: {"vmin": -0.72, "vmax": 0.67}
        },
        "vertical": {
            "vmin": -0.64,
            "vmax": 0.40
        }
    },
    "uv": {
        "horizontal": {
            200: {"vmin": 0.0, "vmax": 93.32},
            500: {"vmin": 0.0, "vmax": 66.51},
            850: {"vmin": 0.0, "vmax": 54.49}
        },
        "vertical": {
            "vmin": 0.0,
            "vmax": 93.32
        }
    }
}


# ---------------------------
# Batched inference helpers
# ---------------------------
def batched_model_forward(model, coords: torch.Tensor, batch_size: int = 50000) -> torch.Tensor:
    """
    Run model inference in CPU-friendly batches.
    Inputs:
        model: model on some device (cpu or cuda)
        coords: coordinates tensor on CPU (N, D)
    Returns:
        preds: CPU tensor of shape (N, n_vars)
    """
    preds_cpu = []
    n = coords.shape[0]
    for i in range(0, n, batch_size):
        batch = coords[i:i + batch_size].to(model.device, non_blocking=True)
        with torch.no_grad():
            out = model(batch)
        preds_cpu.append(out.cpu())
    if preds_cpu:
        return torch.cat(preds_cpu, dim=0)
    else:
        # empty
        return torch.empty((0, 0))


def batched_physics_residuals(
    model,
    coords: torch.Tensor,
    var_names: List[str],
    residual_fn,
    statistics=None,
    mass_balance=True,
    batch_size: int = 20000
):
    """
    Compute physics residuals in batches while preserving gradient requirements per-batch.
    residual_fn must accept (coords_batch, model_outputs_dict, statistics, mass_balance)
    and return tuple of tensors for residuals matching coords_batch length.
    Returns CPU tensors concatenated across batches.
    """
    ns_lon_list = []
    ns_lat_list = []
    mass_list = []

    n = coords.shape[0]
    for i in range(0, n, batch_size):
        cb = coords[i:i + batch_size].detach().requires_grad_(True).to(model.device, non_blocking=True)
        with torch.enable_grad():
            preds_batch = model(cb)
            model_outputs_dict = {var_names[j]: preds_batch[:, j] for j in range(len(var_names))}
            ns_lon_b, ns_lat_b, mass_b = residual_fn(
                cb, model_outputs_dict, statistics=statistics, mass_balance=mass_balance
            )
        ns_lon_list.append(ns_lon_b.detach().cpu())
        ns_lat_list.append(ns_lat_b.detach().cpu())
        mass_list.append(mass_b.detach().cpu())

    ns_lon = torch.cat(ns_lon_list, dim=0) if ns_lon_list else torch.empty((0,))
    ns_lat = torch.cat(ns_lat_list, dim=0) if ns_lat_list else torch.empty((0,))
    mass = torch.cat(mass_list, dim=0) if mass_list else torch.empty((0,))
    return ns_lon, ns_lat, mass


# ---------------------------
# Main plotting functions
# ---------------------------
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
    coords = _extract_coords_from_batch(batch)  # CPU tensor
    targets = _extract_targets_from_batch(batch)  # CPU tensor

    # Ensure targets on CPU
    if targets.is_cuda:
        targets = targets.cpu()

    # Get predictions (batched)
    preds = batched_model_forward(model, coords)

    # Compute errors
    errors = (preds - targets).pow(2)

    # Extract lat/lon (they are in coords)
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

    # Extract coordinates (keep on CPU)
    coords = _extract_coords_from_batch(batch)

    # Compute residuals in batches (preserves grads per-batch inside helper)
    statistics = getattr(model, 'statistics', None)
    mass_balance = getattr(model, 'mass_balance', True)

    ns_longitude, ns_latitude, mass_cont = batched_physics_residuals(
        model,
        coords,
        var_names,
        troposphere_pde_residual,
        statistics=statistics,
        mass_balance=mass_balance,
        batch_size=20000
    )

    # Convert coords to numpy for plotting
    lats = coords[:, 1].detach().cpu().numpy()
    lons = coords[:, 0].detach().cpu().numpy()

    # Create residual maps
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    residuals = {
        'NS Longitude': ns_longitude.abs().numpy(),
        'NS Latitude': ns_latitude.abs().numpy(),
        'Mass Continuity': mass_cont.abs().numpy()
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
    grid_resolution: Optional[dict] = None,
    use_ground_truth_ranges: bool = True
) -> Dict[str, plt.Figure]:
    """
    Plot 2D maps of model predictions at a fixed pressure level.

    Args:
        model: Lightning module
        batch: Validation batch (used only for timestep reference)
        pressure: Pressure level in hPa
        var_names: List of variable names
        grid_resolution: Grid resolution in degrees
        use_ground_truth_ranges: If True, use ground truth color ranges for consistency

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

    # Create dense regular grid at this pressure level (grids are kept on CPU)
    grid_coords, norm_grid_coords, n_lats, n_lons = _create_horizontal_grid(
        pressure,
        timestep=1.0,  # Middle of normalized time range [0, 1]
        resolution=grid_resolution,
        device=model.device,
        statistics=statistics,
        pi_scale=pi_scale
    )

    # Get model predictions on dense grid (batched)
    preds = batched_model_forward(model, norm_grid_coords)

    # Denormalize predictions
    for i in range(len(var_names)):
        var_min = statistics[var_names[i]][0]  # min value
        var_max = statistics[var_names[i]][1]  # max value

        # Reverse the [-1, 1] normalization: x_norm = 2*(x - min)/(max - min) - 1
        # So: x = (x_norm + 1) * (max - min) / 2 + min
        preds[:, i] = (preds[:, i] + 1.0) * (var_max - var_min) / 2.0 + var_min

    # Wind magnitude calculation
    if 'u' in var_names and 'v' in var_names:
        u_idx = var_names.index('u')
        v_idx = var_names.index('v')
        wind_mag = torch.sqrt(preds[:, u_idx]**2 + preds[:, v_idx]**2)

    # Create figures for each variable
    figures = {}

    for i, var_name in enumerate(var_names):
        pred_field = preds[:, i].reshape(n_lons, n_lats).cpu().numpy()

        fig = _create_horizontal_plot(
            pred_field, var_name, pressure,
            use_ground_truth_ranges=use_ground_truth_ranges
        )
        figures[var_name] = fig

    if 'u' in var_names and 'v' in var_names:
        wind_mag_field = wind_mag.reshape(n_lons, n_lats).cpu().numpy()
        fig = _create_horizontal_plot(
            wind_mag_field, 'uv', pressure,
            use_ground_truth_ranges=use_ground_truth_ranges
        )
        figures['uv'] = fig

    return figures


def plot_meridional_slices(
    model,
    batch: Dict,
    longitude: int,
    var_names: List[str],
    pressure_levels: Optional[List[int]] = None,
    grid_resolution: Optional[dict] = None
) -> Dict[str, plt.Figure]:
    """
    Plot vertical cross-sections along a meridian (lat-pressure slice).

    Args:
        model: Lightning module
        batch: Validation batch (used only for timestep reference)
        longitude: Fixed longitude for the slice
        var_names: List of variable names
        pressure_levels: List of pressure levels in hPa. If None, uses standard ERA5 levels.
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

    # Use standard ERA5 pressure levels if not specified
    if pressure_levels is None:
        pressure_levels = [850, 825, 800, 775, 750, 700, 650, 600, 550, 500,
                          450, 400, 350, 300, 250, 225, 200]

    # Create lat-pressure grid at fixed longitude (CPU)
    grid_coords, norm_grid_coords, n_lats = _create_meridional_grid(
        longitude,
        pressure_levels,
        timestep=1.0,
        resolution=grid_resolution,
        device=model.device,
        statistics=statistics,
        pi_scale=pi_scale
    )

    # Get model predictions (batched)
    preds = batched_model_forward(model, norm_grid_coords)

    # Denormalize predictions
    for i in range(len(var_names)):
        var_min = statistics[var_names[i]][0]
        var_max = statistics[var_names[i]][1]
        preds[:, i] = (preds[:, i] + 1.0) * (var_max - var_min) / 2.0 + var_min

    # Create figures for each variable
    figures = {}
    n_pressure = len(pressure_levels)

    for i, var_name in enumerate(var_names):
        pred_field = preds[:, i].reshape(n_lats, n_pressure).cpu().numpy()

        fig = _create_meridional_plot(
            pred_field, var_name, longitude, pressure_levels, grid_resolution, True
        )
        figures[var_name] = fig

    return figures


def plot_zonal_mean(
    model,
    batch: Dict,
    var_names: List[str],
    pressure_levels: Optional[List[int]] = None,
    grid_resolution: Optional[dict] = None
) -> Dict[str, plt.Figure]:
    """
    Plot zonal mean (longitude-averaged) profiles.

    Args:
        model: Lightning module
        batch: Validation batch (used only for timestep reference)
        var_names: List of variable names
        pressure_levels: List of pressure levels in hPa. If None, uses standard ERA5 levels.
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

    # Use standard ERA5 pressure levels if not specified
    if pressure_levels is None:
        pressure_levels = [850, 825, 800, 775, 750, 700, 650, 600, 550, 500,
                          450, 400, 350, 300, 250, 225, 200]

    # Create full lat-lon-pressure grid (CPU)
    grid_coords, norm_grid_coords, n_lats, n_lons, n_pressure = _create_full_grid(
        pressure_levels,
        timestep=1.0,
        resolution=grid_resolution,
        device=model.device,
        statistics=statistics,
        pi_scale=pi_scale
    )

    # Get model predictions (batched)
    preds = batched_model_forward(model, norm_grid_coords)

    # Denormalize predictions
    for i in range(len(var_names)):
        var_min = statistics[var_names[i]][0]
        var_max = statistics[var_names[i]][1]
        preds[:, i] = (preds[:, i] + 1.0) * (var_max - var_min) / 2.0 + var_min

    figures = {}

    for i, var_name in enumerate(var_names):
        pred_field = preds[:, i].reshape(n_lons, n_lats, n_pressure).cpu().numpy()

        # Zonal mean (average over longitude dimension)
        pred_zonal = pred_field.mean(axis=0)  # [n_lats, n_pressure]

        fig = _create_zonal_mean_plot(
            pred_zonal, var_name, pressure_levels, grid_resolution, True
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
    lats = torch.arange(-90, 90 + resolution['latitude'], resolution['latitude'], dtype=torch.float32)

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
    lon_min, lon_max = statistics["longitude"][:2]
    norm_grid[:, 0] = 2.0 * (grid[:, 0] - lon_min) / (lon_max - lon_min) - 1.0
    if pi_scale:
        norm_grid[:, 0] = norm_grid[:, 0] * torch.pi

    # Latitude: normalize to [-1, 1], then optionally scale by π/2
    lat_min, lat_max = statistics["latitude"][:2]
    norm_grid[:, 1] = 2.0 * (grid[:, 1] - lat_min) / (lat_max - lat_min) - 1.0
    if pi_scale:
        norm_grid[:, 1] = norm_grid[:, 1] * (torch.pi / 2)

    # Pressure:
    p_min, p_max = statistics["pressure_level"][:2]
    if p_min == p_max:
        norm_grid[:, 2] = 1.0
    else:
        norm_grid[:, 2] = 2.0 * (grid[:, 2] - p_min) / (p_max - p_min) - 1.0

    # Time: already in [0, 1] range, use as-is
    norm_grid[:, 3] = timestep

    # NOTE: do NOT move big grids to GPU here; do batched sends inside inference helpers.
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

    lats = torch.arange(-90, 90 + resolution['latitude'], resolution['latitude'], dtype=torch.float32)
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

    # Pressure:
    pressure_min = statistics["pressure_level"][0]
    pressure_max = statistics["pressure_level"][1]
    if pressure_min == pressure_max:
        norm_grid[:, 2] = 1.0
    else:
        norm_grid[:, 2] = 2.0 * (grid[:, 2] - pressure_min) / (pressure_max - pressure_min) - 1.0
    # Time: already in [0, 1] range
    norm_grid[:, 3] = timestep

    # NOTE: keep on CPU
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
    lats = torch.arange(-90, 90 + resolution['latitude'], resolution['latitude'], dtype=torch.float32)
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

    # Pressure:
    pressure_min = statistics["pressure_level"][0]
    pressure_max = statistics["pressure_level"][1]
    if pressure_min == pressure_max:
        norm_grid[:, 2] = 1.0
    else:
        norm_grid[:, 2] = 2.0 * (grid[:, 2] - pressure_min) / (pressure_max - pressure_min) - 1.0

    # Time: already in [0, 1] range
    norm_grid[:, 3] = timestep

    # NOTE: do NOT move big grids to GPU here
    return grid, norm_grid, n_lats, n_lons, n_pressures


def _create_horizontal_plot(
    pred_field: np.ndarray,
    var_name: str,
    pressure: int,
    use_ground_truth_ranges: bool = True  # ADD THIS PARAMETER
) -> plt.Figure:
    """Create a single-panel plot showing model prediction."""
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add coastlines and land
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)

    extent = [-180, 180, -90, 90]

    # Determine colormap and limits based on variable
    if use_ground_truth_ranges and var_name in GROUND_TRUTH_RANGES and pressure in GROUND_TRUTH_RANGES[var_name][
        'horizontal']:
        # Use ground truth ranges for consistent comparison
        vmin = GROUND_TRUTH_RANGES[var_name]["horizontal"][pressure]["vmin"]
        vmax = GROUND_TRUTH_RANGES[var_name]["horizontal"][pressure]["vmax"]

        # Still need to determine colormap
        if var_name in ['u', 'v']:
            cmap = 'RdBu_r'
        elif var_name == 'uv':
            cmap = 'cet_CET_R3'
        elif var_name == 't':
            cmap = 'cet_CET_R1'
        elif var_name == 'z':
            cmap = 'cet_rainbow'
        else:
            cmap = 'viridis'
    else:
        # Fall back to original logic if not using ground truth ranges
        if var_name in ['u', 'v']:
            cmap = 'RdBu_r'
            vmax = pred_field.max()
            vmin = pred_field.min()
        elif var_name == 'uv':
            cmap = 'cet_CET_R3'
            vmin = 0
            vmax = None
        elif var_name == 't':
            cmap = 'cet_CET_R1'
            vmin, vmax = None, None
        elif var_name == 'z':
            cmap = 'cet_rainbow'
            vmin, vmax = None, None
        else:
            cmap = 'viridis'
            vmin, vmax = None, None

    im = ax.imshow(
        pred_field.T, origin='lower', extent=extent,
        cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax,
        transform=ccrs.PlateCarree()
    )
    ax.set_title(f'{var_name.upper()} @ {pressure} hPa')

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    plt.colorbar(im, ax=ax, label=_get_var_label(var_name))
    plt.tight_layout()

    return fig


def _create_meridional_plot(
    pred_field: np.ndarray,
    var_name: str,
    longitude: int,
    pressure_levels: List[int],
    resolution: Optional[dict] = None,
    use_ground_truth_ranges: bool = True
) -> plt.Figure:
    """Create meridional cross-section plot."""
    if resolution is None:
        resolution = {'longitude': 2, 'latitude': 2}

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    lats = np.arange(-90, 90 + resolution['latitude'], resolution['latitude'])
    pressures = np.array(pressure_levels)

    # Determine colormap (matching ground truth visualization)
    cmap = _get_colormap(var_name)

    # Determine vmin/vmax
    if use_ground_truth_ranges and var_name in GROUND_TRUTH_RANGES and "vertical" in GROUND_TRUTH_RANGES[var_name]:
        if GROUND_TRUTH_RANGES[var_name] != 'w':
            vmin = GROUND_TRUTH_RANGES[var_name]["vertical"]["vmin"]
            vmax = GROUND_TRUTH_RANGES[var_name]["vertical"]["vmax"]
        else:
            vmin = np.nanpercentile(pred_field, 1)
            vmax = np.nanpercentile(pred_field, 90)
    else:
        # Fall back to data-based limits
        if var_name in ['u', 'v']:
            vmax = np.abs(pred_field).max()
            vmin = -vmax
        elif var_name == 'w':
            vmin = np.nanpercentile(pred_field, 1)
            vmax = np.nanpercentile(pred_field, 90)
        else:
            vmin = pred_field.min()
            vmax = pred_field.max()

    # Contour plot
    levels = np.linspace(vmin, vmax, 21)  # 21 values creates 20 intervals
    if var_name == 'w':
        im = ax.contourf(lats, pressures, pred_field.T, levels=levels, cmap=cmap, extend='both')
    else:
        im = ax.contourf(lats, pressures, pred_field.T, levels=levels, cmap=cmap)
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
    resolution: Optional[dict] = None,
    use_ground_truth_ranges: bool = True
) -> plt.Figure:
    """Create zonal mean plot."""
    if resolution is None:
        resolution = {'longitude': 2, 'latitude': 2}

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    lats = np.arange(-90, 90 + resolution['latitude'], resolution['latitude'])
    pressures = np.array(pressure_levels)

    # Determine colormap (matching ground truth visualization)
    cmap = _get_colormap(var_name)

    # Determine vmin/vmax
    if use_ground_truth_ranges and var_name in GROUND_TRUTH_RANGES and "vertical" in GROUND_TRUTH_RANGES[var_name]:
        vmin = GROUND_TRUTH_RANGES[var_name]["vertical"]["vmin"]
        vmax = GROUND_TRUTH_RANGES[var_name]["vertical"]["vmax"]
    else:
        # Fall back to data-based limits
        if var_name in ['u', 'v']:
            vmax = np.abs(pred_zonal).max()
            vmin = -vmax
        elif var_name == 'w':
            vmin = np.nanpercentile(pred_zonal, 1)
            vmax = np.nanpercentile(pred_zonal, 90)
        else:
            vmin = pred_zonal.min()
            vmax = pred_zonal.max()

    # Contour plot
    levels = np.linspace(vmin, vmax, 21)  # 21 values creates 20 intervals
    if var_name == 'w':
        im = ax.contourf(lats, pressures, pred_zonal.T, levels=levels, cmap=cmap, extend='both')
    else:
        im = ax.contourf(lats, pressures, pred_zonal.T, levels=levels, cmap=cmap)
    ax.set_title(f'{var_name.upper()} Zonal Mean')
    ax.set_xlabel('Latitude (°)')
    ax.set_ylabel('Pressure (hPa)')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    plt.colorbar(im, ax=ax, label=_get_var_label(var_name))

    plt.tight_layout()
    return fig


def _get_colormap(var_name: str) -> str:
    """Get appropriate colormap for variable (matching ground truth visualization)."""
    if var_name in ['u', 'v']:
        return 'RdBu_r'
    elif var_name == 'uv':
        return 'cet_CET_R3'
    elif var_name == 't':
        return 'cet_CET_R1'
    elif var_name == 'z':
        return 'cet_rainbow'
    else:
        return 'viridis'


def _get_var_label(var_name: str) -> str:
    """Get appropriate label for variable."""
    labels = {
        't': 'Temperature (K)',
        'w': 'Vertical Velocity (Pa/s)',
        'u': 'Zonal Wind (m/s)',
        'z': 'Geopotential (m²/s²)',
        'v': 'Meridional Wind (m/s)',
        'uv': 'Wind Magnitude (m/s)'
    }
    return labels.get(var_name, var_name)
