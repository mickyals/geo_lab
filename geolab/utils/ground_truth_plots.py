"""
Ground truth visualization utilities for ERA5 dataset.
Creates horizontal slice plots matching model visualization style.
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import colorcet as cc
from pathlib import Path
from typing import List, Optional, Dict
import json


def plot_era5_horizontal_slices(
    data_path: str,
    var_names: List[str],
    pressure_levels: List[int] = [200, 500, 850],
    output_dir: str = "./ground_truth_plots",
    time_dim: str = "valid_time",
    pressure_dim: str = "pressure_level",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude"
) -> Dict:
    """
    Generate horizontal slice plots from ERA5 data for specified variables and pressure levels.
    
    Args:
        data_path: Path to ERA5 netCDF file
        var_names: List of variable names to plot (e.g., ['t', 'u', 'v', 'z', 'w'])
        pressure_levels: List of pressure levels in hPa
        output_dir: Directory to save plots
        time_dim: Name of time dimension in dataset
        pressure_dim: Name of pressure dimension in dataset
        lat_dim: Name of latitude dimension in dataset
        lon_dim: Name of longitude dimension in dataset
    
    Returns:
        Dictionary containing vmin/vmax ranges for each variable and pressure level
    """
    # Load dataset
    print(f"Loading ERA5 data from {data_path}")
    data = xr.open_dataset(data_path)
    
    # Adjust longitude to [-180, 180] if needed
    if data[lon_dim].min() >= 0:
        data[lon_dim] = (data[lon_dim] + 180) % 360 - 180
        data = data.sortby(lon_dim)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get time steps
    time_steps = data[time_dim].values
    n_times = len(time_steps)
    
    print(f"Found {n_times} time steps")
    print(f"Processing variables: {var_names}")
    print(f"Pressure levels: {pressure_levels} hPa")
    
    # Dictionary to store value ranges
    value_ranges = {}
    
    # Process each variable
    for var_name in var_names:
        if var_name not in data.variables:
            print(f"Warning: Variable '{var_name}' not found in dataset. Skipping.")
            continue
        
        var_dir = output_path / var_name
        var_dir.mkdir(exist_ok=True)
        
        value_ranges[var_name] = {}
        
        # Process each pressure level
        for pressure in pressure_levels:
            pressure_dir = var_dir / f"{pressure}hPa"
            pressure_dir.mkdir(exist_ok=True)
            
            print(f"\nProcessing {var_name} @ {pressure} hPa...")
            
            # Select data at this pressure level
            try:
                data_p = data[var_name].sel({pressure_dim: pressure})
            except KeyError:
                print(f"  Pressure level {pressure} not found. Skipping.")
                continue
            
            # Track min/max across all timesteps
            all_values = []
            
            # Process each time step
            for idx, t in enumerate(time_steps):
                data_t = data_p.sel({time_dim: t})
                field = data_t.values
                all_values.append(field)
                
                # Create figure
                fig = _create_horizontal_ground_truth_plot(
                    data_t,
                    var_name,
                    pressure,
                    t,
                    lat_dim,
                    lon_dim
                )
                
                # Save figure
                time_str = str(t).replace(':', '-').replace(' ', '_')
                filename = f"{var_name}_{pressure}hPa_t{idx:03d}_{time_str}.png"
                filepath = pressure_dir / filename
                
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{n_times} time steps")
            
            # Calculate statistics across all timesteps
            all_values_flat = np.concatenate([v.flatten() for v in all_values])
            vmin_data = float(np.nanmin(all_values_flat))
            vmax_data = float(np.nanmax(all_values_flat))
            
            # Determine vmin/vmax based on variable type
            if var_name in ['u', 'v']:
                # Symmetric around zero, using max absolute value
                vmax = float(np.nanmax(np.abs(all_values_flat)))
                vmin = -vmax
            elif var_name == 'uv':
                vmin = 0.0
                vmax = vmax_data
            else:
                # Use actual data range
                vmin = vmin_data
                vmax = vmax_data
            
            value_ranges[var_name][f"{pressure}hPa"] = {
                'vmin': vmin,
                'vmax': vmax,
                'data_min': vmin_data,
                'data_max': vmax_data
            }
            
            print(f"  Completed {var_name} @ {pressure} hPa")
            print(f"  Value range: [{vmin:.4f}, {vmax:.4f}]")
    
    # Create wind magnitude plots if both u and v are available
    if 'u' in var_names and 'v' in var_names and 'u' in data.variables and 'v' in data.variables:
        print("\nGenerating wind magnitude plots...")
        uv_ranges = _create_wind_magnitude_plots(
            data, pressure_levels, output_path, time_steps,
            time_dim, pressure_dim, lat_dim, lon_dim
        )
        value_ranges['uv'] = uv_ranges
    
    # Print summary
    print("\n" + "="*80)
    print("VALUE RANGES SUMMARY")
    print("="*80)
    for var_name, pressure_data in value_ranges.items():
        print(f"\n{var_name.upper()}:")
        for pressure_str, ranges in pressure_data.items():
            print(f"  {pressure_str}:")
            print(f"    Plot range: vmin={ranges['vmin']:.4f}, vmax={ranges['vmax']:.4f}")
            print(f"    Data range: [{ranges['data_min']:.4f}, {ranges['data_max']:.4f}]")
    
    # Save to JSON file
    json_path = output_path / "value_ranges.json"
    with open(json_path, 'w') as f:
        json.dump(value_ranges, f, indent=2)
    print(f"\nValue ranges saved to {json_path}")
    
    print(f"\nAll plots saved to {output_dir}")
    
    return value_ranges


def _create_horizontal_ground_truth_plot(
    data_array: xr.DataArray,
    var_name: str,
    pressure: int,
    timestamp,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude"
) -> plt.Figure:
    """
    Create a single horizontal plot matching the model visualization style.
    
    Args:
        data_array: xarray DataArray for a single time/pressure slice
        var_name: Variable name
        pressure: Pressure level in hPa
        timestamp: Time value for title
        lat_dim: Name of latitude dimension
        lon_dim: Name of longitude dimension
    
    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add coastlines and land
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    
    # Get data as numpy array
    field = data_array.values
    lons = data_array[lon_dim].values
    lats = data_array[lat_dim].values
    
    # Determine colormap and limits based on variable
    cmap, vmin, vmax = _get_colormap_and_limits(var_name, field)
    
    # Determine the extent
    extent = [lons.min(), lons.max(), lats.min(), lats.max()]
    
    # Check the dimensions - field should be (lon, lat) for correct plotting
    # If field is (lat, lon), we need to transpose
    if field.shape[0] == len(lats):
        # Data is (lat, lon), need to transpose to (lon, lat)
        field = field.T
    
    # Use imshow (matches model viz style)
    # field.T because imshow expects (lat, lon) but we have (lon, lat)
    # origin='upper' because latitude typically goes from 90 to -90
    im = ax.imshow(
        field.T,
        origin='upper',
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
        transform=ccrs.PlateCarree()
    )
    
    ax.set_title(f'{var_name.upper()} @ {pressure} hPa — {timestamp}')
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label=_get_var_label(var_name), shrink=0.8)
    plt.tight_layout()
    
    return fig


def _create_wind_magnitude_plots(
    data: xr.Dataset,
    pressure_levels: List[int],
    output_path: Path,
    time_steps,
    time_dim: str,
    pressure_dim: str,
    lat_dim: str,
    lon_dim: str
) -> Dict:
    """Create wind magnitude plots from u and v components."""
    uv_dir = output_path / "uv"
    uv_dir.mkdir(exist_ok=True)
    
    uv_ranges = {}
    
    for pressure in pressure_levels:
        pressure_dir = uv_dir / f"{pressure}hPa"
        pressure_dir.mkdir(exist_ok=True)
        
        print(f"\nProcessing wind magnitude @ {pressure} hPa...")
        
        try:
            u_data = data['u'].sel({pressure_dim: pressure})
            v_data = data['v'].sel({pressure_dim: pressure})
        except KeyError:
            print(f"  Pressure level {pressure} not found. Skipping.")
            continue
        
        all_wind_mags = []
        
        for idx, t in enumerate(time_steps):
            u_t = u_data.sel({time_dim: t})
            v_t = v_data.sel({time_dim: t})
            
            # Compute wind magnitude
            wind_mag = np.sqrt(u_t.values**2 + v_t.values**2)
            all_wind_mags.append(wind_mag)
            
            # Create DataArray for plotting - keep same dims as u_t
            wind_mag_da = xr.DataArray(
                wind_mag,
                coords={lat_dim: u_t[lat_dim], lon_dim: u_t[lon_dim]},
                dims=u_t.dims  # Use the same dimension order as the original data
            )
            
            # Create figure
            fig = _create_horizontal_ground_truth_plot(
                wind_mag_da, 'uv', pressure, t, lat_dim, lon_dim
            )
            
            # Save figure
            time_str = str(t).replace(':', '-').replace(' ', '_')
            filename = f"uv_{pressure}hPa_t{idx:03d}_{time_str}.png"
            filepath = pressure_dir / filename
            
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(time_steps)} time steps")
        
        # Calculate statistics
        all_wind_mags_flat = np.concatenate([v.flatten() for v in all_wind_mags])
        vmin_data = float(np.nanmin(all_wind_mags_flat))
        vmax_data = float(np.nanmax(all_wind_mags_flat))
        
        # Wind magnitude: vmin=0, vmax=data max
        vmin = 0.0
        vmax = vmax_data
        
        uv_ranges[f"{pressure}hPa"] = {
            'vmin': vmin,
            'vmax': vmax,
            'data_min': vmin_data,
            'data_max': vmax_data
        }
        
        print(f"  Completed wind magnitude @ {pressure} hPa")
        print(f"  Value range: [{vmin:.4f}, {vmax:.4f}]")
    
    return uv_ranges


def _get_colormap_and_limits(var_name: str, field: np.ndarray) -> tuple:
    """
    Get appropriate colormap and value limits for variable.
    Matches the style used in model visualizations.
    """
    if var_name in ['u', 'v']:
        cmap = 'RdBu_r'
        vmax = np.abs(field).max()
        vmin = -vmax
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
    
    return cmap, vmin, vmax


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


# Example usage
if __name__ == "__main__":
    # Configuration
    DATA_PATH = r"C:\Users\freez\Downloads\dfda178b22d3772b9b9b0118dba68a02.nc"
    VARIABLES = ['t', 'u', 'v', 'z', 'w']  # Adjust based on your dataset
    PRESSURE_LEVELS = [200, 500, 850]
    OUTPUT_DIR = "./ground_truth_plots"
    
    # Generate plots and get value ranges
    value_ranges = plot_era5_horizontal_slices(
        data_path=DATA_PATH,
        var_names=VARIABLES,
        pressure_levels=PRESSURE_LEVELS,
        output_dir=OUTPUT_DIR
    )