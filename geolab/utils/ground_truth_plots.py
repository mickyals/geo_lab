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


# Ground truth value ranges - computed once, used for all plots
GROUND_TRUTH_RANGES = {
    "t": {
        200: {"vmin": 193.58, "vmax": 231.82},
        500: {"vmin": 223.57, "vmax": 275.73},
        850: {"vmin": 225.91, "vmax": 305.72}
    },
    "u": {
        200: {"vmin": -92.99, "vmax": 92.99},
        500: {"vmin": -61.90, "vmax": 61.90},
        850: {"vmin": -51.97, "vmax": 51.97}
    },
    "v": {
        200: {"vmin": -63.03, "vmax": 63.03},
        500: {"vmin": -52.41, "vmax": 52.41},
        850: {"vmin": -54.49, "vmax": 54.49}
    },
    "z": {
        200: {"vmin": 102408.81, "vmax": 122770.81},
        500: {"vmin": 45832.16, "vmax": 58609.67},
        850: {"vmin": 6847.90, "vmax": 16639.25}
    },
    "w": {
        200: {"vmin": -9.78, "vmax": 2.07},
        500: {"vmin": -14.14, "vmax": 4.63},
        850: {"vmin": -11.98, "vmax": 8.34}
    },
    "uv": {
        200: {"vmin": 0.0, "vmax": 93.32},
        500: {"vmin": 0.0, "vmax": 66.51},
        850: {"vmin": 0.0, "vmax": 54.49}
    }
}


def plot_era5_horizontal_slices(
    data_path: str,
    var_names: List[str],
    pressure_levels: List[int] = [200, 500, 850],
    output_dir: str = "./ground_truth_plots",
    time_dim: str = "valid_time",
    pressure_dim: str = "pressure_level",
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    use_fixed_ranges: bool = True,
    compute_ranges_first: bool = False
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
        use_fixed_ranges: If True, use GROUND_TRUTH_RANGES for consistent colormaps
        compute_ranges_first: If True, compute ranges from data first (two-pass approach)
    
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
    
    # First pass: compute ranges if requested
    if compute_ranges_first:
        print("\n" + "="*80)
        print("FIRST PASS: Computing value ranges across all timesteps")
        print("="*80)
        value_ranges = _compute_value_ranges(
            data, var_names, pressure_levels, time_steps,
            time_dim, pressure_dim, lat_dim, lon_dim
        )
        print("\nRanges computed. Starting plot generation...")
    
    # Process each variable
    for var_name in var_names:
        if var_name not in data.variables:
            print(f"Warning: Variable '{var_name}' not found in dataset. Skipping.")
            continue
        
        var_dir = output_path / var_name
        var_dir.mkdir(exist_ok=True)
        
        if not compute_ranges_first:
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
            
            # Determine vmin/vmax to use for this variable/pressure
            if use_fixed_ranges and var_name in GROUND_TRUTH_RANGES and pressure in GROUND_TRUTH_RANGES[var_name]:
                vmin = GROUND_TRUTH_RANGES[var_name][pressure]["vmin"]
                vmax = GROUND_TRUTH_RANGES[var_name][pressure]["vmax"]
                print(f"  Using fixed ranges: [{vmin:.4f}, {vmax:.4f}]")
            elif compute_ranges_first:
                vmin = value_ranges[var_name][f"{pressure}hPa"]["vmin"]
                vmax = value_ranges[var_name][f"{pressure}hPa"]["vmax"]
                print(f"  Using computed ranges: [{vmin:.4f}, {vmax:.4f}]")
            else:
                vmin, vmax = None, None
                print(f"  Using auto-scaling per timestep")
            
            # Process each time step
            for idx, t in enumerate(time_steps):
                data_t = data_p.sel({time_dim: t})
                
                # Create figure with fixed or auto vmin/vmax
                fig = _create_horizontal_ground_truth_plot(
                    data_t,
                    var_name,
                    pressure,
                    t,
                    lat_dim,
                    lon_dim,
                    vmin=vmin,
                    vmax=vmax
                )
                
                # Save figure
                time_str = str(t).replace(':', '-').replace(' ', '_')
                filename = f"{var_name}_{pressure}hPa_t{idx:03d}_{time_str}.png"
                filepath = pressure_dir / filename
                
                fig.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                if (idx + 1) % 10 == 0:
                    print(f"  Processed {idx + 1}/{n_times} time steps")
            
            print(f"  Completed {var_name} @ {pressure} hPa")
    
    # Create wind magnitude plots if both u and v are available
    if 'u' in var_names and 'v' in var_names and 'u' in data.variables and 'v' in data.variables:
        print("\nGenerating wind magnitude plots...")
        uv_ranges = _create_wind_magnitude_plots(
            data, pressure_levels, output_path, time_steps,
            time_dim, pressure_dim, lat_dim, lon_dim,
            use_fixed_ranges=use_fixed_ranges,
            compute_ranges_first=compute_ranges_first
        )
        if not compute_ranges_first:
            value_ranges['uv'] = uv_ranges
    
    # Print summary
    if compute_ranges_first or not use_fixed_ranges:
        print("\n" + "="*80)
        print("VALUE RANGES SUMMARY")
        print("="*80)
        for var_name, pressure_data in value_ranges.items():
            print(f"\n{var_name.upper()}:")
            for pressure_str, ranges in pressure_data.items():
                print(f"  {pressure_str}:")
                print(f"    Plot range: vmin={ranges['vmin']:.4f}, vmax={ranges['vmax']:.4f}")
                if 'data_min' in ranges:
                    print(f"    Data range: [{ranges['data_min']:.4f}, {ranges['data_max']:.4f}]")
        
        # Save to JSON file
        json_path = output_path / "value_ranges.json"
        with open(json_path, 'w') as f:
            json.dump(value_ranges, f, indent=2)
        print(f"\nValue ranges saved to {json_path}")
    
    print(f"\nAll plots saved to {output_dir}")
    
    return value_ranges


def _compute_value_ranges(
    data: xr.Dataset,
    var_names: List[str],
    pressure_levels: List[int],
    time_steps,
    time_dim: str,
    pressure_dim: str,
    lat_dim: str,
    lon_dim: str
) -> Dict:
    """Compute value ranges across all timesteps for each variable/pressure."""
    value_ranges = {}
    
    for var_name in var_names:
        if var_name not in data.variables:
            continue
        
        value_ranges[var_name] = {}
        
        for pressure in pressure_levels:
            try:
                data_p = data[var_name].sel({pressure_dim: pressure})
            except KeyError:
                continue
            
            print(f"  Computing ranges for {var_name} @ {pressure} hPa...")
            
            # Collect all values across timesteps
            all_values = []
            for t in time_steps:
                data_t = data_p.sel({time_dim: t})
                all_values.append(data_t.values)
            
            # Calculate statistics
            all_values_flat = np.concatenate([v.flatten() for v in all_values])
            vmin_data = float(np.nanmin(all_values_flat))
            vmax_data = float(np.nanmax(all_values_flat))
            
            # Determine vmin/vmax based on variable type
            if var_name in ['u', 'v']:
                vmax = float(np.nanmax(np.abs(all_values_flat)))
                vmin = -vmax
            elif var_name == 'uv':
                vmin = 0.0
                vmax = vmax_data
            else:
                vmin = vmin_data
                vmax = vmax_data
            
            value_ranges[var_name][f"{pressure}hPa"] = {
                'vmin': vmin,
                'vmax': vmax,
                'data_min': vmin_data,
                'data_max': vmax_data
            }
    
    return value_ranges


def _create_horizontal_ground_truth_plot(
    data_array: xr.DataArray,
    var_name: str,
    pressure: int,
    timestamp,
    lat_dim: str = "latitude",
    lon_dim: str = "longitude",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
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
        vmin: Optional fixed minimum value for colormap
        vmax: Optional fixed maximum value for colormap
    
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
    
    # Determine colormap
    cmap = _get_colormap(var_name)
    
    # If vmin/vmax not provided, compute from field
    if vmin is None or vmax is None:
        field_vmin, field_vmax = _get_field_limits(var_name, field)
        if vmin is None:
            vmin = field_vmin
        if vmax is None:
            vmax = field_vmax
    
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
    lon_dim: str,
    use_fixed_ranges: bool = True,
    compute_ranges_first: bool = False
) -> Dict:
    """Create wind magnitude plots from u and v components."""
    uv_dir = output_path / "uv"
    uv_dir.mkdir(exist_ok=True)
    
    uv_ranges = {}
    
    # First pass: compute ranges if needed
    computed_ranges = {}
    if compute_ranges_first:
        print("  Computing wind magnitude ranges...")
        for pressure in pressure_levels:
            try:
                u_data = data['u'].sel({pressure_dim: pressure})
                v_data = data['v'].sel({pressure_dim: pressure})
            except KeyError:
                continue
            
            all_wind_mags = []
            for t in time_steps:
                u_t = u_data.sel({time_dim: t})
                v_t = v_data.sel({time_dim: t})
                wind_mag = np.sqrt(u_t.values**2 + v_t.values**2)
                all_wind_mags.append(wind_mag)
            
            all_wind_mags_flat = np.concatenate([v.flatten() for v in all_wind_mags])
            vmax_data = float(np.nanmax(all_wind_mags_flat))
            computed_ranges[pressure] = {'vmin': 0.0, 'vmax': vmax_data}
    
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
        
        # Determine vmin/vmax to use
        if use_fixed_ranges and pressure in GROUND_TRUTH_RANGES.get('uv', {}):
            vmin = GROUND_TRUTH_RANGES['uv'][pressure]["vmin"]
            vmax = GROUND_TRUTH_RANGES['uv'][pressure]["vmax"]
            print(f"  Using fixed ranges: [{vmin:.4f}, {vmax:.4f}]")
        elif compute_ranges_first and pressure in computed_ranges:
            vmin = computed_ranges[pressure]['vmin']
            vmax = computed_ranges[pressure]['vmax']
            print(f"  Using computed ranges: [{vmin:.4f}, {vmax:.4f}]")
        else:
            vmin, vmax = None, None
            print(f"  Using auto-scaling per timestep")
        
        for idx, t in enumerate(time_steps):
            u_t = u_data.sel({time_dim: t})
            v_t = v_data.sel({time_dim: t})
            
            # Compute wind magnitude
            wind_mag = np.sqrt(u_t.values**2 + v_t.values**2)
            
            # Create DataArray for plotting - keep same dims as u_t
            wind_mag_da = xr.DataArray(
                wind_mag,
                coords={lat_dim: u_t[lat_dim], lon_dim: u_t[lon_dim]},
                dims=u_t.dims
            )
            
            # Create figure with fixed or auto vmin/vmax
            fig = _create_horizontal_ground_truth_plot(
                wind_mag_da, 'uv', pressure, t, lat_dim, lon_dim,
                vmin=vmin, vmax=vmax
            )
            
            # Save figure
            time_str = str(t).replace(':', '-').replace(' ', '_')
            filename = f"uv_{pressure}hPa_t{idx:03d}_{time_str}.png"
            filepath = pressure_dir / filename
            
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(time_steps)} time steps")
        
        if compute_ranges_first and pressure in computed_ranges:
            uv_ranges[f"{pressure}hPa"] = computed_ranges[pressure]
        
        print(f"  Completed wind magnitude @ {pressure} hPa")
    
    return uv_ranges


def _get_colormap(var_name: str) -> str:
    """Get appropriate colormap for variable."""
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


def _get_field_limits(var_name: str, field: np.ndarray) -> tuple:
    """Get value limits for field based on variable type."""
    if var_name in ['u', 'v']:
        vmax = np.abs(field).max()
        vmin = -vmax
    elif var_name == 'uv':
        vmin = 0
        vmax = field.max()
    else:
        vmin = field.min()
        vmax = field.max()
    return vmin, vmax


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
    VARIABLES = ['t', 'u', 'v', 'z', 'w']
    PRESSURE_LEVELS = [200, 500, 850]
    OUTPUT_DIR = "./ground_truth_plots"
    
    # Generate plots with fixed ranges (using GROUND_TRUTH_RANGES)
    value_ranges = plot_era5_horizontal_slices(
        data_path=DATA_PATH,
        var_names=VARIABLES,
        pressure_levels=PRESSURE_LEVELS,
        output_dir=OUTPUT_DIR,
        use_fixed_ranges=True,  # Use the predefined GROUND_TRUTH_RANGES
        compute_ranges_first=False  # Don't compute ranges, use fixed ones
    )
    
    # Alternative: Compute ranges first, then plot with consistent colors
    # (useful if you don't have GROUND_TRUTH_RANGES yet)
    # value_ranges = plot_era5_horizontal_slices(
    #     data_path=DATA_PATH,
    #     var_names=VARIABLES,
    #     pressure_levels=PRESSURE_LEVELS,
    #     output_dir=OUTPUT_DIR,
    #     use_fixed_ranges=False,
    #     compute_ranges_first=True
    # )