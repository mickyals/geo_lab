import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import cm
import numpy as np
import colorcet as cc
from scipy.ndimage import zoom

# Ground truth ranges for consistent coloring
GROUND_TRUTH_RANGES = {
    "t": {
        "horizontal": {
            200: {"vmin": 193.58, "vmax": 231.82},
            500: {"vmin": 223.57, "vmax": 275.73},
            850: {"vmin": 225.91, "vmax": 305.72}
        },
        "vertical": {
            "vmin": 195.82,
            "vmax": 293.45
        }
    },
    "u": {
        "horizontal": {
            200: {"vmin": -92.99, "vmax": 92.99},
            500: {"vmin": -61.90, "vmax": 61.90},
            850: {"vmin": -51.97, "vmax": 51.97}
        },
        "vertical": {
            "vmin": -68.21,
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
    """Get label for colorbar based on variable name."""
    labels = {
        'u': 'Zonal Wind (m/s)',
        'v': 'Meridional Wind (m/s)',
        'uv': 'Wind Speed (m/s)',
        't': 'Temperature (K)',
        'z': 'Geopotential Height (m)',
        'w': 'Vertical Velocity (Pa/s)'
    }
    return labels.get(var_name, var_name)


def arrange_plots(image_paths, var_name, plot_type='vertical', pressure=None,
                  title=None, subtitles=None, ground_truth_path=None, 
                  ground_truth_subtitle='Ground Truth',
                  output_path='combined_plot.png', dpi=300,
                  crop_box=(35, 600, 0, 820),
                  gt_crop_box=(38, 881, 0, 1220)):
    """
    Arrange multiple PNG plots in a grid with a common colorbar.
    
    Parameters:
    -----------
    image_paths : list of str
        Paths to PNG files (2-7 images, or 1-6 if ground_truth_path provided)
    var_name : str
        Variable name (e.g., 'u', 'v', 't', 'z', 'w', 'uv')
    plot_type : str
        Either 'horizontal' or 'vertical' (for meridional/zonal plots)
    pressure : int, optional
        Pressure level in hPa (required for horizontal plots)
    title : str, optional
        Overall title for the figure
    subtitles : list of str, optional
        Subtitle for each plot in image_paths (same length as image_paths)
    ground_truth_path : str, optional
        Path to ground truth PNG (1390x881). Will be placed in first position.
    ground_truth_subtitle : str, optional
        Subtitle for ground truth plot (default: 'Ground Truth')
    output_path : str
        Path to save the combined figure
    dpi : int
        Resolution for saved figure
    crop_box : tuple
        (top, bottom, left, right) pixel coordinates to crop each image
        Default crops top 35 pixels and rightmost 180 pixels from 1000x600 images
    gt_crop_box : tuple
        (top, bottom, left, right) pixel coordinates to crop ground truth image
        Default crops top 38 pixels and rightmost 170 pixels from 1390x881 images
    """
    
    n_images = len(image_paths)
    
    # Add ground truth to total count if provided
    total_images = n_images + (1 if ground_truth_path else 0)
    
    if total_images < 2 or total_images > 7:
        raise ValueError(f"Total number of images must be between 2 and 7 (got {total_images})")
    
    # Validate subtitles
    if subtitles is not None and len(subtitles) != n_images:
        raise ValueError(f"Number of subtitles ({len(subtitles)}) must match number of image_paths ({n_images})")
    
    # Build combined list of all images and subtitles
    if ground_truth_path:
        all_paths = [ground_truth_path] + image_paths
        all_subtitles = [ground_truth_subtitle] + (subtitles if subtitles else [''] * n_images)
        is_ground_truth = [True] + [False] * n_images
    else:
        all_paths = image_paths
        all_subtitles = subtitles if subtitles else [''] * n_images
        is_ground_truth = [False] * n_images
    
    # Get colormap
    cmap = _get_colormap(var_name)
    
    # Get vmin/vmax based on plot type and variable
    if plot_type == 'horizontal':
        if pressure is None:
            raise ValueError("pressure level required for horizontal plots")
        if var_name in GROUND_TRUTH_RANGES and pressure in GROUND_TRUTH_RANGES[var_name]['horizontal']:
            vmin = GROUND_TRUTH_RANGES[var_name]['horizontal'][pressure]['vmin']
            vmax = GROUND_TRUTH_RANGES[var_name]['horizontal'][pressure]['vmax']
        else:
            raise ValueError(f"No ground truth ranges for {var_name} at {pressure} hPa")
    else:  # vertical (meridional or zonal)
        if var_name in GROUND_TRUTH_RANGES and 'vertical' in GROUND_TRUTH_RANGES[var_name]:
            vmin = GROUND_TRUTH_RANGES[var_name]['vertical']['vmin']
            vmax = GROUND_TRUTH_RANGES[var_name]['vertical']['vmax']
        else:
            raise ValueError(f"No ground truth ranges for {var_name} vertical plots")
    
    # Check if we need 'extend' for colorbar (matching original plotting logic)
    extend = 'both' if var_name == 'w' and plot_type == 'vertical' else 'neither'
    
    # Determine grid layout
    if total_images <= 3:
        nrows, ncols = 1, total_images
        layout = 'simple'
    elif total_images == 4:
        nrows, ncols = 2, 2
        layout = 'simple'
    elif total_images == 5:
        nrows, ncols = 2, 3
        layout = 'simple'
    elif total_images == 6:
        nrows, ncols = 2, 3
        layout = 'simple'
    else:  # total_images == 7
        nrows, ncols = 2, 4
        layout = 'irregular'  # 4 in first row, 3 in second
    
    # Create figure with space for colorbar on right
    fig = plt.figure(figsize=(5 * ncols + 0.5, 4 * nrows))
    
    # Create grid for subplots and colorbar
    gs = fig.add_gridspec(nrows, ncols + 1, 
                          width_ratios=[5] * ncols + [0.3],
                          hspace=0.05, wspace=0.05,
                          left=0.05, right=0.95, top=0.95, bottom=0.05)
    
    # Load and display each image
    for idx, (img_path, subtitle, is_gt) in enumerate(zip(all_paths, all_subtitles, is_ground_truth)):
        # For 7 images: first 4 in row 0, next 3 centered in row 1
        if layout == 'irregular' and total_images == 7:
            if idx < 4:
                row, col = 0, idx
            else:
                row, col = 1, idx - 4 + 0.5  # Center the 3 images in second row
                # We need to handle this differently - use column span
                ax = fig.add_subplot(gs[row, int(col):int(col)+1])
                # Adjust position to center
                pos = ax.get_position()
                offset = 0.125  # Half of one column width to center 3 items in 4 columns
                ax.set_position([pos.x0 + offset, pos.y0, pos.width, pos.height])
        else:
            row = idx // ncols
            col = idx % ncols
            ax = fig.add_subplot(gs[row, col])
        
        if layout != 'irregular' or idx < 4:
            ax = fig.add_subplot(gs[row, col])
        
        # Load image
        img = mpimg.imread(img_path)
        
        # Crop based on image type
        if is_gt:
            gt_top, gt_bottom, gt_left, gt_right = gt_crop_box
            cropped_img = img[gt_top:gt_bottom, gt_left:gt_right]
            
            # Resize to match height of regular images
            regular_top, regular_bottom, _, _ = crop_box
            target_height = regular_bottom - regular_top
            current_height, current_width = cropped_img.shape[:2]
            scale_factor = target_height / current_height
            target_width = int(current_width * scale_factor)
            
            # Resize using array indexing (simple nearest neighbor)
            # For better quality, we'll use a basic resampling approach
            from scipy.ndimage import zoom
            if cropped_img.ndim == 3:  # RGB/RGBA
                zoom_factors = (scale_factor, scale_factor, 1)
            else:  # Grayscale
                zoom_factors = (scale_factor, scale_factor)
            cropped_img = zoom(cropped_img, zoom_factors, order=1)
        else:
            top, bottom, left, right = crop_box
            cropped_img = img[top:bottom, left:right]
        
        # Display image
        ax.imshow(cropped_img)
        ax.axis('off')
        
        # Add subtitle if provided
        if subtitle:
            ax.set_title(subtitle, fontsize=10, pad=5)
    
    # Hide any unused subplots (only for irregular layout)
    if layout == 'irregular' and total_images == 7:
        # Hide the extra column spot in the second row
        ax = fig.add_subplot(gs[1, 3])
        ax.axis('off')
    else:
        # Hide any unused subplots for regular layouts
        for idx in range(total_images, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            ax = fig.add_subplot(gs[row, col])
            ax.axis('off')
    
    # Add colorbar on right side spanning all rows
    cax = fig.add_subplot(gs[:, ncols])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), 
                        cax=cax, orientation='vertical', extend=extend)
    cbar.set_label(_get_var_label(var_name), fontsize=12)
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    
    # Save figure
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved combined plot to {output_path}")


# Example usage:
if __name__ == "__main__":
    # Choose list of up to 6 plots from WandB (5 if used Ground Truth
    test_model_plots = [
        'mlp03_200hPa_z.png',
        'mlp07_200hPa_z.png',
        'mlp05_200hPa_z.png',
        'mlp08_200hPa_z.png',
        'siren10_200hPa_z.png',
        'siren13_200hPa_z.png'
    ]
    model_subtitles = ['MLP', 'MLP+PINN', 'MLP+FF',
                    'MLP+PINN+FF', 'SIREN', 'SIREN+PINN']

    # Optionally, specify ground truth to compare (will always go first)
    ground_plot = 'ground_z_200hPa.png'
    ground_subtitle = 'ERA5 Ground Truth'
    
    # Combined plot parameters
    title = 'Geopotential (z) at 200hPa Model Comparison'
    plot_type = 'horizontal'  # horizontal or vertical
    pressure = 200  # if horizontal, specify pressure
    var_name = 'z'  # specify any output variable (t, u, v, w, z, uv)
    output_path='200hPa_z_comparison.png'

    arrange_plots(
        image_paths=test_model_plots,
        var_name=var_name,
        plot_type=plot_type,
        pressure=pressure,
        subtitles=model_subtitles,
        ground_truth_path=ground_plot,
        ground_truth_subtitle=ground_subtitle,
        title=title,
        output_path=output_path
    )


    # # Example 1: Horizontal plots at 500 hPa with ground truth
    # horizontal_paths = [
    #     'model1_u_500hpa.png',
    #     'model2_u_500hpa.png'
    # ]
    
    # arrange_plots(
    #     image_paths=horizontal_paths,
    #     var_name='u',
    #     plot_type='horizontal',
    #     pressure=500,
    #     subtitles=['Model 1', 'Model 2'],
    #     ground_truth_path='gt_u_500hpa.png',
    #     ground_truth_subtitle='ERA5 Ground Truth',
    #     title='500 hPa Zonal Wind Comparison',
    #     output_path='combined_horizontal.png'
    # )
    
    # # Example 2: Vertical (zonal/meridional) plots without ground truth
    # vertical_paths = [
    #     'u_zonal_mean.png',
    #     'u_meridional_120E.png',
    #     'u_meridional_240E.png'
    # ]
    
    # arrange_plots(
    #     image_paths=vertical_paths,
    #     var_name='u',
    #     plot_type='vertical',
    #     subtitles=['Zonal Mean', 'Meridional 120°E', 'Meridional 240°E'],
    #     title='Zonal Wind Vertical Structure',
    #     output_path='combined_vertical.png'
    # )
    
    # # Example 3: 2x2 grid with ground truth
    # prediction_plots = [
    #     'pred1.png',
    #     'pred2.png',
    #     'pred3.png'
    # ]
    
    # arrange_plots(
    #     image_paths=prediction_plots,
    #     var_name='t',
    #     plot_type='vertical',
    #     subtitles=['Forecast +6h', 'Forecast +12h', 'Forecast +24h'],
    #     ground_truth_path='gt_temperature.png',
    #     title='Temperature Forecast Comparison',
    #     output_path='combined_4panel.png'
    # )