# Visualization Package

This package provides tools for visualizing atmospheric model predictions and data.

## Key Components

### `geometry.py`
- `GeometryGenerator`: Generates coordinate grids for visualization
- Supports points, lines, planes, and volumes in 4D space (lon, lat, pressure, time)
- Handles coordinate transformations and normalization

### `aggregation.py`
- `DataAggregator`: Aggregates data along coordinate dimensions
- Supports zonal means, temporal means, and spatial binning
- Handles both model predictions and ground truth data

### `inference.py`
- `ModelInference`: Wrapper for model inference with batching
- Handles denormalization of outputs
- Computes physics residuals for PINN models

### `visualizer.py`
- High-level visualization functions
- Unified interface for different plot types
- Support for 2D fields, 1D profiles, and scatter plots
- Integration with model predictions and ground truth

## Features
- **Coordinate-Aware**: Handles atmospheric coordinates natively
- **Efficient**: Batched inference for large datasets
- **Flexible**: Works with different model architectures
- **Publication-Quality**: Configurable, high-quality plots

## Usage
```python
# Example: Plotting a horizontal slice
from geolab.viz import plot_field

figs = plot_field(
    pl_module=model,
    geometry_spec={
        'type': 'plane',
        'axes': ['longitude', 'latitude'],
        'pressure_level': 500,
        'valid_time': 0.0,
        'resolution': {'longitude': 2.0, 'latitude': 2.0}
    },
    var_names=['u', 'v', 'w']
)
```
