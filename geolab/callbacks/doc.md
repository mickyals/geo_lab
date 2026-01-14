# Callbacks Package

This package contains callback implementations for the geolab project, primarily focused on visualization and monitoring during training.

## Key Components

### `visualization_callback.py`
A flexible, logger-agnostic callback for atmospheric visualizations during training. Supports:
- 2D field visualizations (horizontal/vertical slices)
- Scatter plots of data distributions
- Error heatmaps
- Physics residuals visualization (for PINN models)
- Multiple logging backends (WandB, TensorBoard)

### Configuration
Visualizations are configured through a declarative YAML/JSON format, allowing for:
- Custom geometry specifications
- Multiple visualization types per training run
- Flexible scheduling (by epoch or step)
- Support for different data splits (train/val/test)

### Features
- Batch caching for efficient visualization
- Support for multiple loggers
- Local figure saving
- Automatic handling of device placement
- Rich error handling and logging

### Usage
See the class docstring in `visualization_callback.py` for detailed configuration examples.
