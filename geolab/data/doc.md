# Data Package

This package handles all data loading, preprocessing, and dataset management for the geolab project.

## Structure

### `datamodule/`
- `__init__.py`: Package initialization
- `mnist_datamodule.py`: MNIST dataset integration
- `troposphere_datamodule.py`: Main datamodule for atmospheric data

### `dataset/`
- `__init__.py`: Package initialization
- `era5multi.py`: ERA5 multi-variable dataset implementation
- `precompute_statistics.py`: Utilities for dataset statistics
- `samplers.py`: Custom samplers for data loading

## Key Features
- **Modular Design**: Separate concerns between data loading (datamodules) and data representation (datasets)
- **Efficient Loading**: Optimized for large-scale atmospheric data
- **Reproducibility**: Seeded data splits and transformations
- **Normalization**: Built-in support for data normalization
- **Multi-dataset Support**: Handles both MNIST (for testing) and atmospheric data

## Usage
```python
# Example: Using the troposphere datamodule
datamodule = TroposphereDataModule(
    data_dir='path/to/data',
    batch_size=32,
    num_workers=4
)
datamodule.setup()
train_loader = datamodule.train_dataloader()
```
