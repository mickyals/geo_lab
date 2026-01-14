# Utilities Package

This package contains various utility modules supporting the geolab project.

## Key Modules

### Core Utilities
- `utils.py`: General utility functions
- `pylogger.py`: Logging configuration and utilities
- `rich_utils.py`: Rich text formatting and console output
- `logging_utils.py`: Enhanced logging functionality

### Data Processing
- `checkpoint_update.py`: Model checkpoint management
- `ground_truth_plots.py`: Utilities for plotting ground truth data
- `meteorology.py`: Meteorological calculations and conversions

### Model Development
- `instantiators.py`: Dynamic class instantiation
- `jit_compiler.py`: Just-in-time compilation utilities

## Features
- **Logging**: Consistent logging across the project
- **Checkpointing**: Model save/load utilities
- **Meteorological Functions**: Common atmospheric calculations
- **Type Hints**: Full type annotations for better IDE support
- **Error Handling**: Consistent error reporting

## Usage
```python
# Example: Using the logger
from geolab.utils.pylogger import get_pylogger
log = get_pylogger(__name__)
log.info("Initializing model...")

# Example: Loading a checkpoint
from geolab.utils.checkpoint_update import load_checkpoint
model = load_checkpoint("path/to/checkpoint.ckpt")
```

## Notes
- All modules are designed to work with PyTorch and PyTorch Lightning
- Most functions include detailed docstrings with usage examples
- Error messages are designed to be descriptive and actionable
