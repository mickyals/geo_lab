# Models Package

This package contains the core model architectures and components for the geolab project.

## Structure

### `components/`
- `physics.py`: Physics-informed neural network (PINN) components
- `position_encoders.py`: Positional encoding modules

### `model/`
- `finer.py`: FINER model implementation
- `gaborwavelet.py`: Gabor wavelet network
- `gaussian.py`: Gaussian-based models
- `simple_dense_net.py`: Basic fully-connected network
- `siren.py`: SIREN (Sinusoidal Representation Networks) implementation

### `modules/`
- `mnist_module.py`: Lightning module for MNIST
- `troposphere_module.py`: Main lightning module for atmospheric modeling

### `on_hold/`
- Experimental components and research code
- Activation functions
- Embedding layers
- Initialization strategies
- Neural Implicit Functions (NIF) implementation

## Key Features
- **Multiple Architectures**: Support for various neural network architectures
- **Physics-Informed**: Built-in support for physics constraints
- **Flexible Training**: PyTorch Lightning integration
- **Positional Encodings**: Various encoding strategies for coordinate inputs

## Usage
```python
# Example: Creating a SIREN model
model = Siren(
    in_features=3,  # x, y, z coordinates
    out_features=4,  # u, v, w, p
    hidden_features=256,
    hidden_layers=3
)
```
