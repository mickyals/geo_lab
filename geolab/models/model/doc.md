# Neural Network Models Documentation

This document provides an overview of the available neural network models in the `geolab.models.model` package.

## Common Parameters

All models support the following optional parameters for positional encoding:

- `position_encoder_type` (str, optional): Type of positional encoding. Options: 'gaussian', 'positional', 'basic'
- `mapping_dim` (int, optional): Dimension of the mapped features. Required if `position_encoder_type` is specified.
- `scale` (float, default=1.0): Scale parameter for positional encoding.

## Available Models

### 1. FCN (Fully Connected Network)
**File:** `simple_dense_net.py`

A standard fully connected neural network with configurable activation functions.

```python
FCN(
    N_in_features: int,               # Input feature dimension
    N_out_features: int,              # Output feature dimension
    N_hidden_features: int,           # Number of hidden units per layer
    N_hidden_layers: int,             # Number of hidden layers
    activation: str = 'relu',         # Options: 'relu', 'silu', 'tanh'
    bias: bool = True,                # Whether to use bias terms
    init_type: str = 'uniform',       # Options: 'uniform', 'normal'
    position_encoder_type: str = None,# Type of positional encoding
    mapping_dim: int = None,          # Dimension for positional encoding
    scale: float = 1.0               # Scale for positional encoding
)
```

### 2. GaussianNet
**File:** `gaussian.py`

A neural network using Gaussian activation functions.

```python
GaussianNet(
    N_in_features: int,               # Input feature dimension
    N_out_features: int,              # Output feature dimension
    N_hidden_features: int,           # Number of hidden units per layer
    N_hidden_layers: int,             # Number of hidden layers
    sigma: float,                     # Standard deviation for Gaussian activation
    init_type: str,                   # Options: 'uniform', 'normal'
    position_encoder_type: str = None,# Type of positional encoding
    mapping_dim: int = None,          # Dimension for positional encoding
    scale: float = 1.0               # Scale for positional encoding
)
```

### 3. SirenNet
**File:** `siren.py`

Sinusoidal Representation Network (SIREN) with periodic activation functions.

```python
SirenNet(
    N_in_features: int,               # Input feature dimension
    N_out_features: int,              # Output feature dimension
    N_hidden_features: int,           # Number of hidden units per layer
    N_hidden_layers: int,             # Number of hidden layers
    first_omega: float = 30.0,        # Frequency for first layer
    hidden_omega: float = 30.0,       # Frequency for hidden layers
    position_encoder_type: str = None,# Type of positional encoding
    mapping_dim: int = None,          # Dimension for positional encoding
    scale: float = 1.0               # Scale for positional encoding
)
```

### 4. FinerNet
**File:** `finer.py`

A variant of SIREN with additional scaling parameters.

```python
FinerNet(
    N_in_features: int,               # Input feature dimension
    N_out_features: int,              # Output feature dimension
    N_hidden_features: int,           # Number of hidden units per layer
    N_hidden_layers: int,             # Number of hidden layers
    first_omega: float,               # Frequency for first layer
    hidden_omega: float,              # Frequency for hidden layers
    spread: float,                    # Spread parameter for initialization
    position_encoder_type: str = None,# Type of positional encoding
    mapping_dim: int = None,          # Dimension for positional encoding
    scale: float = 1.0               # Scale for positional encoding
)
```

### 5. RealWireNet
**File:** `gaborwavelet.py`

Neural network using Gabor wavelet activation functions (real-valued).

```python
RealWireNet(
    N_in_features: int,               # Input feature dimension
    N_out_features: int,              # Output feature dimension
    N_hidden_features: int,           # Number of hidden units per layer
    N_hidden_layers: int,             # Number of hidden layers
    omega_0: float,                   # Initial frequency
    scale_0: float,                   # Initial scale
    init_type: str,                   # Options: 'uniform', 'normal'
    bias: bool,                       # Whether to use bias terms
    position_encoder_type: str = None,# Type of positional encoding
    mapping_dim: int = None,         # Dimension for positional encoding
    scale: float = 1.0               # Scale for positional encoding
)
```

### 6. ComplexWireNet
**File:** `gaborwavelet.py`

Complex-valued Gabor wavelet network with trainable frequency and scale parameters.

```python
ComplexWireNet(
    # Implementation in progress
    # Will support complex-valued Gabor wavelets
)
```

## Usage Examples

```python
from geolab.models.model import FCN, SirenNet, GaussianNet, FinerNet, RealWireNet

# 1. Fully Connected Network
fcn = FCN(
    N_in_features=2,
    N_out_features=1,
    N_hidden_features=64,
    N_hidden_layers=3,
    activation='relu',
    position_encoder_type='positional',
    mapping_dim=256,
    scale=10.0
)

# 2. SIREN Network
siren = SirenNet(
    N_in_features=2,
    N_out_features=1,
    N_hidden_features=64,
    N_hidden_layers=3,
    first_omega=30.0,
    hidden_omega=30.0,
    position_encoder_type='gaussian',
    mapping_dim=256,
    scale=10.0
)

# 3. Gaussian Network
gaussian = GaussianNet(
    N_in_features=2,
    N_out_features=1,
    N_hidden_features=64,
    N_hidden_layers=3,
    sigma=1.0,
    init_type='uniform',
    position_encoder_type='gaussian',
    mapping_dim=256,
    scale=10.0
)

# 4. Finer Network (SIREN variant)
finer = FinerNet(
    N_in_features=2,
    N_out_features=1,
    N_hidden_features=64,
    N_hidden_layers=3,
    first_omega=30.0,
    hidden_omega=30.0,
    spread=0.1,
    position_encoder_type='positional',
    mapping_dim=256,
    scale=10.0
)

# 5. RealWireNet (Gabor Wavelet)
realwire = RealWireNet(
    N_in_features=2,
    N_out_features=1,
    N_hidden_features=64,
    N_hidden_layers=3,
    omega_0=30.0,
    scale_0=0.1,
    init_type='uniform',
    bias=True,
    position_encoder_type='basic',
    mapping_dim=256,
    scale=10.0
)
```

## Notes

- All models inherit from `torch.nn.Module` and can be used like any other PyTorch module.
- Positional encoding is optional but can help with learning high-frequency functions.
- The `mapping_dim` parameter controls the dimension of the encoded features when using positional encoding.
- For all models, if `position_encoder_type` is specified, `mapping_dim` must be provided.
- The `scale` parameter affects the frequency range of the positional encoding.
