# Utils Package Documentation

This package contains various utility modules that provide common functionality across the project. Below is a summary of each module and its purpose.

## Table of Contents

1. [instantiators.py](#instantiatorspy)
2. [jit_compiler.py](#jit_compilerpy)
3. [logging_utils.py](#logging_utilspy)
4. [meteorology.py](#meteorologypy)
5. [pylogger.py](#pyloggerpy)
6. [rich_utils.py](#rich_utilspy)
7. [utils.py](#utilspy)

---

## instantiators.py

Provides utility functions for instantiating callbacks and loggers from configuration.

### Functions

#### `instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]`
Instantiates callbacks from a Hydra configuration object.

**Parameters:**
- `callbacks_cfg`: A DictConfig object containing callback configurations.

**Returns:**
- A list of instantiated callback objects.

#### `instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]`
Instantiates loggers from a Hydra configuration object.

**Parameters:**
- `logger_cfg`: A DictConfig object containing logger configurations.

**Returns:**
- A list of instantiated logger objects.

---

## jit_compiler.py

Provides utilities for JIT (Just-In-Time) compilation of PyTorch models, with special handling for FakeTensors.

### Functions

#### `defake(x)`
Converts FakeTensor instances to real, zero-filled tensors with the same meta-data.

#### `fake_tensor_unsupported(fn)`
Decorator for functions that are incompatible with FakeTensors.

#### `strip_overloads(gm)`
Updates the targets of graph nodes in the given Fx graph module to remove overloads.

#### `jit_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor])`
Compiles the given Fx graph module using JIT, handling fake tensors and overloads appropriately.

---

## logging_utils.py

Provides utilities for logging hyperparameters in PyTorch Lightning.

### Functions

#### `log_hyperparameters(object_dict: Dict[str, Any]) -> None`
Controls which config parts are saved by Lightning loggers and logs hyperparameters.

**Parameters:**
- `object_dict`: A dictionary containing configuration, model, and trainer objects.

---

## meteorology.py

Contains meteorological utility functions for weather and climate data processing.

### Functions

#### `omega_to_w(omega, pressure, temperature)`
Converts vertical velocity from pressure coordinates (ω, Pa/s) to geometric coordinates (w, m/s).

**Parameters:**
- `omega`: Vertical velocity in pressure coordinates (Pa/s)
- `pressure`: Pressure level (Pa)
- `temperature`: Temperature (K)

**Returns:**
- Vertical velocity in geometric coordinates (m/s)

---

## pylogger.py

Implements a multi-GPU-friendly Python command line logger.

### Classes

#### `RankedLogger`
A logging adapter that handles multi-GPU logging with rank prefixes.

**Methods:**
- `__init__(name: str = __name__, rank_zero_only: bool = False, extra: Optional[Mapping[str, object]] = None)`
- `log(level: int, msg: str, rank: Optional[int] = None, *args, **kwargs)`

---

## rich_utils.py

Provides utilities for rich console output and configuration handling.

### Functions

#### `print_config_tree(cfg: DictConfig, print_order: Sequence[str] = (...), resolve: bool = False, save_to_file: bool = False) -> None`
Prints the contents of a DictConfig as a tree structure using the Rich library.

#### `enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None`
Prompts user to input tags from command line if no tags are provided in config.

---

## utils.py

Core utility functions for the project.

### Functions

#### `extras(cfg: DictConfig) -> None`
Applies optional utilities before the task is started (e.g., ignoring warnings, setting tags).

#### `task_wrapper(task_func: Callable) -> Callable`
Optional decorator that controls the failure behavior when executing a task function.

#### `get_metric_value(metric_dict: Dict[str, Any], metric_name: Optional[str]) -> Optional[float]`
Safely retrieves value of the metric logged in LightningModule.

---

## Usage Examples

### Using the Logger
```python
from geolab.utils.pylogger import RankedLogger

log = RankedLogger(__name__)
log.info("This is an info message")
log.error("This is an error message")
```

### Converting Omega to W
```python
from geolab.utils.meteorology import omega_to_w

omega = 0.1  # Pa/s
pressure = 85000  # Pa
temperature = 280  # K
w = omega_to_w(omega, pressure, temperature)
```

### Printing Config Tree
```python
from omegaconf import OmegaConf
from geolab.utils.rich_utils import print_config_tree

config = OmegaConf.load("config.yaml")
print_config_tree(config, resolve=True)
```

## Dependencies

- PyTorch
- PyTorch Lightning
- Omegaconf
- Rich
- Hydra
- Lightning Utilities

## Contributing

When adding new utility functions, please:
1. Add proper docstrings following NumPy style
2. Include type hints
3. Add unit tests
4. Update this README if adding new modules or significant functionality
