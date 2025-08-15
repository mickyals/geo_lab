
# Geoscience Project Test Lab

> I found that my projects tnd to follow the same or similar experimental set up each time but without a clear structure, my code would be confusing and poorly formatted. This repository seeks to mitigate such issues going forward, allowing you to easily build and scale your work with onlu minor changes to the internal code logic. 

### Blueprint
> This blueprint shows the general structure of a project utilising this repository. 

```
geo_lab/
│
├── configs/                      # All configs (Hydra/OmegaConf)
│   ├── model/                     # SIREN, FourierNet, hybrid, DMD, etc.
│   ├── dataset/                   # ERA5, seismic, DEM, etc.
│   ├── training/                   # Optimizers, schedulers, loss settings
│   ├── experiment/                 # Full experiment definitions
│   └── default.yaml                 # Defaults for everything
│
├── data/
│   ├── processors/                 # Preprocessing & transforms
│   │   ├── era5.py                  # Load, normalize, coord prep, scaling
│   │   ├── normalization.py         # Scaling functions
│   │   └── sampling.py              # Custom coordinate sampling strategies
│   │
│   ├── datamodules/                 # PyTorch Lightning DataModules
│   │   ├── geo_datamodule.py        # Base geoscience datamodule
│   │   ├── structured.py            # Gridded (ERA5, DEM)
│   │   └── unstructured.py          # Point clouds
│   │
│   └── dataset_factory.py           # Config-based dataset instantiation
│
├── models/
│   ├── implicit/
│   │   ├── core/                    # Model building blocks
│   │   │   ├── activations.py       # Sine, ReLU, Gaussian, etc.
│   │   │   ├── layers.py            # Fourier layers, modulated layers
│   │   │   ├── initializations.py   # SIREN init, xavier, kaiming, etc.
│   │   │   └── embeddings.py        # Positional/Fourier embedding
│   │   │
│   │   ├── architectures/           # Complete models
│   │   │   ├── siren.py
│   │   │   ├── fourier_net.py
│   │   │   ├── hybrid.py
│   │   │   └── neural_dmd.py        # Multi-network architectures
│   │   │
│   │   └── builders.py              # Model assembly from configs
│   │
│   └── model_factory.py             # Registry + unified creation entry point
│
├── training/
│   ├── experiment.py                # PL LightningModule
│   ├── callbacks.py                  # Physical penalties, activation tracking
│   ├── losses.py                     # MSE, custom physics-informed losses
│   ├── optimizers.py                 # Optimizers
│   └── schedulers.py                 # LR schedulers
│
├── analysis/
│   ├── evaluators/                   # Domain-specific metrics
│   │   ├── metrics.py
│   │   └── physical_constraints.py
│   │
│   └── visualization/                # Output rendering
│       ├── volume_rendering.py       # 3D ERA5 cubes
│       ├── slice_plotter.py          # 2D cross sections
│       ├── datashader_plots.py       # Interactive large-scale views
│       └── embedding_viewer.py       # Latent space visualization
│
├── experiments/
│   ├── train.py                      # Universal training entry point
│   ├── validate.py                   # Evaluation on saved checkpoints
│   └── hyperparameter_tuner/         # Optuna/Ray Tune integration
│		│
│		├── optuna_runner.py           # Main entry point for Optuna sweeps
│		├── ray_tune_runner.py         # Main entry for Ray Tune sweeps
│		├── search_spaces.py           # Search space definitions
│		├── utils.py                   # Helper functions for trial configs
│		├── callbacks.py               # Optional: custom pruning/logging callbacks
│		└── README.md                  # How to run hyperparameter tuning
│
├── lab/                              # Scratchpad / prototype scripts
│   ├── era5_playground.ipynb         # First ERA5 → SIREN experiment
│   ├── activation_explorer.py        # Test activation init strategies
│   └── sampling_experiments.py       # Try custom coord sampling
│
├── utils/
│   ├── io.py                         # NetCDF, GRIB, CSV readers
│   ├── config_utils.py               # Hydra/OmegaConf helpers
│   ├── logging_utils.py              # W&B / TensorBoard setup
│   ├── helpers.py                    # helpers for setting seed, device and getting device as well as model summaries
│   └── math_ops.py                   # Small reusable math helpers
│
└── README.md


```