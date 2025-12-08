# GeoLab: Atmospheric Modeling with Deep Learning

<div align="center">

[![python](https://img.shields.io/badge/Python-3.8%20|%203.9%20|%203.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

</div>

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mickyals/geo_lab.git
   cd geo_lab
   ```

2. **Create and activate a conda environment**:
   ```bash
   conda create -n geolab python=3.9
   conda activate geolab
   ```

3. **Install PyTorch** (with CUDA support recommended):
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🏗 Project Structure

```
geo_lab/
├── configs/               # Configuration files
│   ├── callbacks/         # Callback configurations
│   ├── data/              # Data loading configurations
│   ├── debug/             # Debugging configurations
│   ├── experiment/        # Experiment configurations
│   ├── logger/            # Logger configurations (W&B, TensorBoard)
│   ├── model/             # Model configurations
│   ├── trainer/           # Trainer configurations
│   ├── train.yaml         # Main training configuration
│   └── eval.yaml          # Evaluation configuration
│
├── data/                  # Data directory
│   └── era5/              # ERA5 data files
│
├── geolab/                # Source code
│   ├── data/              # Data loading and processing
│   ├── models/            # Model architectures
│   ├── utils/             # Utility functions
│   └── visualization/     # Plotting and visualization
│
├── notebooks/             # Jupyter notebooks
├── scripts/               # Utility scripts
└── tests/                 # Unit tests
```

## ⚙️ Configuration

### 1. Data Configuration (`configs/data/troposphere.yaml`)
```yaml
_target_: geolab.data.datamodule.troposphere_datamodule.TroposphereDataModule
data_dir: ${paths.data_dir}
solution_vars: ["w", "u", "z", "v"]
time_idx_range: [0,1]      # null for all time steps
pressure_idx_range: null   # null for all pressure levels
latitude_idx_range: null   # null for all latitudes
longitude_idx_range: null  # null for all longitudes
test_split: 0.0
val_split: 0.3
num_virtual: 100000000    # Number of virtual points
include_virtual: false
use_lhs: true            # Use Latin Hypercube Sampling
pi_scale: false          # Scale coordinates
batch_size: 1024
num_workers: 4
persistent_workers: true
pin_memory: true
```

### 2. Model Configuration (`configs/model/troposphere_module.yaml`)
```yaml
_target_: geolab.models.modules.troposphere_module.TroposhpereLightningModule

# Model architecture
model_name: SirenNet
N_in_features: 4          # longitude, latitude, pressure_level, time
N_out_features: 4         # t, w, u, z, v
N_hidden_features: 128
N_hidden_layers: 5
position_encoder_type: null
mapping_dim: 128
scale: 10.0

# Model-specific parameters
model_params:
  first_omega: 30.0
  hidden_omega: 30.0

# Optimizer
optimizer_name: Adam
optimizer_config:
  lr: 1e-4
  weight_decay: 1e-5

# Scheduler
scheduler_name: null
scheduler_config:
  T_max: min
  patience: 3
  factor: 0.1

# Physics-informed settings
train_pinn: true
mass_balance: true
physics_loss_weight: 0.50
```

### 3. Trainer Configuration (`configs/trainer/default.yaml`)
```yaml
_target_: lightning.pytorch.trainer.Trainer

default_root_dir: ${paths.output_dir}
min_epochs: 1
max_epochs: 20
accelerator: auto
devices: 1
log_every_n_steps: 4
check_val_every_n_epoch: 1
deterministic: true
```

### 4. Main Training Configuration (`configs/train.yaml`)
```yaml
# @package _global_
defaults:
  - _self_
  - data: troposphere
  - model: troposphere_module
  - callbacks: default
  - logger: wandb
  - trainer: default
  - paths: default
  - extras: default
  - hydra: default
  - experiment: null
  - hparams_search: null
  - optional local: default
  - debug: null

task_name: "train"
tags: ["non_valid"]
train: True
test: False
ckpt_path: null
seed: 42
```

## 🚀 Training

### Available Models
- **SirenNet**: Physics-informed neural network for atmospheric modeling
- *[Add other models as they become available]*

### Training Command
```bash
python train.py
```

### Training with Custom Configuration
To override any configuration parameter:
```bash
python train.py trainer.max_epochs=50 data.batch_size=2048 model.optimizer_config.lr=1e-5
```

### Creating a New Model Configuration
1. Create a new YAML file in `configs/model/`
2. Extend from an existing configuration or create a new one
3. Reference your new config in `train.py` or via command line

## 📊 Monitoring
Training progress can be monitored using Weights & Biases. Configure your W&B settings in `configs/logger/wandb.yaml`.

## 🤝 Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## 📜 License
This project is licensed under the MIT License.
