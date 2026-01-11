# Data Pipeline Blueprint: Optimizing PINN Training Performance

## Overview

This document consolidates the architectural decisions for eliminating data loading bottlenecks in the troposphere PINN training pipeline. The current implementation suffers from 5-7% GPU utilization due to excessive per-sample Python overhead.

---

## 1. Problem Diagnosis

### Current Bottlenecks

| Issue | Location | Impact |
|-------|----------|--------|
| Nested dict returns | `ERA5MultiDataset.__getitem__` | 8+ dict lookups per sample |
| Per-sample tensor creation | `ERA5MultiDataset.__getitem__` | `torch.tensor()` called 8x per sample |
| Per-sample normalization | `ERA5MultiDataset._norm` | Redundant computation |
| Batch unpacking + stacking | `model_step` | `torch.stack()` called 2x per batch |
| Struct-of-arrays memory layout | `ERA5MultiDataset.data` | Cache-unfriendly access |

### Current Flow (Slow)

```
ERA5MultiDataset.__getitem__(i):
    j = self.idx[i]
    lon = self.data["longitude"][j]        # dict + index
    lat = self.data["latitude"][j]         # dict + index
    ...
    lon_n = self._norm(lon, ...)           # normalize per sample
    ...
    return {
        "coords": {
            "longitude": torch.tensor(lon_n),   # 4 tensor creations
            "latitude": torch.tensor(lat_n),
            ...
        },
        "variables": {
            "u": torch.tensor(u_n),             # 4 more tensor creations
            ...
        },
        "classification": torch.tensor(...)
    }
        │
        ▼
DataLoader collates 4096 nested dicts
        │
        ▼
model_step unpacks and stacks EVERY BATCH:
    coords = batch['coords']
    variables = batch['variables']
    coord_list = [coords['longitude'], coords['latitude'], ...]
    inputs = torch.stack(coord_list, dim=1).float()
    targets = torch.stack(list(variables.values()), dim=1).float()
```

---

## 2. Solution Architecture

### Design Principles

1. **Load once, tensorize once**: All data processing happens in `setup()`, not `__getitem__`
2. **Array-of-structs layout**: Single contiguous `[N, 4]` and `[N, V]` tensors
3. **Index-only datasets**: Datasets hold references + indices, not data copies
4. **Label dictionaries**: Named access via `coord_labels["latitude"]` instead of dict returns
5. **Train-only statistics**: Avoid data leakage by computing stats on train split only

### Fixed Flow (Fast)

```
setup() does ONCE:
    coords = normalize(coords_raw)         # normalize all
    targets = normalize(targets_raw)       # normalize all
    _coords = torch.from_numpy(coords)     # tensorize once
    _targets = torch.from_numpy(targets)   # tensorize once

ERA5MultiDataset.__getitem__(i):
    j = self.indices[i]
    return self.coords[j], self.targets[j] # just indexing, no dicts
        │
        ▼
DataLoader stacks tensors directly (fast default collate)
        │
        ▼
model_step receives ready tensors:
    coords, targets = batch                # no unpacking needed
    preds = self.forward(coords)           # direct use
```

---

## 3. Component Specifications

### 3.1 ERA5MultiData (Data Processor)

**Purpose**: Load NetCDF, slice, flatten to numpy arrays, build label dictionaries.

```
┌───────────────────────────────────────────────────────────────┐
│                       ERA5MultiData                           │
│                      (Data Processor)                         │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ATTRIBUTES (set by load)                                     │
│  ────────────────────────                                     │
│  coord_labels: Dict[str, int]                                 │
│      {"longitude": 0, "latitude": 1,                          │
│       "pressure_level": 2, "valid_time": 3}                   │
│                                                               │
│  var_labels: Dict[str, int]                                   │
│      {"u": 0, "v": 1, "w": 2, "z": 3}                         │
│                                                               │
│  METHODS                                                      │
│  ───────                                                      │
│  __init__(data_path: str)                                     │
│                                                               │
│  load(variables, time_slice, pressure_slice,                  │
│       lat_slice, lon_slice)                                   │
│      → coords: np.ndarray[N, 4]                               │
│      → targets: np.ndarray[N, V]                              │
│                                                               │
│  _build_coord_array(lon, lat, pressure, time) → [N, 4]        │
│  _build_target_array(ds, variables, N) → [N, V]               │
│  _make_slice(idx_range, dim_size) → slice                     │
│                                                               │
│  DELETED METHODS                                              │
│  ───────────────                                              │
│  run()              → replaced by load()                      │
│  _load_dataset()    → inline in load()                        │
│  subset_data()      → merge into load()                       │
│  generate_virtual_samples() → moved to LightningModule        │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 3.2 ERA5MultiDataset (Index Wrapper)

**Purpose**: Minimal wrapper that indexes into shared tensors.

```
┌───────────────────────────────────────────────────────────────┐
│                      ERA5MultiDataset                         │
│                       (Index Wrapper)                         │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ATTRIBUTES                                                   │
│  ──────────                                                   │
│  coords: Tensor[N, 4]        (reference to DataModule)        │
│  targets: Tensor[N, V]       (reference to DataModule)        │
│  indices: Tensor[n_split]    (train/val/test indices)         │
│                                                               │
│  METHODS                                                      │
│  ───────                                                      │
│  __init__(coords, targets, indices)                           │
│      self.coords = coords      # reference, not copy          │
│      self.targets = targets    # reference, not copy          │
│      self.indices = indices                                   │
│                                                               │
│  __len__()                                                    │
│      return len(self.indices)                                 │
│                                                               │
│  __getitem__(i)                                               │
│      j = self.indices[i]                                      │
│      return self.coords[j], self.targets[j]                   │
│                                                               │
│  DELETED                                                      │
│  ───────                                                      │
│  data: dict              (was struct-of-arrays)               │
│  statistics: dict        (moved to DataModule)                │
│  variables: list         (not needed)                         │
│  include_virtual         (not needed)                         │
│  pi_scale                (not needed)                         │
│  _norm()                 (moved to DataModule)                │
│  _timedelta_norm()       (moved to DataModule)                │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 3.3 TroposphereDataModule (Orchestrator)

**Purpose**: Coordinate loading, splitting, normalization, and dataset creation.

```
┌───────────────────────────────────────────────────────────────┐
│                    TroposphereDataModule                      │
│                       (Orchestrator)                          │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ATTRIBUTES                                                   │
│  ──────────                                                   │
│  coord_labels: Dict[str, int]                                 │
│  var_labels: Dict[str, int]                                   │
│  statistics: Dict[str, Tuple[float, float, float, float]]     │
│      {name: (min, max, mean, std), ...}                       │
│                                                               │
│  _coords: Tensor[N, 4]       (single copy, normalized)        │
│  _targets: Tensor[N, V]      (single copy, normalized)        │
│                                                               │
│  train_dataset: ERA5MultiDataset                              │
│  val_dataset: ERA5MultiDataset                                │
│  test_dataset: ERA5MultiDataset                               │
│                                                               │
│  METHODS                                                      │
│  ───────                                                      │
│  setup(stage)                  # main orchestration           │
│  _split_indices(N)             # random or structured split   │
│  _compute_statistics(coords, targets, train_idx)              │
│  _normalize(coords, targets)   # in-place using statistics    │
│  get_labels()                  # for LightningModule access   │
│                                                               │
│  train_dataloader()            # unchanged                    │
│  val_dataloader()              # unchanged                    │
│  test_dataloader()             # unchanged                    │
│                                                               │
│  DELETED METHODS                                              │
│  ───────────────                                              │
│  _load_statistics()            → stats computed in setup      │
│  _load_and_prepare_data()      → merged into setup            │
│  _get_real_virtual_indices()   → no classification tensor     │
│  _create_dataset()             → inline in setup              │
│                                                               │
│  PROPERTIES                                                   │
│  ──────────                                                   │
│  num_train_samples                                            │
│  num_val_samples                                              │
│  num_test_samples                                             │
│  input_dim → 4                                                │
│  output_dim → len(var_labels)                                 │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow

```
TroposphereDataModule.setup()
       │
       ├── processor = ERA5MultiData(data_path)
       │
       ├── coords, targets = processor.load(variables, slices)
       │       coords: np.ndarray[N, 4]
       │       targets: np.ndarray[N, V]
       │
       ├── self.coord_labels = processor.coord_labels
       ├── self.var_labels = processor.var_labels
       │
       ├── train_idx, val_idx, test_idx = _split_indices(N)
       │
       ├── self.statistics = _compute_statistics(
       │       coords,                    # full domain for coords
       │       targets[train_idx],        # train only for targets
       │   )
       │
       ├── _normalize(coords, targets)    # in-place, using statistics
       │
       ├── self._coords = torch.from_numpy(coords).float().contiguous()
       ├── self._targets = torch.from_numpy(targets).float().contiguous()
       │
       └── create datasets (refs to tensors + split indices)


Training Loop
       │
       ├── DataLoader(train_dataset)
       │       │
       │       └── __getitem__(i) → _coords[j], _targets[j]
       │                            (pure tensor indexing)
       │
       └── model_step(batch)
               coords, targets = batch     # [B, 4], [B, V] ready
               preds = model(coords)       # no preprocessing
```

---

## 5. Memory Layout

```
TroposphereDataModule
       │
       ├── coord_labels = {"longitude": 0, "latitude": 1, ...}
       ├── var_labels = {"u": 0, "v": 1, "w": 2, "z": 3}
       ├── statistics = {"longitude": (min,max,mean,std), ...}
       │
       ├── _coords: Tensor[N, 4] ◄────────┬───────────┬───────────┐
       │                                  │           │           │
       ├── _targets: Tensor[N, V] ◄───────┼───────────┼───────────┤
       │                             (ref only)  (ref only)  (ref only)
       │                                  │           │           │
       ├── train_dataset ─────────────────┘           │           │
       │       └── indices: Tensor[n_train]           │           │
       │                                              │           │
       ├── val_dataset ───────────────────────────────┘           │
       │       └── indices: Tensor[n_val]                         │
       │                                                          │
       └── test_dataset ──────────────────────────────────────────┘
               └── indices: Tensor[n_test]

ONE COPY of data. Datasets hold references + indices only.
```

---

## 6. Normalization Strategy

### Key Insight: Avoid Data Leakage

- **Inputs (coordinates)**: Use full domain bounds (physical facts, not learned)
- **Outputs (fields)**: Use train split statistics only

### Implementation

```
_compute_statistics(coords, targets, train_idx):
    │
    ├── # Input stats: use FULL DOMAIN (defines coordinate system)
    ├── for name, idx in coord_labels.items():
    │       col = coords[:, idx]  # all data
    │       statistics[name] = (col.min(), col.max(), col.mean(), col.std())
    │
    └── # Output stats: TRAIN ONLY (avoid leak)
        for name, idx in var_labels.items():
            col = targets[train_idx, idx]  # train only
            statistics[name] = (col.min(), col.max(), col.mean(), col.std())
```

### Normalization Functions

```python
def normalize(x, lo, hi):
    """Normalize to [-1, 1]"""
    return 2 * (x - lo) / (hi - lo) - 1

def denormalize(x_norm, lo, hi):
    """Denormalize from [-1, 1] to physical units"""
    return (x_norm + 1) * (hi - lo) / 2 + lo
```

---

## 7. Virtual Sampling

Virtual samples are generated fresh each batch in `training_step`, not stored in the dataset.

```
TroposphereLitModule
       │
       ├── setup(stage):
       │       # Get labels from datamodule
       │       self.coord_labels = self.trainer.datamodule.coord_labels
       │       self.var_labels = self.trainer.datamodule.var_labels
       │       self.statistics = self.trainer.datamodule.statistics
       │
       ├── _sample_virtual(n, device):
       │       return torch.rand(n, 4, device=device) * 2 - 1
       │
       └── training_step(batch):
               coords_real, targets = batch
               n_real = coords_real.shape[0]
               
               # Fresh virtual samples
               coords_virtual = self._sample_virtual(n_virtual, coords_real.device)
               
               # Single forward pass
               coords_all = torch.cat([coords_real, coords_virtual], dim=0)
               coords_all.requires_grad_(True)
               preds_all = self(coords_all)
               
               # Data loss (real only)
               data_loss = F.mse_loss(preds_all[:n_real], targets)
               
               # Physics loss (all or virtual only)
               physics_loss = self._pde_residual(coords_all, preds_all)
               
               return data_loss + lambda * physics_loss
```

---

## 8. LightningModule Changes

### Before (Current)

```python
def model_step(self, batch):
    coords = batch['coords']
    variables = batch['variables']
    classification = batch['classification']
    
    coord_list = [
        coords['longitude'],
        coords['latitude'],
        coords['pressure_level'],
        coords['time']
    ]
    
    inputs = torch.stack(coord_list, dim=1).float()
    targets = torch.stack(list(variables.values()), dim=1).float()
    ...
```

### After (Fixed)

```python
def model_step(self, batch):
    coords, targets = batch  # already [B, 4] and [B, V]
    
    if self.train_pinn:
        coords = coords.requires_grad_(True)
    
    preds = self.forward(coords)
    ...

# Access by name using labels
lat = coords[:, self.coord_labels["latitude"]]
u_pred = preds[:, self.var_labels["u"]]
u_min = self.statistics["u"][0]
u_max = self.statistics["u"][1]
```

---

## 9. YAML Configuration Updates

### Data Configuration

```yaml
data:
  _target_: geolab.data.datamodule.troposphere_datamodule.TroposphereDataModule
  data_dir: ${paths.data_dir}
  
  # Variables to load
  solution_vars: ["w", "u", "z", "v"]
  
  # Slicing (null = all)
  time_idx_range: [0, 1]
  pressure_idx_range: null
  latitude_idx_range: null
  longitude_idx_range: null
  
  # Splitting
  val_split: 0.3
  test_split: 0.0
  split_seed: 42
  
  # REMOVED: num_virtual, include_virtual, use_lhs
  # Virtual sampling now happens in LightningModule.training_step
  
  # REMOVED: pi_scale
  # Normalization handled internally using train statistics
  
  # DataLoader settings
  batch_size: 4096  # can increase now that bottleneck is removed
  num_workers: 4
  persistent_workers: true
  pin_memory: true
```

### Model Configuration

```yaml
model:
  _target_: geolab.models.modules.troposphere_module.TroposphereLitModule
  
  # Model architecture
  model_name: FCN  # or SirenNet
  N_in_features: 4
  N_out_features: 4
  N_hidden_features: 128
  N_hidden_layers: 5
  
  # Position encoding
  position_encoder_type: gaussian  # gaussian, positional, basic, null
  mapping_dim: 128
  scale: 3.0
  
  # Model-specific params
  model_params:
    activation: relu
    bias: true
    init_type: uniform
  
  # Optimizer
  optimizer_name: Adam
  optimizer_config:
    lr: 1e-4
    weight_decay: 1e-5
  
  # Scheduler
  scheduler_name: null
  scheduler_config: {}
  
  # PINN settings
  train_pinn: true
  mass_balance: true
  physics_loss_weight: 0.5
  
  # NEW: Virtual sampling (moved from data config)
  n_virtual: 1024  # virtual samples per batch, not total
  physics_on_real: true  # apply physics loss to real data points too
  
  # REMOVED: statistics, pi_scale
  # Now obtained from datamodule in setup()
```

### Key Config Changes Summary

| Parameter | Old Location | New Location | Notes |
|-----------|--------------|--------------|-------|
| `num_virtual` | data | model | Now per-batch, not total |
| `include_virtual` | data | REMOVED | Always generated in training_step |
| `use_lhs` | data | model (optional) | Can add as sampling strategy |
| `pi_scale` | data + model | REMOVED | Normalization is automatic |
| `statistics` | model | REMOVED | Obtained from datamodule |
| `n_virtual` | N/A | model | NEW: virtual samples per batch |
| `physics_on_real` | N/A | model | NEW: control physics loss scope |
| `split_seed` | N/A | data | NEW: reproducibility |

---

## 10. Expected Performance Impact

| Metric | Before | After |
|--------|--------|-------|
| GPU utilization | 5-7% | 70-90% |
| Dict lookups per batch | 8+ × batch_size | 0 |
| `torch.tensor()` calls per batch | 8 × batch_size | 0 |
| `torch.stack()` calls per batch | 2 | 0 |
| Normalization calls per batch | batch_size × 8 | 0 |
| Memory copies | Multiple | 1 (at setup) |
| Recommended batch_size | 1024 | 4096+ |

---

## 11. Label Flow

```
ERA5MultiData.load()
       │
       ├── self.coord_labels = {"longitude": 0, "latitude": 1,
       │                        "pressure_level": 2, "valid_time": 3}
       ├── self.var_labels = {"u": 0, "v": 1, "w": 2, "z": 3}
       └── (statistics computed later by DataModule)
                  │
                  │ (reference assignment)
                  ▼
TroposphereDataModule.setup()
       │
       ├── self.coord_labels = processor.coord_labels
       ├── self.var_labels = processor.var_labels
       └── self.statistics = self._compute_statistics(...)
                  │
                  │ trainer.datamodule accessible
                  ▼
TroposphereLitModule.setup()
       │
       ├── self.coord_labels = self.trainer.datamodule.coord_labels
       ├── self.var_labels = self.trainer.datamodule.var_labels
       └── self.statistics = self.trainer.datamodule.statistics
```

---

## 12. Sampling Strategies (Optional Extension)

For more flexible splitting, inject a sampler into the DataModule:

```yaml
data:
  _target_: geolab.data.datamodule.troposphere_datamodule.TroposphereDataModule
  ...
  sampler:
    _target_: geolab.data.samplers.RandomIndexSampler
    train_frac: 0.7
    val_frac: 0.15
    test_frac: 0.15
    index_file: null  # optional: load pre-computed split
```

Available samplers:
- `RandomIndexSampler`: Random shuffle and split
- `TemporalSplitSampler`: Hold out later time steps for val/test
- `SpatialSplitSampler`: Hold out spatial regions for val/test
- `SensorFieldSampler`: DeepONet-style fixed sensor locations

---

## 13. Files to Modify

| File | Changes |
|------|---------|
| `era5multi.py` | Replace `run()` with `load()`, return `[N,4]` and `[N,V]` arrays, build label dicts |
| `troposphere_datamodule.py` | Refactor `setup()`, add `_compute_statistics`, `_normalize`, `get_labels` |
| `troposphere_module.py` | Simplify `model_step` to receive tensors directly, use label dicts |
| `configs/data/*.yaml` | Remove virtual/pi_scale params, add split_seed |
| `configs/model/*.yaml` | Add n_virtual, physics_on_real, remove statistics |

---

## 14. Implementation Checklist

- [ ] Update `ERA5MultiData.load()` to return numpy arrays
- [ ] Simplify `ERA5MultiDataset` to pure indexing
- [ ] Refactor `TroposphereDataModule.setup()` with new flow
- [ ] Add `_compute_statistics()` with train-only output stats
- [ ] Add `_normalize()` for in-place normalization
- [ ] Update `TroposphereLitModule.model_step()` for tensor input
- [ ] Add virtual sampling to `training_step()`
- [ ] Update YAML configs
- [ ] Test GPU utilization improvement
- [ ] Verify training convergence matches previous results
