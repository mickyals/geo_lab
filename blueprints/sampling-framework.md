# Spatiotemporal Train/Val/Test Splitting Framework

## Conceptual Framework: Data Splitting

### Goal: Test different generalization capabilities

1. **Temporal extrapolation**: Can model predict unseen future?
2. **Spatial interpolation**: Can model fill gaps in observed field?
3. **Spatiotemporal robustness**: Performance on sparse observations

### Data Structure: 4D Tensor

Shape: `(T, P, Lat, Lon) × Variables`
- T: time steps (e.g., 365 days)
- P: pressure levels (17 levels)
- Lat: latitude points (181 points)
- Lon: longitude points (360 points)

### Splitting Strategy

#### STEP 1: Temporal Split (hold out last N timesteps)

```
Time axis: [t₀, t₁, t₂, ..., t_{T-N-1} | t_{T-N}, ..., t_{T-1}]
            ├─────── Available ────────┤ ├───── Test ─────┤
```

**Test set**: All spatial points from last N timesteps
- Shape: `(N, P, Lat, Lon)` full fields
- Tests: Temporal extrapolation
- Use: Full dataloader for dense visualization

**Available for train/val**: First `(T - N)` timesteps
- Shape: `(T-N, P, Lat, Lon)`

#### STEP 2: Spatial Split (train vs val on available timesteps)

From available timesteps, create spatial sampling patterns:

### Spatial Sampling Strategies

#### A. Random Sampling (baseline)

**Each epoch:**
- Sample X random `(t, p, lat, lon)` indices
- Val gets different random Y indices

**Pros**: Maximum coverage over time  
**Cons**: Can't test interpolation to fixed unseen spatial locations

#### B. Static Hold-Out (spatial interpolation)

**One-time split:**
- Define fixed val spatial mask
- Train never sees these `(t, p, lat, lon)`
- Val always uses same spatial points

**Pros**: True test of spatial interpolation  
**Cons**: Val points never change, less coverage

#### C. Checkerboard Pattern

**Create alternating pattern:**
- Val mask: `(lat + lon) % 2 == 0`
- Train mask: complement

**Pros**: Regular spatial coverage  
**Cons**: Pattern may be too regular

#### D. Regional Hold-Out

**Hold out geographic regions:**
- Val: `lat ∈ [30°S, 30°N]` (tropics)
- Or: `lon ∈ [0°, 90°E]` (specific sector)

**Pros**: Tests regional generalization  
**Cons**: Biased by regional dynamics

#### E. Stratified Sampling

**Ensure coverage across:**
- Pressure levels (upper/mid/lower)
- Latitudes (tropics/midlat/polar)
- Times (seasons if applicable)

**Pros**: Balanced representation  
**Cons**: Complex implementation

#### F. Hybrid: Static + Random

**Combine approaches:**
- Val_static: Fixed spatial points
- Val_random: Random samples each epoch
- Report both metrics

**Pros**: Best of both worlds  
**Cons**: More complex tracking

---

## Data Module Architecture Redesign

### Class Signature

```python
class TroposphereDataModule:
    def __init__(self,
                 # Temporal split config
                 test_time_steps: int = 30,  # Last N timesteps for test
                 # Spatial split config
                 val_split_strategy: str = 'random',
                 val_fraction: float = 0.15,
                 # Strategy-specific params
                 spatial_seed: int = 42,  # For reproducible static splits
                 checkerboard_offset: int = 0,
                 regional_bounds: dict = None,
                 # Sampling config
                 train_samples_per_epoch: int = 100000,
                 val_samples_per_epoch: int = 20000,
                 **kwargs):
```

### Strategy Options

- `'random'`: New random samples each epoch (current)
- `'static_random'`: Fixed random spatial points, consistent across time
- `'checkerboard'`: Alternating spatial pattern
- `'regional'`: Geographic regions held out
- `'stratified'`: Balanced across dimensions
- `'hybrid'`: Combination of static + random

### Attributes after setup

```python
self.train_time_indices: np.ndarray  # [0, 1, ..., T-N-1]
self.test_time_indices: np.ndarray   # [T-N, T-N+1, ..., T-1]

self.train_spatial_mask: np.ndarray  # (P, Lat, Lon) boolean mask
self.val_spatial_mask: np.ndarray    # (P, Lat, Lon) boolean mask
```

For random strategies: masks regenerated each epoch  
For static strategies: masks fixed at setup

---

## Splitting Implementation Details

### setup() Method

```python
def setup(self, stage: str):
    """Setup train/val/test splits."""
    
    # ──────────────────────────────────────
    # STEP 1: Temporal split
    # ──────────────────────────────────────
    total_time_steps = self.data.coordinates['valid_time'].shape[0]
    train_val_time_steps = total_time_steps - self.test_time_steps
    
    self.train_time_indices = np.arange(0, train_val_time_steps)
    self.test_time_indices = np.arange(train_val_time_steps,
                                       total_time_steps)
    
    # ──────────────────────────────────────
    # STEP 2: Spatial split
    # ──────────────────────────────────────
    spatial_shape = (
        len(self.data.coordinates['pressure_level']),
        len(self.data.coordinates['latitude']),
        len(self.data.coordinates['longitude'])
    )  # (P, Lat, Lon)
    
    if self.val_split_strategy == 'random':
        # No fixed masks, generate fresh each epoch
        self.train_spatial_mask = None
        self.val_spatial_mask = None
    
    elif self.val_split_strategy == 'static_random':
        self._create_static_random_masks(spatial_shape)
    
    elif self.val_split_strategy == 'checkerboard':
        self._create_checkerboard_masks(spatial_shape)
    
    elif self.val_split_strategy == 'regional':
        self._create_regional_masks(spatial_shape)
    
    elif self.val_split_strategy == 'stratified':
        self._create_stratified_masks(spatial_shape)
    
    # ──────────────────────────────────────
    # STEP 3: Create datasets
    # ──────────────────────────────────────
    if stage == 'fit':
        self.train_dataset = SpatiotemporalDataset(
            data=self.data,
            time_indices=self.train_time_indices,
            spatial_mask=self.train_spatial_mask,
            num_samples=self.train_samples_per_epoch,
            mode='train'
        )
        
        self.val_dataset = SpatiotemporalDataset(
            data=self.data,
            time_indices=self.train_time_indices,  # Same times as train!
            spatial_mask=self.val_spatial_mask,
            num_samples=self.val_samples_per_epoch,
            mode='val'
        )
    
    if stage == 'test':
        self.test_dataset = FullFieldDataset(
            data=self.data,
            time_indices=self.test_time_indices,
            mode='test'
        )
```

### Mask Creation Methods

#### Static Random Masks

```python
def _create_static_random_masks(self, shape):
    """Create fixed random spatial masks."""
    rng = np.random.RandomState(self.spatial_seed)
    
    # Generate all possible spatial indices
    total_spatial = np.prod(shape)
    all_indices = np.arange(total_spatial)
    
    # Shuffle and split
    rng.shuffle(all_indices)
    val_size = int(total_spatial * self.val_fraction)
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:]
    
    # Convert to boolean masks
    self.val_spatial_mask = np.zeros(shape, dtype=bool)
    self.train_spatial_mask = np.zeros(shape, dtype=bool)
    
    # Unravel indices to 3D
    val_3d = np.unravel_index(val_indices, shape)
    train_3d = np.unravel_index(train_indices, shape)
    
    self.val_spatial_mask[val_3d] = True
    self.train_spatial_mask[train_3d] = True
```

#### Checkerboard Masks

```python
def _create_checkerboard_masks(self, shape):
    """Create checkerboard pattern."""
    P, Lat, Lon = shape
    
    # Create coordinate grids
    p_grid, lat_grid, lon_grid = np.meshgrid(
        np.arange(P),
        np.arange(Lat),
        np.arange(Lon),
        indexing='ij'
    )
    
    # Checkerboard: alternate in lat-lon
    checker = (lat_grid + lon_grid + self.checkerboard_offset) % 2
    
    self.val_spatial_mask = (checker == 0)
    self.train_spatial_mask = (checker == 1)
```

#### Regional Masks

```python
def _create_regional_masks(self, shape):
    """Hold out specific geographic region."""
    P, Lat, Lon = shape
    
    # Get coordinate values
    lats = self.data.coordinates['latitude']
    lons = self.data.coordinates['longitude']
    
    # Create masks based on bounds
    lat_mask = (
        (lats >= self.regional_bounds['lat_min']) &
        (lats <= self.regional_bounds['lat_max'])
    )
    lon_mask = (
        (lons >= self.regional_bounds['lon_min']) &
        (lons <= self.regional_bounds['lon_max'])
    )
    
    # Broadcast to full spatial shape
    self.val_spatial_mask = np.zeros(shape, dtype=bool)
    self.val_spatial_mask[:, lat_mask, :] = True
    self.val_spatial_mask[:, :, lon_mask] = True
    
    self.train_spatial_mask = ~self.val_spatial_mask
```

#### Stratified Masks

```python
def _create_stratified_masks(self, shape):
    """Stratified sampling across dimensions."""
    # Implementation: Bin each dimension, sample within bins
    # Ensure representation across pressure, latitude zones, etc.
    # TODO: Implement stratification logic
    pass
```

---

## Dataset Implementations

### SpatiotemporalDataset

```python
class SpatiotemporalDataset(Dataset):
    """Dataset for train/val with spatiotemporal sampling."""
    
    def __init__(self, data, time_indices, spatial_mask,
                 num_samples, mode='train'):
        self.data = data
        self.time_indices = time_indices
        self.spatial_mask = spatial_mask
        self.num_samples = num_samples
        self.mode = mode
        
        # Precompute valid indices if using static mask
        if spatial_mask is not None:
            self._precompute_valid_indices()
    
    def _precompute_valid_indices(self):
        """Build lookup table for valid spatial indices."""
        # Get (p, lat, lon) indices where mask is True
        self.valid_spatial_indices = np.argwhere(self.spatial_mask)
        # Shape: (N_valid, 3)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Sample a single spatiotemporal point."""
        
        if self.spatial_mask is None:
            # Random strategy: sample anywhere
            t_idx = np.random.choice(self.time_indices)
            p_idx = np.random.randint(0, len(self.data.coordinates[
                'pressure_level']))
            lat_idx = np.random.randint(0, len(self.data.coordinates[
                'latitude']))
            lon_idx = np.random.randint(0, len(self.data.coordinates[
                'longitude']))
        else:
            # Static strategy: sample from valid spatial points
            t_idx = np.random.choice(self.time_indices)
            spatial_idx = np.random.randint(0,
                len(self.valid_spatial_indices))
            p_idx, lat_idx, lon_idx = self.valid_spatial_indices[
                spatial_idx]
        
        # Extract coordinates
        coords = np.array([
            self.data.coordinates['valid_time'][t_idx],
            self.data.coordinates['pressure_level'][p_idx],
            self.data.coordinates['latitude'][lat_idx],
            self.data.coordinates['longitude'][lon_idx]
        ])
        
        # Extract values
        values = np.array([
            self.data.data_arrays[var][t_idx, p_idx, lat_idx, lon_idx]
            for var in self.data.var_order
        ])
        
        return {
            'coords': torch.from_numpy(coords).float(),
            'values': torch.from_numpy(values).float()
        }
```

### FullFieldDataset

```python
class FullFieldDataset(Dataset):
    """Dataset for test set: full spatiotemporal fields."""
    
    def __init__(self, data, time_indices, mode='test'):
        self.data = data
        self.time_indices = time_indices
        self.mode = mode
        
        # Precompute all indices
        self.all_indices = self._generate_all_indices()
    
    def _generate_all_indices(self):
        """Generate all (t, p, lat, lon) combinations for test times."""
        indices = []
        for t_idx in self.time_indices:
            for p_idx in range(len(self.data.coordinates[
                'pressure_level'])):
                for lat_idx in range(len(self.data.coordinates[
                    'latitude'])):
                    for lon_idx in range(len(self.data.coordinates[
                        'longitude'])):
                        indices.append((t_idx, p_idx, lat_idx, lon_idx))
        return np.array(indices)
    
    def __len__(self):
        return len(self.all_indices)
    
    def __getitem__(self, idx):
        """Return single point from full field."""
        t_idx, p_idx, lat_idx, lon_idx = self.all_indices[idx]
        
        coords = np.array([
            self.data.coordinates['valid_time'][t_idx],
            self.data.coordinates['pressure_level'][p_idx],
            self.data.coordinates['latitude'][lat_idx],
            self.data.coordinates['longitude'][lon_idx]
        ])
        
        values = np.array([
            self.data.data_arrays[var][t_idx, p_idx, lat_idx, lon_idx]
            for var in self.data.var_order
        ])
        
        return {
            'coords': torch.from_numpy(coords).float(),
            'values': torch.from_numpy(values).float()
        }
```

---

## Metrics & Evaluation Strategy

### Different metrics for different capabilities

#### Train Metrics (spatial interpolation within seen times)

- MSE per variable
- Physics residuals (if PINN)
- Convergence rate

#### Val Metrics (spatial generalization within seen times)

- MSE on held-out spatial locations
- If static mask: consistent spatial interpolation error
- If random mask: average interpolation capability
- Regional breakdown (tropics vs midlat vs polar)

#### Test Metrics (temporal extrapolation to unseen future)

- Full-field reconstruction error
- Spatial correlation with ground truth
- Spectral analysis (power spectrum)
- Conservation of mass/energy
- Per-timestep evolution (does error grow over time?)

### Logging Strategy

```python
wandb.log({
    'train/mse': ...,
    'val/mse_interpolation': ...,  # Spatial interpolation
    'test/mse_extrapolation': ..., # Temporal extrapolation
    'test/spatial_correlation': ...,
    'test/mse_per_timestep': [...]  # Track error evolution
})
```

---

## Visualization Integration

### Test Set Visualization (dense reconstruction)

```python
def on_test_epoch_end(self, trainer, pl_module):
    """Visualize full field reconstruction on test timesteps."""
    
    test_loader = trainer.test_dataloaders
    
    # Collect all predictions for one timestep
    timestep_to_plot = self.datamodule.test_time_indices[0]
    
    # Filter test loader for this timestep
    coords_list = []
    preds_list = []
    targets_list = []
    
    for batch in test_loader:
        # Filter for specific timestep
        time_mask = (batch['coords'][:, 0] ==
            self.datamodule.data.coordinates['valid_time'][timestep_to_plot])
        
        if time_mask.any():
            coords = batch['coords'][time_mask]
            targets = batch['values'][time_mask]
            preds = pl_module(coords)
            
            coords_list.append(coords)
            preds_list.append(preds)
            targets_list.append(targets)
    
    # Concatenate
    all_coords = torch.cat(coords_list)
    all_preds = torch.cat(preds_list)
    all_targets = torch.cat(targets_list)
    
    # Now we have full field for one timestep
    # Can create dense visualizations
    
    # Reshape to grid
    shape = (len(self.data.coordinates['pressure_level']),
             len(self.data.coordinates['latitude']),
             len(self.data.coordinates['longitude']))
    
    # Create comparison plots
    for var_idx, var_name in enumerate(self.data.var_order):
        pred_field = all_preds[:, var_idx].reshape(shape)
        target_field = all_targets[:, var_idx].reshape(shape)
        error_field = (pred_field - target_field).abs()
        
        # Plot horizontal slices at each pressure level
        for p_idx, pressure in enumerate(
            self.data.coordinates['pressure_level']):
            
            fig = self.plotter.plot_comparison(
                data1=pred_field[p_idx].cpu().numpy(),
                data2=target_field[p_idx].cpu().numpy(),
                error=error_field[p_idx].cpu().numpy(),
                var_name=var_name,
                title=f'Test: {var_name} at {pressure}hPa'
            )
            self._log_figure(fig, f'test/{var_name}_{pressure}hPa')
```

### Val Set Visualization (sparse samples)

```python
def on_validation_epoch_end(self, trainer, pl_module):
    """Show where val samples are located."""
    
    # If using static spatial mask, visualize the mask
    if self.datamodule.val_spatial_mask is not None:
        fig = self._plot_spatial_mask(
            self.datamodule.val_spatial_mask,
            title='Validation Spatial Sampling Pattern'
        )
        self._log_figure(fig, 'val/spatial_mask')
    
    # Scatter plot of val sample locations
    val_batch = next(iter(trainer.val_dataloaders))
    coords = val_batch['coords']
    
    fig = self.plotter.plot_scatter(
        coords=coords.cpu().numpy(),
        projection='3d',
        color_by='pressure_level',
        title='Validation Sample Distribution'
    )
    self._log_figure(fig, 'val/sample_distribution')

def _plot_spatial_mask(self, mask, title):
    """Visualize 3D boolean mask."""
    # Sum over pressure to get 2D lat-lon map
    mask_2d = mask.sum(axis=0)  # (Lat, Lon)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6),
                          subplot_kw={'projection': ccrs.PlateCarree()})
    
    lats = self.datamodule.data.coordinates['latitude']
    lons = self.datamodule.data.coordinates['longitude']
    
    im = ax.pcolormesh(lons, lats, mask_2d.T,
                       transform=ccrs.PlateCarree(),
                       cmap='RdYlGn', vmin=0, vmax=mask.shape[0])
    ax.coastlines()
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Number of pressure levels sampled')
    
    return fig
```

---

## Recommended Implementation Plan

### PHASE 1: Basic temporal split (TOMORROW)

- [ ] Add test_time_steps parameter to datamodule
- [ ] Implement temporal splitting in setup()
- [ ] Create FullFieldDataset for test set
- [ ] Test: Verify test times are completely held out
- [ ] Test: Verify train/val use only earlier times

### PHASE 2: Random spatial sampling (current behavior, ensure it works)

- [ ] Refactor current random sampling into SpatiotemporalDataset
- [ ] Ensure train/val sample from same time range
- [ ] Test: Train for a few epochs, verify works as before

### PHASE 3: Static spatial split (NEW capability)

- [ ] Implement _create_static_random_masks()
- [ ] Update SpatiotemporalDataset to use precomputed masks
- [ ] Visualize train/val spatial distributions
- [ ] Test: Verify val points never appear in training

### PHASE 4: Alternative spatial strategies (LATER)

- [ ] Implement checkerboard pattern
- [ ] Implement regional hold-out
- [ ] Implement stratified sampling
- [ ] Compare performance across strategies

### PHASE 5: Test visualization (LATER)

- [ ] Implement full-field reconstruction plots
- [ ] Add comparison plots (pred vs target vs error)
- [ ] Add temporal evolution tracking
- [ ] Generate paper-quality figures

### Recommended starting point

1. Start with temporal split + random spatial (Phase 1+2)
2. Get training working, establish baseline metrics
3. Add static spatial split to test interpolation (Phase 3)
4. Compare random vs static validation performance
5. Decide on best strategy for your research questions
