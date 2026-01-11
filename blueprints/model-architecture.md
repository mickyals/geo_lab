# Model Architecture: Property-Based Access & Visualization

## Property-Based Access Pattern

### ERA5MultiData - Core Data Container

#### Stored Attributes

```python
self.coordinates: Dict[name, np.ndarray]
    {'valid_time': array, 'pressure_level': array, ...}

self.data_arrays: Dict[var, np.ndarray]
    {'w': array, 'u': array, 'z': array, 'v': array}

self.coord_labels: Dict[name, int]
    {'valid_time': 0, 'pressure_level': 1, 'latitude': 2, 'longitude': 3}
    # Maps coordinate name → column index in tensor

self.variable_labels: Dict[name, int]
    {'w': 0, 'u': 1, 'z': 2, 'v': 3}
    # Maps variable name → column index in predictions
```

#### Computed Properties

```python
@property
def input_dim(self) -> int:
    """Number of input coordinates."""
    return len(self.coord_labels)  # 4

@property
def output_dim(self) -> int:
    """Number of output variables."""
    return len(self.variable_labels)  # 4 for [w, u, z, v]

@property
def coord_order(self) -> List[str]:
    """Return coordinate names sorted by index."""
    return sorted(self.coord_labels.keys(),
                  key=lambda k: self.coord_labels[k])
    # Returns: ['valid_time', 'pressure_level', 'latitude', 'longitude']

@property
def var_order(self) -> List[str]:
    """Return variable names sorted by index."""
    return sorted(self.variable_labels.keys(),
                  key=lambda k: self.variable_labels[k])
    # Returns: ['w', 'u', 'z', 'v']
```

#### Usage Cascade

```
ERA5MultiData → TroposphereDataModule → LightningModule → Physics
      ↓                    ↓                     ↓               ↓
Properties defined    Properties used    Properties passed  Labels used
```

---

## Physics Code Refactor - Label-Aware Interface

### OLD APPROACH (coordinate reordering in Lightning module)

❌ Lightning module needs to know physics expects `[lon, lat, p, time]`  
❌ Hardcoded column swaps: `coords[:, [3, 2, 1, 0]]`  
❌ Breaks if coord order changes  
❌ Physics code is rigid  

### NEW APPROACH (physics code accepts labels)

✅ Physics code extracts coordinates by name  
✅ No reordering needed in Lightning module  
✅ Works with any coordinate ordering  
✅ Self-documenting, testable  

### Updated Function Signatures

```python
def compute_troposphere_gradients(
    inputs_tensor: torch.Tensor,     # (B, 4) in ANY order
    model_outputs: Dict[str, Tensor],
    coord_labels: Dict[str, int]     # NEW: tells us column order
) -> Dict[str, Tensor]:

def troposphere_pde_residual(
    inputs_tensor: torch.Tensor,
    outputs: Dict[str, Tensor],
    statistics: Dict,
    coord_labels: Dict[str, int],    # NEW
    var_labels: Dict[str, int],      # NEW (optional, for validation)
    mass_balance: bool = True
) -> Tuple[Tensor, Tensor, Tensor]:
```

### Implementation Pattern

```python
def troposphere_pde_residual(inputs_tensor, outputs, statistics,
                              coord_labels, mass_balance=True):
    
    # Extract coordinates by name, not by hardcoded index
    lon_idx = coord_labels['longitude']
    lat_idx = coord_labels['latitude']
    p_idx = coord_labels['pressure_level']
    t_idx = coord_labels['valid_time']
    
    # Now use these indices
    longitude = inputs_tensor[:, lon_idx]
    latitude = inputs_tensor[:, lat_idx]
    pressure = inputs_tensor[:, p_idx]
    time = inputs_tensor[:, t_idx]
    
    # Rest of physics computation unchanged
    ...
```

### Lightning Module Simplified

```python
def model_step(self, batch):
    coords = batch['coords']  # (B, 4) in dataset order
    preds = self.forward(coords)
    
    if self.train_pinn:
        # Denormalize coords for physics
        coords_denorm = self.datamodule.denormalize_coords(coords)
        
        # Build outputs dict
        outputs = {var: preds[:, i]
                   for i, var in enumerate(self.datamodule.data.var_order)}
        
        # Pass coordinate labels to physics
        ns_lon, ns_lat, mass = troposphere_pde_residual(
            coords_denorm,
            outputs,
            statistics=self._convert_statistics_format(),
            coord_labels=self.datamodule.data.coord_labels,  # NEW
            mass_balance=self.mass_balance
        )
```

### Benefits

✓ Lightning module doesn't need `_reorder_coords_for_physics()`  
✓ Physics code works with any coordinate system  
✓ Easy to add new coordinates (e.g., height, potential temp)  
✓ Physics code can be unit tested independently  
✓ Clear separation of concerns  

---

## Visualization Architecture - 4 Layers

### Layer 1: GEOMETRY GENERATION

**Responsibility**: Create coordinate grids to sample  
**Input**: Coordinate ranges, resolution  
**Output**: Coordinate tensor `(N, 4)`

```python
class GeometryGenerator:
    def __init__(self, coord_ranges, coord_labels):
        self.coord_ranges = coord_ranges
        self.coord_labels = coord_labels
    
    def point(self, **coords) -> Tensor:
        """Single point: (1, 4)"""
        # point(lon=0, lat=45, p=500, time=0.5)
    
    def line(self, axis: str, values: array, **fixed) -> Tensor:
        """1D line: (N, 4)"""
        # line(axis='time', values=linspace(0,1,100),
        #      lon=0, lat=45, p=500)
    
    def plane(self, axes: List[str], resolutions: Dict,
              **fixed) -> Tensor:
        """2D plane: (N, 4)"""
        # plane(axes=['longitude', 'latitude'],
        #       resolutions={'longitude': 2, 'latitude': 2},
        #       pressure_level=500, valid_time=0.5)
    
    def volume(self, axes: List[str], resolutions: Dict,
               **fixed) -> Tensor:
        """3D volume: (N, 4)"""
        # volume(axes=['longitude', 'latitude', 'pressure_level'],
        #        resolutions={...}, valid_time=0.5)
    
    def random_samples(self, n: int, seed: int = None) -> Tensor:
        """Random samples: (N, 4)"""
    
    def from_indices(self, indices: np.ndarray,
                     data: ERA5MultiData) -> Tensor:
        """Convert dataset indices to coordinates"""
        # For plotting training/val/test distributions
```

### Layer 2: MODEL INFERENCE

**Responsibility**: Get model predictions for coordinates  
**Input**: Coordinates `(N, 4)`, model  
**Output**: Predictions `(N, num_vars)` or derived quantities

```python
class ModelInference:
    def __init__(self, model, datamodule):
        self.model = model
        self.datamodule = datamodule
    
    def predict(self, coords: Tensor,
                normalize: bool = True,
                denormalize_output: bool = True) -> Tensor:
        """Basic prediction: (N, num_vars)"""
        if normalize:
            coords = self.datamodule.normalize_coords(coords)
        preds = batched_forward(self.model, coords)
        if denormalize_output:
            preds = self.denormalize_predictions(preds)
        return preds
    
    def predict_with_physics(self, coords: Tensor) -> Dict:
        """Predictions + physics residuals"""
        preds = self.predict(coords, denormalize_output=False)
        coords_norm = self.datamodule.normalize_coords(coords)
        coords_denorm = coords  # Already physical
        
        # Build outputs dict
        outputs = {var: preds[:, i]
                   for i, var in enumerate(self.datamodule.data.var_order)}
        
        # Compute residuals
        ns_lon, ns_lat, mass = troposphere_pde_residual(
            coords_denorm, outputs,
            statistics=...,
            coord_labels=self.datamodule.data.coord_labels
        )
        
        return {
            'predictions': self.denormalize_predictions(preds),
            'ns_longitude': ns_lon,
            'ns_latitude': ns_lat,
            'mass_continuity': mass
        }
    
    def compute_errors(self, coords: Tensor,
                       targets: Tensor) -> Tensor:
        """MSE per variable: (N, num_vars)"""
        preds = self.predict(coords)
        return (preds - targets).pow(2)
    
    def get_ground_truth(self, coords: Tensor) -> Tensor:
        """Extract ground truth from data if available"""
        # Use datamodule to interpolate/lookup actual values
```

### Layer 3: AGGREGATION

**Responsibility**: Reduce dimensions via aggregation  
**Input**: Data `(N, num_vars)`, coordinates `(N, 4)`  
**Output**: Aggregated data with reduced shape

```python
class DataAggregator:
    def __init__(self, coord_labels):
        self.coord_labels = coord_labels
    
    def reduce_axis(self, data: Tensor, coords: Tensor,
                    axis: str, method: str = 'mean') -> Tensor:
        """Reduce along one coordinate axis."""
        # axis: 'longitude', 'latitude', 'pressure_level', 'valid_time'
        # method: 'mean', 'std', 'min', 'max', 'median'
        
        axis_idx = self.coord_labels[axis]
        # Group by other coordinates, aggregate over axis
        # Returns: (M, num_vars) where M < N
    
    def zonal_mean(self, data: Tensor, coords: Tensor) -> Tensor:
        """Average over longitude: (N, num_vars) → (M, num_vars)"""
        return self.reduce_axis(data, coords, 'longitude', 'mean')
    
    def temporal_mean(self, data: Tensor, coords: Tensor) -> Tensor:
        """Average over time"""
        return self.reduce_axis(data, coords, 'valid_time', 'mean')
    
    def spatial_stats(self, data: Tensor, coords: Tensor) -> Dict:
        """Aggregate over all spatial dimensions"""
        # Returns dict with 'mean', 'std', 'min', 'max' over space
    
    def temporal_evolution(self, data: Tensor, coords: Tensor,
                           spatial_agg: str = 'mean') -> Tensor:
        """Reduce to time series by aggregating space"""
        # Input: (N, num_vars) 4D data
        # Output: (T, num_vars) time series
```

### Layer 4: PLOTTING

**Responsibility**: Create matplotlib/cartopy figures  
**Input**: Aggregated data, coordinates, styling  
**Output**: Figure object

```python
class Plotter:
    def __init__(self, coord_labels, var_labels):
        self.coord_labels = coord_labels
        self.var_labels = var_labels
    
    def plot_2d_field(self, data: np.ndarray,
                      coords: np.ndarray,
                      var_name: str,
                      projection: str = 'cartopy',
                      **style) -> Figure:
        """2D contour/pcolormesh plot"""
        # Handles: horizontal slices, meridional slices
    
    def plot_1d_profile(self, data: np.ndarray,
                        coords: np.ndarray,
                        var_name: str,
                        **style) -> Figure:
        """Line plot: data vs coordinate"""
        # Handles: vertical profiles, time series
    
    def plot_scatter(self, coords: np.ndarray,
                     values: Optional[np.ndarray] = None,
                     projection: str = '3d',
                     **style) -> Figure:
        """Scatter plot of points"""
        # Handles: training point distribution, error maps
    
    def plot_error_heatmap(self, errors: np.ndarray,
                           coords: np.ndarray,
                           var_names: List[str],
                           **style) -> Figure:
        """Multi-panel error distribution"""
    
    def plot_comparison(self, data1: np.ndarray,
                        data2: np.ndarray,
                        coords: np.ndarray,
                        var_name: str,
                        labels: Tuple[str, str],
                        **style) -> Figure:
        """Side-by-side or difference plots"""
    
    def plot_animated(self, data: np.ndarray,
                      coords: np.ndarray,
                      time_axis: int,
                      var_name: str,
                      **style) -> Animation:
        """Create animation over time dimension"""
```

---

## Orchestration - Visualization Callback

```python
class AtmosphericVisualizationCallback(Callback):
    def __init__(self, config):
        # Initialize all layers
        self.geometry = None  # Created in setup()
        self.inference = None
        self.aggregator = None
        self.plotter = None
    
    def setup(self, trainer, pl_module, stage):
        """Initialize visualization pipeline"""
        datamodule = trainer.datamodule
        
        self.geometry = GeometryGenerator(
            coord_ranges=datamodule.coordinate_ranges,
            coord_labels=datamodule.data.coord_labels
        )
        
        self.inference = ModelInference(
            model=pl_module,
            datamodule=datamodule
        )
        
        self.aggregator = DataAggregator(
            coord_labels=datamodule.data.coord_labels
        )
        
        self.plotter = Plotter(
            coord_labels=datamodule.data.coord_labels,
            var_labels=datamodule.data.variable_labels
        )
    
    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._should_plot(trainer):
            return
        
        # Example: Horizontal slice at 500 hPa
        self._plot_horizontal_slice(pl_module, pressure=500)
        
        # Example: Zonal mean
        self._plot_zonal_mean(pl_module)
        
        # Example: Training data distribution
        self._plot_training_distribution(trainer)
        
        # Example: Physics residuals
        if pl_module.train_pinn:
            self._plot_physics_residuals(pl_module)
```

### Example: Horizontal Slice

```python
def _plot_horizontal_slice(self, pl_module, pressure):
    """Horizontal slice at fixed pressure level"""
    
    # 1. Generate 2D plane
    coords = self.geometry.plane(
        axes=['longitude', 'latitude'],
        resolutions={'longitude': 2, 'latitude': 2},
        pressure_level=pressure,
        valid_time=0.5  # Middle of time range
    )  # Returns: (N, 4)
    
    # 2. Get predictions
    preds = self.inference.predict(coords)  # (N, num_vars)
    
    # 3. No aggregation needed (already 2D)
    
    # 4. Plot each variable
    for i, var in enumerate(self.inference.datamodule.data.var_order):
        fig = self.plotter.plot_2d_field(
            data=preds[:, i].cpu().numpy(),
            coords=coords.cpu().numpy(),
            var_name=var,
            projection='cartopy',
            title=f'{var.upper()} at {pressure} hPa'
        )
        self._log_figure(fig, f'horizontal_{pressure}hPa_{var}')
```

### Example: Zonal Mean

```python
def _plot_zonal_mean(self, pl_module):
    """Latitude-pressure cross-section (zonal mean)"""
    
    # 1. Generate 3D volume (all lon, lat, pressure at fixed time)
    coords = self.geometry.volume(
        axes=['longitude', 'latitude', 'pressure_level'],
        resolutions={'longitude': 2, 'latitude': 2},
        pressure_levels=[850, 700, 500, 300, 200],
        valid_time=0.5
    )  # Returns: (N, 4) where N = n_lon * n_lat * n_pressure
    
    # 2. Get predictions
    preds = self.inference.predict(coords)  # (N, num_vars)
    
    # 3. Aggregate: mean over longitude
    zonal_data, zonal_coords = self.aggregator.zonal_mean(
        preds, coords
    )  # Returns: (M, num_vars) where M = n_lat * n_pressure
    
    # 4. Plot
    for i, var in enumerate(self.inference.datamodule.data.var_order):
        fig = self.plotter.plot_2d_field(
            data=zonal_data[:, i].cpu().numpy(),
            coords=zonal_coords.cpu().numpy(),
            var_name=var,
            projection='lat_pressure',  # Special projection
            title=f'{var.upper()} Zonal Mean'
        )
        self._log_figure(fig, f'zonal_mean_{var}')
```

### Example: Training Distribution

```python
def _plot_training_distribution(self, trainer):
    """Scatter plot of training point locations"""
    
    # 1. Get training indices
    train_indices = trainer.datamodule.train_dataset.indices
    # (N, 4) with integer indices
    
    # 2. Convert indices to physical coordinates
    coords = self.geometry.from_indices(
        train_indices,
        data=trainer.datamodule.data
    )  # Returns: (N, 4) in physical units
    
    # 3. No prediction or aggregation needed
    
    # 4. Plot as 3D scatter
    fig = self.plotter.plot_scatter(
        coords=coords.cpu().numpy(),
        projection='3d',
        color_by='pressure_level',
        title='Training Data Distribution'
    )
    self._log_figure(fig, 'training_distribution')
    
    # Also plot 2D projections
    for proj in ['lat_lon', 'lat_pressure', 'lon_pressure']:
        fig = self.plotter.plot_scatter(
            coords=coords.cpu().numpy(),
            projection=proj,
            alpha=0.1,
            title=f'Training Points ({proj})'
        )
        self._log_figure(fig, f'training_dist_{proj}')
```

### Example: Physics Residuals

```python
def _plot_physics_residuals(self, pl_module):
    """Plot spatial distribution of physics residuals"""
    
    # 1. Generate random samples across domain
    coords = self.geometry.random_samples(n=10000, seed=42)
    
    # 2. Get predictions + physics residuals
    results = self.inference.predict_with_physics(coords)
    # Returns: {
    #   'predictions': (N, num_vars),
    #   'ns_longitude': (N,),
    #   'ns_latitude': (N,),
    #   'mass_continuity': (N,)
    # }
    
    # 3. Aggregate: bin into lat-lon grid and take mean
    residual_map = self.aggregator.spatial_bin(
        data=results['mass_continuity'].abs(),
        coords=coords,
        bins={'longitude': 36, 'latitude': 18},
        method='mean'
    )
    
    # 4. Plot heatmap
    fig = self.plotter.plot_2d_field(
        data=residual_map.cpu().numpy(),
        coords=...,  # Grid coords from binning
        var_name='mass_residual',
        projection='cartopy',
        colormap='Reds',
        title='Mass Continuity Residual'
    )
    self._log_figure(fig, 'physics_mass_residual')
```

---

## Implementation Roadmap

### PHASE 1: Core Infrastructure (TOMORROW - 2-3 hours)

- [ ] Add properties to ERA5MultiData
    - input_dim, output_dim, coord_order, var_order
    - Test: verify properties return correct values

- [ ] Update TroposphereDataModule
    - Use data.input_dim instead of hardcoded 4
    - Use data.output_dim instead of len(solution_vars)
    - Test: verify dimensions are correct

- [ ] Refactor physics code to accept coord_labels
    - Update compute_troposphere_gradients()
    - Update troposphere_pde_residual()
    - Test: physics with coords in different orders

- [ ] Update LightningModule
    - Remove _reorder_coords_for_physics()
    - Pass coord_labels to physics functions
    - Use var_order for building outputs dict
    - Test: training step runs without errors

### PHASE 2: Geometry Layer (TOMORROW - 2 hours)

- [ ] Implement GeometryGenerator class
    - point(), line(), plane(), volume()
    - random_samples(), from_indices()
    - Test: generate grids of expected shapes

- [ ] Implement ModelInference class
    - predict(), predict_with_physics()
    - Batched inference for large grids
    - Test: predictions match direct model calls

### PHASE 3: Aggregation + Plotting (LATER - 3-4 hours)

- [ ] Implement DataAggregator class
    - reduce_axis(), zonal_mean(), spatial_stats()
    - Test: aggregated data has correct shape

- [ ] Implement Plotter class
    - plot_2d_field(), plot_1d_profile(), plot_scatter()
    - Test: generates valid matplotlib figures

- [ ] Refactor AtmosphericVisualizationCallback
    - Use new 4-layer architecture
    - Add training distribution plots
    - Test: callback runs during validation

### PHASE 4: Advanced Visualizations (LATER - ongoing)

- [ ] Temporal evolution plots (error over time)
- [ ] Animated plots (rotating 3D, time-lapse)
- [ ] Model comparison plots (A vs B)
- [ ] Uncertainty quantification
- [ ] Interactive plots (plotly/bokeh)

### KEY MILESTONES

✓ End of Phase 1: Model trains successfully with new architecture  
✓ End of Phase 2: Can generate any geometry and get predictions  
✓ End of Phase 3: Basic visualizations working in callback  
✓ End of Phase 4: Publication-quality visualizations  

---

## Example Usage Flows

### Example 1: Quick horizontal slice

```python
geometry = GeometryGenerator(coord_ranges, coord_labels)
inference = ModelInference(model, datamodule)
plotter = Plotter(coord_labels, var_labels)

coords = geometry.plane(axes=['longitude', 'latitude'],
                        resolutions={'longitude': 2, 'latitude': 2},
                        pressure_level=500, valid_time=0.5)
preds = inference.predict(coords)
fig = plotter.plot_2d_field(preds[:, 0], coords, 'w')
```

### Example 2: Zonal mean with aggregation

```python
coords = geometry.volume(axes=['longitude', 'latitude', 'pressure_level'],
                         resolutions={...}, valid_time=0.5)
preds = inference.predict(coords)
zonal_data, zonal_coords = aggregator.zonal_mean(preds, coords)
fig = plotter.plot_2d_field(zonal_data[:, 1], zonal_coords, 'u')
```

### Example 3: Error evolution over time

```python
# Generate 4D volume
coords = geometry.volume(axes=['longitude', 'latitude',
                               'pressure_level', 'valid_time'],
                         resolutions={...})
errors = inference.compute_errors(coords, targets)
time_series = aggregator.temporal_evolution(errors, coords,
                                            spatial_agg='mean')
fig = plotter.plot_1d_profile(time_series, 'time', 'mse')
```

### Example 4: Physics residuals on validation set

```python
val_indices = datamodule.val_dataset.indices
coords = geometry.from_indices(val_indices, datamodule.data)
results = inference.predict_with_physics(coords)
fig = plotter.plot_scatter(coords, results['mass_continuity'].abs(),
                           projection='3d', colormap='Reds')
```
