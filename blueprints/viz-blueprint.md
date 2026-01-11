# Visualization Refactoring Blueprint

**Goal:** Refactor existing visualization code into clean 4-layer architecture without starting from scratch.

---

## Current State

✅ **Already have:**
- Grid generation functions (`_create_horizontal_grid`, `_create_meridional_grid`, `_create_full_grid`)
- Batched inference (`batched_model_forward`, `batched_physics_residuals`)
- Plotting functions (`_create_horizontal_plot`, `_create_meridional_plot`, etc.)
- Working callback that generates visualizations

❌ **Problems:**
- Logic scattered across callback and utils
- Hard to test individual components
- Difficult to add new visualization types
- Normalization logic mixed with geometry
- Can't reuse components in eval scripts

---

## Target Architecture

**4 Layers (each in separate file):**

### Layer 1: Geometry Generation (`geometry.py`)
**Responsibility:** Create coordinate grids to sample  
**Input:** Coordinate ranges, resolution  
**Output:** Coordinate tensor (N, 4) in **physical coordinates**

```python
class GeometryGenerator:
    def point(self, **coords) -> Tensor
    def line(self, axis: str, values: array, **fixed) -> Tensor
    def plane(self, axes: List[str], resolutions: Dict, **fixed) -> Tensor
    def volume(self, axes: List[str], resolutions: Dict, **fixed) -> Tensor
    def random_samples(self, n: int, seed: int = None) -> Tensor
    def from_indices(self, indices: np.ndarray, data) -> Tensor
```

### Layer 2: Model Inference (`inference.py`)
**Responsibility:** Get model predictions for coordinates  
**Input:** Coordinates (N, 4), model  
**Output:** Predictions (N, num_vars) or derived quantities

```python
class ModelInference:
    def predict(self, coords: Tensor, denormalize: bool = True) -> Tensor
    def predict_with_physics(self, coords: Tensor) -> Dict
    def compute_errors(self, coords: Tensor, targets: Tensor) -> Tensor
```

### Layer 3: Data Aggregation (`aggregation.py`)
**Responsibility:** Reduce dimensions via aggregation  
**Input:** Data (N, num_vars), coordinates (N, 4)  
**Output:** Aggregated data with reduced shape

```python
class DataAggregator:
    def reduce_axis(self, data, coords, axis: str, method: str) -> Tensor
    def zonal_mean(self, data, coords) -> Tuple[Tensor, Tensor]
    def spatial_bin(self, data, coords, bins: Dict, method: str) -> Tensor
    def temporal_evolution(self, data, coords, spatial_agg: str) -> Tensor
```

### Layer 4: Plotting (`plotting.py`)
**Responsibility:** Create matplotlib/cartopy figures  
**Input:** Aggregated data, coordinates, styling  
**Output:** Figure object

```python
class Plotter:
    def plot_2d_field(self, data, coords, var_name: str, **style) -> Figure
    def plot_1d_profile(self, data, coords, var_name: str, **style) -> Figure
    def plot_scatter(self, coords, values, **style) -> Figure
    def plot_error_heatmap(self, errors, coords, var_names, **style) -> Figure
    def plot_comparison(self, data1, data2, coords, var_name, **style) -> Figure
```

### Orchestration: Callback (`callback.py`)
**Combines all 4 layers**

```python
class AtmosphericVisualizationCallback(Callback):
    def setup(self, trainer, pl_module, stage):
        self.geometry = GeometryGenerator(...)
        self.inference = ModelInference(...)
        self.aggregator = DataAggregator(...)
        self.plotter = Plotter(...)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Use pipeline to create visualizations
```

---

## Migration Plan

### **PHASE 1: Extract Geometry Layer** (1-2 hours)

**Create `visualization/geometry.py`:**

1. **Extract existing grid functions:**
   - `_create_horizontal_grid` → `plane(axes=['longitude', 'latitude'], ...)`
   - `_create_meridional_grid` → `plane(axes=['latitude', 'pressure_level'], ...)`
   - `_create_full_grid` → `volume(axes=['longitude', 'latitude', 'pressure_level'], ...)`

2. **Key changes:**
   - ✅ Remove normalization logic (moves to Layer 2)
   - ✅ Always return physical coordinates
   - ✅ Make API flexible: `plane()` handles any 2D slice, not just horizontal
   - ✅ Add `coord_order` property to ensure consistent ordering

3. **Implementation:**

```python
class GeometryGenerator:
    def __init__(self, coord_ranges: Dict, coord_labels: Dict):
        self.coord_ranges = coord_ranges
        self.coord_labels = coord_labels
        self.coord_order = sorted(coord_labels.keys(), 
                                  key=lambda k: coord_labels[k])
    
    def plane(self, axes: List[str], resolutions: Dict, 
              **fixed) -> torch.Tensor:
        """Create 2D plane by varying two coordinates, fixing others.
        
        Args:
            axes: Two coordinate names to vary, e.g., ['longitude', 'latitude']
            resolutions: Resolution for each axis, e.g., {'longitude': 2, 'latitude': 2}
            **fixed: Fixed values for other coordinates, e.g., pressure_level=500
        
        Returns:
            (N, 4) tensor in physical coordinates, ordered by coord_order
        
        Examples:
            # Horizontal slice
            plane(axes=['longitude', 'latitude'],
                  resolutions={'longitude': 2, 'latitude': 2},
                  pressure_level=500, valid_time=0.5)
            
            # Meridional slice
            plane(axes=['latitude', 'pressure_level'],
                  resolutions={'latitude': 2},
                  pressure_levels=[850, 500, 200],
                  longitude=0, valid_time=0.5)
        """
        # YOUR EXISTING GRID CREATION LOGIC HERE
        # But generalized to handle any 2D slice
        pass
    
    def volume(self, axes: List[str], resolutions: Dict,
               **fixed) -> torch.Tensor:
        """Create 3D volume - generalized _create_full_grid."""
        pass
    
    def random_samples(self, n: int, seed: Optional[int] = None) -> torch.Tensor:
        """Random LHS samples - similar to generate_virtual_points."""
        from pyDOE3 import lhs
        
        if seed is not None:
            np.random.seed(seed)
        
        # LHS sample in [0, 1]^4
        samples = lhs(n=len(self.coord_order), samples=n)
        
        # Map to physical ranges
        coords = np.zeros_like(samples)
        for i, coord_name in enumerate(self.coord_order):
            min_val, max_val = self.coord_ranges[coord_name]
            coords[:, i] = samples[:, i] * (max_val - min_val) + min_val
        
        return torch.from_numpy(coords).float()
    
    def from_indices(self, indices: np.ndarray, 
                     data) -> torch.Tensor:
        """Convert dataset indices to physical coordinates."""
        # Use data.get_coords_at_index() from ERA5MultiData
        pass
```

4. **Test:**
```python
# Test that plane() produces expected shape
geometry = GeometryGenerator(coord_ranges, coord_labels)
coords = geometry.plane(
    axes=['longitude', 'latitude'],
    resolutions={'longitude': 2, 'latitude': 2},
    pressure_level=500,
    valid_time=0.5
)
assert coords.shape == (180*91, 4)  # 180 lons × 91 lats
```

---

### **PHASE 2: Extract Inference Layer** (1 hour)

**Create `visualization/inference.py`:**

1. **Move batched inference functions:**
   - `batched_model_forward` → `ModelInference.predict()`
   - `batched_physics_residuals` → `ModelInference.predict_with_physics()`

2. **Key changes:**
   - ✅ Accepts physical coordinates, handles normalization internally
   - ✅ Uses datamodule for normalization/denormalization
   - ✅ Returns denormalized predictions by default

3. **Implementation:**

```python
class ModelInference:
    def __init__(self, model, datamodule):
        self.model = model
        self.datamodule = datamodule
    
    def predict(self, coords: torch.Tensor,
                denormalize_output: bool = True,
                batch_size: int = 50000) -> torch.Tensor:
        """Get model predictions with batching.
        
        Args:
            coords: (N, 4) in physical coordinates
            denormalize_output: Whether to denormalize predictions
            batch_size: Batch size for inference
            
        Returns: (N, num_vars) predictions
        """
        # Normalize coords
        coords_norm = self.datamodule.normalize_coords(coords)
        
        # Batched forward pass (YOUR EXISTING CODE)
        preds = []
        n = coords_norm.shape[0]
        for i in range(0, n, batch_size):
            batch = coords_norm[i:i+batch_size].to(self.model.device)
            with torch.no_grad():
                out = self.model(batch)
            preds.append(out.cpu())
        
        preds = torch.cat(preds, dim=0)
        
        # Denormalize if requested
        if denormalize_output:
            for i, var in enumerate(self.datamodule.data.var_order):
                preds[:, i] = self.datamodule.denormalize(preds[:, i], var)
        
        return preds
    
    def predict_with_physics(self, coords: torch.Tensor,
                            batch_size: int = 20000) -> Dict:
        """Predictions + physics residuals - YOUR batched_physics_residuals."""
        # Similar to existing code, but uses datamodule for normalization
        pass
    
    def compute_errors(self, coords: torch.Tensor,
                      targets: torch.Tensor) -> torch.Tensor:
        """Compute MSE per variable."""
        preds = self.predict(coords)
        return (preds - targets).pow(2)
```

4. **Test:**
```python
# Test that predictions match old batched_model_forward
inference = ModelInference(model, datamodule)
coords = geometry.plane(...)
preds_new = inference.predict(coords)
preds_old = batched_model_forward(model, coords_normalized_manually)
# After denormalization, should match
```

---

### **PHASE 3: Minimal Aggregation** (30 min)

**Create `visualization/aggregation.py`:**

Start with just what you need now, expand later:

```python
class DataAggregator:
    def __init__(self, coord_labels: Dict):
        self.coord_labels = coord_labels
    
    def spatial_bin(self, data: torch.Tensor, coords: torch.Tensor,
                   bins: Dict, method: str = 'mean') -> np.ndarray:
        """Bin data spatially for heatmaps - YOUR histogram2d logic.
        
        Args:
            data: (N,) or (N, num_vars) values to aggregate
            coords: (N, 4) coordinates
            bins: Dict like {'longitude': 72, 'latitude': 36}
            method: 'mean', 'sum', 'count'
            
        Returns: Gridded data array
        """
        lon_idx = self.coord_labels['longitude']
        lat_idx = self.coord_labels['latitude']
        
        lons = coords[:, lon_idx].cpu().numpy()
        lats = coords[:, lat_idx].cpu().numpy()
        values = data.cpu().numpy()
        
        # YOUR EXISTING histogram2d LOGIC
        H, xedges, yedges = np.histogram2d(
            lons, lats,
            bins=[bins['longitude'], bins['latitude']],
            weights=values
        )
        
        if method == 'mean':
            counts, _, _ = np.histogram2d(lons, lats, 
                                         bins=[bins['longitude'], bins['latitude']])
            H = np.divide(H, counts, where=counts > 0, out=np.zeros_like(H))
        
        return H
    
    def zonal_mean(self, data: torch.Tensor, coords: torch.Tensor) -> Tuple:
        """Average over longitude - implement later."""
        pass
```

---

### **PHASE 4: Keep Existing Plotting** (30 min)

**Create `visualization/plotting.py`:**

Just reorganize, minimal changes:

```python
class Plotter:
    def __init__(self, coord_labels: Dict, var_labels: Dict):
        self.coord_labels = coord_labels
        self.var_labels = var_labels
    
    def plot_2d_field(self, data: np.ndarray, coords: np.ndarray,
                     var_name: str, projection: str = 'cartopy',
                     **kwargs) -> plt.Figure:
        """2D field plot - routes to appropriate sub-function.
        
        Args:
            data: 2D array to plot
            coords: Coordinate array (used for extent)
            var_name: Variable name
            projection: 'cartopy', 'lat_pressure', or 'lon_pressure'
            **kwargs: Additional plot styling
        """
        if projection == 'cartopy':
            return self._plot_horizontal(data, var_name, **kwargs)
        elif projection == 'lat_pressure':
            return self._plot_meridional(data, var_name, **kwargs)
        else:
            raise ValueError(f"Unknown projection: {projection}")
    
    def _plot_horizontal(self, data, var_name, **kwargs):
        """YOUR EXISTING _create_horizontal_plot code - NO CHANGES."""
        pass
    
    def _plot_meridional(self, data, var_name, **kwargs):
        """YOUR EXISTING _create_meridional_plot code - NO CHANGES."""
        pass
    
    def plot_error_heatmap(self, errors, coords, var_names, **kwargs):
        """YOUR EXISTING plot_error_heatmap code - NO CHANGES."""
        pass
```

---

### **PHASE 5: Update Callback** (1 hour)

**Update `visualization/callback.py`:**

Replace old functions with new pipeline:

```python
from geolab.visualization.geometry import GeometryGenerator
from geolab.visualization.inference import ModelInference
from geolab.visualization.aggregation import DataAggregator
from geolab.visualization.plotting import Plotter


class AtmosphericVisualizationCallback(Callback):
    def __init__(self, plot_every_n_epochs: int = 1, ...):
        # Config only, no object creation
        self.plot_every_n_epochs = plot_every_n_epochs
        # ... other config params
        
        # These are created in setup()
        self.geometry = None
        self.inference = None
        self.aggregator = None
        self.plotter = None
    
    def setup(self, trainer, pl_module, stage):
        """Initialize 4-layer pipeline."""
        if stage != 'fit':
            return
        
        dm = trainer.datamodule
        
        self.geometry = GeometryGenerator(
            coord_ranges=dm.coordinate_ranges,
            coord_labels=dm.data.coord_labels
        )
        
        self.inference = ModelInference(
            model=pl_module,
            datamodule=dm
        )
        
        self.aggregator = DataAggregator(
            coord_labels=dm.data.coord_labels
        )
        
        self.plotter = Plotter(
            coord_labels=dm.data.coord_labels,
            var_labels=dm.data.variable_labels
        )
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Use new pipeline for visualizations."""
        if trainer.current_epoch % self.plot_every_n_epochs != 0:
            return
        if not trainer.is_global_zero:
            return
        
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is None:
            return
        
        # OLD CODE:
        # self._plot_and_log_horizontal_slices(wandb_logger, pl_module, val_batch)
        
        # NEW CODE:
        for pressure in self.pressure_levels:
            coords = self.geometry.plane(
                axes=['longitude', 'latitude'],
                resolutions=self.grid_resolution,
                pressure_level=pressure,
                valid_time=0.5
            )
            
            preds = self.inference.predict(coords)
            
            for i, var in enumerate(pl_module.datamodule.data.var_order):
                # Reshape for plotting (geometry gives flat array)
                n_lons = 180 // self.grid_resolution['longitude']
                n_lats = 91
                pred_2d = preds[:, i].reshape(n_lons, n_lats).cpu().numpy()
                
                fig = self.plotter.plot_2d_field(
                    data=pred_2d,
                    coords=coords.cpu().numpy(),
                    var_name=var,
                    projection='cartopy',
                    pressure=pressure
                )
                
                wandb_logger.log_image(
                    key=f'val/horizontal_{pressure}hPa_{var}',
                    images=[fig]
                )
                plt.close(fig)
        
        # Similarly for other visualization types...
```

---

## Timeline

### **Week 1 (Today/Tomorrow):**
- [x] Phase 1: Create `geometry.py` - implement `plane()` and `volume()`
- [x] Phase 2: Create `inference.py` - move batched functions
- [ ] Test: Geometry generates correct shapes, inference matches old code

### **Week 2:**
- [ ] Phase 3: Create `aggregation.py` - just `spatial_bin()` for now
- [ ] Phase 4: Create `plotting.py` - copy existing plot functions
- [ ] Phase 5: Update callback to use new pipeline for 1-2 plot types
- [ ] Test: Callback produces same plots as before

### **Later:**
- [ ] Add `point()`, `line()`, `from_indices()` to geometry
- [ ] Add `zonal_mean()`, `reduce_axis()` to aggregation
- [ ] Expand plotter with new visualization types
- [ ] Gradually migrate all callback methods to use pipeline

---

## Key Principles

1. **Don't break existing code** - migrate incrementally, one visualization type at a time
2. **Test at each phase** - verify new code produces identical results to old
3. **Keep old code until migration complete** - comment out, don't delete
4. **Start simple** - implement minimal versions first, expand later
5. **One layer at a time** - finish geometry before starting inference

---

## Success Criteria

✅ **Phase 1 Complete:** Can generate coordinate grids without calling old functions  
✅ **Phase 2 Complete:** Can get predictions without manually normalizing coords  
✅ **Phase 3 Complete:** Callback successfully uses new pipeline for ≥1 visualization  
✅ **Full Migration:** All old helper functions deleted, callback only uses 4 layers  
✅ **Testing:** Can create standalone eval scripts that reuse visualization components

---

## Example Usage (After Migration)

**Creating a horizontal slice:**

```python
# OLD (scattered logic)
grid_coords, norm_coords, n_lats, n_lons = _create_horizontal_grid(...)
preds = batched_model_forward(model, norm_coords)
# Manual denormalization
pred_2d = preds.reshape(...)
fig = _create_horizontal_plot(pred_2d, ...)

# NEW (clean pipeline)
coords = geometry.plane(axes=['longitude', 'latitude'],
                       resolutions={'longitude': 2, 'latitude': 2},
                       pressure_level=500, valid_time=0.5)
preds = inference.predict(coords)  # Auto-denormalized
pred_2d = preds[:, var_idx].reshape(n_lons, n_lats).numpy()
fig = plotter.plot_2d_field(pred_2d, coords, var_name='u')
```

**Computing zonal mean:**

```python
# NEW (will be possible after full migration)
coords = geometry.volume(axes=['longitude', 'latitude', 'pressure_level'],
                        resolutions={...}, valid_time=0.5)
preds = inference.predict(coords)
zonal_data, zonal_coords = aggregator.zonal_mean(preds, coords)
fig = plotter.plot_2d_field(zonal_data[:, var_idx], zonal_coords,
                            var_name='u', projection='lat_pressure')
```

---

## Notes

- Keep existing `GROUND_TRUTH_RANGES` dict in `plotting.py`
- Keep existing colormap/styling logic intact
- Geometry layer should be pure coordinate generation - no matplotlib
- Inference layer handles all normalization - geometry and aggregation stay in physical space
- Each layer should be independently testable
