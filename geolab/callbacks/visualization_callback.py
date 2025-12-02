"""
Callback for visualizing atmospheric model predictions during training.
"""
from lightning import Callback
import torch
import matplotlib.pyplot as plt
from typing import List, Optional
from lightning.pytorch.loggers import WandbLogger

from geolab.utils import visualiser


class AtmosphericVisualizationCallback(Callback):
    """Callback for visualizing atmospheric model predictions during training."""
    
    def __init__(
        self,
        plot_every_n_epochs: int = 1,
        plot_every_n_steps: Optional[int] = None,
        _last_log_step: Optional[int] = None,
        pressure_levels: List[int] = [850, 500, 200],
        meridional_longitudes: List[int] = [0, 180],
        grid_resolution: Optional[dict] = None,
        enable_horizontal_slices: bool = True,
        enable_meridional_slices: bool = True,
        enable_zonal_mean: bool = True,
        enable_error_heatmap: bool = True,
        enable_physics_residuals: bool = True,
    ):
        """
        Initialize the visualization callback.
        
        Args:
            plot_every_n_epochs: Plot visualizations every N epochs
            pressure_levels: List of pressure levels (hPa) for horizontal slices
            meridional_longitudes: List of longitudes for meridional cross-sections
            grid_resolution: Grid resolution in degrees for visualization
            enable_horizontal_slices: Whether to plot horizontal slices
            enable_meridional_slices: Whether to plot meridional cross-sections
            enable_zonal_mean: Whether to plot zonal means
            enable_error_heatmap: Whether to plot error distribution heatmap
            enable_physics_residuals: Whether to plot physics residual maps (PINN only)
        """
        super().__init__()
        if grid_resolution is None:
            grid_resolution = {"longitude": 2, "latitude": 2}
            
        self.plot_every_n_epochs = plot_every_n_epochs
        self.pressure_levels = pressure_levels
        self.meridional_longitudes = meridional_longitudes
        self.grid_resolution = grid_resolution
        
        # Flags to enable/disable specific visualizations
        self.enable_horizontal_slices = enable_horizontal_slices
        self.enable_meridional_slices = enable_meridional_slices
        self.enable_zonal_mean = enable_zonal_mean
        self.enable_error_heatmap = enable_error_heatmap
        self.enable_physics_residuals = enable_physics_residuals
        
        # Variable names matching model output order: ['t', 'w', 'u', 'z', 'v']
        self.var_names = ['w', 'u', 'z', 'v']
        self.var_labels = {
            't': 'Temperature (K)',
            'w': 'Vertical Velocity (Pa/s)',
            'u': 'Zonal Wind (m/s)',
            'z': 'Geopotential (m)',
            'v': 'Meridional Wind (m/s)'
        }        
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when validation epoch ends."""
        # Only plot every N epochs and on rank 0
        if trainer.current_epoch % self.plot_every_n_epochs != 0:
            return
        if not trainer.is_global_zero:
            return
        
        # Verify model has statistics
        if not hasattr(pl_module, 'statistics') or pl_module.statistics is None:
            print("WARNING: Model does not have statistics! Cannot create visualizations.")
            print("Make sure to set model.statistics in train.py before training starts.")
            return
        
        # Get wandb logger
        wandb_logger = self._get_wandb_logger(trainer)
        if wandb_logger is None:
            print("Warning: WandB logger not found, skipping visualizations")
            return
        
        # Get a validation batch
        try:
            val_batch = next(iter(trainer.val_dataloaders))
        except StopIteration:
            print("Warning: No validation data available")
            return
        
        # Move to device
        val_batch = self._move_batch_to_device(val_batch, pl_module.device)
        
        # Set model to eval
        pl_module.eval()
        
        with torch.no_grad():
            # 1. Error distribution heatmap
            if self.enable_error_heatmap:
                self._plot_and_log_error_heatmap(
                    wandb_logger, pl_module, val_batch
                )
            
            # 2. Horizontal slices at key pressure levels
            if self.enable_horizontal_slices:
                self._plot_and_log_horizontal_slices(
                    wandb_logger, pl_module, val_batch
                )
            
            # 3. Meridional cross-sections
            if self.enable_meridional_slices:
                self._plot_and_log_meridional_slices(
                    wandb_logger, pl_module, val_batch
                )
            
            # 4. Zonal mean profiles
            if self.enable_zonal_mean:
                self._plot_and_log_zonal_mean(
                    wandb_logger, pl_module, val_batch
                )
            
            # 5. Physics residual maps (PINN only)
            if self.enable_physics_residuals and pl_module.train_pinn:
                self._plot_and_log_physics_residuals(
                    wandb_logger, pl_module, val_batch
                )
        
        pl_module.train()
    
    def _plot_and_log_error_heatmap(self, logger, pl_module, val_batch):
        """Plot and log error distribution heatmap."""
        try:
            fig = visualiser.plot_error_heatmap(
                pl_module,
                val_batch,
                self.var_names
            )
            if fig is not None:
                logger.log_image(key="val/error_heatmap", images=[fig])
                plt.close(fig)
        except Exception as e:
            print(f"Error creating error heatmap: {e}")
    
    def _plot_and_log_horizontal_slices(self, logger, pl_module, val_batch):
        """Plot and log horizontal slices at key pressure levels."""
        for pressure in self.pressure_levels:
            try:
                figs = visualiser.plot_horizontal_slices(
                    pl_module,
                    val_batch,
                    pressure,
                    self.var_names,
                    self.grid_resolution
                )
                for var_name, fig in figs.items():
                    if fig is not None:
                        logger.log_image(
                            key=f"val/horizontal_{pressure}hPa_{var_name}",
                            images=[fig]
                        )
                        plt.close(fig)
            except Exception as e:
                print(f"Error creating horizontal slice at {pressure} hPa: {e}")
    
    def _plot_and_log_meridional_slices(self, logger, pl_module, val_batch):
        """Plot and log meridional cross-sections."""
        for lon in self.meridional_longitudes:
            try:
                figs = visualiser.plot_meridional_slices(
                    pl_module,
                    val_batch,
                    lon,
                    self.var_names,
                    self.pressure_levels,
                    self.grid_resolution
                )
                for var_name, fig in figs.items():
                    if fig is not None:
                        logger.log_image(
                            key=f"val/meridional_{lon}E_{var_name}",
                            images=[fig]
                        )
                        plt.close(fig)
            except Exception as e:
                print(f"Error creating meridional slice at {lon}°: {e}")
    
    def _plot_and_log_zonal_mean(self, logger, pl_module, val_batch):
        """Plot and log zonal mean profiles."""
        try:
            figs = visualiser.plot_zonal_mean(
                pl_module,
                val_batch,
                self.var_names,
                self.pressure_levels,
                self.grid_resolution
            )
            for var_name, fig in figs.items():
                if fig is not None:
                    logger.log_image(
                        key=f"val/zonal_mean_{var_name}",
                        images=[fig]
                    )
                    plt.close(fig)
        except Exception as e:
            print(f"Error creating zonal mean: {e}")
    
    def _plot_and_log_physics_residuals(self, logger, pl_module, val_batch):
        """Plot and log physics residual maps (PINN only)."""
        try:
            fig = visualiser.plot_physics_residuals(
                pl_module,
                val_batch,
                self.var_names
            )
            if fig is not None:
                logger.log_image(key="val/physics_residuals", images=[fig])
                plt.close(fig)
        except Exception as e:
            print(f"Error creating physics residuals: {e}")
    
    def _get_wandb_logger(self, trainer) -> Optional[WandbLogger]:
        """Get WandB logger from trainer."""
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                return logger
        return None
    
    def _move_batch_to_device(self, batch, device):
        """Recursively move batch to device."""
        if isinstance(batch, dict):
            return {k: self._move_batch_to_device(v, device) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, list):
            return [self._move_batch_to_device(item, device) for item in batch]
        else:
            return batch