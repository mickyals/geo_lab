"""
Callback for visualizing atmospheric model predictions during training.
"""
from lightning import Callback
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Any, Union
from pathlib import Path

from geolab.viz import visualizer as visualiser


class AtmosphericVisualizationCallback(Callback):
    """Flexible, logger-agnostic callback for atmospheric visualizations."""

    def __init__(
        self,
        visualizations: List[Dict[str, Any]],
        plot_every_n_epochs: Optional[int] = None,
        plot_every_n_steps: Optional[int] = None,
        save_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the visualization callback.

        Args:
            visualizations: List of visualization configs. Each dict must have 'type' key.

                Config types and their parameters:

                'field':
                    - geometry: Dict with geometry specification
                    - vars: List[str] - variables to plot
                    - aggregation: Optional[Dict] - aggregation spec
                    - plot_type: str - '2d_field' (default), '1d_profile', 'scatter'
                    - data_source: str - 'model', 'ground_truth', 'comparison'
                    - projection: str - 'cartopy', 'meridional', etc.
                    - split: str - 'train', 'val', 'test' (default 'val')
                    - epoch_filter: Optional[List[int]] - only plot on these epochs

                'scatter':
                    - split: str - 'train', 'val', or 'test'
                    - projection: str - 'lat_lon', 'lat_pressure', '3d'
                    - color_by: Optional[str] - variable name to color by
                    - epoch_filter: Optional[List[int]]

                'error_heatmap':
                    - split: str - 'train', 'val', or 'test'
                    - vars: List[str]
                    - epoch_filter: Optional[List[int]]

                'physics_residuals':
                    - split: str - 'train' or 'val' (default 'val')
                    - vars: List[str] (not used, for consistency)
                    - epoch_filter: Optional[List[int]]

            plot_every_n_epochs: Plot every N epochs (mutually exclusive with n_steps)
            plot_every_n_steps: Plot every N steps (mutually exclusive with n_epochs)
            save_dir: Directory to save figures locally (in addition to logger)

        Example:
            callback = AtmosphericVisualizationCallback(
                visualizations=[
                    # Model predictions - horizontal slice
                    {
                        'type': 'field',
                        'geometry': {
                            'type': 'plane',
                            'axes': ['longitude', 'latitude'],
                            'pressure_level': 500,
                            'valid_time': 0.0,
                            'resolution': {'longitude': 2.0, 'latitude': 2.0}
                        },
                        'vars': ['u', 'v'],
                        'projection': 'cartopy',
                        'data_source': 'model'
                    },

                    # Ground truth - only plot once at epoch 0
                    {
                        'type': 'field',
                        'geometry': {
                            'type': 'plane',
                            'axes': ['longitude', 'latitude'],
                            'pressure_level': 500,
                            'valid_time': 0.0,
                            'resolution': {'longitude': 2.0, 'latitude': 2.0}
                        },
                        'vars': ['u', 'v'],
                        'projection': 'cartopy',
                        'data_source': 'ground_truth',
                        'epoch_filter': [0]
                    },

                    # Meridional slice
                    {
                        'type': 'field',
                        'geometry': {
                            'type': 'plane',
                            'axes': ['latitude', 'pressure_level'],
                            'longitude': 0,
                            'valid_time': 0.0,
                            'resolution': {'latitude': 2.0, 'pressure_level': 50.0}
                        },
                        'vars': ['w'],
                        'projection': 'meridional'
                    },

                    # Zonal mean
                    {
                        'type': 'field',
                        'geometry': {
                            'type': 'plane',
                            'axes': ['longitude', 'latitude'],
                            'pressure_level': 500,
                            'valid_time': 0.0,
                            'resolution': {'longitude': 2.0, 'latitude': 2.0}
                        },
                        'vars': ['u', 'v'],
                        'aggregation': {'type': 'zonal_mean'},
                        'plot_type': '1d_profile'
                    },

                    # Train data distribution
                    {
                        'type': 'scatter',
                        'split': 'train',
                        'projection': 'lat_lon',
                        'epoch_filter': [0, 10, 50]
                    },

                    # Val data distribution
                    {
                        'type': 'scatter',
                        'split': 'val',
                        'projection': 'lat_lon'
                    },

                    # Error heatmaps
                    {
                        'type': 'error_heatmap',
                        'split': 'val',
                        'vars': ['u', 'v', 'w', 'z']
                    },

                    # Physics residuals (PINN only)
                    {
                        'type': 'physics_residuals',
                        'split': 'val'
                    }
                ],
                plot_every_n_epochs=5,
                save_dir='./visualizations'
            )
        """
        super().__init__()

        # Validation
        if plot_every_n_epochs is None and plot_every_n_steps is None:
            raise ValueError("Must specify either plot_every_n_epochs or plot_every_n_steps")
        if plot_every_n_epochs is not None and plot_every_n_steps is not None:
            raise ValueError("Cannot specify both plot_every_n_epochs and plot_every_n_steps")

        self.visualizations = visualizations
        self.plot_every_n_epochs = plot_every_n_epochs
        self.plot_every_n_steps = plot_every_n_steps

        # Save directory
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Track last logged step for step-based logging
        self._last_log_step = -1

        # Cache for batches by split
        self._cached_batches = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called when validation epoch ends."""
        # Check if we should plot
        if not self._should_plot(trainer):
            return

        # Only plot on rank 0
        if not trainer.is_global_zero:
            return

        # Set model to eval
        pl_module.eval()

        with torch.no_grad():
            # Get batches for different splits as needed
            self._cache_batches(trainer, pl_module)

            # Loop through all requested visualizations
            for viz_config in self.visualizations:
                self._create_and_log_visualization(
                    trainer, pl_module, viz_config
                )

        # Clear batch cache to free memory
        self._cached_batches.clear()

        pl_module.train()

    def _should_plot(self, trainer) -> bool:
        """Determine if we should plot this epoch/step."""
        if self.plot_every_n_epochs is not None:
            return trainer.current_epoch % self.plot_every_n_epochs == 0

        elif self.plot_every_n_steps is not None:
            global_step = trainer.global_step
            if global_step - self._last_log_step >= self.plot_every_n_steps:
                self._last_log_step = global_step
                return True

        return False

    def _cache_batches(self, trainer, pl_module):
        """Cache batches for all needed splits."""
        self._cached_batches = {}

        # Determine which splits are needed
        needed_splits = set()
        for viz_config in self.visualizations:
            split = viz_config.get('split', 'val')
            needed_splits.add(split)

        # Get batches
        for split in needed_splits:
            if split == 'train':
                batch = self._get_batch(trainer.train_dataloader, pl_module.device)
            elif split == 'val':
                batch = self._get_batch(trainer.val_dataloaders, pl_module.device)
            elif split == 'test' and hasattr(trainer, 'test_dataloaders'):
                batch = self._get_batch(trainer.test_dataloaders, pl_module.device)
            else:
                batch = None

            if batch is not None:
                self._cached_batches[split] = batch

    def _get_batch(self, dataloader, device):
        """Get a batch from dataloader and move to device."""
        try:
            if hasattr(dataloader, '__iter__'):
                batch = next(iter(dataloader))
            else:
                batch = next(iter(dataloader[0]))  # Handle multiple dataloaders
            return self._move_batch_to_device(batch, device)
        except (StopIteration, AttributeError, IndexError, TypeError):
            return None

    def _create_and_log_visualization(
        self,
        trainer,
        pl_module,
        config: Dict[str, Any]
    ):
        """Create and log a single visualization based on config."""
        # Check epoch filter
        if 'epoch_filter' in config:
            if trainer.current_epoch not in config['epoch_filter']:
                return

        viz_type = config['type']
        split = config.get('split', 'val')

        # Get appropriate batch
        batch = self._cached_batches.get(split)
        if batch is None and viz_type != 'field':
            print(f"Warning: No {split} batch available for {viz_type}")
            return

        try:
            # Route to appropriate visualization function
            if viz_type == 'field':
                fig_dict = self._plot_field(pl_module, batch, config)

            elif viz_type == 'scatter':
                fig_dict = self._plot_scatter(pl_module, batch, config)

            elif viz_type == 'error_heatmap':
                fig_dict = self._plot_error_heatmap(pl_module, batch, config)

            elif viz_type == 'physics_residuals':
                fig_dict = self._plot_physics_residuals(pl_module, batch, config)

            else:
                print(f"Warning: Unknown visualization type: {viz_type}")
                return

            # Log figures
            self._log_figures(trainer, fig_dict, split)

        except Exception as e:
            print(f"Error creating {viz_type} visualization: {e}")
            import traceback
            traceback.print_exc()

    def _plot_field(self, pl_module, batch, config):
        """Plot field visualization."""
        geometry_spec = config['geometry']
        vars_to_plot = config.get('vars', ['u', 'v', 'w', 'z'])
        aggregation_spec = config.get('aggregation', None)
        plot_type = config.get('plot_type', '2d_field')
        data_source = config.get('data_source', 'model')
        projection = config.get('projection', 'cartopy')

        figs = visualiser.plot_field(
            pl_module=pl_module,
            geometry_spec=geometry_spec,
            var_names=vars_to_plot,
            aggregation_spec=aggregation_spec,
            plot_type=plot_type,
            data_source=data_source,
            batch=batch,
            projection=projection
        )

        # Generate keys based on geometry and aggregation
        key_prefix = self._generate_key_prefix(config)

        return {
            f"{key_prefix}_{var}": fig
            for var, fig in figs.items()
        }

    def _plot_scatter(self, pl_module, batch, config):
        """Plot scatter visualization."""
        split = config.get('split', 'val')
        projection = config.get('projection', 'lat_lon')
        color_by = config.get('color_by', None)

        fig = visualiser.plot_scatter(
            pl_module=pl_module,
            batch=batch,
            split_name=split,
            projection=projection,
            color_by=color_by
        )

        key = f"{split}/scatter_{projection}"
        if color_by:
            key += f"_by_{color_by}"

        return {key: fig}

    def _plot_error_heatmap(self, pl_module, batch, config):
        """Plot error heatmap."""
        split = config.get('split', 'val')
        vars_to_plot = config.get('vars', ['u', 'v', 'w', 'z'])

        fig = visualiser.plot_error_heatmap(
            pl_module=pl_module,
            batch=batch,
            var_names=vars_to_plot
        )

        return {f"{split}/error_heatmap": fig}

    def _plot_physics_residuals(self, pl_module, batch, config):
        """Plot physics residuals."""
        split = config.get('split', 'val')
        vars_to_plot = config.get('vars', ['u', 'v', 'w', 'z'])

        fig = visualiser.plot_physics_residuals(
            pl_module=pl_module,
            batch=batch,
            var_names=vars_to_plot
        )

        return {f"{split}/physics_residuals": fig}

    def _generate_key_prefix(self, config):
        """Generate logging key prefix from config."""
        split = config.get('split', 'val')
        data_source = config.get('data_source', 'model')
        geometry = config['geometry']
        aggregation = config.get('aggregation')

        # Base prefix
        prefix = f"{split}/{data_source}"

        # Add aggregation info
        if aggregation:
            agg_type = aggregation['type']
            prefix += f"_{agg_type}"
        else:
            # Add geometry info
            geom_type = geometry['type']
            axes = geometry.get('axes', [])

            if geom_type == 'plane':
                if set(axes) == {'longitude', 'latitude'}:
                    # Horizontal slice
                    pressure = geometry.get('pressure_level', 'surface')
                    prefix += f"_horizontal_{pressure}hPa"
                elif set(axes) == {'latitude', 'pressure_level'}:
                    # Meridional slice
                    lon = geometry.get('longitude', 0)
                    prefix += f"_meridional_{lon}E"
                elif set(axes) == {'longitude', 'pressure_level'}:
                    # Zonal slice
                    lat = geometry.get('latitude', 0)
                    prefix += f"_zonal_{lat}N"
                else:
                    prefix += f"_plane_{'_'.join(axes)}"
            else:
                prefix += f"_{geom_type}"

        return prefix

    def _log_figures(self, trainer, fig_dict: Dict[str, plt.Figure], split: str):
        """Log figures to all available loggers and optionally save locally."""
        for key, fig in fig_dict.items():
            if fig is None:
                continue

            # Save locally if requested
            if self.save_dir:
                epoch = trainer.current_epoch
                step = trainer.global_step
                filename = f"{key.replace('/', '_')}_epoch{epoch:04d}_step{step:06d}.png"
                fig.savefig(self.save_dir / filename, dpi=150, bbox_inches='tight')

            # Log to all loggers
            for logger in trainer.loggers:
                try:
                    self._log_to_logger(logger, key, fig, trainer)
                except Exception as e:
                    print(f"Warning: Could not log to {type(logger).__name__}: {e}")

            # Close figure
            plt.close(fig)

    def _log_to_logger(self, logger, key: str, fig: plt.Figure, trainer):
        """Log figure to specific logger type."""
        logger_type = type(logger).__name__

        if 'WandbLogger' in logger_type:
            logger.log_image(key=key, images=[fig])

        elif 'TensorBoardLogger' in logger_type:
            # TensorBoard expects images as arrays
            import io
            from PIL import Image
            import numpy as np

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            img_array = np.array(img)
            # Add batch dimension and transpose to CHW
            img_array = img_array.transpose(2, 0, 1)
            logger.experiment.add_image(
                key, img_array,
                global_step=trainer.global_step
            )

        elif 'CSVLogger' in logger_type:
            # CSV logger can't log images, skip
            pass

        else:
            # Unknown logger, try generic approaches
            if hasattr(logger, 'log_image'):
                logger.log_image(key=key, images=[fig])
            elif hasattr(logger.experiment, 'add_figure'):
                logger.experiment.add_figure(
                    key, fig,
                    global_step=getattr(trainer, 'global_step', 0)
                )
    
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