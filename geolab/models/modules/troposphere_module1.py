from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
import torch.nn as nn
from PyDOE3 import lhs
import torch.optim as optim
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from torchmetrics import MinMetric, MeanMetric
from geolab.models.components import troposphere_pde_residual
from geolab.models.model import FCN, SirenNet, GaussianNet, FinerNet, RealWireNet

class TroposhpereLightningModule(LightningModule):
    def __init__(
            self,
            # Basic network configs
            model_name: str,
            N_in_features: int,
            N_out_features: int,
            N_hidden_features: int,
            N_hidden_layers: int,
            # Model specific params
            model_params: Dict[str, Any],
            # Position encoder configs
            position_encoder_type: str,
            mapping_dim: int,
            scale: int,
            # Optimizer configs
            optimizer_name: str,
            optimizer_config: Dict,
            # Scheduler configs
            scheduler_name: str,
            scheduler_config: Dict,
            # PINN configs
            train_pinn: bool,
            mass_balance: bool,
            physics_loss_weight: float = None,
            # Virtual sampling configs
            include_virtual: bool = False,
            num_virtual_per_batch: int = 1000,
            # DataModule reference (passed from trainer)
            datamodule=None,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=['datamodule'])

        self.train_pinn = train_pinn
        self.mass_balance = mass_balance
        self.physics_loss_weight = physics_loss_weight
        self.include_virtual = include_virtual
        self.num_virtual_per_batch = num_virtual_per_batch
        self.datamodule = datamodule

        self.model = self._init_model()

        # Loss functions
        self.criterion = torch.nn.MSELoss(reduction="none")
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        if self.train_pinn:
            self.train_physics_loss = MeanMetric()
            self.train_data_loss = MeanMetric()
            self.train_mass_cont = MeanMetric()
            self.train_ns_longitude = MeanMetric()
            self.train_ns_latitude = MeanMetric()

            self.val_physics_loss = MeanMetric()
            self.val_data_loss = MeanMetric()
            self.val_mass_cont = MeanMetric()
            self.val_ns_longitude = MeanMetric()
            self.val_ns_latitude = MeanMetric()

            self.test_physics_loss = MeanMetric()
            self.test_data_loss = MeanMetric()
            self.test_mass_cont = MeanMetric()
            self.test_ns_longitude = MeanMetric()
            self.test_ns_latitude = MeanMetric()

            self.train_physics_tropical = MeanMetric()
            self.train_physics_midlat = MeanMetric()
            self.train_physics_polar = MeanMetric()

            self.val_physics_tropical = MeanMetric()
            self.val_physics_midlat = MeanMetric()
            self.val_physics_polar = MeanMetric()

        self.val_best = MinMetric()

        # Per-variable reconstruction metrics
        self.train_mse_w = MeanMetric()
        self.train_mse_u = MeanMetric()
        self.train_mse_z = MeanMetric()
        self.train_mse_v = MeanMetric()

        self.val_mse_w = MeanMetric()
        self.val_mse_u = MeanMetric()
        self.val_mse_z = MeanMetric()
        self.val_mse_v = MeanMetric()

    def _init_model(self) -> nn.Module:
        """Initialize the neural network model."""
        common_params = {
            "N_in_features": self.hparams.N_in_features,
            "N_out_features": self.hparams.N_out_features,
            "N_hidden_features": self.hparams.N_hidden_features,
            "N_hidden_layers": self.hparams.N_hidden_layers,
            "position_encoder_type": self.hparams.position_encoder_type,
            "mapping_dim": self.hparams.mapping_dim,
            "scale": self.hparams.scale,
            **self.hparams.model_params
        }

        model_map = {
            "FCN": FCN,
            "SirenNet": SirenNet,
            "GaussianNet": GaussianNet,
            "FinerNet": FinerNet,
            "RealWireNet": RealWireNet
        }

        if self.hparams.model_name not in model_map:
            raise ValueError(f"Unknown model name: {self.hparams.model_name}")

        model = model_map[self.hparams.model_name](**common_params)
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x.float())

    def generate_virtual_points(self, n: int) -> torch.Tensor:
        """Generate random virtual points using LHS sampling.

        Generates points in physical coordinate space, then normalizes them
        using the datamodule's coordinate normalization scheme.

        Args:
            n: Number of virtual points to generate

        Returns:
            Tensor of shape (n, 4) with coordinates normalized by datamodule
        """
        if self.datamodule is None:
            raise RuntimeError("DataModule not set. Cannot generate virtual points.")

        # LHS sample in unit hypercube [0, 1]^4
        samples = lhs(n=4, samples=n)  # (n, 4)

        # Map [0, 1] to physical coordinate ranges
        # Order: [time, pressure, lat, lon] (dataset order)
        coord_names = ['valid_time', 'pressure_level', 'latitude', 'longitude']
        physical_coords = torch.zeros_like(samples)

        for i, name in enumerate(coord_names):
            min_val, max_val = self.datamodule.coordinate_ranges[name]
            # Map [0, 1] → [min, max]
            physical_coords[:, i] = samples[:, i] * (max_val - min_val) + min_val

        # Convert to tensor
        coords_tensor = torch.from_numpy(physical_coords).float().to(self.device)

        # Normalize using datamodule's normalization scheme
        coords_normalized = self.datamodule.normalize_coords(coords_tensor)

        return coords_normalized

    def configure_optimizers(self):
        """Configure and return the optimizer and learning rate scheduler."""
        optimizer_map = {
            "SGD": optim.SGD,
            "Adam": optim.Adam,
            "AdamW": optim.AdamW
        }

        if not hasattr(self.hparams, 'optimizer_name') or self.hparams.optimizer_name not in optimizer_map:
            raise ValueError(
                f"optimizer_name must be one of {list(optimizer_map.keys())}, "
                f"got {getattr(self.hparams, 'optimizer_name', None)}"
            )

        optimizer_config = getattr(self.hparams, 'optimizer_config', {})

        optimizer = optimizer_map[self.hparams.optimizer_name](
            params=self.parameters(),
            **optimizer_config
        )

        if hasattr(self.hparams, 'scheduler_name') and self.hparams.scheduler_name is not None:
            scheduler_map = {
                "CosineAnnealingWarmRestarts": optim.lr_scheduler.CosineAnnealingWarmRestarts,
                "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
                "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau
            }

            if self.hparams.scheduler_name not in scheduler_map:
                raise ValueError(
                    f"scheduler_name must be one of {list(scheduler_map.keys())}, "
                    f"got {self.hparams.scheduler_name}"
                )

            scheduler_config = getattr(self.hparams, 'scheduler_config', {})
            scheduler = scheduler_map[self.hparams.scheduler_name](optimizer, **scheduler_config)

            if self.hparams.scheduler_name == "ReduceLROnPlateau":
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": scheduler_config.get("monitor", "val/loss"),
                        "interval": scheduler_config.get("interval", "epoch"),
                        "frequency": scheduler_config.get("frequency", 1)
                    }
                }

            return [optimizer], [scheduler]

        return optimizer