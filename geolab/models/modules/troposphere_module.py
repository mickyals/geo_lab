from typing import Any, Dict, Tuple, Optional, List
import numpy as np
import torch
from lightning import LightningModule
import torch.nn as nn
from pyDOE3 import lhs
import torch.optim as optim
from torchmetrics import MinMetric, MeanMetric

from geolab.models.components import troposphere_pde_residual
from geolab.models.model import FCN, SirenNet, GaussianNet, FinerNet, RealWireNet


class TroposphereLightningModule(LightningModule):
    """Lightning Module for troposphere reconstruction using implicit neural fields."""

    def __init__(
            self,
            # Basic network configs (dimensions come from datamodule)
            model_name: str,
            N_hidden_features: int,
            N_hidden_layers: int,
            # Model specific params
            model_params: Dict[str, Any],
            # Position encoder configs
            position_encoder_type: str,
            mapping_dim: int,
            scale: int,
            encode_coords: Optional[List[str]] = None,
            # Optimizer configs
            optimizer_name: str = 'Adam',
            optimizer_config: Dict = None,
            # Scheduler configs
            scheduler_name: str = None,
            scheduler_config: Dict = None,
            # PINN configs
            train_pinn: bool = False,
            mass_balance: bool = True,
            physics_loss_weight: float = 0.5,
            # Virtual sampling configs
            include_virtual: bool = False,
            num_virtual_per_batch: int = 1000,
            # DataModule reference (REQUIRED - dimensions extracted from here)
            datamodule=None,
            ) -> None:
        super().__init__()


        # Validate datamodule is provided
        if datamodule is None:
            raise ValueError(
                "datamodule is required. The Lightning module extracts input/output "
                "dimensions, coordinate labels, and normalization info from it."
            )

        self.datamodule = datamodule

        # Extract dimensions from datamodule
        self.N_in_features = datamodule.input_dim
        self.N_out_features = datamodule.output_dim


        # Store hyperparameters (exclude datamodule, add extracted dims)
        self.save_hyperparameters(
            ignore=['datamodule'],
            logger=True
        )

        # PINN configs
        self.train_pinn = train_pinn
        self.mass_balance = mass_balance
        self.physics_loss_weight = physics_loss_weight
        self.include_virtual = include_virtual
        self.num_virtual_per_batch = num_virtual_per_batch

        # Initialize model (uses self.N_in_features and self.N_out_features)
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

        # Convert coordinate names to indices
        encode_dims = None
        if self.hparams.position_encoder_type is not None:
            # Get coordinate names to encode
            if self.hparams.encode_coords is not None:
                # User specified which coords to encode
                encode_coord_names = self.hparams.encode_coords
            else:
                # Default: encode spatial coordinates
                encode_coord_names = self.datamodule.spatial_coords

            # Convert names to indices using coord_labels
            encode_dims = [
                self.datamodule.data.coord_labels[coord_name]
                for coord_name in encode_coord_names
            ]

        common_params = {
            "N_in_features": self.N_in_features,
            "N_out_features": self.N_out_features,
            "N_hidden_features": self.hparams.N_hidden_features,
            "N_hidden_layers": self.hparams.N_hidden_layers,
            "position_encoder_type": self.hparams.position_encoder_type,
            "mapping_dim": self.hparams.mapping_dim,
            "scale": self.hparams.scale,
            "encode_dims": encode_dims,  # List of INTEGER indices or None
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

        print(f"Initialized {self.hparams.model_name} with:")
        print(f"  Input dim: {self.N_in_features}")
        print(f"  Output dim: {self.N_out_features}")
        print(f"  Hidden features: {self.hparams.N_hidden_features}")
        print(f"  Hidden layers: {self.hparams.N_hidden_layers}")
        if encode_dims is not None:
            coord_names = [self.datamodule.data.coord_order[i] for i in encode_dims]
            print(f"  Fourier encoding: {coord_names} (dims {encode_dims})")

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
            Tensor of shape (n, input_dim) with coordinates normalized by datamodule
        """
        # LHS sample in unit hypercube [0, 1]^input_dim
        samples = lhs(n=self.N_in_features, samples=n)  # (n, input_dim)

        # Map [0, 1] to physical coordinate ranges
        # Order matches coord_order from datamodule
        coord_order = self.datamodule.data.coord_order
        physical_coords = np.zeros_like(samples)

        for i, name in enumerate(coord_order):
            min_val, max_val = self.datamodule.coordinate_ranges[name]
            # Map [0, 1] → [min, max]
            physical_coords[:, i] = samples[:, i] * (max_val - min_val) + min_val

        # Convert to tensor
        coords_tensor = torch.from_numpy(physical_coords).float().to(self.device)

        # Normalize using datamodule's normalization scheme
        coords_normalized = self.datamodule.normalize_coords(coords_tensor)

        return coords_normalized

    def augment_batch_with_virtual(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add virtual points to batch for PINN training.

        Args:
            batch: Dict with 'coords' (B, input_dim) and 'values' (B, output_dim)

        Returns:
            Augmented batch with virtual points and classification mask
        """
        B = batch['coords'].shape[0]

        # Generate virtual coordinates
        virtual_coords = self.generate_virtual_points(self.num_virtual_per_batch)

        # Virtual targets are zeros (not used in data loss anyway)
        virtual_values = torch.zeros(
            self.num_virtual_per_batch,
            self.N_out_features,
            device=self.device,
            dtype=batch['values'].dtype
        )

        # Concatenate real and virtual
        augmented_coords = torch.cat([batch['coords'], virtual_coords], dim=0)
        augmented_values = torch.cat([batch['values'], virtual_values], dim=0)

        # Create classification mask (True = real, False = virtual)
        classification = torch.cat([
            torch.ones(B, dtype=torch.bool, device=self.device),
            torch.zeros(self.num_virtual_per_batch, dtype=torch.bool, device=self.device)
        ])

        return {
            'coords': augmented_coords,
            'values': augmented_values,
            'classification': classification
        }

    def _convert_statistics_format(self) -> Dict:
        """Convert datamodule statistics format to physics code format.

        DataModule format: stats[var] = {'min': tensor, 'max': tensor, ...}
        Physics format: stats[var] = [min, max, mean, std]

        Returns:
            Statistics dict in list format
        """
        stats_list = {}

        # Convert variable statistics
        for var, stats in self.datamodule.statistics.items():
            stats_list[var] = [
                stats['min'].item(),
                stats['max'].item(),
                stats['mean'].item(),
                stats['std'].item()
            ]

        # Convert coordinate ranges
        for coord, (min_val, max_val) in self.datamodule.coordinate_ranges.items():
            # Compute mean and std for completeness (physics code may not use these)
            mean_val = (min_val + max_val) / 2
            std_val = (max_val - min_val) / 2
            stats_list[coord] = [min_val, max_val, mean_val, std_val]

        return stats_list

    def model_step(self, batch: Dict[str, torch.Tensor]):
        """Perform a single model step on a batch of data.

        Args:
            batch: Dict with 'coords' (B, input_dim) and 'values' (B, output_dim), both NORMALIZED

        Returns:
            Tuple of losses and metrics
        """
        # Extract data
        coords = batch['coords']  # (B, input_dim) in dataset order, NORMALIZED
        values = batch['values']  # (B, output_dim) NORMALIZED
        classification = batch.get('classification', None)

        # If no classification, all are real points
        if classification is None:
            classification = torch.ones(coords.shape[0], dtype=torch.bool, device=coords.device)

        # Forward pass
        if self.train_pinn:
            coords = coords.detach().requires_grad_(True)

        preds = self.forward(coords)  # (B, output_dim)

        # Separate real and virtual
        real_mask = classification.bool()

        # Data loss (only on real points)
        mse_per_sample = (preds - values).pow(2)  # (B, output_dim)
        data_loss = mse_per_sample[real_mask].mean()

        # Per-variable MSE
        per_var_mse = mse_per_sample[real_mask].mean(dim=0)  # (output_dim,)
        mse_w, mse_u, mse_z, mse_v = per_var_mse

        if self.train_pinn:

            # Build outputs dict using var_order
            outputs_dict = {var: preds[:, i]
                            for i, var in enumerate(self.datamodule.data.var_order)}

            # Convert statistics format
            stats_physics = self._convert_statistics_format()

            # Compute physics residuals
            ns_longitude, ns_latitude, mass_cont = troposphere_pde_residual(
                inputs_tensor=coords,
                outputs=outputs_dict,
                statistics=stats_physics,
                coord_labels=self.datamodule.data.coord_labels,
                var_labels=self.datamodule.data.variable_labels,
                mass_balance=self.mass_balance
            )

            # Physics loss (computed on ALL points including virtual)
            physics_loss = (
                    ns_longitude.pow(2).mean() +
                    ns_latitude.pow(2).mean() +
                    mass_cont.pow(2).mean()
            )

            # Regional physics using label-based indexing
            lat_idx = self.datamodule.data.coord_labels['latitude']
            lat_deg = self.datamodule.denormalize_coords(coords)[:, lat_idx]
            abs_lat_deg = torch.abs(lat_deg)

            # Regional masks
            tropical_mask = abs_lat_deg < 30.0
            midlat_mask = (abs_lat_deg >= 30.0) & (abs_lat_deg < 60.0)
            polar_mask = abs_lat_deg >= 60.0

            def compute_regional_physics(mask):
                if mask.any():
                    return (
                            ns_longitude[mask].pow(2).mean() +
                            ns_latitude[mask].pow(2).mean() +
                            mass_cont[mask].pow(2).mean()
                    )
                return torch.tensor(0.0, device=coords.device, dtype=torch.float32)

            physics_tropical = compute_regional_physics(tropical_mask)
            physics_midlat = compute_regional_physics(midlat_mask)
            physics_polar = compute_regional_physics(polar_mask)

            # Total loss
            total_loss = (
                    (1 - self.physics_loss_weight) * data_loss +
                    self.physics_loss_weight * physics_loss
            )

            return (
                total_loss.float(),
                data_loss.float(),
                physics_loss.float(),
                mass_cont.mean().float(),
                ns_longitude.mean().float(),
                ns_latitude.mean().float(),
                mse_w.float(),
                mse_u.float(),
                mse_z.float(),
                mse_v.float(),
                physics_tropical.float(),
                physics_midlat.float(),
                physics_polar.float()
            )

        return (
            data_loss.float(),
            mse_w.float(),
            mse_u.float(),
            mse_z.float(),
            mse_v.float()
        )

    def on_train_start(self) -> None:
        """Lightning hook called when training begins."""
        self.val_loss.reset()
        self.val_best.reset()

    def training_step(self, batch, batch_idx):
        """Perform a single training step."""

        # Normalize batch (coords and values)
        batch = self.datamodule.normalize_batch(batch)

        # Add virtual points if PINN training
        if self.train_pinn and self.include_virtual:
            batch = self.augment_batch_with_virtual(batch)

        if self.train_pinn:
            (total_loss, data_loss, physics_loss, mass_cont, ns_longitude, ns_latitude,
             mse_w, mse_u, mse_z, mse_v,
             physics_tropical, physics_midlat, physics_polar) = self.model_step(batch)

            # Update and log metrics
            self.train_loss(total_loss)
            self.train_physics_loss(physics_loss)
            self.train_data_loss(data_loss)
            self.train_mass_cont(mass_cont)
            self.train_ns_longitude(ns_longitude)
            self.train_ns_latitude(ns_latitude)

            self.log("train/loss", self.train_loss, on_epoch=True, on_step=True, prog_bar=True)
            self.log("train/physics_loss", self.train_physics_loss, on_epoch=True, on_step=True)
            self.log("train/data_loss", self.train_data_loss, on_epoch=True, on_step=True)
            self.log("train/mass_cont", self.train_mass_cont, on_epoch=True, on_step=False)
            self.log("train/ns_longitude", self.train_ns_longitude, on_epoch=True, on_step=False)
            self.log("train/ns_latitude", self.train_ns_latitude, on_epoch=True, on_step=False)

            # Per-variable MSE
            self.train_mse_w(mse_w)
            self.train_mse_u(mse_u)
            self.train_mse_z(mse_z)
            self.train_mse_v(mse_v)

            self.log("train/mse_w", self.train_mse_w, on_epoch=True, on_step=False)
            self.log("train/mse_u", self.train_mse_u, on_epoch=True, on_step=False)
            self.log("train/mse_z", self.train_mse_z, on_epoch=True, on_step=False)
            self.log("train/mse_v", self.train_mse_v, on_epoch=True, on_step=False)

            # Regional physics
            self.train_physics_tropical(physics_tropical)
            self.train_physics_midlat(physics_midlat)
            self.train_physics_polar(physics_polar)

            self.log("train/physics_tropical", self.train_physics_tropical, on_epoch=True, on_step=False)
            self.log("train/physics_midlat", self.train_physics_midlat, on_epoch=True, on_step=False)
            self.log("train/physics_polar", self.train_physics_polar, on_epoch=True, on_step=False)

            return total_loss

        else:
            data_loss, mse_w, mse_u, mse_z, mse_v = self.model_step(batch)

            # Update and log metrics
            self.train_loss(data_loss)
            self.log("train/loss", self.train_loss, on_epoch=True, on_step=True, prog_bar=True)

            # Per-variable MSE
            self.train_mse_w(mse_w)
            self.train_mse_u(mse_u)
            self.train_mse_z(mse_z)
            self.train_mse_v(mse_v)

            self.log("train/mse_w", self.train_mse_w, on_epoch=True, on_step=False)
            self.log("train/mse_u", self.train_mse_u, on_epoch=True, on_step=False)
            self.log("train/mse_z", self.train_mse_z, on_epoch=True, on_step=False)
            self.log("train/mse_v", self.train_mse_v, on_epoch=True, on_step=False)

            return data_loss

    def on_train_epoch_end(self) -> None:
        """Lightning hook called when a training epoch ends."""
        if self.train_pinn and self.train_data_loss.compute() > 0:
            ratio = self.train_physics_loss.compute() / (self.train_data_loss.compute() + 1e-8)
            self.log("train/physics_data_ratio", ratio, on_epoch=True)

    def on_before_optimizer_step(self, optimizer):
        """Called before optimizer.step(), gradients are available here."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self.log("train/grad_norm", total_norm, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step."""

        # Normalize batch (no virtual points in validation)
        batch = self.datamodule.normalize_batch(batch)

        if self.train_pinn:
            with torch.enable_grad():
                (total_loss, data_loss, physics_loss, mass_cont, ns_longitude, ns_latitude,
                 mse_w, mse_u, mse_z, mse_v,
                 physics_tropical, physics_midlat, physics_polar) = self.model_step(batch)

            self.val_loss(total_loss)
            self.val_physics_loss(physics_loss)
            self.val_data_loss(data_loss)
            self.val_mass_cont(mass_cont)
            self.val_ns_longitude(ns_longitude)
            self.val_ns_latitude(ns_latitude)

            self.log("val/loss", self.val_loss, on_epoch=True, on_step=False, prog_bar=True)
            self.log("val/physics_loss", self.val_physics_loss, on_epoch=True, on_step=False)
            self.log("val/data_loss", self.val_data_loss, on_epoch=True, on_step=False)
            self.log("val/mass_cont", self.val_mass_cont, on_epoch=True, on_step=False)
            self.log("val/ns_longitude", self.val_ns_longitude, on_epoch=True, on_step=False)
            self.log("val/ns_latitude", self.val_ns_latitude, on_epoch=True, on_step=False)

            self.val_mse_w(mse_w)
            self.val_mse_u(mse_u)
            self.val_mse_z(mse_z)
            self.val_mse_v(mse_v)

            self.log("val/mse_w", self.val_mse_w, on_epoch=True, on_step=False)
            self.log("val/mse_u", self.val_mse_u, on_epoch=True, on_step=False)
            self.log("val/mse_z", self.val_mse_z, on_epoch=True, on_step=False)
            self.log("val/mse_v", self.val_mse_v, on_epoch=True, on_step=False)

            self.val_physics_tropical(physics_tropical)
            self.val_physics_midlat(physics_midlat)
            self.val_physics_polar(physics_polar)

            self.log("val/physics_tropical", self.val_physics_tropical, on_epoch=True, on_step=False)
            self.log("val/physics_midlat", self.val_physics_midlat, on_epoch=True, on_step=False)
            self.log("val/physics_polar", self.val_physics_polar, on_epoch=True, on_step=False)

            return total_loss

        else:
            data_loss, mse_w, mse_u, mse_z, mse_v = self.model_step(batch)

            self.val_loss(data_loss)
            self.log("val/loss", self.val_loss, on_epoch=True, on_step=False, prog_bar=True)

            self.val_mse_w(mse_w)
            self.val_mse_u(mse_u)
            self.val_mse_z(mse_z)
            self.val_mse_v(mse_v)

            self.log("val/mse_w", self.val_mse_w, on_epoch=True, on_step=False)
            self.log("val/mse_u", self.val_mse_u, on_epoch=True, on_step=False)
            self.log("val/mse_z", self.val_mse_z, on_epoch=True, on_step=False)
            self.log("val/mse_v", self.val_mse_v, on_epoch=True, on_step=False)

            return data_loss

    def on_validation_epoch_end(self) -> None:
        """Lightning hook called when a validation epoch ends."""
        val_loss = self.val_loss.compute()
        self.val_best.update(val_loss)

        self.log("val/best_loss", self.val_best.compute(), prog_bar=True)

    def test_step(self, batch, batch_idx):
        """Perform a single test step."""
        pass

    def on_test_epoch_end(self) -> None:
        """Lightning hook called when a test epoch ends."""
        pass

    # ========================================================================
    # EVALUATION METHODS
    # ========================================================================

    def evaluate_on_grid(self, coords: torch.Tensor,
                         denormalize: bool = True) -> torch.Tensor:
        """Standalone evaluation method for arbitrary coordinates.

        Args:
            coords: (N, input_dim) tensor in physical units matching coord_order
            denormalize: Whether to denormalize outputs

        Returns:
            preds: (N, output_dim) predictions
        """
        self.eval()

        # Normalize coords
        coords_norm = self.datamodule.normalize_coords(coords)

        with torch.no_grad():
            preds = self.forward(coords_norm)

        if denormalize:
            # Denormalize each variable
            denorm_preds = []
            for i, var in enumerate(self.datamodule.data.var_order):
                denorm_preds.append(
                    self.datamodule.denormalize(preds[:, i], var)
                )
            preds = torch.stack(denorm_preds, dim=1)

        return preds

    def evaluate_with_physics(self, coords: torch.Tensor) -> Dict:
        """Evaluate predictions + compute physics residuals.

        Args:
            coords: (N, input_dim) tensor in physical units

        Returns:
            Dict with predictions and residuals
        """
        self.eval()

        coords_norm = self.datamodule.normalize_coords(coords)
        coords_norm = coords_norm.requires_grad_(True)

        with torch.enable_grad():  # Need grads for physics
            preds = self.forward(coords_norm)

            # Build outputs dict
            outputs = {var: preds[:, i]
                       for i, var in enumerate(self.datamodule.data.var_order)}

            # Compute residuals
            ns_lon, ns_lat, mass = troposphere_pde_residual(
                coords,  # Physical coords
                outputs,
                statistics=self._convert_statistics_format(),
                coord_labels=self.datamodule.data.coord_labels,
                var_labels=self.datamodule.data.variable_labels,
                mass_balance=self.mass_balance
            )

        # Denormalize predictions
        denorm_preds = []
        for i, var in enumerate(self.datamodule.data.var_order):
            denorm_preds.append(
                self.datamodule.denormalize(preds[:, i].detach(), var)
            )
        denorm_preds = torch.stack(denorm_preds, dim=1)

        return {
            'predictions': denorm_preds,
            'ns_longitude': ns_lon.detach(),
            'ns_latitude': ns_lat.detach(),
            'mass_continuity': mass.detach()
        }

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