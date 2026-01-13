"""Model inference for visualization."""
import torch
from typing import Dict, Optional, Tuple
import numpy as np


class ModelInference:
    """Handle model predictions with batching and normalization.

    All methods accept coordinates in PHYSICAL units and return predictions
    in PHYSICAL units (unless denormalize=False is specified).
    """

    def __init__(self, model, datamodule):
        """
        Args:
            model: TroposphereLightningModule instance
            datamodule: TroposphereDataModule instance
        """
        self.model = model
        self.datamodule = datamodule

        # Put model in eval mode
        self.model.eval()

    def predict(self,
                coords: torch.Tensor,
                denormalize_output: bool = True,
                batch_size: int = 50000) -> torch.Tensor:
        """Get model predictions with batching.

        Args:
            coords: (N, 4) tensor in PHYSICAL coordinates
            denormalize_output: Whether to denormalize predictions to physical units
            batch_size: Batch size for inference (adjust based on GPU memory)

        Returns:
            (N, num_vars) predictions in physical units (if denormalize=True)

        Example:
            >>> coords = geometry.plane(axes=['longitude', 'latitude'],
            ...                        pressure_level=500, valid_time=0.5)
            >>> preds = inference.predict(coords)
            >>> # preds is now (N, 4) with denormalized values
        """
        # Normalize coordinates
        coords_norm = self.datamodule.normalize_coords(coords)

        # Batched forward pass
        preds = self._batched_forward(coords_norm, batch_size)

        # Denormalize if requested
        if denormalize_output:
            preds = self._denormalize_predictions(preds)

        return preds

    def predict_with_physics(self,
                             coords: torch.Tensor,
                             batch_size: int = 20000) -> Dict[str, torch.Tensor]:
        """Get predictions AND physics residuals."""
        if not self.model.train_pinn:
            raise RuntimeError(
                "Model was not trained with PINN. Cannot compute physics residuals."
            )

        # Process in batches
        all_preds = []
        all_ns_lon = []
        all_ns_lat = []
        all_mass = []

        n = coords.shape[0]
        for i in range(0, n, batch_size):
            batch_coords = coords[i:i + batch_size]

            # Normalize and enable gradients
            batch_coords_norm = self.datamodule.normalize_coords(batch_coords)
            batch_coords_norm = batch_coords_norm.requires_grad_(True)

            with torch.enable_grad():
                # Forward pass
                preds_batch = self.model(batch_coords_norm)

                # Build outputs dict
                outputs = {
                    var: preds_batch[:, j]
                    for j, var in enumerate(self.datamodule.data.var_order)
                }

                # Compute residuals - PASS NORMALIZED COORDS!
                from geolab.models.components import troposphere_pde_residual

                ns_lon, ns_lat, mass = troposphere_pde_residual(
                    inputs_tensor=batch_coords_norm,  # CHANGED: pass normalized coords with gradients
                    outputs=outputs,
                    statistics=self.model._convert_statistics_format(),
                    coord_labels=self.datamodule.data.coord_labels,
                    var_labels=self.datamodule.data.variable_labels,
                    mass_balance=self.model.mass_balance
                )

            # Store results (detach gradients)
            all_preds.append(preds_batch.detach())
            all_ns_lon.append(ns_lon.detach())
            all_ns_lat.append(ns_lat.detach())
            all_mass.append(mass.detach())

        # Concatenate batches
        preds = torch.cat(all_preds, dim=0)
        ns_longitude = torch.cat(all_ns_lon, dim=0)
        ns_latitude = torch.cat(all_ns_lat, dim=0)
        mass_continuity = torch.cat(all_mass, dim=0)

        # Denormalize predictions
        preds_denorm = self._denormalize_predictions(preds)

        return {
            'predictions': preds_denorm,
            'ns_longitude': ns_longitude,
            'ns_latitude': ns_latitude,
            'mass_continuity': mass_continuity
        }

    def compute_errors(self,
                       coords: torch.Tensor,
                       targets: torch.Tensor,
                       batch_size: int = 50000) -> torch.Tensor:
        """Compute squared errors between predictions and targets.

        Args:
            coords: (N, 4) tensor in PHYSICAL coordinates
            targets: (N, num_vars) ground truth in PHYSICAL units
            batch_size: Batch size for inference

        Returns:
            (N, num_vars) squared errors in physical units

        Example:
            >>> errors = inference.compute_errors(coords, targets)
            >>> mse_per_var = errors.mean(dim=0)  # (num_vars,)
        """
        # Get predictions (denormalized)
        preds = self.predict(coords, denormalize_output=True, batch_size=batch_size)

        # Compute squared errors
        errors = (preds - targets).pow(2)

        return errors

    def _batched_forward(self,
                         coords_normalized: torch.Tensor,
                         batch_size: int) -> torch.Tensor:
        """Run batched forward pass (internal helper).

        Args:
            coords_normalized: (N, 4) tensor with NORMALIZED coordinates
            batch_size: Batch size

        Returns:
            (N, num_vars) predictions (still normalized)
        """
        preds_list = []
        n = coords_normalized.shape[0]

        for i in range(0, n, batch_size):
            batch = coords_normalized[i:i + batch_size].to(self.model.device)

            with torch.no_grad():
                out = self.model(batch)

            preds_list.append(out.cpu())

        return torch.cat(preds_list, dim=0)

    def _denormalize_predictions(self, preds: torch.Tensor) -> torch.Tensor:
        """Denormalize predictions using datamodule statistics.

        Args:
            preds: (N, num_vars) normalized predictions

        Returns:
            (N, num_vars) denormalized predictions
        """
        denorm_preds = torch.zeros_like(preds)

        for i, var in enumerate(self.datamodule.data.var_order):
            denorm_preds[:, i] = self.datamodule.denormalize(preds[:, i], var)

        return denorm_preds

    @property
    def var_order(self):
        """Get variable order from datamodule."""
        return self.datamodule.data.var_order