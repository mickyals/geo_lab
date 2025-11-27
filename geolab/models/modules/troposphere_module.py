from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
import torch.nn as nn
import torch.optim as optim
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from torchmetrics import MinMetric, MeanMetric
from geolab.models.components import troposphere_pde_residual
from geolab.models.model import FCN, SirenNet, GaussianNet, FinerNet, RealWireNet

class TroposhpereLightningModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        # basic network configs
        model_name: str,
        N_in_features: int,
        N_out_features: int,
        N_hidden_features: int,
        N_hidden_layers: int,
        # model specific params
        model_params: Dict[str, Any],
        # position encoder configs,
        position_encoder_type: str,
        mapping_dim: int,
        scale: int,
        # optimizer configs
        optimizer_name: str,
        optimizer_config,
        # scheduler configs
        scheduler_name: str,
        scheduler_config,
        # Train PINN
        train_pinn: bool,
        mass_balance: bool,
        physics_loss_weight: float=None,
        statistics: Dict[str, list] = None,
        pi_scale: bool = False,
    ) -> None:
        """Initialize a `ERA5LightningModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters()
        self.train_pinn = train_pinn
        self.mass_balance = mass_balance
        self.physics_loss_weight = physics_loss_weight

        self.statistics = statistics
        self.pi_scale = pi_scale

        self.model = self._init_model()

        # loss functions
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

            self.train_physics_tropical = MeanMetric()  # |lat| < 30
            self.train_physics_midlat = MeanMetric()    # 30 < |lat| < 60
            self.train_physics_polar = MeanMetric()     # |lat| > 60
            
            self.val_physics_tropical = MeanMetric()
            self.val_physics_midlat = MeanMetric()
            self.val_physics_polar = MeanMetric()


        self.val_best = MinMetric() # for best validation loss

        # metrics
        # Per-variable reconstruction metrics
        self.train_mse_t = MeanMetric()
        self.train_mse_w = MeanMetric() 
        self.train_mse_u = MeanMetric()
        self.train_mse_z = MeanMetric()
        self.train_mse_v = MeanMetric()

        self.val_mse_t = MeanMetric()
        self.val_mse_w = MeanMetric()
        self.val_mse_u = MeanMetric()
        self.val_mse_z = MeanMetric()
        self.val_mse_v = MeanMetric()


    def _init_model(self) -> nn.Module:

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
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.model(x.float())

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

        self.val_loss.reset()
        self.val_best.reset()

    def model_step(self, batch):
        """Perform a single model step on a batch of data."""
        coords = batch['coords']
        variables = batch['variables']
        classification = batch['classification']

        # === Build input tensor ===
        coord_list = [
            coords['longitude'],
            coords['latitude'],
            coords['pressure_level'],
            coords['time']
        ]

        inputs = torch.stack(coord_list, dim=1).float()

        targets = torch.stack(list(variables.values()), dim=1).float()

        # === Forward pass ===
        # For PINN, we need gradients even during validation/testing
        if self.train_pinn:
            # Enable gradients for physics computation
            inputs = inputs.detach().requires_grad_(True)
            preds = self.forward(inputs)
        else:
            preds = self.forward(inputs)

        # === Separate real and virtual points ===
        real_mask = classification.bool()

        # === Data loss (MSE for real samples) ===
        all_loss = self.criterion(preds, targets)
        data_loss = all_loss[real_mask].mean()

        # === Per-variable MSE ===
        per_var_losses = all_loss[real_mask].mean(dim=0)  # Shape: [5]
        mse_t, mse_w, mse_u, mse_z, mse_v = per_var_losses

        if self.train_pinn:
            variable_names = list(variables.keys())
            model_outputs_dict = {k: preds[:, i] for i, k in enumerate(variable_names)}

            # Compute physics residuals
            ns_longitude, ns_latitude, mass_cont = troposphere_pde_residual(
                inputs_tensor=inputs, outputs=model_outputs_dict, statistics=self.statistics, mass_balance=self.mass_balance
            )

            # === Regional physics residuals ===
            # Extract latitude for regional binning (column 1 is latitude)
            lat = inputs[:, 1]
            abs_lat = torch.abs(lat)

            # Regional masks - THESE MAY NEED TO BE NORMALISED
            tropical_mask = abs_lat < 30
            midlat_mask = (abs_lat >= 30) & (abs_lat < 60)
            polar_mask = abs_lat >= 60

            # Helper function to compute regional physics loss
            def compute_regional_physics(mask):
                if mask.any():
                    return (
                        ns_longitude[mask].pow(2).mean() + 
                        ns_latitude[mask].pow(2).mean() + 
                        mass_cont[mask].pow(2).mean()
                    )
                else:
                    # Return zero on same device if no samples in region
                    return torch.tensor(0.0, device=inputs.device, dtype=torch.float32)
            
            physics_tropical = compute_regional_physics(tropical_mask)
            physics_midlat = compute_regional_physics(midlat_mask)
            physics_polar = compute_regional_physics(polar_mask)

            # Physics loss from residuals (global)
            physics_loss = ns_longitude.pow(2).mean() + ns_latitude.pow(2).mean() + mass_cont.pow(2).mean()

            total_loss = ((1 - self.physics_loss_weight) * data_loss) + (self.physics_loss_weight * physics_loss)

            # Ensure loss is Float32 to match model parameters
            return (
                total_loss.float(), 
                data_loss.float(), 
                physics_loss.float(), 
                mass_cont.float(), 
                ns_longitude.float(), 
                ns_latitude.float(),
                mse_t.float(),
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
            mse_t.float(),
            mse_w.float(),
            mse_u.float(),
            mse_z.float(),
            mse_v.float()
            )




    def training_step(
        self, batch, batch_idx
    ):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        if self.train_pinn:
            (total_loss, data_loss, physics_loss, mass_cont, ns_longitude, ns_latitude,
            mse_t, mse_w, mse_u, mse_z, mse_v,
            physics_tropical, physics_midlat, physics_polar) = self.model_step(batch)

            # update and log metrics
            self.train_loss(total_loss)
            self.train_physics_loss(physics_loss)
            self.train_data_loss(data_loss)
            self.train_mass_cont(mass_cont)
            self.train_ns_longitude(ns_longitude)
            self.train_ns_latitude(ns_latitude)

            self.log("train_loss", self.train_loss, on_epoch=True, on_step=True)
            self.log("train_physics_loss", self.train_physics_loss, on_epoch=True, on_step=True)
            self.log("train_data_loss", self.train_data_loss, on_epoch=True, on_step=True)
            self.log("train_mass_cont", self.train_mass_cont, on_epoch=True, on_step=True)
            self.log("train_ns_longitude", self.train_ns_longitude, on_epoch=True, on_step=True)
            self.log("train_ns_latitude", self.train_ns_latitude, on_epoch=True, on_step=True)

            # Per-variable MSE
            self.train_mse_t(mse_t)
            self.train_mse_w(mse_w)
            self.train_mse_u(mse_u)
            self.train_mse_z(mse_z)
            self.train_mse_v(mse_v)

            self.log("train_mse_t", self.train_mse_t, on_epoch=True, on_step=False)
            self.log("train_mse_w", self.train_mse_w, on_epoch=True, on_step=False)
            self.log("train_mse_u", self.train_mse_u, on_epoch=True, on_step=False)
            self.log("train_mse_z", self.train_mse_z, on_epoch=True, on_step=False)
            self.log("train_mse_v", self.train_mse_v, on_epoch=True, on_step=False)

            # Regional physics
            self.train_physics_tropical(physics_tropical)
            self.train_physics_midlat(physics_midlat)
            self.train_physics_polar(physics_polar)

            self.log("train_physics_tropical", self.train_physics_tropical, on_epoch=True, on_step=False)
            self.log("train_physics_midlat", self.train_physics_midlat, on_epoch=True, on_step=False)
            self.log("train_physics_polar", self.train_physics_polar, on_epoch=True, on_step=False)

            return total_loss

        else:
            data_loss, mse_t, mse_w, mse_u, mse_z, mse_v = self.model_step(batch)

            # update and log metrics
            self.train_loss(data_loss)
            self.log("train_loss", self.train_loss, on_epoch=True, on_step=True)

            # Per-variable MSE
            self.train_mse_t(mse_t)
            self.train_mse_w(mse_w)
            self.train_mse_u(mse_u)
            self.train_mse_z(mse_z)
            self.train_mse_v(mse_v)

            self.log("train_mse_t", self.train_mse_t, on_epoch=True, on_step=False)
            self.log("train_mse_w", self.train_mse_w, on_epoch=True, on_step=False)
            self.log("train_mse_u", self.train_mse_u, on_epoch=True, on_step=False)
            self.log("train_mse_z", self.train_mse_z, on_epoch=True, on_step=False)
            self.log("train_mse_v", self.train_mse_v, on_epoch=True, on_step=False)

            return data_loss


    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        # Log loss component ratio (diagnostic for PINN balance)
        if self.train_pinn and self.train_data_loss.compute() > 0:
            ratio = self.train_physics_loss.compute() / self.train_data_loss.compute()
            self.log("train_physics_data_ratio", ratio, on_epoch=True)
        
        # Gradient norm monitoring (helps diagnose training instability)
        if hasattr(self, '_last_grad_norm'):
            self.log("train_grad_norm", self._last_grad_norm, on_epoch=True)



    def on_before_optimizer_step(self, optimizer):
        """Called before optimizer.step(), gradients are available here."""
        # Compute gradient norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
    
        # Log immediately
        self.log("train/grad_norm", total_norm, on_step=True, on_epoch=False)        


    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""

        if self.train_pinn:
            # Enable gradients for physics loss computation during validation
            with torch.enable_grad():
                (total_loss, data_loss, physics_loss, mass_cont, ns_longitude, ns_latitude,
                mse_t, mse_w, mse_u, mse_z, mse_v,
                physics_tropical, physics_midlat, physics_polar) = self.model_step(batch)

            # update and log metrics
            self.val_loss(total_loss)
            self.val_physics_loss(physics_loss)
            self.val_data_loss(data_loss)
            self.val_mass_cont(mass_cont)
            self.val_ns_longitude(ns_longitude)
            self.val_ns_latitude(ns_latitude)

            self.log("val_loss", self.val_loss, on_epoch=True, on_step=True)
            self.log("val_physics_loss", self.val_physics_loss, on_epoch=True, on_step=True)
            self.log("val_data_loss", self.val_data_loss, on_epoch=True, on_step=True)
            self.log("val_mass_cont", self.val_mass_cont, on_epoch=True, on_step=True)
            self.log("val_ns_longitude", self.val_ns_longitude, on_epoch=True, on_step=True)
            self.log("val_ns_latitude", self.val_ns_latitude, on_epoch=True, on_step=True)

            # Per-variable MSE
            self.val_mse_t(mse_t)
            self.val_mse_w(mse_w)
            self.val_mse_u(mse_u)
            self.val_mse_z(mse_z)
            self.val_mse_v(mse_v)

            self.log("val_mse_t", self.val_mse_t, on_epoch=True, on_step=False)
            self.log("val_mse_w", self.val_mse_w, on_epoch=True, on_step=False)
            self.log("val_mse_u", self.val_mse_u, on_epoch=True, on_step=False)
            self.log("val_mse_z", self.val_mse_z, on_epoch=True, on_step=False)
            self.log("val_mse_v", self.val_mse_v, on_epoch=True, on_step=False)

            # Regional physics
            self.val_physics_tropical(physics_tropical)
            self.val_physics_midlat(physics_midlat)
            self.val_physics_polar(physics_polar)

            self.log("val_physics_tropical", self.val_physics_tropical, on_epoch=True, on_step=False)
            self.log("val_physics_midlat", self.val_physics_midlat, on_epoch=True, on_step=False)
            self.log("val_physics_polar", self.val_physics_polar, on_epoch=True, on_step=False)


            return total_loss

        else:
            data_loss, mse_t, mse_w, mse_u, mse_z, mse_v = self.model_step(batch)

            # update and log metrics
            self.val_loss(data_loss)
            self.log("val_loss", self.val_loss, on_epoch=True, on_step=True)

            # Per-variable MSE
            self.val_mse_t(mse_t)
            self.val_mse_w(mse_w)
            self.val_mse_u(mse_u)
            self.val_mse_z(mse_z)
            self.val_mse_v(mse_v)

            self.log("val_mse_t", self.val_mse_t, on_epoch=True, on_step=False)
            self.log("val_mse_w", self.val_mse_w, on_epoch=True, on_step=False)
            self.log("val_mse_u", self.val_mse_u, on_epoch=True, on_step=False)
            self.log("val_mse_z", self.val_mse_z, on_epoch=True, on_step=False)
            self.log("val_mse_v", self.val_mse_v, on_epoch=True, on_step=False)
            
            return data_loss

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."

        val_loss = self.val_loss.compute()
        self.val_best.update(val_loss)

        # Log both current and best validation losses
        self.log("val_loss_epoch", val_loss, prog_bar=True)
        self.log("val_loss_best", self.val_best.compute(), prog_bar=True)



    

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        # if self.train_pinn:
        #     with torch.enable_grad():
        #         print('here at testing')
        #         total_loss, data_loss, physics_loss, mass_cont, ns_longitude, ns_latitude = self.model_step(batch)
        #
        #     # update and log metrics
        #     self.test_loss(total_loss)
        #     self.test_physics_loss(physics_loss)
        #     self.test_data_loss(data_loss)
        #     self.test_mass_cont(mass_cont)
        #     self.test_ns_longitude(ns_longitude)
        #     self.test_ns_latitude(ns_latitude)
        #
        #     self.log("test_loss", self.test_loss, on_epoch=True, on_step=True)
        #     self.log("test_physics_loss", self.test_physics_loss, on_epoch=True, on_step=True)
        #     self.log("test_data_loss", self.test_data_loss, on_epoch=True, on_step=True)
        #     self.log("test_mass_cont", self.test_mass_cont, on_epoch=True, on_step=True)
        #     self.log("test_ns_longitude", self.test_ns_longitude, on_epoch=True, on_step=True)
        #     self.log("test_ns_latitude", self.test_ns_latitude, on_epoch=True, on_step=True)
        #
        #     return total_loss
        #
        # else:
        #     data_loss = self.model_step(batch)
        #
        #     # update and log metrics
        #     self.test_loss(data_loss)
        #     self.log("test_loss", self.test_loss, on_epoch=True, on_step=True)
        #
        #     return data_loss

        pass
        
    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""

        pass

    def on_train_end(self):
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        pass

    def configure_optimizers(self):
        """Configure and return the optimizer and learning rate scheduler.

        Expected hyperparameters in self.hparams:
            - optimizer_name: str, one of ['SGD', 'Adam', 'AdamW']
            - optimizer_config: dict, parameters for the optimizer
            - scheduler_name: Optional[str], one of ['CosineAnnealingWarmRestarts',
                          'CosineAnnealingLR', 'ReduceLROnPlateau']
            - scheduler_config: Optional[dict], parameters for the scheduler

        Returns:
            Union[Optimizer, Tuple[List[Optimizer], List[LRScheduler]]]:
                Single optimizer or a tuple of optimizers and schedulers
        """
        # Validate optimizer configuration
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

        # Initialize optimizer
        optimizer = optimizer_map[self.hparams.optimizer_name](
            params=self.parameters(),
            **optimizer_config
        )

        # Configure scheduler if specified
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

            # Special case for ReduceLROnPlateau
            if self.hparams.scheduler_name == "ReduceLROnPlateau":
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": scheduler_config.get("monitor", "val_loss"),
                        "interval": scheduler_config.get("interval", "epoch"),
                        "frequency": scheduler_config.get("frequency", 1)
                    }
                }

            return [optimizer], [scheduler]

        return optimizer