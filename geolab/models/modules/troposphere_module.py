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
        physics_loss_weight: float=None,
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
        self.physics_loss_weight = physics_loss_weight

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


        self.val_best = MinMetric() # for best validation loss

        # metrics
        # TO DO


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

        if self.train_pinn:
            variable_names = list(variables.keys())
            model_outputs_dict = {k: preds[:, i] for i, k in enumerate(variable_names)}

            # Compute physics residuals
            ns_longitude, ns_latitude, mass_cont = troposphere_pde_residual(
                inputs, model_outputs_dict
            )

            # Physics loss from residuals
            physics_loss = ns_longitude.pow(2).mean() + ns_latitude.pow(2).mean() + mass_cont.pow(2).mean()

            total_loss = ((1 - self.physics_loss_weight) * data_loss) + (self.physics_loss_weight * physics_loss)

            # Ensure loss is Float32 to match model parameters
            return total_loss.float(), data_loss.float(), physics_loss.float(), mass_cont.float(), ns_longitude.float(), ns_latitude.float()

        return data_loss.float()




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
            total_loss, data_loss, physics_loss, mass_cont, ns_longitude, ns_latitude = self.model_step(batch)

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

            return total_loss

        else:
            data_loss = self.model_step(batch)

            # update and log metrics
            self.train_loss(data_loss)
            self.log("train_loss", self.train_loss, on_epoch=True, on_step=True)

            return data_loss


    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set."""

        if self.train_pinn:
            # Enable gradients for physics loss computation during validation
            with torch.set_grad_enabled(True):
                total_loss, data_loss, physics_loss, mass_cont, ns_longitude, ns_latitude = self.model_step(batch)

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

            return total_loss

        else:
            data_loss = self.model_step(batch)

            # update and log metrics
            self.val_loss(data_loss)
            self.log("val_loss", self.val_loss, on_epoch=True, on_step=True)

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
        if self.train_pinn:
            total_loss, data_loss, physics_loss, mass_cont, ns_longitude, ns_latitude = self.model_step(batch)
            
            # update and log metrics
            self.test_loss(total_loss)
            self.test_physics_loss(physics_loss)
            self.test_data_loss(data_loss)
            self.test_mass_cont(mass_cont)
            self.test_ns_longitude(ns_longitude)
            self.test_ns_latitude(ns_latitude)

            self.log("test_loss", self.test_loss, on_epoch=True, on_step=True)
            self.log("test_physics_loss", self.test_physics_loss, on_epoch=True, on_step=True)
            self.log("test_data_loss", self.test_data_loss, on_epoch=True, on_step=True)
            self.log("test_mass_cont", self.test_mass_cont, on_epoch=True, on_step=True)
            self.log("test_ns_longitude", self.test_ns_longitude, on_epoch=True, on_step=True)
            self.log("test_ns_latitude", self.test_ns_latitude, on_epoch=True, on_step=True)

            return total_loss
        
        else:
            data_loss = self.model_step(batch)
            
            # update and log metrics
            self.test_loss(data_loss)
            self.log("test_loss", self.test_loss, on_epoch=True, on_step=True)

            return data_loss
        
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