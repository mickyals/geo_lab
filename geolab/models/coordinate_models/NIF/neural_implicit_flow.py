"""
Neural Implicit Flow (NIF) - Main LightningModule.

Combines ParamNet, WeightsEmbeddingLayer, and ShapeNet for troposphere modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from typing import Dict, Any, Optional

from geolab.models.coordinate_models.nif.param_net import ParamNet
from geolab.models.coordinate_models.nif.shape_net import ShapeNet
from geolab.models.coordinate_models.nif.weights_embedding import WeightsEmbeddingLayer
from geolab.models.components.core.physics import troposphere_pde_residual


class NeuralImplicitFlow(LightningModule):
    """Neural Implicit Flow for troposphere modeling with optional PINN loss.
    
    Args:
        param_net_config: Configuration for ParamNet
            - input_dim: Input dimension (time, pressure_level)
            - latent_dim: Latent embedding dimension
            - hidden_dims: List of hidden layer sizes
            - activation: Activation function
            - net_type: 'mlp' or 'siren'
            - omega_0: SIREN frequency (if applicable)
        
        shape_net_config: Configuration for ShapeNet
            - input_dim: Input dimension (longitude, latitude)
            - output_dim: Output dimension (u, v, w, z)
            - units: Hidden layer width
            - nlayers: Number of hidden layers
            - activation: Activation function
            - net_type: 'mlp' or 'siren'
            - omega_0: SIREN frequency (if applicable)
        
        use_pinn_loss: Whether to include physics-informed loss
        pinn_weight: Weight for physics loss term
        mass_balance: Whether to include mass continuity in PDE residual
        n_collocation: Number of collocation points per batch
        learning_rate: Learning rate for optimizer
        optimizer: Optimizer type ('adam', 'adamw', 'sgd')
    """
    
    def __init__(
        self,
        param_net_config: Dict[str, Any],
        shape_net_config: Dict[str, Any],
        use_pinn_loss: bool = False,
        pinn_weight: float = 0.1,
        mass_balance: bool = True,
        n_collocation: int = 1024,
        learning_rate: float = 1e-3,
        optimizer: str = 'adam'
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Build networks
        self.param_net = ParamNet(
            input_dim=param_net_config['input_dim'],
            latent_dim=param_net_config['latent_dim'],
            hidden_dims=param_net_config['hidden_dims'],
            activation=param_net_config.get('activation', 'gelu'),
            net_type=param_net_config.get('net_type', 'mlp'),
            omega_0=param_net_config.get('omega_0', 30.0)
        )
        
        self.weights_embedding = WeightsEmbeddingLayer(
            latent_dim=param_net_config['latent_dim'],
            shape_net_config=shape_net_config
        )
        
        self.shape_net = ShapeNet(
            input_dim=shape_net_config['input_dim'],
            output_dim=shape_net_config['output_dim'],
            hidden_dim=shape_net_config['units'],
            nlayers=shape_net_config['nlayers'],
            net_type=shape_net_config.get('net_type', 'mlp'),
            activation=shape_net_config.get('activation', 'gelu'),
            omega_0=shape_net_config.get('omega_0', 30.0)
        )
        
        self.param_input_dim = param_net_config['input_dim']
        self.shape_input_dim = shape_net_config['input_dim']
    
    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through NIF.
        
        Args:
            x: Dictionary with keys:
                - 'time': [batch_size]
                - 'pressure_level': [batch_size]
                - 'longitude': [batch_size]
                - 'latitude': [batch_size]
        
        Returns:
            Dictionary with predicted outputs:
                - 'u': [batch_size]
                - 'v': [batch_size]
                - 'w': [batch_size]
                - 'z': [batch_size]
        """
        # Split inputs
        x_param = torch.stack([x['time'], x['pressure_level']], dim=1) #[B, 2]
        x_shape = torch.stack([x['longitude'], x['latitude']], dim=1) #[B, 2]
        
        # ParamNet: encode context to latent
        z = self.param_net(x_param) # [B, latent_dim]
        
        # WeightsEmbeddingLayer: map latent to ShapeNet weights
        layer_params = self.weights_embedding(z) # List of (W, b) tuples
        
        # ShapeNet: compute outputs with dynamic weights
        outputs = self.shape_net(x_shape, layer_params) # [B, 4]
        
        # Package outputs
        return {
            'u': outputs[:, 0],
            'v': outputs[:, 1],
            'w': outputs[:, 2],
            'z': outputs[:, 3]
        }
    
    def training_step(self, batch, batch_idx):
        """Training step with optional PINN loss."""
        x, y_true = batch
        
        # Data loss on labeled points
        y_pred = self.forward(x)
        
        data_loss = sum([
            F.mse_loss(y_pred[var], y_true[var])
            for var in ['u', 'v', 'w', 'z']
        ]) / 4.0
        
        self.log('train/data_loss', data_loss, prog_bar=True)
        
        # Physics loss on collocation points
        if self.hparams.use_pinn_loss:
            physics_loss = self._compute_physics_loss(batch_idx)
            self.log('train/physics_loss', physics_loss, prog_bar=True)
            
            total_loss = data_loss + self.hparams.pinn_weight * physics_loss
            self.log('train/total_loss', total_loss, prog_bar=True)
        else:
            total_loss = data_loss
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y_true = batch
        y_pred = self.forward(x)
        
        # Compute per-variable losses
        losses = {}
        for var in ['u', 'v', 'w', 'z']:
            losses[f'val/{var}_loss'] = F.mse_loss(y_pred[var], y_true[var])
        
        # Total validation loss
        val_loss = sum(losses.values()) / len(losses)
        losses['val/loss'] = val_loss
        
        self.log_dict(losses, prog_bar=True)
        return val_loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        x, y_true = batch
        y_pred = self.forward(x)
        
        # Compute per-variable losses
        losses = {}
        for var in ['u', 'v', 'w', 'z']:
            losses[f'test/{var}_loss'] = F.mse_loss(y_pred[var], y_true[var])
        
        # Total test loss
        test_loss = sum(losses.values()) / len(losses)
        losses['test/loss'] = test_loss
        
        self.log_dict(losses)
        return test_loss
    
    def _compute_physics_loss(self, batch_idx):
        """Compute physics-informed loss on collocation points."""
        # Sample collocation points from datamodule
        x_col = self.trainer.datamodule.get_collocation_batch(
            self.hparams.n_collocation
        )
        
        # Move to device and enable gradients
        x_col = {k: v.to(self.device).requires_grad_(True) 
                 for k, v in x_col.items()}
        
        # Forward pass
        y_col = self.forward(x_col)
        
        # Compute PDE residuals
        pde_residuals = troposphere_pde_residual(
            inputs=x_col,
            outputs=y_col,
            mass_balance=self.hparams.mass_balance
        )
        
        # Mean squared residual
        physics_loss = sum([r.pow(2).mean() for r in pde_residuals]) / len(pde_residuals)
        
        return physics_loss
    
    def configure_optimizers(self):
        """Configure optimizer."""
        if self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate
            )
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate
            )
        elif self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")
        
        return optimizer