from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from geolab.data.components.coordinate_data.mesh import ERA5MultiData
from geolab.data.components.coordinate_data.troposphere_dataset import TroposphereDataset
import xarray as xr


class TroposphereDataModule(LightningDataModule):

    def __init__(
            self,
            root_dir: str,
            read_data_fn=xr.open_dataset,
            solution_vars: List[str] = ["u", "v", "w", "z"],
            prc_points: float = 0.3,
            prc_virtual: float = 0.1,
            batch_size: int = 32,
            val_split: float = 0.15,
            test_split: float = 0.15,
            num_workers: int = 4,
            pin_memory: bool = True,
    ):
        """Initialize a TroposphereDataModule.

        Args:
            root_dir: Path to the directory containing the data files
            read_data_fn: Function to read the data files (default: xr.open_dataset)
            solution_vars: List of variable names to include in the dataset
            prc_points: Percentage of total points to use (0.0 to 1.0)
            prc_virtual: Percentage of virtual points to use (0.0 to 1.0) based on the real points so 0.1 means 10% of the real points
            batch_size: Batch size for the dataloaders
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            num_workers: Number of workers for the dataloaders
            pin_memory: Whether to pin memory for the dataloaders
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.save_hyperparameters(ignore=["read_data_fn"])

        self.root_dir = Path(root_dir)
        self.read_data_fn = read_data_fn
        self.solution_vars = solution_vars
        self.prc_data_points = prc_points
        self.prc_virtual = prc_virtual
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.era5_data = None

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU. Do not use it to assign state (self.x = y).
        """
        # You can add data downloading logic here if needed
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.train_dataset`, `self.val_dataset`, [self.test_dataset](cci:1://file:///C:/Users/micke/OneDrive%20-%20University%20of%20Toronto/geo_lab/tests/test_era5_vertical_velocity.py:18:0-55:13)."""
        # Initialize ERA5 data
        self.era5_data = ERA5MultiData(
            root_dir=str(self.root_dir),
            read_data_fn=self.read_data_fn,
            solution_vars=self.solution_vars
        )

        # Get indices for all available points
        real_num_points = self.era5_data.num_points
        virtual_num_points = int(self.prc_virtual * real_num_points)
        num_points = real_num_points + virtual_num_points
        all_idx = np.arange(num_points)
        num_samples = int(self.prc_data_points * num_points) + virtual_num_points

        # Sample indices
        rng = np.random.default_rng()
        sampled_idx = rng.choice(all_idx, min(num_samples, num_points), replace=False)

        # Split indices
        num_val = int(self.val_split * len(sampled_idx))
        num_test = int(self.test_split * len(sampled_idx))
        num_train = len(sampled_idx) - num_val - num_test

        train_idx = sampled_idx[:num_train]
        val_idx = sampled_idx[num_train:num_train + num_val]
        test_idx = sampled_idx[num_train + num_val:]

        # Create datasets
        self.train_dataset = TroposphereDataset(
            root_dir=str(self.root_dir),
            read_data_fn=self.read_data_fn,
            solution_vars=self.solution_vars,
            indices=train_idx
        )

        self.val_dataset = TroposphereDataset(
            root_dir=str(self.root_dir),
            read_data_fn=self.read_data_fn,
            solution_vars=self.solution_vars,
            indices=val_idx
        )

        self.test_dataset = TroposphereDataset(
            root_dir=str(self.root_dir),
            read_data_fn=self.read_data_fn,
            solution_vars=self.solution_vars,
            indices=test_idx
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    @property
    def num_train_samples(self) -> int:
        """Number of training samples."""
        return len(self.train_dataset) if self.train_dataset else 0

    @property
    def num_val_samples(self) -> int:
        """Number of validation samples."""
        return len(self.val_dataset) if self.val_dataset else 0

    @property
    def num_test_samples(self) -> int:
        """Number of test samples."""
        return len(self.test_dataset) if self.test_dataset else 0

    @property
    def input_dim(self) -> int:
        """Dimensionality of input features (longitude, latitude, pressure_level, time)."""
        return 4

    @property
    def output_dim(self) -> int:
        """Dimensionality of output variables."""
        return len(self.solution_vars)