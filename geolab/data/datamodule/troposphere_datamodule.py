from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split

from geolab.data.dataset import ERA5MultiData, ERA5MultiDataset
import xarray as xr


class TroposphereDataModule(LightningDataModule):

    def __init__(
            self,
            data_dir: str,
            read_data_fn=xr.open_dataset,
            solution_vars: List[str] = ["u", "v", "w", "z"],
            time_idx_range: List[int] = None,
            pressure_idx_range: List[int] = None,
            latitude_idx_range: List[int] = None,
            longitude_idx_range: List[int] = None,
            include_virtual=False,
            indexing = 'ij',
            num_virtual = 20000,
            use_lhs = True,
            batch_size: int = 32,
            val_split: float = 0.15,
            test_split: float = 0.15,
            scale: bool = True,
            num_workers: int = 4,
            pin_memory: bool = True,
    ):
        """Initialize a TroposphereDataModule.

        Args:
            data_dir: Path to the directory containing the data files
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

        self.data_dir = data_dir
        self.read_data_fn = read_data_fn
        self.solution_vars = solution_vars
        self.time_idx_range = time_idx_range if time_idx_range is not None else None
        self.pressure_idx_range = pressure_idx_range if pressure_idx_range is not None else None
        self.latitude_idx_range = latitude_idx_range if latitude_idx_range is not None else None
        self.longitude_idx_range = longitude_idx_range if longitude_idx_range is not None else None
        self.include_virtual = include_virtual
        self.indexing = indexing
        self.num_virtual = num_virtual
        self.use_lhs = use_lhs
        self.batch_size = batch_size
        self.test_split = test_split
        self.scale = scale
        self.val_split = val_split
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
        self.era5 = ERA5MultiData(
            data_dir=str(self.data_dir),
            read_data_fn=self.read_data_fn,
            variables=self.solution_vars
        )

        data, statistics = self.era5.run(
            self.time_idx_range, 
            self.pressure_idx_range, 
            self.latitude_idx_range,
            self.longitude_idx_range, 
            indexing=self.indexing, 
            num_samples=self.num_virtual,
            include_virtual=self.include_virtual, 
            use_lhs=self.use_lhs
        )

        # Create shuffled indices for all data points
        self.rng = np.random.default_rng()  # Fixed seed for reproducibility
        total_points = data['count'][0]
        shuffled_idx = self.rng.permutation(total_points)
        
        # Shuffle all arrays in the data dictionary
        for k, v in data['data'].items():
            v[:] = v[shuffled_idx]
        
        # Calculate split sizes
        test_size = int(total_points * self.test_split)
        val_size = int(total_points * self.val_split)
        train_size = total_points - val_size - test_size
        
        # Create index arrays for each split
        train_idx = np.arange(0, train_size)
        val_idx = np.arange(train_size, train_size + val_size)
        test_idx = np.arange(train_size + val_size, total_points)

        self.statistics = statistics

        
        # Create datasets with the shuffled data and corresponding indices
        self.train_dataset = ERA5MultiDataset(
            data=data,
            statistics=statistics,
            indices=train_idx,
            include_virtual=self.include_virtual,
            variables=self.solution_vars,
            scale=self.scale
        )

        self.val_dataset = ERA5MultiDataset(
            data=data,
            statistics=statistics,
            indices=val_idx,
            include_virtual=self.include_virtual,
            variables=self.solution_vars,
            scale=self.scale
        )

        self.test_dataset = ERA5MultiDataset(
            data=data,
            statistics=statistics,
            indices=test_idx,
            include_virtual=self.include_virtual,
            variables=self.solution_vars,
            scale=self.scale
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