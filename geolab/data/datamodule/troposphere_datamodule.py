from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import pickle

import numpy as np
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import xarray as xr

from geolab.data.dataset import ERA5MultiData, ERA5MultiDataset


class TroposphereDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: Union[str, Path],
            solution_vars: List[str] = ["u", "v", "w", "z"],
            read_data_fn=None,
            time_idx_range: Optional[List[int]] = None,
            pressure_idx_range: Optional[List[int]] = None,
            latitude_idx_range: Optional[List[int]] = None,
            longitude_idx_range: Optional[List[int]] = None,
            include_virtual: bool = False,
            indexing: str = 'ij',
            num_virtual: int = 20000,
            use_lhs: bool = True,
            batch_size: int = 32,
            val_split: float = 0.15,
            test_split: float = 0.70,
            pi_scale: bool = True,
            num_workers: int = 4,
            pin_memory: bool = True,
            persistent_workers: bool = False,
            seed: int = 42,
            statistics_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize a TroposphereDataModule with improved memory management.
        
        Args:
            statistics_dir: Directory containing precomputed statistics files.
                          If None, defaults to data_dir/statistics
        """
        super().__init__()
        self.save_hyperparameters(ignore=["read_data_fn"])

        # Set default read function if not provided
        self.read_data_fn = read_data_fn if read_data_fn is not None else xr.open_dataset

        # Store parameters
        self.data_dir = Path(data_dir)
        self.solution_vars = solution_vars
        self.time_idx_range = time_idx_range
        self.pressure_idx_range = pressure_idx_range
        self.latitude_idx_range = latitude_idx_range
        self.longitude_idx_range = longitude_idx_range
        self.include_virtual = include_virtual
        self.indexing = indexing
        self.num_virtual = num_virtual
        self.use_lhs = use_lhs
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.pi_scale = pi_scale
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed = seed
        
        # Set statistics directory
        # Default to geolab/data/dataset/ where precompute_statistics.py lives
        if statistics_dir is None:
            # Assume this file is in geolab/data/datamodule/
            # Navigate to geolab/data/dataset/
            module_dir = Path(__file__).parent.parent / "dataset"
            self.statistics_dir = module_dir
        else:
            self.statistics_dir = Path(statistics_dir)

        # Initialize dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self._data_counts = None
        self.statistics = None
        self.full_data = None

        # Set random seed for reproducibility
        self.rng = np.random.default_rng(seed)
        
        # Load statistics immediately
        self._load_statistics()

    def _load_statistics(self):
        """Load precomputed statistics based on time_idx_range configuration."""
        # Determine which statistics file to load
        if self.time_idx_range is not None and self.time_idx_range == [0, 1]:
            config_name = "1_timeslice"
        else:
            config_name = "all_timeslices"
        
        stats_file = self.statistics_dir / f"statistics_{config_name}.pkl"
        
        if not stats_file.exists():
            raise FileNotFoundError(
                f"Statistics file not found: {stats_file}\n"
                f"Please run precompute_statistics.py first to generate statistics files."
            )
        
        print(f"Loading precomputed statistics from: {stats_file}")
        
        with open(stats_file, 'rb') as f:
            self.statistics = pickle.load(f)
        
        print(f"Statistics loaded successfully! Keys: {self.statistics.keys()}")

    def prepare_data(self):
        """
        Download or validate ERA5 data if needed.
        
        Note: Statistics are now loaded in __init__, so we don't need to
        compute them here. This method now just validates data availability.
        """
        print("Validating ERA5 data availability...")
        
        # Just check that data directory exists and is accessible
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        print("Data directory validated.")

    def _load_and_prepare_data(self):
        """
        Load and prepare data using precomputed statistics.
        
        This method now only loads the data once since statistics are
        already available from __init__.
        """
        print("Loading data for training...")
        
        era5 = ERA5MultiData(
            data_dir=str(self.data_dir),
            read_data_fn=self.read_data_fn,
            variables=self.solution_vars
        )
        
        # Load data with precomputed statistics
        data, _ = era5.run(
            self.time_idx_range,
            self.pressure_idx_range,
            self.latitude_idx_range,
            self.longitude_idx_range,
            indexing=self.indexing,
            num_samples=self.num_virtual,
            include_virtual=self.include_virtual,
            use_lhs=self.use_lhs
        )
        
        # Use precomputed statistics instead of newly computed ones
        self.full_data = data['data']
        self._data_counts = data['count']

    def _get_real_virtual_indices(self):
        """Get indices for real and virtual samples."""
        if 'classification' in self.full_data:
            classification = self.full_data['classification']
            real_indices = np.where(classification)[0]
            virtual_indices = np.where(~classification)[0] if self.include_virtual else np.array([], dtype=int)
        else:
            real_count = self._data_counts[1] if self.include_virtual else self._data_counts[0]
            real_indices = np.arange(real_count)
            virtual_indices = np.arange(real_count,
                                        len(self.full_data['longitude'])) if self.include_virtual else np.array([],
                                                                                                                dtype=int)

        return real_indices, virtual_indices

    def _split_indices(self, real_indices, virtual_indices):
        """Split real indices into train/val/test sets and combine with virtual samples."""
        shuffled_real = self.rng.permutation(real_indices)

        n_val = int(len(real_indices) * self.val_split)
        n_test = int(len(real_indices) * self.test_split)
        n_train_real = len(real_indices) - n_val - n_test

        real_train_idx = shuffled_real[:n_train_real]
        real_val_idx = shuffled_real[n_train_real:n_train_real + n_val]
        real_test_idx = shuffled_real[n_train_real + n_val:]

        train_idx = np.concatenate([virtual_indices, real_train_idx])
        train_idx = self.rng.permutation(train_idx)

        return train_idx, real_val_idx, real_test_idx

    def _create_dataset(self, indices: np.ndarray, include_virtual: bool) -> Dataset:
        """Create a dataset with the given indices."""
        return ERA5MultiDataset(
            data=self.full_data,
            statistics=self.statistics,
            indices=indices,
            include_virtual=include_virtual,
            variables=self.solution_vars,
            pi_scale=self.pi_scale
        )

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for training, validation, and testing."""
        if hasattr(self, 'full_data') and self.full_data is not None:
            print(f"Data already loaded, skipping reload for stage: {stage}")
            return

        print(f"Setting up datasets for stage: {stage}")

        self._load_and_prepare_data()
        real_indices, virtual_indices = self._get_real_virtual_indices()

        total_points = len(self.full_data['longitude'])
        real_points = len(real_indices)
        virtual_points = len(virtual_indices)
        print(f"Dataset statistics - Total: {total_points}, Real: {real_points}, Virtual: {virtual_points}")

        train_idx, val_idx, test_idx = self._split_indices(real_indices, virtual_indices)

        print(f"Split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        print(f"  (Real in train: {len(train_idx) - len(virtual_indices)}, Virtual in train: {len(virtual_indices)})")

        self.train_dataset = self._create_dataset(train_idx, include_virtual=True)
        self.val_dataset = self._create_dataset(val_idx, include_virtual=False)

        # Uncomment if test set is needed
        # self.test_dataset = self._create_dataset(test_idx, include_virtual=False)

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader with consistent settings."""
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=shuffle,
            prefetch_factor=2 if self.num_workers > 0 else None
        )

    def train_dataloader(self) -> DataLoader:
        """Create and return the training DataLoader."""
        if self.train_dataset is None:
            self.setup('fit')
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader."""
        if self.val_dataset is None:
            self.setup('fit')
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Create and return the test DataLoader."""
        if self.test_dataset is None:
            self.setup('test')
        return self._create_dataloader(self.val_dataset, shuffle=False)  # or self.test_dataset if using test set

    def teardown(self, stage: Optional[str] = None) -> None:
        """Clean up resources."""
        if hasattr(self, 'full_data'):
            del self.full_data
        if hasattr(self, 'train_dataset'):
            del self.train_dataset
        if hasattr(self, 'val_dataset'):
            del self.val_dataset
        if hasattr(self, 'test_dataset'):
            del self.test_dataset
        if hasattr(self, '_prepared_data'):
            del self._prepared_data

    @property
    def num_train_samples(self) -> int:
        """Number of training samples."""
        return len(self.train_dataset) if self.train_dataset is not None else 0

    @property
    def num_val_samples(self) -> int:
        """Number of validation samples."""
        return len(self.val_dataset) if self.val_dataset is not None else 0

    @property
    def num_test_samples(self) -> int:
        """Number of test samples."""
        return len(self.test_dataset) if hasattr(self, 'test_dataset') and self.test_dataset is not None else 0

    @property
    def input_dim(self) -> int:
        """Dimensionality of input features (longitude, latitude, pressure_level, time)."""
        return 4

    @property
    def output_dim(self) -> int:
        """Dimensionality of output variables."""
        return len(self.solution_vars)