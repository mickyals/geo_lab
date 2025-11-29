from typing import Any, Dict, Optional, Tuple, List
from pathlib import Path
import pickle

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
            test_split: float = 0.70,
            pi_scale: bool = True,
            num_workers: int = 4,
            pin_memory: bool = True,
            persistent_workers: bool = False,
            cache_dir: str = None,
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
            cache_dir: Directory to cache preprocessed data (default: data_dir/cache)
        """
        super().__init__()
        self.save_hyperparameters()

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
        self.pi_scale = pi_scale
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        # Set up cache directory
        data_path = Path(data_dir)
        if data_path.is_file() or data_path.suffix:  # If it's a file or has an extension
            cache_base = data_path.parent  # Use parent directory
        else:
            cache_base = data_path  # Use data_dir itself
            
        self.cache_dir = Path(cache_dir) if cache_dir else cache_base / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache file path
        self.cache_file = self.cache_dir / "preprocessed_data.pkl"

        # Dataset attributes
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.era5_data = None
        self.full_data = None
        self.statistics = None
        
        # Try to load statistics from cache immediately
        cached_stats = self._load_statistics()
        if cached_stats is not None:
            self.statistics = cached_stats

    def prepare_data(self):
        """Download or load ERA5 data if needed.

        This method is called only once (on rank 0).
        Preprocesses data and saves to disk for later loading.
        """
        # Skip if cache already exists
        if self.cache_file.exists():
            print(f"Cache file already exists at {self.cache_file}, skipping preparation.")
            return

        print("Preparing ERA5 data...")

        era5 = ERA5MultiData(
            data_dir=str(self.data_dir),
            read_data_fn=self.read_data_fn,
            variables=self.solution_vars
        )

        # Run the data preparation/loading
        data, statistics = era5.run(
            self.time_idx_range,
            self.pressure_idx_range,
            self.latitude_idx_range,
            self.longitude_idx_range,
            indexing=self.indexing,
            num_samples=self.num_virtual,
            include_virtual=self.include_virtual,
            use_lhs=self.use_lhs
        )

        # Save preprocessed data to disk
        print(f"Saving preprocessed data to {self.cache_file}...")
        with open(self.cache_file, 'wb') as f:
            pickle.dump({'data': data, 'statistics': statistics}, f)
        
        print("Data prepared and cached successfully.")


    def _load_statistics(self) -> Optional[Dict]:
        """Load only statistics from cache if available."""
        if self.cache_file.exists():
            try:
                print(f"Loading statistics from {self.cache_file}...")
                with open(self.cache_file, 'rb') as f:
                    cached = pickle.load(f)
                    print(f"Cache loaded successfully. Keys: {cached.keys()}")
                    stats = cached.get('statistics')
                    if stats is not None:
                        print("Statistics loaded successfully!")
                    else:
                        print("Warning: 'statistics' key not found in cache")
                    return stats
            except Exception as e:
                print(f"Warning: Could not load statistics from cache: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            print(f"Cache file does not exist at {self.cache_file}")
        return None


    def setup(self, stage: Optional[str] = None):
        """Split data into train/val/test and create datasets."""
        if hasattr(self, 'full_data') and self.full_data is not None:
            print(f"Data already loaded, skipping reload for stage: {stage}")
            return

        print(f"Setting up datasets for stage: {stage}")

        # Load from cache file
        if self.cache_file.exists():
            print(f"Loading preprocessed data from {self.cache_file}...")
            with open(self.cache_file, 'rb') as f:
                cached = pickle.load(f)
                data = cached['data']
                statistics = cached['statistics']
        else:
            # Fallback: regenerate if cache doesn't exist
            print("Warning: Cache file not found, regenerating data...")
            era5 = ERA5MultiData(
                data_dir=str(self.data_dir),
                read_data_fn=self.read_data_fn,
                variables=self.solution_vars
            )
            data, statistics = era5.run(
                self.time_idx_range,
                self.pressure_idx_range,
                self.latitude_idx_range,
                self.longitude_idx_range,
                indexing=self.indexing,
                num_samples=self.num_virtual,
                include_virtual=self.include_virtual,
                use_lhs=self.use_lhs
            )

        self.statistics = statistics
        self.full_data = data['data']

        if self.include_virtual:
            total_points = data['count'][0]
            real_points = data['count'][1]
            virtual_points = data['count'][2]
        else:
            real_points = data['count'][0]
            total_points = real_points
            virtual_points = 0

        print(f"Total={total_points}, Real={real_points}, Virtual={virtual_points}")

        # === Identify real and virtual samples ===
        if 'classification' in self.full_data:
            classification = self.full_data['classification']
            real_indices = np.where(classification)[0]
            virtual_indices = np.where(~classification)[0]
        else:
            # fallback if not explicitly labeled
            real_indices = np.arange(real_points)
            virtual_indices = np.arange(real_points, total_points)

        # === Split only the real indices ===
        self.rng = np.random.default_rng()
        shuffled_real = self.rng.permutation(real_indices)

        n_val = int(len(real_indices) * self.val_split)
        n_test = int(len(real_indices) * self.test_split)
        n_train_real = len(real_indices) - n_val - n_test

        real_train_idx = shuffled_real[:n_train_real]
        real_val_idx = shuffled_real[n_train_real:n_train_real + n_val]
        real_test_idx = shuffled_real[n_train_real + n_val:]

        # === Construct splits ===
        train_idx = np.concatenate([virtual_indices, real_train_idx])
        train_idx = self.rng.permutation(train_idx)

        val_idx = real_val_idx
        test_idx = real_test_idx

        print(f"Split sizes -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        print(f"  (Real in train: {len(real_train_idx)}, Virtual in train: {len(virtual_indices)})")

        # === Build dataset objects ===
        self.train_dataset = ERA5MultiDataset(
            data=self.full_data,
            statistics=self.statistics,
            indices=train_idx,
            include_virtual=True,
            variables=self.solution_vars,
            pi_scale=self.pi_scale
        )

        self.val_dataset = ERA5MultiDataset(
            data=self.full_data,
            statistics=self.statistics,
            indices=val_idx,
            include_virtual=False,  # real-only validation
            variables=self.solution_vars,
            pi_scale=self.pi_scale
        )

        # self.test_dataset = ERA5MultiDataset(
        #     data=self.full_data,
        #     statistics=self.statistics,
        #     indices=test_idx,
        #     include_virtual=False,  # real-only test
        #     variables=self.solution_vars,
        #     pi_scale=self.pi_scale
        # )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            prefetch_factor=None,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            persistent_workers=self.persistent_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return None
        #return None #DataLoader(
        #     dataset=self.test_dataset,
        #     batch_size=self.batch_size,
        #     num_workers=self.num_workers,
        #     pin_memory=self.pin_memory,
        #     shuffle=False)

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