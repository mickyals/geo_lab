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
    def __init__(self,
                 data_dir: Union[str, Path],
                 solution_vars: List[str],
                 time_idx_range: Optional[List[int]] = None,
                 pressure_idx_range: Optional[List[int]] = None,
                 latitude_idx_range: Optional[List[int]] = None,
                 longitude_idx_range: Optional[List[int]] = None,
                 val_split: float = 0.15,
                 test_split: float = 0.70,
                 split_type: str = "random",
                 norm_type: str = "default",
                 batch_size: int = 32,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 persistent_workers: bool = False,
                 seed: int = 42,
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.solution_vars = solution_vars
        self.time_idx_range = time_idx_range
        self.pressure_idx_range = pressure_idx_range
        self.latitude_idx_range = latitude_idx_range
        self.longitude_idx_range = longitude_idx_range
        self.val_split = val_split
        self.test_split = test_split
        self.split_type = split_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.norm_type = norm_type

        # Set stats cache path here, not in prepare_data
        self.stats_cache = self.data_dir.parent / "statistics.pkl"

        # Will be populated in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.statistics = None
        self.coordinate_ranges = None

    def prepare_data(self):
        """Validate data availability."""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")

        if not self.stats_cache.exists():
            print(f"Statistics cache not found at {self.stats_cache}.")
            print("Will compute during setup().")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Load the full data
            self.data = ERA5MultiData(
                data_dir=self.data_dir,
                variables=self.solution_vars,
                time_idx_range=self.time_idx_range,
                pressure_idx_range=self.pressure_idx_range,
                latitude_idx_range=self.latitude_idx_range,
                longitude_idx_range=self.longitude_idx_range
            )

            # Split indices
            train_indices, val_indices, test_indices = self._split_indices()

            # Compute or load statistics
            if not self.stats_cache.exists():
                self.statistics = self._compute_statistics(train_indices)
            else:
                self.statistics = self._load_statistics()

            # Store coordinate ranges for reference
            self.coordinate_ranges = {
                'valid_time': (self.data.times.min(), self.data.times.max()),
                'pressure_level': (self.data.pressure_levels.min(), self.data.pressure_levels.max()),
                'latitude': (self.data.latitudes.min(), self.data.latitudes.max()),
                'longitude': (self.data.longitudes.min(), self.data.longitudes.max()),
            }

            # Create datasets
            self.train_dataset = ERA5MultiDataset(
                data=self.data,
                indices=train_indices,
                statistics=self.statistics,
                variables=self.solution_vars,
            )

            self.val_dataset = ERA5MultiDataset(
                data=self.data,
                indices=val_indices,
                statistics=self.statistics,
                variables=self.solution_vars,
            )

            self.test_dataset = ERA5MultiDataset(
                data=self.data,
                indices=test_indices,
                statistics=self.statistics,
                variables=self.solution_vars,
            )

    def _split_indices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split 4D indices (time, pressure, lat, lon) into train/val/test."""
        np.random.seed(self.seed)

        n_times = self.data.num_times
        n_pressure = self.data.num_pressure_levels
        n_lats = self.data.num_latitudes
        n_lons = self.data.num_longitudes

        total_points = n_times * n_pressure * n_lats * n_lons

        print(f"Total 4D grid points: {total_points:,}")
        print(f"  Shape: (time={n_times}, pressure={n_pressure}, lat={n_lats}, lon={n_lons})")

        all_indices = np.arange(total_points)
        np.random.shuffle(all_indices)

        n_val = int(total_points * self.val_split)
        n_test = int(total_points * self.test_split)
        n_train = total_points - n_val - n_test

        train_flat = all_indices[:n_train]
        val_flat = all_indices[n_train:n_train + n_val]
        test_flat = all_indices[n_train + n_val:]

        train_indices = np.stack(np.unravel_index(
            train_flat, (n_times, n_pressure, n_lats, n_lons)
        ), axis=-1)

        val_indices = np.stack(np.unravel_index(
            val_flat, (n_times, n_pressure, n_lats, n_lons)
        ), axis=-1)

        test_indices = np.stack(np.unravel_index(
            test_flat, (n_times, n_pressure, n_lats, n_lons)
        ), axis=-1)

        print(f"Split complete:")
        print(f"  Train: {len(train_indices):,} points ({n_train / total_points:.1%})")
        print(f"  Val:   {len(val_indices):,} points ({n_val / total_points:.1%})")
        print(f"  Test:  {len(test_indices):,} points ({n_test / total_points:.1%})")

        return train_indices, val_indices, test_indices

    def _compute_statistics(self, train_indices):
        """Compute statistics from training indices."""
        print("Computing normalisation statistics from training data...")

        statistics = {}

        for var in self.solution_vars:
            t_idx = train_indices[:, 0]
            p_idx = train_indices[:, 1]
            lat_idx = train_indices[:, 2]
            lon_idx = train_indices[:, 3]

            var_values = self.data.get_values_at_index(
                var, t_idx, p_idx, lat_idx, lon_idx
            )

            minimum = var_values.min()
            maximum = var_values.max()
            mean = var_values.mean()
            std = var_values.std()

            statistics[var] = {
                'min': torch.tensor(minimum, dtype=torch.float32),
                'max': torch.tensor(maximum, dtype=torch.float32),
                'mean': torch.tensor(mean, dtype=torch.float32),
                'std': torch.tensor(std, dtype=torch.float32)
            }
            print(f"   {var}: min={minimum:.4f}, max={maximum:.4f}, mean={mean:.4f}, std={std:.4f}")

        with open(self.stats_cache, 'wb') as f:
            pickle.dump(statistics, f)

        print(f"Cached statistics to {self.stats_cache}")

        return statistics

    def _load_statistics(self):
        """Load precomputed statistics from cache."""
        print(f"Loading statistics from {self.stats_cache}")

        with open(self.stats_cache, 'rb') as f:
            statistics = pickle.load(f)

        print(f"Statistics loaded successfully!")
        return statistics

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers if self.num_workers > 0 else False,
            shuffle=shuffle,
            drop_last=True,
        )

    def normalize(self, data: torch.Tensor, var: str) -> torch.Tensor:
        """Normalize a single variable tensor."""
        if self.statistics is None:
            raise RuntimeError("Statistics not computed. Call setup() first.")
        if var not in self.statistics:
            raise ValueError(f"Variable {var} not found in statistics.")

        stats = self.statistics[var]

        if self.norm_type == "default":
            if stats["max"] == stats["min"]:
                return torch.ones_like(data)
            return 2 * (data - stats["min"]) / (stats["max"] - stats["min"]) - 1

        elif self.norm_type == "standard":
            return (data - stats["mean"]) / (stats["std"] + 1e-8)

        else:  # min-max [0, 1]
            if stats["max"] == stats["min"]:
                return torch.ones_like(data)
            return (data - stats["min"]) / (stats["max"] - stats["min"])

    def denormalize(self, data: torch.Tensor, var: str) -> torch.Tensor:
        """Denormalize a single variable tensor."""
        if self.statistics is None:
            raise RuntimeError("Statistics not computed. Call setup() first.")
        if var not in self.statistics:
            raise ValueError(f"Variable {var} not found in statistics.")

        stats = self.statistics[var]

        if self.norm_type == "default":
            if stats["max"] == stats["min"]:
                return torch.ones_like(data)
            return (data + 1) / 2 * (stats["max"] - stats["min"]) + stats["min"]

        elif self.norm_type == "standard":
            return data * (stats["std"] + 1e-8) + stats["mean"]

        else:  # min-max
            if stats["max"] == stats["min"]:
                return torch.ones_like(data)
            return data * (stats["max"] - stats["min"]) + stats["min"]

    def normalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates to [-1, 1].

        Args:
            coords: Tensor of shape (..., 4) with [time, pressure, lat, lon]

        Returns:
            Normalized coordinates in range [-1, 1]
        """
        if self.coordinate_ranges is None:
            raise RuntimeError("Coordinate ranges not set. Call setup() first.")

        normalized = coords.clone()
        coord_names = ['valid_time', 'pressure_level', 'latitude', 'longitude']

        for i, name in enumerate(coord_names):
            min_val, max_val = self.coordinate_ranges[name]
            normalized[..., i] = 2 * (coords[..., i] - min_val) / (max_val - min_val + 1e-8) - 1

        return normalized

    def denormalize_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Denormalize coordinates from [-1, 1] back to original range.

        Args:
            coords: Normalized tensor of shape (..., 4)

        Returns:
            Denormalized coordinates
        """
        if self.coordinate_ranges is None:
            raise RuntimeError("Coordinate ranges not set. Call setup() first.")

        denormalized = coords.clone()
        coord_names = ['valid_time', 'pressure_level', 'latitude', 'longitude']

        for i, name in enumerate(coord_names):
            min_val, max_val = self.coordinate_ranges[name]
            denormalized[..., i] = (coords[..., i] + 1) / 2 * (max_val - min_val) + min_val

        return denormalized

    def normalize_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize an entire batch (coords + all variables).

        Args:
            batch: Dict with 'coords' (B, 4) and 'values' (B, num_vars)

        Returns:
            Dict with normalized coords and values
        """
        normalized_batch = {}

        # Normalize coordinates
        normalized_batch['coords'] = self.normalize_coords(batch['coords'])

        # Normalize each variable
        normalized_values = []
        for i, var in enumerate(self.solution_vars):
            normalized_values.append(
                self.normalize(batch['values'][:, i], var)
            )
        normalized_batch['values'] = torch.stack(normalized_values, dim=1)

        return normalized_batch

    def denormalize_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Denormalize an entire batch.

        Args:
            batch: Dict with normalized 'coords' and 'values'

        Returns:
            Dict with denormalized coords and values
        """
        denormalized_batch = {}

        # Denormalize coordinates
        denormalized_batch['coords'] = self.denormalize_coords(batch['coords'])

        # Denormalize each variable
        denormalized_values = []
        for i, var in enumerate(self.solution_vars):
            denormalized_values.append(
                self.denormalize(batch['values'][:, i], var)
            )
        denormalized_batch['values'] = torch.stack(denormalized_values, dim=1)

        return denormalized_batch

    def get_variable_labels(self) -> List[str]:
        """Get list of variable names."""
        return self.solution_vars

    def get_coordinate_labels(self) -> List[str]:
        """Get list of coordinate names."""
        return ['valid_time', 'pressure_level', 'latitude', 'longitude']

    def train_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        return self._create_dataloader(self.test_dataset, shuffle=False)

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
        """Input coordinate dimension (time, pressure, lat, lon)."""
        return 4

    @property
    def output_dim(self) -> int:
        """Output dimension (number of atmospheric variables)."""
        return len(self.solution_vars)

    @property
    def spatial_coords(self) -> List[str]:
        """Return names of spatial coordinates."""
        return ['latitude', 'longitude']

    @property
    def temporal_coords(self) -> List[str]:
        """Return names of temporal coordinates."""
        return ['valid_time']

    @property
    def vertical_coords(self) -> List[str]:
        """Return names of vertical coordinates."""
        return ['pressure_level']