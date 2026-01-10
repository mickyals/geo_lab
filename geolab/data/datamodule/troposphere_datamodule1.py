from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import pickle

import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset
import xarray as xr

from geolab.data.dataset import ERA5MultiData, ERA5MultiDataset


class TroposphereDataModule(LightningDataModule):
    def __init__(self,
                 data_dir: Union[str, Path],
                 var_labels: Dict[str, int],
                 statistics: Dict[str, Tuple[float, float, float, float]] = {name: (mean, std, min, max)}
                 ):
        pass

    def prepare_data(self) :
        pass

    def setup(self, stage: Optional[str]):
        pass

    def _split_indices(N):
        pass

    def _compute_statistics(self):
        pass

    def _create_dataloader(self):
        pass

    def normalize(self, data):
        pass

    def denormalize(self, data):
        pass

    def get_labels(self):
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def val_dataloader(self) -> DataLoader:
        pass

    def test_dataloader(self) -> DataLoader:
        pass

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
        """Input dimension."""
        return self.input_dim

    @property
    def output_dim(self) -> int:
        """Output dimension."""
        return len(var_labels)