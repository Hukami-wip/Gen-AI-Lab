from lit.data.sonar_parquet_datamodule import Wiki40bDataModule
from lit.data.sorting_datamodule import SortingDataModule

from .mnist_datamodule import MNISTDataModule

__all__ = ["MNISTDataModule", "SortingDataModule", "Wiki40bDataModule"]
