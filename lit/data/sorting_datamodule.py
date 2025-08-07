import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset


class SortingDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        sequence_length: int = 7,
        num_samples: int = 10000,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_samples = num_samples
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

    def setup(self, stage: str = None):
        """Generates the dataset."""
        unsorted, sorted_indices = self._generate_data(
            self.num_samples, self.sequence_length
        )

        # The model expects input features of shape (batch, seq, features)
        # and targets of shape (batch, seq)
        self.dataset = TensorDataset(unsorted.unsqueeze(-1), sorted_indices)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        # For validation, we can use a fixed batch of data
        val_inputs, val_targets = self._generate_data(
            self.batch_size, self.sequence_length
        )
        val_dataset = TensorDataset(val_inputs.unsqueeze(-1), val_targets)
        return DataLoader(val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # For testing, we can use a fixed batch of data
        test_inputs, test_targets = self._generate_data(
            self.batch_size, self.sequence_length
        )
        test_dataset = TensorDataset(test_inputs.unsqueeze(-1), test_targets)
        return DataLoader(test_dataset, batch_size=self.batch_size)

    def _generate_data(self, num_samples, seq_len):
        unsorted = torch.rand(num_samples, seq_len)
        sorted_indices = torch.argsort(unsorted, dim=1)
        return unsorted, sorted_indices
