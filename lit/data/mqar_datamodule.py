import random

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset


class MQARDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_pairs: int = 8,
        vocab_size: int = 100,
        sequence_length: int = 50,
        num_samples: int = 10000,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_pairs = num_pairs
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.num_samples = num_samples

    def setup(self, stage: str = None):
        """Generates the dataset."""
        sequences, targets = self._generate_data(
            self.num_samples, self.num_pairs, self.vocab_size, self.sequence_length
        )

        # Convert to tensor format
        sequence_tensors = torch.tensor(sequences, dtype=torch.long)
        target_tensors = torch.tensor(targets, dtype=torch.long)

        self.dataset = TensorDataset(sequence_tensors, target_tensors)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # For validation, we can use a fixed batch of data
        val_sequences, val_targets = self._generate_data(
            self.batch_size, self.num_pairs, self.vocab_size, self.sequence_length
        )
        val_sequence_tensors = torch.tensor(val_sequences, dtype=torch.long)
        val_target_tensors = torch.tensor(val_targets, dtype=torch.long)
        val_dataset = TensorDataset(val_sequence_tensors, val_target_tensors)
        return DataLoader(val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # For testing, we can use a fixed batch of data
        test_sequences, test_targets = self._generate_data(
            self.batch_size, self.num_pairs, self.vocab_size, self.sequence_length
        )
        test_sequence_tensors = torch.tensor(test_sequences, dtype=torch.long)
        test_target_tensors = torch.tensor(test_targets, dtype=torch.long)
        test_dataset = TensorDataset(test_sequence_tensors, test_target_tensors)
        return DataLoader(test_dataset, batch_size=self.batch_size)

    def _generate_data(self, num_samples, num_pairs, vocab_size, seq_len):
        sequences = []
        targets = []

        for _ in range(num_samples):
            seq, target = self._generate_mqar_sequence(num_pairs, vocab_size, seq_len)
            sequences.append(seq)
            targets.append(target)

        return sequences, targets

    def _generate_mqar_sequence(self, num_pairs, vocab_size, seq_len):
        """Generate a MQAR sequence with key-value pairs and queries."""
        # Generate unique key-value pairs
        keys = random.sample(range(vocab_size), num_pairs)
        values = random.sample(range(vocab_size), num_pairs)

        # Create key-value mapping
        kv_pairs = list(zip(keys, values))

        # Generate sequence: [key1, value1, key2, value2, ..., query_key, target_value]
        sequence = []

        # Add key-value pairs
        for key, value in kv_pairs:
            sequence.extend([key, value])

        # Add separator token (vocab_size)
        sequence.append(vocab_size)

        # Randomly select a query key
        query_key = random.choice(keys)
        sequence.append(query_key)

        # Find the corresponding value
        target_value = dict(kv_pairs)[query_key]

        # Pad sequence to desired length
        while len(sequence) < seq_len:
            sequence.append(vocab_size + 1)  # Padding token

        # Truncate if too long
        sequence = sequence[:seq_len]

        return sequence, target_value
