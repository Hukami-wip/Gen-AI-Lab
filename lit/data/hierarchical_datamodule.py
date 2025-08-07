import random

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset


class HierarchicalDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        max_depth: int = 3,
        max_length: int = 20,
        num_samples: int = 10000,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_depth = max_depth
        self.max_length = max_length
        self.num_samples = num_samples

    def setup(self, stage: str = None):
        """Generates the dataset."""
        sequences, parse_trees = self._generate_data(
            self.num_samples, self.max_depth, self.max_length
        )

        # Convert sequences to tensor format
        # Each token: 0=open_paren, 1=close_paren, 2=token
        sequence_tensors = torch.tensor(sequences, dtype=torch.long)
        parse_tree_tensors = torch.tensor(parse_trees, dtype=torch.long)

        self.dataset = TensorDataset(sequence_tensors, parse_tree_tensors)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        # For validation, we can use a fixed batch of data
        val_sequences, val_parse_trees = self._generate_data(
            self.batch_size, self.max_depth, self.max_length
        )
        val_sequence_tensors = torch.tensor(val_sequences, dtype=torch.long)
        val_parse_tree_tensors = torch.tensor(val_parse_trees, dtype=torch.long)
        val_dataset = TensorDataset(val_sequence_tensors, val_parse_tree_tensors)
        return DataLoader(val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # For testing, we can use a fixed batch of data
        test_sequences, test_parse_trees = self._generate_data(
            self.batch_size, self.max_depth, self.max_length
        )
        test_sequence_tensors = torch.tensor(test_sequences, dtype=torch.long)
        test_parse_tree_tensors = torch.tensor(test_parse_trees, dtype=torch.long)
        test_dataset = TensorDataset(test_sequence_tensors, test_parse_tree_tensors)
        return DataLoader(test_dataset, batch_size=self.batch_size)

    def _generate_data(self, num_samples, max_depth, max_length):
        sequences = []
        parse_trees = []

        for _ in range(num_samples):
            seq, parse_tree = self._generate_hierarchical_sequence(
                max_depth, max_length
            )
            sequences.append(seq)
            parse_trees.append(parse_tree)

        return sequences, parse_trees

    def _generate_hierarchical_sequence(self, max_depth, max_length):
        """Generate a hierarchical sequence with nested parentheses."""
        if max_depth <= 0:
            # Base case: just tokens
            length = random.randint(1, max_length // 2)
            seq = [2] * length  # 2 represents a token
            parse_tree = [0] * length  # 0 represents depth level
            return seq, parse_tree

        # Generate sequence with potential nesting
        seq = []
        parse_tree = []
        current_depth = 0

        while len(seq) < max_length:
            if current_depth < max_depth and random.random() < 0.3:
                # Add opening parenthesis
                seq.append(0)  # 0 = open parenthesis
                parse_tree.append(current_depth)
                current_depth += 1
            elif current_depth > 0 and random.random() < 0.4:
                # Add closing parenthesis
                seq.append(1)  # 1 = close parenthesis
                current_depth -= 1
                parse_tree.append(current_depth)
            else:
                # Add token
                seq.append(2)  # 2 = token
                parse_tree.append(current_depth)

            # Ensure we don't exceed max_length
            if len(seq) >= max_length:
                break

        # Close any remaining open parentheses
        while current_depth > 0 and len(seq) < max_length:
            seq.append(1)
            current_depth -= 1
            parse_tree.append(current_depth)

        # Pad sequences to max_length
        while len(seq) < max_length:
            seq.append(3)  # 3 = padding token
            parse_tree.append(-1)  # -1 = padding

        return seq, parse_tree
