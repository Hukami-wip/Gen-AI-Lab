import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class MQARDataset(Dataset):
    """
    Multi-Query Associative Recall Dataset.

    Generates sequences of key-value pairs followed by queries.
    The model must recall the value associated with each query key.

    Format: [k1, v1, k2, v2, ..., kN, vN, SEP, q1, q2, ..., qM]
    Target: [-, -, -, -, ..., -, -, -, a1, a2, ..., aM]

    Where qi is a query key and ai is the corresponding value.
    """

    def __init__(
        self,
        num_samples: int,
        num_pairs: int,
        num_queries: int,
        vocab_size: int,
    ):
        self.num_samples = num_samples
        self.num_pairs = num_pairs
        self.num_queries = num_queries
        self.vocab_size = vocab_size

        # Special tokens
        self.sep_token = vocab_size
        self.pad_token = vocab_size + 1
        self.ignore_index = -100

        # Sequence length: 2*num_pairs (kv) + 1 (sep) + num_queries
        self.seq_len = 2 * num_pairs + 1 + num_queries

        # Pre-generate all data for reproducibility
        self.data = [self._generate_sample() for _ in range(num_samples)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx]

    def _generate_sample(self):
        # Generate unique keys and values from disjoint sets
        half_vocab = self.vocab_size // 2
        keys = torch.randperm(half_vocab)[: self.num_pairs]
        values = torch.randperm(half_vocab)[: self.num_pairs] + half_vocab

        # Interleave keys and values
        kv_seq = torch.stack([keys, values], dim=1).flatten()

        # Select random query keys
        query_indices = torch.randperm(self.num_pairs)[: self.num_queries]
        query_keys = keys[query_indices]
        query_answers = values[query_indices]

        # Build input sequence: [kv_seq, SEP, queries]
        input_seq = torch.cat(
            [
                kv_seq,
                torch.tensor([self.sep_token]),
                query_keys,
            ]
        )

        # Build target: ignore kv portion, predict answers after queries
        target = torch.full((self.seq_len,), self.ignore_index, dtype=torch.long)
        target[-self.num_queries :] = query_answers

        return input_seq, target


class MQARDataModule(LightningDataModule):
    """
    Lightning DataModule for MQAR task.

    Args:
        num_pairs: Number of key-value pairs to memorize
        num_queries: Number of queries per sequence
        vocab_size: Size of key/value vocabulary (actual vocab = vocab_size + 2 for special tokens)
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        batch_size: Batch size for all dataloaders
        num_workers: Number of dataloader workers
    """

    def __init__(
        self,
        num_pairs: int = 8,
        num_queries: int = 8,
        vocab_size: int = 64,
        train_samples: int = 50000,
        val_samples: int = 5000,
        test_samples: int = 5000,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_pairs = num_pairs
        self.num_queries = num_queries
        self.vocab_size = vocab_size
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Computed properties
        self.total_vocab_size = vocab_size + 2  # +2 for SEP and PAD tokens
        self.sequence_length = 2 * num_pairs + 1 + num_queries

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = MQARDataset(
                num_samples=self.train_samples,
                num_pairs=self.num_pairs,
                num_queries=self.num_queries,
                vocab_size=self.vocab_size,
            )
            self.val_dataset = MQARDataset(
                num_samples=self.val_samples,
                num_pairs=self.num_pairs,
                num_queries=self.num_queries,
                vocab_size=self.vocab_size,
            )

        if stage == "test" or stage is None:
            self.test_dataset = MQARDataset(
                num_samples=self.test_samples,
                num_pairs=self.num_pairs,
                num_queries=self.num_queries,
                vocab_size=self.vocab_size,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
