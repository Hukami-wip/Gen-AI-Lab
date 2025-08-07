import itertools
import multiprocessing
import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq
import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset


class FileManager:
    """Process-safe manager for distributing files across workers."""

    def __init__(
        self,
        parquet_files,
        files_per_batch,
        shared_file_idx,
        lock,
        shared_indices,
        shuffle_files,
    ):
        self.parquet_files = parquet_files
        self.files_per_batch = files_per_batch
        self.current_file_idx = shared_file_idx
        self.lock = lock
        self.shared_indices = shared_indices
        self.shuffle_files = shuffle_files

    def get_next_batch(self):
        """Get the next batch of files for processing."""
        with self.lock:
            if self.current_file_idx.value >= len(self.parquet_files):
                self.current_file_idx.value = 0
                # Shuffle the indices for the new epoch
                if self.shuffle_files:
                    # This works for both multiprocessing.Manager.list and regular lists
                    indices = list(self.shared_indices)
                    random.shuffle(indices)
                    self.shared_indices[:] = indices

            start_idx = self.current_file_idx.value
            end_idx = min(start_idx + self.files_per_batch, len(self.parquet_files))

            if start_idx == end_idx:
                return []

            batch_indices = self.shared_indices[start_idx:end_idx]
            batch = [self.parquet_files[i] for i in batch_indices]
            self.current_file_idx.value = end_idx
            return batch


class SonarSentenceDataset(IterableDataset):
    def __init__(
        self,
        parquet_files,
        sequence_length,
        stride,
        shared_file_idx,
        lock,
        shared_indices,
        files_per_batch: int = 10,
        shuffle_sentences: bool = True,
        shuffle_files: bool = True,
        return_article_embedding: bool = False,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.stride = stride
        self.shuffle_sentences = shuffle_sentences
        self.return_article_embedding = return_article_embedding

        # Process-safe file manager
        self.file_manager = FileManager(
            parquet_files,
            files_per_batch,
            shared_file_idx,
            lock,
            shared_indices,
            shuffle_files,
        )

        # Double-buffering state
        self.buffers = [[], []]
        self.active_buffer_idx = 0
        self.buffer_lock = threading.Lock()
        self.data_exhausted = threading.Event()
        self.prefetch_thread = None

    def _fill_buffer(self, buffer_idx):
        """Loads a batch of files into the specified buffer."""
        file_batch = self.file_manager.get_next_batch()
        if not file_batch:
            return True  # No more files

        loaded_sentences = self._load_sentences_from_files(file_batch)

        with self.buffer_lock:
            self.buffers[buffer_idx] = loaded_sentences
        return False

    def _prefetch_loop(self):
        """Background thread loop to keep the inactive buffer full."""
        while not self.data_exhausted.is_set():
            inactive_buffer_idx = 1 - self.active_buffer_idx
            if not self.buffers[inactive_buffer_idx]:
                is_empty = self._fill_buffer(inactive_buffer_idx)
                if is_empty:
                    self.data_exhausted.set()
                    break
            time.sleep(0.5)  # Avoid busy-waiting

    def _load_sentences_from_files(self, file_paths):
        """Load sentences from a list of files."""
        all_data = []

        for file_path in file_paths:
            table = pq.read_table(file_path)

            if self.return_article_embedding:
                # New format: sentence_embedding and article_embedding columns
                sentence_embeddings = table.column("sentence_embedding").to_pylist()
                article_embeddings = table.column("article_embedding").to_pylist()
                for s_emb, a_emb in zip(sentence_embeddings, article_embeddings):
                    all_data.append(
                        (
                            torch.tensor(s_emb, dtype=torch.float),
                            torch.tensor(a_emb, dtype=torch.float),
                        )
                    )
            else:
                embeddings = table.column("sentence_embedding").to_pylist()

                for e in embeddings:
                    all_data.append(torch.tensor(e, dtype=torch.float))

        if self.shuffle_sentences:
            random.shuffle(all_data)

        return all_data

    def __iter__(self):
        # Initial fill of the first buffer
        if self._fill_buffer(self.active_buffer_idx):
            return  # No data to process

        # Start the prefetching thread
        self.prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.prefetch_thread.start()

        sentence_cache = deque()
        # Loop as long as there's potential for data from the prefetch thread,
        # the buffers, or the existing cache.
        while (
            not self.data_exhausted.is_set()
            or any(self.buffers)
            or len(sentence_cache) >= self.sequence_length
        ):
            # 1. Fill sentence_cache from the active buffer
            with self.buffer_lock:
                active_buffer = self.buffers[self.active_buffer_idx]
                if active_buffer:
                    sentence_cache.extend(active_buffer)
                    active_buffer.clear()

            # 2. Swap buffers if the pre-fetched one has data and cache is running low
            if (
                len(sentence_cache) < self.sequence_length
                and not self.data_exhausted.is_set()
            ):
                with self.buffer_lock:
                    if self.buffers[1 - self.active_buffer_idx]:
                        self.active_buffer_idx = 1 - self.active_buffer_idx
                        # Go back to top to fill cache from the new active buffer
                        continue

            # 3. Yield all possible sequences from the cache
            while len(sentence_cache) >= self.sequence_length:
                # Grab the sequence *before* advancing the window
                sequence = list(
                    itertools.islice(sentence_cache, 0, self.sequence_length)
                )
                if self.return_article_embedding:
                    # Unzip list of tuples into two lists
                    sentence_sequence, article_sequence = zip(*sequence)
                    yield torch.stack(sentence_sequence), torch.stack(article_sequence)
                else:
                    yield torch.stack(sequence)

                # Now, advance the window by the stride
                for _ in range(self.stride):
                    sentence_cache.popleft()

            # 4. If all data sources are exhausted and cache is too small, break
            if (
                self.data_exhausted.is_set()
                and not any(self.buffers)
                and len(sentence_cache) < self.sequence_length
            ):
                break

            # 5. If we are waiting for data, sleep a bit to prevent busy-waiting
            if len(sentence_cache) < self.sequence_length:
                time.sleep(0.1)


class Wiki40bDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/wiki40b_sonar_sentences",
        batch_size: int = 32,
        val_split: float = 0.1,
        sequence_length: int = 128,
        stride: int = 64,
        max_samples: Optional[int] = None,
        num_workers: int = 4,
        files_per_batch: int = 10,
        shuffle_sentences: bool = True,
        shuffle_files: bool = True,
        return_article_embedding: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.val_split = val_split
        self.sequence_length = sequence_length
        self.stride = stride
        self.max_samples = max_samples
        self.num_workers = num_workers
        self.files_per_batch = files_per_batch
        self.shuffle_sentences = shuffle_sentences
        self.shuffle_files = shuffle_files
        self.return_article_embedding = return_article_embedding
        self.train_dataset = None
        self.val_dataset = None

        if self.num_workers > 0:
            self.manager = multiprocessing.Manager()
            self.train_shared_file_idx = self.manager.Value("i", 0)
            self.train_lock = self.manager.Lock()
            self.val_shared_file_idx = self.manager.Value("i", 0)
            self.val_lock = self.manager.Lock()

    def setup(self, stage: Optional[str] = None):
        data_dir = self.data_dir / "flat"
        if not os.path.exists(data_dir):
            raise FileNotFoundError(
                f"Flattened data directory not found at {data_dir}.\n"
                f"Please run scripts/flatten_sonar_dataset.py first."
            )

        train_parquet_files = sorted(
            [
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".parquet")
            ]
        )
        if not train_parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        # Split files for train/val
        total_files = len(train_parquet_files)
        val_file_count = int(total_files * self.val_split)
        train_file_count = total_files - val_file_count

        train_files = (
            train_parquet_files[:train_file_count] if train_file_count > 0 else []
        )
        val_files = train_parquet_files[train_file_count:] if val_file_count > 0 else []

        # Create dummy shared state for single-process mode
        if self.num_workers > 0:
            train_indices = self.manager.list(range(len(train_files)))
            val_indices = self.manager.list(range(len(val_files)))
        else:
            self.train_shared_file_idx = multiprocessing.Value("i", 0)
            self.train_lock = multiprocessing.Lock()
            self.val_shared_file_idx = multiprocessing.Value("i", 0)
            self.val_lock = multiprocessing.Lock()
            train_indices = list(range(len(train_files)))
            val_indices = list(range(len(val_files)))

        if self.shuffle_files:
            random.shuffle(train_indices)

        if train_files:
            self.train_dataset = SonarSentenceDataset(
                train_files,
                sequence_length=self.sequence_length,
                stride=self.stride,
                files_per_batch=self.files_per_batch,
                shuffle_sentences=self.shuffle_sentences,
                shuffle_files=self.shuffle_files,
                shared_file_idx=self.train_shared_file_idx,
                lock=self.train_lock,
                shared_indices=train_indices,
                return_article_embedding=self.return_article_embedding,
            )

        if val_files:
            self.val_dataset = SonarSentenceDataset(
                val_files,
                sequence_length=self.sequence_length,
                stride=self.stride,
                files_per_batch=self.files_per_batch,
                shuffle_sentences=False,  # Don't shuffle validation
                shuffle_files=False,
                shared_file_idx=self.val_shared_file_idx,
                lock=self.val_lock,
                shared_indices=val_indices,
                return_article_embedding=self.return_article_embedding,
            )

    def train_dataloader(self):
        if self.train_dataset is None:
            return None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Shuffling handled within dataset
            num_workers=self.num_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )
