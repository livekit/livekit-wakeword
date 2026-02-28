"""Memory-mapped dataset for training with mixed positive/negative batches."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


def mmap_batch_generator(
    data_files: dict[str, str | Path],
    n_per_class: dict[str, int],
    label_funcs: dict[str, Callable[[np.ndarray], int]],
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Generate mixed batches from memory-mapped .npy files.

    Each batch contains samples from each class according to n_per_class.
    Files are memory-mapped so data larger than RAM can be used.

    Yields:
        (features, labels) where features is (batch_size, 16, 96)
        and labels is (batch_size,) with 0/1 values.
    """
    # Memory-map all files
    mmaps: dict[str, np.ndarray] = {}
    for name, path in data_files.items():
        path = Path(path)
        if not path.exists():
            logger.warning(f"Data file not found: {path}, skipping class '{name}'")
            continue
        data = np.load(str(path), mmap_mode="r")
        # Reshape 2D (N, 96) → 3D (N//16, 16, 96) for pre-extracted embeddings
        if data.ndim == 2 and data.shape[1] == 96:
            n_full = (data.shape[0] // 16) * 16
            data = data[:n_full].reshape(-1, 16, 96)
        mmaps[name] = data
        logger.info(f"Loaded {name}: shape={mmaps[name].shape} from {path}")

    if not mmaps:
        raise FileNotFoundError("No data files found for training")

    # Warn about requested classes that have no loaded data
    for name, n in n_per_class.items():
        if n > 0 and name not in mmaps:
            logger.warning(
                f"Class '{name}' requested {n} samples per batch but no data file was loaded"
            )

    # Track position in each file
    positions: dict[str, int] = {name: 0 for name in mmaps}

    while True:
        batch_features: list[np.ndarray] = []
        batch_labels: list[int] = []

        for name, data in mmaps.items():
            n = n_per_class.get(name, 0)
            if n == 0:
                continue

            label_fn = label_funcs[name]
            total = data.shape[0]
            pos = positions[name]

            # Collect n samples, wrapping around if needed
            indices = np.arange(pos, pos + n) % total
            samples = data[indices]

            for sample in samples:
                batch_features.append(sample)
                batch_labels.append(label_fn(sample))

            positions[name] = (pos + n) % total

        if not batch_features:
            break

        features = np.stack(batch_features, axis=0)
        labels = np.array(batch_labels, dtype=np.float32)

        # Shuffle within batch
        perm = np.random.permutation(len(labels))
        yield features[perm], labels[perm]


class WakeWordDataset(IterableDataset):  # type: ignore[type-arg]
    """IterableDataset wrapping mmap_batch_generator for DataLoader."""

    def __init__(
        self,
        data_files: dict[str, str | Path],
        n_per_class: dict[str, int],
        label_funcs: dict[str, Callable[[np.ndarray], int]],
    ):
        self.data_files = data_files
        self.n_per_class = n_per_class
        self.label_funcs = label_funcs

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        gen = mmap_batch_generator(
            data_files=self.data_files,
            n_per_class=self.n_per_class,
            label_funcs=self.label_funcs,
        )
        for features, labels in gen:
            yield (
                torch.from_numpy(features.copy()),
                torch.from_numpy(labels.copy()),
            )


def create_dataloader(
    data_files: dict[str, str | Path],
    n_per_class: dict[str, int],
    label_funcs: dict[str, Callable[[np.ndarray], int]],
    prefetch_factor: int = 16,
    num_workers: int = 0,
) -> DataLoader:  # type: ignore[type-arg]
    """Create a DataLoader from memory-mapped feature files."""
    dataset = WakeWordDataset(
        data_files=data_files,
        n_per_class=n_per_class,
        label_funcs=label_funcs,
    )
    return DataLoader(
        dataset,
        batch_size=None,  # Dataset yields pre-batched data
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
