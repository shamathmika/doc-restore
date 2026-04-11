# =============================================================================
# File: data/dataloader.py
# Owner: Shamathmika
#
# Purpose:
#   Provides a PyTorch Dataset and DataLoader for the DocRestore training
#   pipeline. Used by both train_docres.py and train_nafnet.py — the interface
#   here must not change without coordination.
#
# Exports:
#   - DocRestoreDataset(csv_path, transform=None)
#       torch.utils.data.Dataset subclass.
#       csv_path: path to train.csv / val.csv / test.csv (produced by
#                 data/split.py). Columns: clean_path, degraded_path.
#       __len__: returns number of image pairs.
#       __getitem__(idx): loads both images as PIL, applies transform,
#                         returns (degraded_tensor, clean_tensor) as float32
#                         tensors in [0, 1], shape (3, H, W).
#
#   - get_dataloader(csv_path, batch_size, shuffle, num_workers,
#                    transform=None) -> DataLoader
#       Wraps DocRestoreDataset in a DataLoader.
#       Default transform (when None): Resize(256, 256) then ToTensor().
#
# Dependencies:
#   - torch, torchvision
#   - Pillow (PIL)
#   - pandas
#
# Notes:
#   - All random ops use seed=42 for reproducibility
#   - Images are loaded as RGB
# =============================================================================

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DocRestoreDataset(Dataset):
    """
    Dataset of clean/degraded document image pairs for DocRestore training.

    Reads a CSV file with columns `clean_path` and `degraded_path` and exposes
    each row as a (degraded_tensor, clean_tensor) pair suitable for supervised
    image restoration training.
    """

    def __init__(self, csv_path, transform=None):
        """
        Args:
            csv_path: Path to a CSV file with columns clean_path, degraded_path.
                      Produced by data/split.py.
            transform: Optional torchvision transform applied to both images.
                       When None, caller is responsible for converting to tensor.
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        """Returns the number of image pairs in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Loads and returns a single (degraded, clean) image pair.

        Both images are opened as RGB PIL Images, then the transform (if any)
        is applied to each independently before returning.

        Returns:
            Tuple of (degraded_tensor, clean_tensor), each float32 in [0, 1]
            with shape (3, H, W) when the default transform is used.
        """
        row = self.df.iloc[idx]
        degraded = Image.open(row["degraded_path"]).convert("RGB")
        clean = Image.open(row["clean_path"]).convert("RGB")

        if self.transform:
            degraded = self.transform(degraded)
            clean = self.transform(clean)

        return degraded, clean


def get_dataloader(csv_path, batch_size, shuffle, num_workers, transform=None):
    """
    Builds and returns a DataLoader wrapping DocRestoreDataset.

    Args:
        csv_path:    Path to a split CSV file (train.csv / val.csv / test.csv).
        batch_size:  Number of image pairs per batch.
        shuffle:     Whether to shuffle the dataset each epoch.
        num_workers: Number of worker processes for data loading.
        transform:   torchvision transform to apply to each image. Defaults to
                     Resize(256, 256) followed by ToTensor().

    Returns:
        torch.utils.data.DataLoader yielding (degraded, clean) batches.
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    dataset = DocRestoreDataset(csv_path, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=lambda _: torch.manual_seed(42),
    )
