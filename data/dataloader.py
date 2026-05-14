import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ConditionalResize:
    def __init__(self, size: tuple[int, int]):
        self.size = size
        self._resize = transforms.Resize(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        if img.size == (self.size[1], self.size[0]):
            return img
        return self._resize(img)


class DocRestoreDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        degraded = Image.open(row["degraded_path"]).convert("RGB")
        clean    = Image.open(row["clean_path"]).convert("RGB")

        if self.transform:
            # Sync random state so spatial ops (RandomCrop, RandomFlip, etc.)
            # produce identical results for both images in the pair.
            seed = torch.randint(0, 2**31, (1,)).item()
            random.seed(seed)
            torch.manual_seed(seed)
            degraded = self.transform(degraded)
            random.seed(seed)
            torch.manual_seed(seed)
            clean = self.transform(clean)

        return degraded, clean


def get_dataloader(
    csv_path,
    batch_size,
    shuffle,
    num_workers,
    transform=None,
    input_size: int = 512,
    crop_mode: str = "resize",
):
    """
    crop_mode:
        'random_crop' — RandomCrop at native resolution (use for training).
        'resize'      — ConditionalResize to input_size (use for val/test).
    """
    if transform is None:
        if crop_mode == "random_crop":
            transform = transforms.Compose([
                transforms.RandomCrop(input_size, pad_if_needed=True),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                ConditionalResize((input_size, input_size)),
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
