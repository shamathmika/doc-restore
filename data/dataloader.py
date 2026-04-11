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
