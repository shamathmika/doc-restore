# =============================================================================
# File: data/split.py
# Owner: Apoorva Adimulam
#
# Purpose:
#   Splits the full ShabbyPages dataset into train / val / test CSV files
#   used by the DataLoader (data/dataloader.py).
#
# Exports:
#   - split_dataset(shabby_dir: str, output_dir: str,
#                   train_ratio=0.8, val_ratio=0.1) -> None
#       Scans shabby_dir for clean/degraded pairs, shuffles with seed=42,
#       and writes three CSV files to output_dir:
#           train.csv, val.csv, test.csv
#       Each CSV has columns: clean_path, degraded_path
#
# Dependencies:
#   - pandas
#   - pathlib
#
# Notes:
#   - Output CSVs are consumed by data/dataloader.py
#   - Paths in CSVs are relative to project root
# =============================================================================
