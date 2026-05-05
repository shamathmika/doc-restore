# =============================================================================
# File: train/colab_utils.py
# Owner: Shamathmika
#
# Purpose:
#   Google Drive sync utilities for Colab training sessions.
#   Mounts Drive and copies checkpoints and loss logs so they survive
#   after the Colab session ends.
#
# Usage:
#   from train.colab_utils import sync_checkpoints_to_drive
#   sync_checkpoints_to_drive()
#
# Exports:
#   sync_checkpoints_to_drive(
#       checkpoint_dir="checkpoints",
#       drive_dir="MyDrive/doc-restore/checkpoints"
#   )
# =============================================================================

import shutil
import sys
from pathlib import Path


def sync_checkpoints_to_drive(
    checkpoint_dir: str = "checkpoints",
    drive_dir: str = "MyDrive/doc-restore/checkpoints",
) -> None:
    """
    Mount Google Drive (if not already mounted) and copy all checkpoint
    files and loss logs to Drive so they persist after the session ends.

    Safe to call multiple times - only copies files that exist locally.
    No-op when not running in Colab.

    Args:
        checkpoint_dir: Local path to the checkpoints directory.
        drive_dir:      Destination path relative to /content/drive/.
    """
    if "google.colab" not in sys.modules:
        print("[colab_utils] Not running in Colab - skipping Drive sync.")
        return

    from google.colab import drive

    mount_point = Path("/content/drive")
    if not (mount_point / "MyDrive").exists():
        print("[colab_utils] Mounting Google Drive ...")
        drive.mount(str(mount_point))

    src = Path(checkpoint_dir)
    dst = mount_point / drive_dir
    dst.mkdir(parents=True, exist_ok=True)

    copied = []
    for f in src.iterdir():
        if f.suffix in {".pth", ".csv"}:
            shutil.copy2(f, dst / f.name)
            copied.append(f.name)

    if copied:
        print(f"[colab_utils] Synced to Drive ({dst}):")
        for name in copied:
            print(f"  {name}")
    else:
        print(f"[colab_utils] Nothing to sync in {src}")


if __name__ == "__main__":
    sync_checkpoints_to_drive()
