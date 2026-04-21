# =============================================================================
# File: train/scheduler.py
# Owner: Shamathmika
#
# Purpose:
#   Learning rate scheduler factory for DocRestore training.
#   Imported by both train/train_docres.py and train/train_nafnet.py.
#
# Exports:
#   - get_scheduler(optimizer, config: dict) -> torch LR scheduler
#
# Dependencies:
#   - torch
# =============================================================================

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR


def get_scheduler(optimizer: Optimizer, config: dict):
    """
    Returns the appropriate learning rate scheduler based on config["scheduler"].

    Supported schedulers:

        "cosine"  — CosineAnnealingLR
            Required keys: epochs
            Decays LR from initial value to 0 over the full training run.

        "step"    — StepLR
            Required keys: step_size, gamma
            Multiplies LR by gamma every step_size epochs.

        "plateau" — ReduceLROnPlateau
            No extra keys required (patience fixed at 5).
            Reduces LR when validation loss stops improving.

    Args:
        optimizer: The optimizer whose LR will be scheduled.
        config:    Training config dict, typically loaded from a YAML file.

    Returns:
        A torch LR scheduler instance.

    Raises:
        ValueError: If config["scheduler"] is not one of the supported values.
    """
    scheduler_type = config.get("scheduler", "").lower()

    if scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=config["epochs"],
        )

    if scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=config["step_size"],
            gamma=config["gamma"],
        )

    if scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            patience=5,
        )

    raise ValueError(
        f"Unknown scheduler '{scheduler_type}'. "
        "Choose from: cosine, step, plateau."
    )
