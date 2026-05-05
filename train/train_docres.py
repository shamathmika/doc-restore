# =============================================================================
# File: train/train_docres.py
# Owner: Apoorva Adimulam
#
# Purpose:
#   Training script for DocRes on the DocRestore dataset.
#   Identical loop structure to train_nafnet.py so both can be compared.
#
# Usage:
#   # From project root:
#   python train/train_docres.py --config configs/docres.yaml
#
#   # On Colab (run once to add project root to path):
#   import sys; sys.path.insert(0, "/content/doc-restore")
#   !python train/train_docres.py --config configs/docres.yaml
#
# Outputs:
#   checkpoints/docres_best.pth          — best val-loss checkpoint
#   checkpoints/docres_loss_log.csv      — per-epoch train/val losses
# =============================================================================

import argparse
import csv
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path regardless of cwd
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch
import torch.optim as optim
import yaml

from data.dataloader import get_dataloader
from models.docres_wrapper import DocResModel
from train.losses import get_loss
from train.scheduler import get_scheduler

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _open_log(log_path: Path):
    """Open (or append to) a CSV loss log. Returns (file_handle, csv.writer)."""
    is_new = not log_path.exists()
    fh = open(log_path, "a", newline="")
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(["epoch", "train_loss", "val_loss"])
    return fh, writer


# ---------------------------------------------------------------------------
# Per-epoch loops
# ---------------------------------------------------------------------------

def _train_epoch(
    model: DocResModel,
    loader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    for degraded, clean in loader:
        degraded = degraded.to(device, non_blocking=True)
        clean    = clean.to(device, non_blocking=True)

        optimizer.zero_grad()
        restored = model(degraded)
        loss = criterion(restored, clean)
        loss.backward()
        optimizer.step()

        total += loss.item() * degraded.size(0)

    return total / len(loader.dataset)


@torch.no_grad()
def _val_epoch(
    model: DocResModel,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for degraded, clean in loader:
        degraded = degraded.to(device, non_blocking=True)
        clean    = clean.to(device, non_blocking=True)
        restored = model(degraded)
        loss = criterion(restored, clean)
        total += loss.item() * degraded.size(0)
    return total / len(loader.dataset)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config_path: str) -> None:
    torch.manual_seed(SEED)

    cfg    = _load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_docres] Device : {device}")

    # -- Model ---------------------------------------------------------------
    model = DocResModel(config_path).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train_docres] DocResModel  params={n_params:,}")

    # -- Data ----------------------------------------------------------------
    # num_workers=0 required on Windows (spawn can't pickle lambdas); Colab uses 2
    num_workers = 0 if sys.platform == "win32" else 2
    train_loader = get_dataloader(
        cfg["train_csv"],
        batch_size  = cfg["batch_size"],
        shuffle     = True,
        num_workers = num_workers,
    )
    val_loader = get_dataloader(
        cfg["val_csv"],
        batch_size  = cfg["batch_size"],
        shuffle     = False,
        num_workers = num_workers,
    )
    print(f"[train_docres] Train batches={len(train_loader)}  "
          f"Val batches={len(val_loader)}")

    # -- Loss ----------------------------------------------------------------
    loss_kwargs = {}
    if cfg.get("loss") == "combined":
        loss_kwargs["lambda_perceptual"] = cfg.get("lambda_perceptual", 0.1)
    criterion = get_loss(cfg["loss"], **loss_kwargs).to(device)

    # -- Optimizer / scheduler -----------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    scheduler = get_scheduler(optimizer, cfg)

    # -- Outputs -------------------------------------------------------------
    ckpt_dir  = Path(cfg.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "docres_best.pth"
    log_path  = ckpt_dir / "docres_loss_log.csv"

    log_fh, log_writer = _open_log(log_path)
    best_val_loss = float("inf")

    print(f"[train_docres] Training for {cfg['epochs']} epoch(s) ...")
    print(f"[train_docres] Loss={cfg['loss']}  "
          f"LR={cfg['learning_rate']}  Scheduler={cfg['scheduler']}")

    try:
        for epoch in range(1, cfg["epochs"] + 1):
            train_loss = _train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss   = _val_epoch(model, val_loader, criterion, device)

            # Step the LR scheduler (plateau needs the metric)
            if cfg.get("scheduler") == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch:3d}/{cfg['epochs']}]  "
                f"train={train_loss:.6f}  val={val_loss:.6f}  lr={lr:.2e}"
            )

            log_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}"])
            log_fh.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_checkpoint(str(best_ckpt), epoch=epoch, loss=val_loss)

    finally:
        log_fh.close()

    print(
        f"\n[train_docres] Done.  "
        f"Best val loss={best_val_loss:.6f}  Checkpoint={best_ckpt}"
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DocRes for DocRestore")
    parser.add_argument(
        "--config",
        default="configs/docres.yaml",
        help="Path to training config YAML (default: configs/docres.yaml)",
    )
    args = parser.parse_args()
    train(args.config)
