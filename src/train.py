"""
src/train.py — BraTS 2021 3D U-Net training script.

Features
--------
  • --resume <ckpt>     explicit断点续训（恢复 model / optimizer / scheduler /
                        scaler / epoch / best_dice / early-stopping 状态 / wandb run）
  • wandb logging       (loss, per-class Dice, mean Dice, LR, epoch time)
                        resume 时用原 run_id 续接，曲线不断
  • AMP mixed precision (torch.cuda.amp  GradScaler + autocast)
  • CosineAnnealingLR
  • Early stopping      (patience=50, monitors val mean Dice，counter 随 checkpoint 持久化)
  • tqdm progress bars  (per-batch inside each epoch)
  • Checkpoint strategy
      checkpoints/latest.pth          每 epoch 覆盖（crash 保护）
      checkpoints/epoch_NNNN.pth      每 10 epoch 留存
      checkpoints/best_model.pth      val mean Dice 最高时覆盖

Usage
-----
    # 从头训练
    python -m src.train --config configs/train_3d_unet.yaml

    # 断点续训
    python -m src.train --config configs/train_3d_unet.yaml \
                        --resume checkpoints/latest.pth

    # 不用 wandb
    python -m src.train --config configs/train_3d_unet.yaml --no-wandb

Label convention
----------------
After ConvertToMultiChannelBasedOnBratsClassesd the "seg" key has 3 binary
channels:  0=TC (Tumor Core), 1=WT (Whole Tumor), 2=ET (Enhancing Tumor).
The model therefore outputs 3 channels with sigmoid activation.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.utils import set_determinism
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.data import get_data_loaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

NUM_SEG_CHANNELS = 3   # TC / WT / ET


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BraTS 2021 3D U-Net trainer")
    p.add_argument(
        "--config",
        default="configs/train_3d_unet.yaml",
        help="Path to YAML config file",
    )
    p.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help="Path to a checkpoint (.pth) to resume training from "
             "(e.g. checkpoints/latest.pth)",
    )
    p.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        metavar="N",
        help="Override max_epochs from config (handy for dry-run, e.g. --epochs 1)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast / dry-run mode: use only 4 cases, cache_rate=0, batch_size=1. "
             "Automatically enabled when --epochs is given.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Early stopping  (counter state persisted inside checkpoints)
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(
        self,
        patience: int = 50,
        min_delta: float = 1e-4,
        *,
        initial_counter: int = 0,
        initial_best: float = -float("inf"),
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = initial_counter
        self.best_score = initial_best
        self.should_stop = False

    def step(self, score: float) -> bool:
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            log.info(
                "EarlyStopping: no improvement %d/%d (best=%.4f)",
                self.counter, self.patience, self.best_score,
            )
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    best_dice: float,
    early_stopping: EarlyStopping,
    wandb_run_id: str | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_dice": best_dice,
            # early-stopping state
            "es_counter": early_stopping.counter,
            "es_best_score": early_stopping.best_score,
            # wandb run id for resuming the same run
            "wandb_run_id": wandb_run_id,
        },
        path,
    )
    log.info("Checkpoint saved → %s", path)


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
) -> dict[str, Any]:
    """
    Load checkpoint and restore in-place.

    Returns a dict with the scalar fields that callers need:
        start_epoch, best_dice, es_counter, es_best_score, wandb_run_id
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    log.info("Loading checkpoint: %s", path)
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    scaler.load_state_dict(ckpt["scaler_state_dict"])

    resumed_epoch = ckpt["epoch"]
    log.info(
        "Resumed from epoch %d  |  best_dice=%.4f  |  es_counter=%d",
        resumed_epoch,
        ckpt.get("best_dice", 0.0),
        ckpt.get("es_counter", 0),
    )

    return {
        "start_epoch": resumed_epoch + 1,
        "best_dice": ckpt.get("best_dice", 0.0),
        "es_counter": ckpt.get("es_counter", 0),
        "es_best_score": ckpt.get("es_best_score", -float("inf")),
        "wandb_run_id": ckpt.get("wandb_run_id", None),
    }


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    scaler: GradScaler,
    device: torch.device,
    modalities: list[str],
    label_key: str,
    epoch: int,
    max_epochs: int,
) -> float:
    model.train()
    epoch_loss = 0.0

    with tqdm(
        loader,
        desc=f"Epoch {epoch:>3}/{max_epochs} [train]",
        unit="batch",
        leave=False,
    ) as pbar:
        for batch in pbar:
            inputs = torch.cat(
                [batch[mod].to(device) for mod in modalities], dim=1
            )
            labels = batch[label_key].to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                loss = loss_fn(model(inputs), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / len(loader)


@torch.no_grad()
def validate(
    *,
    model: torch.nn.Module,
    loader,
    loss_fn: torch.nn.Module,
    dice_metric: DiceMetric,
    device: torch.device,
    modalities: list[str],
    label_key: str,
    roi_size: list[int],
    sw_batch_size: int,
    epoch: int,
    max_epochs: int,
) -> tuple[float, float, list[float]]:
    """Returns (val_loss, mean_dice, [tc_dice, wt_dice, et_dice])."""
    model.eval()
    val_loss = 0.0
    dice_metric.reset()

    with tqdm(
        loader,
        desc=f"Epoch {epoch:>3}/{max_epochs} [val]  ",
        unit="vol",
        leave=False,
    ) as pbar:
        for batch in pbar:
            inputs = torch.cat(
                [batch[mod].to(device) for mod in modalities], dim=1
            )
            labels = batch[label_key].to(device)

            with autocast("cuda"):
                outputs = sliding_window_inference(
                    inputs=inputs,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                )
                loss = loss_fn(outputs, labels)

            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            dice_metric(y_pred=preds, y=labels)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

    per_class = dice_metric.aggregate()   # shape (3,)
    dice_metric.reset()

    tc, wt, et = per_class[0].item(), per_class[1].item(), per_class[2].item()
    return val_loss / len(loader), per_class.mean().item(), [tc, wt, et]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    ckpt_cfg  = cfg["checkpointing"]
    log_cfg   = cfg["logging"]
    tf_cfg    = cfg.get("transforms", {})

    modalities: list[str] = data_cfg["modalities"]
    label_key:  str       = data_cfg.get("label_key", "seg")
    roi_size:   list[int] = tf_cfg.get("roi_size", [128, 128, 128])
    max_epochs: int       = args.epochs if args.epochs is not None else train_cfg["max_epochs"]

    # fast / dry-run mode: small dataset, no caching, batch_size=1
    fast_mode: bool = args.fast or (args.epochs is not None)
    if fast_mode:
        log.info("*** FAST MODE: 4 cases, cache_rate=0, batch_size=1 ***")
        train_cfg = {**train_cfg, "batch_size": 1}
        cfg = {**cfg, "training": train_cfg}   # propagate to get_data_loaders
    save_dir              = Path(ckpt_cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Reproducibility ───────────────────────────────────────────────────
    set_determinism(seed=args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ── Build objects (before loading checkpoint) ─────────────────────────
    model = UNet(
        spatial_dims=model_cfg["spatial_dims"],
        in_channels=model_cfg["in_channels"],
        out_channels=NUM_SEG_CHANNELS,
        channels=model_cfg["channels"],
        strides=model_cfg["strides"],
        num_res_units=model_cfg.get("num_res_units", 2),
        norm=model_cfg.get("norm", "instance"),
    ).to(device)
    log.info("Parameters: %s", f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    loss_fn = DiceLoss(
        sigmoid=True,
        include_background=True,
        to_onehot_y=False,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
        batch=True,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg["scheduler_params"]["T_max"],
        eta_min=train_cfg["scheduler_params"]["eta_min"],
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    dice_metric = DiceMetric(
        include_background=True,
        reduction="mean_batch",
        get_not_nans=False,
    )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch   = 1
    best_dice     = 0.0
    wandb_run_id: str | None = None
    es_counter    = 0
    es_best_score = -float("inf")

    if args.resume:
        state = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        start_epoch   = state["start_epoch"]
        best_dice     = state["best_dice"]
        wandb_run_id  = state["wandb_run_id"]
        es_counter    = state["es_counter"]
        es_best_score = state["es_best_score"]

    early_stopping = EarlyStopping(
        patience=50,
        initial_counter=es_counter,
        initial_best=es_best_score,
    )

    # ── wandb ─────────────────────────────────────────────────────────────
    use_wandb = log_cfg.get("use_wandb", True) and not args.no_wandb
    if use_wandb:
        import wandb as _wandb

        if wandb_run_id:
            # Reconnect to the same run so the loss curve is continuous
            _wandb.init(
                project=log_cfg.get("project", "brats-biohub"),
                id=wandb_run_id,
                resume="must",
            )
            log.info("wandb: resumed run %s", wandb_run_id)
        else:
            _wandb.init(
                project=log_cfg.get("project", "brats-biohub"),
                config=cfg,
            )
            log.info("wandb: new run %s", _wandb.run.id)

        wandb_run_id = _wandb.run.id   # store/update for subsequent checkpoints

    # ── Data ──────────────────────────────────────────────────────────────
    log.info("Building data loaders …")
    train_loader, val_loader = get_data_loaders(
        cfg,
        cache_rate=0.0 if fast_mode else data_cfg.get("cache_rate", 0.1),
        seed=args.seed,
        max_cases=4 if fast_mode else None,
    )
    log.info("Train batches: %d  |  Val batches: %d", len(train_loader), len(val_loader))

    # ── Training loop ─────────────────────────────────────────────────────
    log.info(
        "Training epochs %d → %d  (early-stop counter starts at %d/%d)",
        start_epoch, max_epochs, es_counter, early_stopping.patience,
    )

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.perf_counter()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scaler=scaler,
            device=device,
            modalities=modalities,
            label_key=label_key,
            epoch=epoch,
            max_epochs=max_epochs,
        )

        val_loss, mean_dice, (tc_dice, wt_dice, et_dice) = validate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            dice_metric=dice_metric,
            device=device,
            modalities=modalities,
            label_key=label_key,
            roi_size=roi_size,
            sw_batch_size=train_cfg.get("batch_size", 2),
            epoch=epoch,
            max_epochs=max_epochs,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        elapsed = time.perf_counter() - t0

        log.info(
            "Epoch %3d/%d | train=%.4f val=%.4f | "
            "Dice mean=%.4f TC=%.4f WT=%.4f ET=%.4f | LR=%.2e | %.0fs",
            epoch, max_epochs,
            train_loss, val_loss,
            mean_dice, tc_dice, wt_dice, et_dice,
            current_lr, elapsed,
        )

        if use_wandb:
            _wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/dice_mean": mean_dice,
                    "val/dice_TC": tc_dice,
                    "val/dice_WT": wt_dice,
                    "val/dice_ET": et_dice,
                    "lr": current_lr,
                    "epoch_time_s": elapsed,
                },
                step=epoch,
            )

        # Shared kwargs for all save_checkpoint calls this epoch
        ckpt_kwargs = dict(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_dice=best_dice,
            early_stopping=early_stopping,
            wandb_run_id=wandb_run_id,
        )

        # Best model
        if mean_dice > best_dice:
            best_dice = mean_dice
            ckpt_kwargs["best_dice"] = best_dice   # update the shared value
            save_checkpoint(save_dir / "best_model.pth", **ckpt_kwargs)
            log.info("  ↑ New best val mean Dice: %.4f", best_dice)

        # Periodic checkpoint every N epochs
        if epoch % ckpt_cfg.get("save_interval", 10) == 0:
            save_checkpoint(
                save_dir / f"epoch_{epoch:04d}.pth", **ckpt_kwargs
            )

        # Always overwrite latest (crash safety)
        save_checkpoint(save_dir / "latest.pth", **ckpt_kwargs)

        # Early stopping
        if early_stopping.step(mean_dice):
            log.info(
                "Early stopping at epoch %d  (best Dice=%.4f)",
                epoch, best_dice,
            )
            break

    log.info("Done. Best val mean Dice: %.4f", best_dice)
    if use_wandb:
        _wandb.finish()


if __name__ == "__main__":
    main()
