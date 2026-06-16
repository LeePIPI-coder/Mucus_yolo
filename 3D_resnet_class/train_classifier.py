"""Train 2nd-stage FP reduction classifier (3D ResNet).

A 3D ResNet classifier for reducing false positives from the detector.
Training is organized in two phases:
  Phase 1 (Pilot): Train on folds 1-4, validate on fold 0, test one hyperparameter set.
  Phase 2 (Full 5-fold): Run the best configuration on all 5 folds.

Each epoch performs random downsampling of TPs (one random jitter copy per GT group)
so all GT instances contribute uniformly while maintaining consistent pos:neg ratios.
"""

import argparse
import sys
from pathlib import Path

# Add the current script's directory to path for importing dataset and model modules
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast  # Mixed-precision training
from tqdm import tqdm

from sklearn.metrics import average_precision_score

from dataset import PatchDataset  # Custom 3D patch dataset
from model import build_fp_classifier  # 3D ResNet classifier builder


# ---------------------------------------------------------------------------
# Epoch-level balanced sampling
# ---------------------------------------------------------------------------

def get_epoch_indices(dataset, pos_neg_ratio, epoch, base_seed):
    """Build balanced sample indices for a single epoch. With TP count as baseline: FP = TP * ratio.

    Each epoch keeps all TPs (one random jitter copy per GT group, so each nodule
    contributes exactly one positive sample). FPs are randomly downsampled to
    n_tp * pos_neg_ratio to control the pos:neg ratio.

    Parameters
    ----------
    dataset : PatchDataset
    pos_neg_ratio : int
        Denominator multiplier for pos:neg ratio. 1 = TP:FP = 1:1, 2 = TP:FP = 1:2.
    epoch : int
        Current epoch number, used for seed offset so each epoch samples differently.
    base_seed : int
        Base random seed.

    Returns
    -------
    list[int] — Balanced and shuffled list of dataset indices.
    """
    # Epoch-specific random seed: same seed for same epoch, ensuring reproducibility
    rng = np.random.RandomState(base_seed + epoch)

    fp_indices = list(dataset.fp_indices)
    tp_groups = dataset.tp_gt_groups
    n_fp = len(fp_indices)
    n_gts = len(tp_groups)

    # Randomly select one jitter copy per GT group as the representative positive
    sampled_tp = [rng.choice(group, size=1).item() for group in tp_groups]

    # Sample FPs proportionally to TP count, without replacement
    target_n_fp = n_gts * pos_neg_ratio
    target_n_fp = min(target_n_fp, n_fp)  # If not enough FPs, take all
    sampled_fp = rng.choice(fp_indices, size=target_n_fp, replace=False).tolist()

    # Merge and shuffle to ensure balanced pos/neg mixing per batch
    all_indices = sampled_tp + sampled_fp
    rng.shuffle(all_indices)
    return all_indices


def get_val_indices(dataset, pos_neg_ratio, seed):
    """Build GT-level validation indices: first jitter copy per GT + proportionally sampled FPs.

    Unlike training, TPs use the first copy of each group (deterministic) for
    reproducible validation. FPs are downsampled to n_gts * pos_neg_ratio so
    val_loss is on the same scale as train_loss — preventing class-imbalance
    bias in model selection.

    Parameters
    ----------
    dataset : PatchDataset
    pos_neg_ratio : int
        Denominator multiplier for pos:neg ratio.
    seed : int
        Random seed for reproducible FP sampling.

    Returns
    -------
    list[int] — GT-deduplicated validation indices.
    """
    rng = np.random.RandomState(seed)

    tp_groups = dataset.tp_gt_groups
    fp_indices = list(dataset.fp_indices)
    n_gts = len(tp_groups)

    sampled_tp = [group[0] for group in tp_groups]  # Deterministic: take the first copy

    target_n_fp = n_gts * pos_neg_ratio
    target_n_fp = min(target_n_fp, len(fp_indices))
    sampled_fp = rng.choice(fp_indices, size=target_n_fp, replace=False).tolist()

    return sampled_tp + sampled_fp


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scaler, device):
    """Single epoch training with mixed precision (AMP) and gradient scaling (GradScaler).

    Returns the average BCE loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    n = 0

    for patches, labels in tqdm(loader, desc="  Train", leave=False):
        patches = patches.to(device)
        labels = labels.float().to(device).unsqueeze(1)  # [B, 1] for BCE target shape

        optimizer.zero_grad()
        # Mixed-precision forward: auto-cast to float16 for speed and memory savings
        with autocast('cuda'):
            preds = model(patches)
            # BCEWithLogitsLoss = sigmoid + BCE, numerically more stable
            loss = nn.functional.binary_cross_entropy_with_logits(preds, labels)

        # Gradient scaling: prevents small gradient underflow in float16
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * patches.size(0)
        n += patches.size(0)

    return total_loss / n


@torch.no_grad()
def validate(model, loader, device):
    """Validate: compute BCE loss and AUPRC on the validation set."""
    model.eval()
    total_loss = 0.0
    n = 0
    all_probs = []
    all_labels = []

    for patches, labels in loader:
        patches = patches.to(device)
        labels_flat = labels.float().to(device).unsqueeze(1)

        with autocast('cuda'):
            preds = model(patches)
            loss = nn.functional.binary_cross_entropy_with_logits(preds, labels_flat)

        total_loss += loss.item() * patches.size(0)
        n += patches.size(0)

        probs = torch.sigmoid(preds).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    auprc = average_precision_score(all_labels, all_probs)

    return total_loss / n, auprc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Training entry point: build dataset, model, training loop; save best model and history.

    Training flow:
    1. Load train/validation data for the specified fold
    2. Build 3D ResNet classifier (with optional ImageNet/MedicalNet pretrained weights)
    3. Per epoch: balanced sampling -> train -> validate -> save best model (by val_loss)
    4. Output history.csv (per-epoch metrics) and best_metrics.csv (best epoch summary)
    """
    parser = argparse.ArgumentParser(description="Train 2nd-stage FP reduction classifier")

    # ---- Data parameters ----
    parser.add_argument("--patch_root", default="/data/Class_3D_patch/TP-FP-jitter_4_260521",
                        help="3D patch dataset root directory")
    parser.add_argument("--val_fold", type=int, default=0,
                        help="Fold to hold out for validation (0-4)")

    # ---- Model parameters ----
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"],
                        help="3D ResNet backbone depth")
    parser.add_argument("--pretrained", default="/workspace/3D_resnet_class/MedicalNet_res_weight/resnet_18_23dataset.pth",
                        help="MedicalNet .pth pretrained weights path (None = train from scratch)")

    # ---- Training strategy parameters ----
    parser.add_argument("--pos_neg_ratio", type=int, default=1, choices=[1, 2],
                        help="Pos:neg ratio denominator: 1 = TP:FP = 1:1, 2 = 1:2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Initial learning rate (with CosineAnnealingLR decay)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience: max epochs without val_loss improvement")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                        help="Minimum absolute decrease in val_loss to count as improvement")

    # ---- Regularization parameters ----
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability for the classification head")

    # ---- Training control ----
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", default="/workspace/3D_resnet_class/experiments/test_TP_FP_260521",
                        help="Experiment output root directory")
    parser.add_argument("--exp_name", default="re18_pretrain_neg_1",
                        help="Experiment subdirectory name for distinguishing runs")
    args = parser.parse_args()

    # ---- Set random seeds for reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: backbone={args.backbone}, pretrained={args.pretrained}, "
          f"val_fold={args.val_fold}, ratio=1:{args.pos_neg_ratio}, "
          f"batch={args.batch_size}, epochs={args.epochs}, lr={args.lr}, "
          f"patience={args.patience}, min_delta={args.min_delta}")

    # ---- Build train/validation datasets ----
    all_folds = set(range(5))
    train_folds = sorted(all_folds - {args.val_fold})  # Train folds: 0-4 excluding val fold

    train_ds = PatchDataset(args.patch_root, train_folds, augment=True)   # Train set with data augmentation
    val_ds = PatchDataset(args.patch_root, [args.val_fold], augment=False)  # Val set without augmentation

    val_indices = get_val_indices(val_ds, args.pos_neg_ratio, args.seed)
    val_subset = Subset(val_ds, val_indices)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Train folds: {train_folds}  |  TP={len(train_ds.tp_indices)}, "
          f"FP={len(train_ds.fp_indices)}, GT groups={len(train_ds.tp_gt_groups)}")
    print(f"Val fold:   {args.val_fold}  |  TP={len(val_ds.tp_indices)}, "
          f"FP={len(val_ds.fp_indices)}, GT groups={len(val_ds.tp_gt_groups)}, "
          f"val samples={len(val_subset)}")

    # ---- Build model ----
    model = build_fp_classifier(args.backbone, pretrained=args.pretrained,
                                dropout=args.dropout)
    model = model.to(device)

    # Adam + CosineAnnealing: learning rate decays from lr to near zero
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()  # Mixed-precision gradient scaler

    # ---- Output directory ----
    out_dir = Path(args.output_dir) / args.exp_name / f"fold_{args.val_fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []              # Per-epoch metrics
    best_val_loss = float("inf")  # Best val_loss (lower is better)
    best_epoch = 0
    patience_counter = 0      # Early stopping counter

    # ---- Training loop ----
    for epoch in range(args.epochs):
        # Re-sample training data each epoch (TP sub-sampling + FP random downsampling)
        epoch_indices = get_epoch_indices(
            train_ds, args.pos_neg_ratio, epoch, args.seed)
        epoch_subset = Subset(train_ds, epoch_indices)
        train_loader = DataLoader(epoch_subset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True,
                                  drop_last=True)

        # ---- Train one epoch ----
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)

        # ---- Validate ----
        val_loss, val_auprc = validate(model, val_loader, device)

        scheduler.step()  # Update learning rate each epoch

        # ---- Log metrics ----
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auprc": val_auprc,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(history_row)

        print(f"Epoch {epoch:3d}  |  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_auprc={val_auprc:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # ---- Save best model (by val_loss) ----
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_auprc": val_auprc,
                "train_loss": train_loss,
                "args": vars(args),
            }, out_dir / "best_model.pth")
        else:
            patience_counter += 1

        # ---- Save periodic checkpoint every 10 epochs ----
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_auprc": val_auprc,
                "train_loss": train_loss,
                "args": vars(args),
            }, out_dir / f"checkpoint_epoch_{epoch}.pth")

        # ---- Early stopping check ----
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}: val_loss did not improve "
                  f"for {args.patience} epochs (best={best_val_loss:.4f} at epoch {best_epoch})")
            break

        # Save history after every epoch to prevent loss on crash
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # ---- Save last model ----
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_auprc": val_auprc,
        "train_loss": train_loss,
        "args": vars(args),
    }, out_dir / "last_model.pth")

    # ---- Training complete: print best epoch summary ----
    best = history[best_epoch]
    print(f"Done. Best epoch {best_epoch}  |  "
          f"train_loss={best['train_loss']:.4f}  "
          f"val_loss={best['val_loss']:.4f}  "
          f"val_auprc={best['val_auprc']:.4f}")


if __name__ == "__main__":
    main()
