"""Train 2nd-stage FP reduction classifier (3D ResNet).

第二阶段假阳性抑制分类器（3D ResNet），用于降低检测器产出的误报。
训练分为两个阶段：
  Phase 1 (Pilot): 在 fold 1-4 上训练，fold 0 上验证，每次仅测试一组超参数。
  Phase 2 (Full 5-fold): 将最优配置在所有 5 个 fold 上完整运行。

每个 epoch 对 TP 进行随机下采样（每个 GT 组随机选一个 jitter 副本），
确保所有 GT 节点均匀贡献，同时 pos:neg 比例在每个 epoch 保持一致。
"""

import argparse
import sys
from pathlib import Path

# 将当前脚本所在目录加入 path，便于直接 import 同目录下的 dataset 和 model 模块
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.amp import GradScaler, autocast  # 混合精度训练
from tqdm import tqdm

from sklearn.metrics import average_precision_score

from dataset import PatchDataset  # 自定义 3D patch 数据集
from model import build_fp_classifier  # 3D ResNet 分类器构建函数



# ---------------------------------------------------------------------------
# Epoch-level balanced sampling
# ---------------------------------------------------------------------------

def get_epoch_indices(dataset, pos_neg_ratio, epoch, base_seed):
    """构建单个 epoch 的均衡样本索引。以 TP 数量为基准：FP = TP × ratio。

    每个 epoch 保留所有 TP（每个 GT 组随机选一个 jitter 副本，保证每个结节
    只贡献一个正样本）。FP 随机下采样到 n_tp × pos_neg_ratio，用于控制正负比例。

    Parameters
    ----------
    dataset : PatchDataset
    pos_neg_ratio : int
        正负比的分母倍数。1 = TP:FP = 1:1，2 = TP:FP = 1:2。
    epoch : int
        当前 epoch 编号，用于随机种子偏移，使每个 epoch 的采样不同。
    base_seed : int
        基础随机种子。

    Returns
    -------
    list[int] — 均衡采样并打乱后的数据集索引列表。
    """
    # epoch 相关随机种子：同一 epoch 用相同种子，保证可复现
    rng = np.random.RandomState(base_seed + epoch)

    fp_indices = list(dataset.fp_indices)
    tp_groups = dataset.tp_gt_groups
    n_fp = len(fp_indices)
    n_gts = len(tp_groups)

    # 每个 GT 组随机选一个 jitter 副本作为该结节的代表正样本
    sampled_tp = [rng.choice(group, size=1).item() for group in tp_groups]

    # 根据 TP 数量按比例采样 FP，不重复采样
    target_n_fp = n_gts * pos_neg_ratio
    target_n_fp = min(target_n_fp, n_fp)  # 若 FP 不足，取全部 FP
    sampled_fp = rng.choice(fp_indices, size=target_n_fp, replace=False).tolist()

    # 合并后打乱，确保每个 batch 中正负样本混合均匀
    all_indices = sampled_tp + sampled_fp
    rng.shuffle(all_indices)
    return all_indices


def get_val_indices(dataset, pos_neg_ratio, seed):
    """构建验证集的 GT-level 采样索引：每个 GT 取第一个 jitter 副本 + 按比例采样的 FP。

    与训练集不同的是：TP 取每组第一个（确定性），保证验证可复现。
    FP 按 n_gts × pos_neg_ratio 下采样，使 val_loss 与 train_loss 在同一尺度上，
    避免模型选择被多数类主导。

    Parameters
    ----------
    dataset : PatchDataset
    pos_neg_ratio : int
        正负比的分母倍数。
    seed : int
        随机种子，用于 FP 采样的可复现性。

    Returns
    -------
    list[int] — GT 去重后的验证集索引列表。
    """
    rng = np.random.RandomState(seed)

    tp_groups = dataset.tp_gt_groups
    fp_indices = list(dataset.fp_indices)
    n_gts = len(tp_groups)

    sampled_tp = [group[0] for group in tp_groups]  # 确定性取第一个

    target_n_fp = n_gts * pos_neg_ratio
    target_n_fp = min(target_n_fp, len(fp_indices))
    sampled_fp = rng.choice(fp_indices, size=target_n_fp, replace=False).tolist()

    return sampled_tp + sampled_fp


# ---------------------------------------------------------------------------
# Training / validation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scaler, device):
    """单 epoch 训练，使用混合精度 (AMP) 和梯度缩放 (GradScaler)。

    返回该 epoch 的平均 BCE 损失。
    """
    model.train()
    total_loss = 0.0
    n = 0

    for patches, labels in tqdm(loader, desc="  Train", leave=False):
        patches = patches.to(device)
        labels = labels.float().to(device).unsqueeze(1)  # [B, 1]，BCE 目标形状

        optimizer.zero_grad()
        # 混合精度前向：自动将计算转为 float16 以加速和节省显存
        with autocast('cuda'):
            preds = model(patches)
            # BCEWithLogitsLoss = sigmoid + BCE，数值更稳定
            loss = nn.functional.binary_cross_entropy_with_logits(preds, labels)

        # 梯度缩放：防止 float16 下小梯度下溢
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * patches.size(0)
        n += patches.size(0)

    return total_loss / n


@torch.no_grad()
def validate(model, loader, device):
    """验证函数：计算验证集的 BCE 损失和 AUPRC。"""
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
    """训练主函数：构建数据集、模型、训练循环，输出最佳模型和历史记录。

    训练流程：
    1. 加载指定 fold 的训练/验证数据
    2. 构建 3D ResNet 分类器（可选 ImageNet/MedicalNet 预训练权重）
    3. 每个 epoch：均衡采样 → 训练 → 验证 → 保存最佳模型（按 Sens@1,2,4 FP 均值）
    4. 输出 history.csv（每轮指标）和 best_metrics.csv（最佳 epoch 汇总）
    """
    parser = argparse.ArgumentParser(description="Train 2nd-stage FP reduction classifier")

    # ---- 数据参数 ----
    parser.add_argument("--patch_root", default="/data/Class_3D_patch/TP-FP-jitter_4_260521",
                        help="3D patch 数据集根目录")
    parser.add_argument("--val_fold", type=int, default=0,
                        help="留作验证的 fold 编号 (0-4)")

    # ---- 模型参数 ----
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18", "resnet34"],
                        help="3D ResNet 主干网络深度")
    parser.add_argument("--pretrained", default="/workspace/3D_resnet_class/MedicalNet_res_weight/resnet_18_23dataset.pth",
                        help="MedicalNet .pth 预训练权重路径 (None = 从头训练)")

    # ---- 训练策略参数 ----
    parser.add_argument("--pos_neg_ratio", type=int, default=1, choices=[1, 2],
                        help="正负样本比的分母倍数: 1 = TP:FP = 1:1, 2 = 1:2")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="初始学习率（配合 CosineAnnealingLR 衰减）")
    parser.add_argument("--patience", type=int, default=20,
                        help="早停耐心值：val_loss 连续不改善的 epoch 数上限")
    parser.add_argument("--min_delta", type=float, default=1e-4,
                        help="val_loss 改善的最小绝对下降量阈值")

    # ---- 正则化参数 ----
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="分类头 Dropout 概率")

    # ---- 训练控制 ----
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，用于可复现性")
    parser.add_argument("--output_dir", default="/workspace/3D_resnet_class/experiments/test_TP_FP_260521",
                        help="实验输出根目录")
    parser.add_argument("--exp_name", default="re18_pretrain_neg_1",
                        help="实验子目录名，用于区分不同配置的运行结果")
    args = parser.parse_args()

    # ---- 设定随机种子，保证可复现 ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: backbone={args.backbone}, pretrained={args.pretrained}, "
          f"val_fold={args.val_fold}, ratio=1:{args.pos_neg_ratio}, "
          f"batch={args.batch_size}, epochs={args.epochs}, lr={args.lr}, "
          f"patience={args.patience}, min_delta={args.min_delta}")

    # ---- 构建训练/验证数据集 ----
    all_folds = set(range(5))
    train_folds = sorted(all_folds - {args.val_fold})  # 训练 fold：0-4 中排除验证 fold

    train_ds = PatchDataset(args.patch_root, train_folds, augment=True)   # 训练集开启数据增强
    val_ds = PatchDataset(args.patch_root, [args.val_fold], augment=False)  # 验证集关闭增强

    val_indices = get_val_indices(val_ds, args.pos_neg_ratio, args.seed)
    val_subset = Subset(val_ds, val_indices)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    print(f"Train folds: {train_folds}  |  TP={len(train_ds.tp_indices)}, "
          f"FP={len(train_ds.fp_indices)}, GT groups={len(train_ds.tp_gt_groups)}")
    print(f"Val fold:   {args.val_fold}  |  TP={len(val_ds.tp_indices)}, "
          f"FP={len(val_ds.fp_indices)}, GT groups={len(val_ds.tp_gt_groups)}, "
          f"val samples={len(val_subset)}")

    # ---- 构建模型 ----
    model = build_fp_classifier(args.backbone, pretrained=args.pretrained,
                                dropout=args.dropout)
    model = model.to(device)

    # Adam + CosineAnnealing：学习率从 lr 衰减到接近 0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()  # 混合精度梯度缩放器

    # ---- 输出目录 ----
    out_dir = Path(args.output_dir) / args.exp_name / f"fold_{args.val_fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []              # 记录每个 epoch 的指标
    best_val_loss = float("inf")  # 最佳 val_loss（越低越好）
    best_epoch = 0
    patience_counter = 0      # 早停计数器

    # ---- 训练循环 ----
    for epoch in range(args.epochs):
        # 每个 epoch 重新均衡采样训练数据（TP 子采样 + FP 随机下采样）
        epoch_indices = get_epoch_indices(
            train_ds, args.pos_neg_ratio, epoch, args.seed)
        epoch_subset = Subset(train_ds, epoch_indices)
        train_loader = DataLoader(epoch_subset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=True,
                                  drop_last=True)

        # ---- 训练一个 epoch ----
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)

        # ---- 验证 ----
        val_loss, val_auprc = validate(model, val_loader, device)

        scheduler.step()  # 每个 epoch 更新学习率

        # ---- 记录指标 ----
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

        # ---- 保存最佳模型（依据 val_loss） ----
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

        # ---- 每 10 轮保存 periodic checkpoint ----
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

        # ---- 早停检查 ----
        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}: val_loss did not improve "
                  f"for {args.patience} epochs (best={best_val_loss:.4f} at epoch {best_epoch})")
            break

        # 每个 epoch 结束后都保存历史记录，防止中途崩溃丢失
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    # ---- 保存 last model ----
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_auprc": val_auprc,
        "train_loss": train_loss,
        "args": vars(args),
    }, out_dir / "last_model.pth")

    # ---- 训练完成，输出最佳 epoch 信息 ----
    best = history[best_epoch]
    print(f"Done. Best epoch {best_epoch}  |  "
          f"train_loss={best['train_loss']:.4f}  "
          f"val_loss={best['val_loss']:.4f}  "
          f"val_auprc={best['val_auprc']:.4f}")


if __name__ == "__main__":
    main()
