import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    """3D CT patch dataset for 2nd-stage FP reduction classifier.

    Loads pre-extracted .npy patches from the data preparation pipeline.
    Returns (1, 64, 64, 32) float32 tensors with [0,1] range.

    Parameters
    ----------
    patch_root : str
        Root directory containing fold_0/ ... fold_4/ subdirectories.
    fold_indices : list[int]
        Which folds to include, e.g. [1,2,3,4] for training, [0] for validation.
    augment : bool
        If True, apply random 3D flip augmentation (default: False).
    """

    def __init__(self, patch_root, fold_indices, augment=False):
        self.patch_root = Path(patch_root)
        self.fold_indices = fold_indices
        self.augment = augment

        self.samples = []
        self.patient_keys = []

        for fold in fold_indices:
            meta_path = self.patch_root / f"fold_{fold}" / "metadata.csv"
            df = pd.read_csv(meta_path)
            for _, row in df.iterrows():
                patch_path = self.patch_root / f"fold_{fold}" / row["patch_file"]
                label = int(row["label"])
                s = {
                    "path": str(patch_path),
                    "label": label,
                }
                if label == 1:
                    s["gt_key"] = (
                        str(row["patient_key"]),
                        round(float(row["center_world_x"]), 3),
                        round(float(row["center_world_y"]), 3),
                        round(float(row["center_world_z"]), 3),
                    )
                self.samples.append(s)
                self.patient_keys.append(str(row["patient_key"]))

        self.patient_keys = np.array(self.patient_keys)
        self._build_tp_groups()

    def _build_tp_groups(self):
        """Group TP samples by unique GT: (patient_key, cx, cy, cz).

        Each GT has up to n_jitter copies. Groups are stored as lists of
        dataset indices so the training script can sample 1 per GT per epoch,
        guaranteeing every GT contributes evenly.
        """
        from collections import defaultdict
        groups = defaultdict(list)
        for idx, s in enumerate(self.samples):
            if s["label"] == 1:
                groups[s["gt_key"]].append(idx)
        self._tp_gt_groups = list(groups.values())

    @property
    def labels(self):
        return np.array([s["label"] for s in self.samples])

    @property
    def tp_indices(self):
        """Indices of all TP samples (for epoch-level sub-sampling)."""
        return np.where(self.labels == 1)[0]

    @property
    def fp_indices(self):
        """Indices of all FP samples (for epoch-level sub-sampling)."""
        return np.where(self.labels == 0)[0]

    @property
    def tp_gt_groups(self):
        """List of lists: each inner list is all jitter-copy indices of one GT.

        Use this for epoch-level sampling that guarantees every GT gets
        exactly k copies picked, rather than random-sampling all TP indices
        which can over-represent some GTs and skip others entirely.
        """
        return self._tp_gt_groups

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        patch = np.load(s["path"]).astype(np.float32) / 255.0  # uint8 → [0,1]

        if self.augment:
            patch = self._random_flip(patch)

        patch = patch[np.newaxis, ...]  # (64, 64, 32) → (1, 64, 64, 32)
        return patch, s["label"]

    @staticmethod
    def _random_flip(patch):
        """Random flip along each spatial axis with p=0.5."""
        if np.random.rand() < 0.5:
            patch = patch[::-1, :, :].copy()
        if np.random.rand() < 0.5:
            patch = patch[:, ::-1, :].copy()
        if np.random.rand() < 0.5:
            patch = patch[:, :, ::-1].copy()
        return patch
