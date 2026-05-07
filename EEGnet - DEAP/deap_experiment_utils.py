import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from prepare_data import prepare_dataset_single_subject


class DE_Net(nn.Module):
    def __init__(self, nb_classes=2, chans=32, bands=5):
        super().__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 16, (chans, 1), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(0.5),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(16 * bands, 64),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(64, nb_classes),
        )

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.flatten(x)
        return self.fc(x)


def normalize_fold(x_train, x_other):
    mu = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(x_train, axis=0, keepdims=True) + 1e-8
    return (x_train - mu) / std, (x_other - mu) / std


def binarize_and_filter(y_raw_task, threshold=5.0, ambiguous_margin=0.0):
    y_cont = y_raw_task.astype(np.float32)
    keep = np.ones(len(y_cont), dtype=bool)
    if ambiguous_margin > 0:
        keep = np.abs(y_cont - threshold) >= ambiguous_margin
    y_bin = (y_cont > threshold).astype(np.int64)
    return y_bin, keep


def calc_acc_f1(y_true, y_pred):
    acc = float(np.mean(y_true == y_pred))
    f1 = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
    return acc, f1


def make_groupkfold_splits(X, y_bin, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(X, y_bin, groups=groups))


def _fmt_float(x):
    s = f"{x:.4f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def _make_config_key(window_size, step_size, trim_post_sec, use_ratio_start, use_ratio_end):
    return (
        f"w{_fmt_float(window_size)}"
        f"_s{_fmt_float(step_size)}"
        f"_t{_fmt_float(trim_post_sec)}"
        f"_r{_fmt_float(use_ratio_start)}-{_fmt_float(use_ratio_end)}"
    )


def _resolve_cache_path(file_path, cache_root, window_size, step_size, trim_post_sec, use_ratio_start, use_ratio_end):
    cfg_key = _make_config_key(window_size, step_size, trim_post_sec, use_ratio_start, use_ratio_end)
    stem = os.path.splitext(os.path.basename(file_path))[0]
    return os.path.join(cache_root, cfg_key, f"{stem}.npz")


def _try_load_cache(cache_path):
    if not os.path.exists(cache_path):
        return None
    data = np.load(cache_path)
    return data["X"], data["y"], data["groups"]


def _save_cache(cache_path, X, y, groups):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, X=X, y=y, groups=groups)


def load_subject_windows(
    file_path,
    trim_post_sec=3.0,
    use_ratio_start=0.5,
    use_ratio_end=1.0,
    window_size=1.0,
    step_size=0.5,
    cache_root="deap_prepared_cache",
    use_cache=True,
    auto_write_cache=True,
):
    cache_path = _resolve_cache_path(
        file_path,
        cache_root,
        window_size,
        step_size,
        trim_post_sec,
        use_ratio_start,
        use_ratio_end,
    )

    if use_cache:
        cached = _try_load_cache(cache_path)
        if cached is not None:
            return cached

    X, y, groups = prepare_dataset_single_subject(
        file_path,
        window_size=window_size,
        step_size=step_size,
        trim_post_sec=trim_post_sec,
        use_ratio_start=use_ratio_start,
        use_ratio_end=use_ratio_end,
    )

    if X is not None and use_cache and auto_write_cache:
        _save_cache(cache_path, X, y, groups)

    return X, y, groups


def save_binary_confusion_figure(cm, save_path, title, class_names=("Low", "High")):
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    if cm.shape != (2, 2):
        raise ValueError(f"Expected 2x2 confusion matrix, got shape={cm.shape}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1], class_names)
    ax.set_yticks([0, 1], class_names)

    vmax = max(float(np.max(cm)), 1.0)
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            color = "white" if cm[i, j] > (0.55 * vmax) else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=10)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)
