import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from deap_experiment_utils import load_subject_windows


class EEGNetRegressor(nn.Module):
    def __init__(self, chans=32, bands=5, f1=16, d=2, f2=32, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * d, (chans, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1 * d, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            z = torch.zeros(1, 1, chans, bands)
            z = self.block1(z)
            z = self.block2(z)
            flat = z.numel()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.head(x)


def map_labels(y, label_space):
    y = np.asarray(y, dtype=np.float32)
    if label_space == "0_9":
        return np.clip(y, 0.0, 9.0)
    if label_space == "neg1_1":
        return (y - 5.0) / 4.0
    raise ValueError("label_space must be one of: 0_9, neg1_1")


def to_neg1_1(y, label_space):
    y = np.asarray(y, dtype=np.float32)
    if label_space == "neg1_1":
        return y
    # 0..9 -> -1..1
    return np.clip((y / 4.5) - 1.0, -1.0, 1.0)


def fit_scaler(X_train):
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-8
    return mean, std


def apply_scaler(X, mean, std):
    return (X - mean) / std


def regression_metrics(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred, multioutput="uniform_average"))

    mse_v = float(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
    mae_v = float(mean_absolute_error(y_true[:, 0], y_pred[:, 0]))
    rmse_v = float(np.sqrt(mse_v))
    r2_v = float(r2_score(y_true[:, 0], y_pred[:, 0]))

    mse_a = float(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
    mae_a = float(mean_absolute_error(y_true[:, 1], y_pred[:, 1]))
    rmse_a = float(np.sqrt(mse_a))
    r2_a = float(r2_score(y_true[:, 1], y_pred[:, 1]))

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "valence": {"mse": mse_v, "mae": mae_v, "rmse": rmse_v, "r2": r2_v},
        "arousal": {"mse": mse_a, "mae": mae_a, "rmse": rmse_a, "r2": r2_a},
    }




def high_low_accuracy(y_true, y_pred, label_space):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)

    if label_space == "0_9":
        t_true = y_true > 5.0
        t_pred = y_pred > 5.0
    else:
        t_true = y_true > 0.0
        t_pred = y_pred > 0.0

    acc_v = float(np.mean(t_true[:, 0] == t_pred[:, 0]))
    acc_a = float(np.mean(t_true[:, 1] == t_pred[:, 1]))
    acc_overall = float(np.mean(t_true == t_pred))
    return {"overall": acc_overall, "valence": acc_v, "arousal": acc_a}

def eval_reg(model, loader, device, loss_fn):
    model.eval()
    losses = []
    y_true, y_pred = [], []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            loss = loss_fn(out, by)
            losses.append(float(loss.item()))
            y_true.append(by.cpu().numpy())
            y_pred.append(out.cpu().numpy())
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    metrics = regression_metrics(y_true, y_pred)
    return float(np.mean(losses)), metrics, y_true, y_pred


def build_s01_dataset_from_cache(
    data_dir,
    subject_file,
    cache_root,
    window_sec,
    step_sec,
    trim_post_sec,
    use_ratio_start,
    use_ratio_end,
    channel_indices,
):
    file_path = os.path.join(data_dir, subject_file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    X, y, groups = load_subject_windows(
        file_path,
        trim_post_sec=trim_post_sec,
        use_ratio_start=use_ratio_start,
        use_ratio_end=use_ratio_end,
        window_size=window_sec,
        step_size=step_sec,
        cache_root=cache_root,
        use_cache=True,
        auto_write_cache=True,
    )

    if X is None or len(X) == 0:
        raise RuntimeError(f"No valid samples for {subject_file}")

    if channel_indices is not None:
        X = X[:, channel_indices, :]

    return X.astype(np.float32), y[:, :2].astype(np.float32), groups.astype(np.int64)


def _quadrant_mismatch_mask(actual, pred):
    return np.sign(actual) != np.sign(pred)


def plot_actual_vs_pred_quadrants(y_true, y_pred, label_space, save_path):
    yt = to_neg1_1(y_true, label_space)
    yp = to_neg1_1(y_pred, label_space)

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), dpi=150)
    tasks = [(1, "Arousal"), (0, "Valence")]

    for ax, (idx, name) in zip(axes, tasks):
        x = yt[:, idx]
        y = yp[:, idx]
        mismatch = _quadrant_mismatch_mask(x, y)

        ax.axvspan(0, 1, ymin=0.5, ymax=1.0, color="#8ecae6", alpha=0.35)
        ax.axvspan(-1, 0, ymin=0.0, ymax=0.5, color="#8ecae6", alpha=0.35)

        ax.scatter(x[~mismatch], y[~mismatch], s=24, c="#3b82f6", alpha=0.75)
        if np.any(mismatch):
            ax.scatter(x[mismatch], y[mismatch], s=28, c="#ef4444", alpha=0.9)

        ax.plot([-1, 1], [-1, 1], color="#1d4ed8", linewidth=1.2)
        ax.axhline(0, color="#222", linewidth=0.8)
        ax.axvline(0, color="#222", linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(f"Actual {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"Residual for {name}")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train EEGNet regression for DEAP single-subject (S01 by default)")
    parser.add_argument("--data-dir", type=str, default="C:/Users/xinji/Desktop/data_preprocessed_python")
    parser.add_argument("--subject-file", type=str, default="s01.dat")
    parser.add_argument("--cache-root", type=str, default="deap_prepared_cache")
    parser.add_argument("--out-dir", type=str, default="realtime_eegnet_regression_deap_s01")

    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--trim-post-sec", type=float, default=3.0)
    parser.add_argument("--use-ratio-start", type=float, default=0.5)
    parser.add_argument("--use-ratio-end", type=float, default=1.0)

    parser.add_argument("--channel-indices", type=str, default="", help="e.g. 0,1,2 ; empty means all channels")
    parser.add_argument("--label-space", type=str, default="0_9", choices=["0_9", "neg1_1"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    channel_indices = None
    if args.channel_indices.strip():
        channel_indices = [int(x.strip()) for x in args.channel_indices.split(",") if x.strip()]

    X, y, groups = build_s01_dataset_from_cache(
        data_dir=args.data_dir,
        subject_file=args.subject_file,
        cache_root=args.cache_root,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        trim_post_sec=args.trim_post_sec,
        use_ratio_start=args.use_ratio_start,
        use_ratio_end=args.use_ratio_end,
        channel_indices=channel_indices,
    )

    y = map_labels(y, args.label_space)
    n_chans = X.shape[1]

    print(f"Loaded subject: {args.subject_file}")
    print(f"X={X.shape}, y={y.shape}, n_trials={len(np.unique(groups))}, chans={n_chans}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    tr_idx, va_idx = next(gss.split(X, y, groups=groups))

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]

    y_train_mean = np.mean(y_tr, axis=0, keepdims=True)

    mean, std = fit_scaler(X_tr)
    X_tr = apply_scaler(X_tr, mean, std)
    X_va = apply_scaler(X_va, mean, std)

    tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1), torch.tensor(y_tr, dtype=torch.float32))
    va_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32).unsqueeze(1), torch.tensor(y_va, dtype=torch.float32))
    tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EEGNetRegressor(chans=n_chans, bands=5).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_mse = float("inf")
    best_train_metrics = None
    bad = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(bx)
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
            tr_losses.append(float(loss.item()))

        train_mse = float(np.mean(tr_losses))
        _, train_metrics, _, _ = eval_reg(model, tr_loader, device, loss_fn)
        _, val_metrics, _, _ = eval_reg(model, va_loader, device, loss_fn)
        print(
            f"Epoch {epoch:03d} | Train MSE {train_mse:.4f} R2 {train_metrics['r2']:.4f} || "
            f"Val MSE {val_metrics['mse']:.4f} MAE {val_metrics['mae']:.4f} "
            f"RMSE {val_metrics['rmse']:.4f} R2 {val_metrics['r2']:.4f}"
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_train_metrics = train_metrics
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, val_metrics, y_true, y_pred = eval_reg(model, va_loader, device, loss_fn)
    hl_acc = high_low_accuracy(y_true, y_pred, args.label_space)

    y_pred_baseline = np.repeat(y_train_mean, repeats=len(y_true), axis=0)
    baseline_metrics = regression_metrics(y_true, y_pred_baseline)

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, "eegnet_regressor_best.pt")
    scaler_path = os.path.join(args.out_dir, "eegnet_regressor_scaler.npz")
    pred_path = os.path.join(args.out_dir, "val_predictions.npz")
    fig_path = os.path.join(args.out_dir, "actual_vs_pred_quadrants.png")
    summary_path = os.path.join(args.out_dir, "summary.json")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "chans": n_chans,
            "bands": 5,
            "label_space": args.label_space,
            "window_sec": args.window_sec,
            "step_sec": args.step_sec,
            "trim_post_sec": args.trim_post_sec,
            "use_ratio_start": args.use_ratio_start,
            "use_ratio_end": args.use_ratio_end,
            "channel_indices": channel_indices if channel_indices is not None else list(range(n_chans)),
            "subject_file": args.subject_file,
        },
        model_path,
    )
    np.savez_compressed(scaler_path, mean=mean.astype(np.float32), std=std.astype(np.float32))
    np.savez_compressed(pred_path, y_true=y_true.astype(np.float32), y_pred=y_pred.astype(np.float32))

    plot_actual_vs_pred_quadrants(y_true, y_pred, args.label_space, fig_path)

    summary = {
        "subject_file": args.subject_file,
        "train_metrics_at_best_val": best_train_metrics,
        "val_metrics": val_metrics,
        "baseline_metrics": baseline_metrics,
        "high_low_accuracy": hl_acc,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "label_space": args.label_space,
        "train_target_mean": y_train_mean.squeeze(0).tolist(),
        "model_path": model_path,
        "scaler_path": scaler_path,
        "plot_path": fig_path,
        "cache_root": args.cache_root,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Final Validation ===")
    if best_train_metrics is not None:
        print(
            f"Train@best | MSE={best_train_metrics['mse']:.4f} MAE={best_train_metrics['mae']:.4f} "
            f"RMSE={best_train_metrics['rmse']:.4f} R2={best_train_metrics['r2']:.4f}"
        )
    print(
        f"Model    | MSE={val_metrics['mse']:.4f} MAE={val_metrics['mae']:.4f} "
        f"RMSE={val_metrics['rmse']:.4f} R2={val_metrics['r2']:.4f}"
    )
    print(
        f"Baseline | MSE={baseline_metrics['mse']:.4f} MAE={baseline_metrics['mae']:.4f} "
        f"RMSE={baseline_metrics['rmse']:.4f} R2={baseline_metrics['r2']:.4f}"
    )
    print(
        f"Valence  | R2={val_metrics['valence']['r2']:.4f} "
        f"(baseline {baseline_metrics['valence']['r2']:.4f})"
    )
    print(
        f"Arousal  | R2={val_metrics['arousal']['r2']:.4f} "
        f"(baseline {baseline_metrics['arousal']['r2']:.4f})"
    )

    print("\n=== Indicator Table (Model) ===")
    print(f"MSE: {val_metrics['mse']:.4f}")
    print(f"RMSE: {val_metrics['rmse']:.4f}")
    print(f"MAE: {val_metrics['mae']:.4f}")
    print(f"R2: {val_metrics['r2']:.4f}")
    print(f"Accuracy of High/Low (overall): {hl_acc['overall'] * 100:.1f}%")
    print(f"Accuracy of High/Low (valence): {hl_acc['valence'] * 100:.1f}%")
    print(f"Accuracy of High/Low (arousal): {hl_acc['arousal'] * 100:.1f}%")

    print(f"Saved model: {model_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Saved plot: {fig_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
