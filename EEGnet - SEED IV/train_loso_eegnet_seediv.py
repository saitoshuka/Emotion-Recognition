import os
import copy
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit
from sklearn.metrics import f1_score, confusion_matrix

from load_data import SEEDIVLoader


class EEGNetDE(nn.Module):
    def __init__(
        self,
        nb_classes=4,
        chans=62,
        samples=5,
        dropout=0.5,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 3), padding=(0, 1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.depthwise_conv = nn.Conv2d(16, 32, (chans, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.activation = nn.ELU()
        self.separable_conv = nn.Conv2d(32, 32, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(32 * 1 * samples, nb_classes)

    def forward(self, x):
        x = self.batchnorm1(self.conv1(x))
        x = self.activation(self.batchnorm2(self.depthwise_conv(x)))
        x = self.dropout(x)
        x = self.activation(self.batchnorm3(self.separable_conv(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def build_window_tensor(features_list, labels, scaler=None, target_time=42, trial_id_offset=0):
    mats, y_list, tid_list = [], [], []
    for i, trial in enumerate(features_list):
        if trial.shape[1] > target_time:
            trial = trial[:, :target_time, :]
        elif trial.shape[1] < target_time:
            pad = np.repeat(trial[:, -1:, :], target_time - trial.shape[1], axis=1)
            trial = np.concatenate([trial, pad], axis=1)
        trial_windows = trial.transpose(1, 0, 2)  # (T, 62, 5)
        mats.append(trial_windows)
        y_list.append(np.full(target_time, int(labels[i]), dtype=np.int64))
        tid_list.append(np.full(target_time, i + trial_id_offset, dtype=np.int64))

    x = np.concatenate(mats, axis=0)  # (n_trials * T, 62, 5)
    y = np.concatenate(y_list, axis=0)
    tids = np.concatenate(tid_list, axis=0)

    if scaler is None:
        mean = x.mean(axis=0, keepdims=True)  # (1, 62, 5)
        std = x.std(axis=0, keepdims=True) + 1e-7
        scaler = (mean, std)
    else:
        mean, std = scaler
    x = (x - mean) / std

    x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y, dtype=torch.long)
    tid_t = torch.tensor(tids, dtype=torch.long)
    return x_t, y_t, tid_t, scaler


@torch.no_grad()
def eval_loader(model, loader, device, criterion=None):
    model.eval()
    y_true_win, y_pred_win = [], []
    trial_sum_logits = {}
    trial_count = {}
    trial_true = {}
    correct, total = 0, 0
    total_loss = 0.0
    for bx, by, bt in loader:
        bx = bx.to(device)
        by = by.to(device)
        out = model(bx)
        pred = out.argmax(1)
        if criterion is not None:
            total_loss += criterion(out, by).item() * bx.size(0)
        correct += (pred == by).sum().item()
        total += by.numel()
        y_true_win.append(by.cpu().numpy())
        y_pred_win.append(pred.cpu().numpy())

        out_np = out.cpu().numpy()
        by_np = by.cpu().numpy()
        bt_np = bt.cpu().numpy()
        for i in range(len(bt_np)):
            tid = int(bt_np[i])
            if tid not in trial_sum_logits:
                trial_sum_logits[tid] = out_np[i].copy()
                trial_count[tid] = 1
                trial_true[tid] = int(by_np[i])
            else:
                trial_sum_logits[tid] += out_np[i]
                trial_count[tid] += 1

    y_true_win = np.concatenate(y_true_win)
    y_pred_win = np.concatenate(y_pred_win)
    window_acc = correct / total if total else 0.0
    window_f1 = f1_score(y_true_win, y_pred_win, average="macro", zero_division=0)
    window_cm = confusion_matrix(y_true_win, y_pred_win, labels=[0, 1, 2, 3])

    y_true_trial, y_pred_trial = [], []
    for tid in sorted(trial_sum_logits.keys()):
        logits = trial_sum_logits[tid] / trial_count[tid]
        y_true_trial.append(trial_true[tid])
        y_pred_trial.append(int(np.argmax(logits)))
    y_true_trial = np.array(y_true_trial)
    y_pred_trial = np.array(y_pred_trial)
    trial_acc = float(np.mean(y_true_trial == y_pred_trial))
    trial_f1 = f1_score(y_true_trial, y_pred_trial, average="macro", zero_division=0)
    trial_cm = confusion_matrix(y_true_trial, y_pred_trial, labels=[0, 1, 2, 3])

    return {
        "window_loss": total_loss / total if total and criterion is not None else 0.0,
        "window_acc": window_acc,
        "window_f1": window_f1,
        "window_cm": window_cm,
        "trial_acc": trial_acc,
        "trial_f1": trial_f1,
        "trial_cm": trial_cm,
    }


def train_one_fold(
    x_train_trials,
    y_train_trials,
    x_test_trials,
    y_test_trials,
    device,
    batch_size=64,
    max_epochs=120,
    lr=1e-3,
    weight_decay=1e-3,
    patience=20,
    min_delta=1e-3,
    warmup_epochs=10,
    val_ratio=0.15,
    seed=42,
):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr_idx, va_idx = next(sss.split(np.zeros_like(y_train_trials), y_train_trials))

    x_tr_trials = [x_train_trials[i] for i in tr_idx]
    y_tr = y_train_trials[tr_idx]
    x_va_trials = [x_train_trials[i] for i in va_idx]
    y_va = y_train_trials[va_idx]

    x_tr, y_tr_t, tid_tr, scaler = build_window_tensor(x_tr_trials, y_tr, scaler=None)
    x_va, y_va_t, tid_va, _ = build_window_tensor(x_va_trials, y_va, scaler=scaler)
    x_te, y_te_t, tid_te, _ = build_window_tensor(x_test_trials, y_test_trials, scaler=scaler)

    train_loader = DataLoader(TensorDataset(x_tr, y_tr_t), batch_size=batch_size, shuffle=True)
    train_eval_loader = DataLoader(TensorDataset(x_tr, y_tr_t, tid_tr), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(x_va, y_va_t, tid_va), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_te, y_te_t, tid_te), batch_size=batch_size, shuffle=False)

    model = EEGNetDE(nb_classes=4, chans=62, samples=5, dropout=0.5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=7)

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        tr_loss = 0.0
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * bx.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for bx, by, _ in val_loader:
                bx = bx.to(device)
                by = by.to(device)
                out = model(bx)
                va_loss += criterion(out, by).item() * bx.size(0)
        va_loss /= len(val_loader.dataset)
        scheduler.step(va_loss)

        if epoch == 1 or epoch % 10 == 0:
            tr_metrics = eval_loader(model, train_eval_loader, device, criterion=criterion)
            va_metrics = eval_loader(model, val_loader, device, criterion=criterion)
            print(
                f"Epoch {epoch:03d} | Train Loss {tr_loss:.4f} Trial Acc {tr_metrics['trial_acc']:.4f} "
                f"Trial F1 {tr_metrics['trial_f1']:.4f} || "
                f"Val Loss {va_loss:.4f} Trial Acc {va_metrics['trial_acc']:.4f} "
                f"Trial F1 {va_metrics['trial_f1']:.4f} | LR {optimizer.param_groups[0]['lr']:.6f}"
            )

        if epoch >= warmup_epochs:
            if va_loss < best_val_loss - min_delta:
                best_val_loss = va_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
        else:
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = eval_loader(model, train_eval_loader, device)
    test_metrics = eval_loader(model, test_loader, device)
    return {
        "train_acc": train_metrics["trial_acc"],
        "train_f1": train_metrics["trial_f1"],
        "test_acc": test_metrics["trial_acc"],
        "test_f1": test_metrics["trial_f1"],
        "test_cm": test_metrics["trial_cm"],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def _plot_confusion(cm, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    names = ["Neut", "Sad", "Fear", "Happy"]
    plt.xticks(np.arange(4), names)
    plt.yticks(np.arange(4), names)
    for i in range(4):
        for j in range(4):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def save_loso_report(subjects, accs, f1s, cms, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))
    mean_f1 = float(np.mean(f1s))
    std_f1 = float(np.std(f1s))

    x = np.arange(len(subjects))
    plt.figure(figsize=(10, 4))
    plt.bar(x - 0.2, accs, width=0.4, label="Acc")
    plt.bar(x + 0.2, f1s, width=0.4, label="Macro-F1")
    plt.xticks(x, [str(s) for s in subjects])
    plt.ylim(0, 1)
    plt.xlabel("Held-out Subject")
    plt.title("LOSO Per-Fold Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loso_per_fold_bar.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.boxplot([accs, f1s], labels=["Test Acc", "Test Macro-F1"])
    plt.ylim(0, 1)
    plt.title("LOSO Performance Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loso_boxplot.png"), dpi=180)
    plt.close()

    cm_sum = np.sum(np.stack(cms, axis=0), axis=0)
    _plot_confusion(
        cm_sum,
        title="LOSO Aggregated Confusion Matrix",
        save_path=os.path.join(output_dir, "loso_confusion_agg.png"),
    )

    np.savez(
        os.path.join(output_dir, "loso_results.npz"),
        subjects=np.array(subjects, dtype=int),
        accs=np.array(accs, dtype=float),
        f1s=np.array(f1s, dtype=float),
        cm_sum=cm_sum,
    )
    with open(os.path.join(output_dir, "loso_summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def run_loso_eegnet(base_path, subject_ids=None, seed=42):
    loader = SEEDIVLoader()
    features, labels, subjects = loader.load_all_subjects(base_path=base_path, subject_ids=subject_ids)

    feats = np.empty(len(features), dtype=object)
    for i, trial in enumerate(features):
        feats[i] = trial
    y = np.asarray(labels).astype(int)
    groups = np.asarray(subjects).astype(int)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logo = LeaveOneGroupOut()
    unique_subjects = np.unique(groups)

    print(f"\nLOSO subjects: {unique_subjects.tolist()}")
    fold_subjects, fold_accs, fold_f1s, fold_cms = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(logo.split(feats, y, groups=groups), start=1):
        test_subject = int(groups[test_idx[0]])
        print(f"\n===== LOSO Fold {fold}/{len(unique_subjects)} | Test Subject={test_subject:02d} =====")

        x_train_trials = [feats[i] for i in train_idx]
        y_train_trials = y[train_idx]
        x_test_trials = [feats[i] for i in test_idx]
        y_test_trials = y[test_idx]

        result = train_one_fold(
            x_train_trials=x_train_trials,
            y_train_trials=y_train_trials,
            x_test_trials=x_test_trials,
            y_test_trials=y_test_trials,
            device=device,
            seed=seed + fold,
        )

        print(
            f"Best epoch={result['best_epoch']} | best val_loss={result['best_val_loss']:.4f} | "
            f"Train Acc={result['train_acc']:.4f} F1={result['train_f1']:.4f} | "
            f"Test Acc={result['test_acc']:.4f} F1={result['test_f1']:.4f}"
        )

        fold_subjects.append(test_subject)
        fold_accs.append(result["test_acc"])
        fold_f1s.append(result["test_f1"])
        fold_cms.append(result["test_cm"])

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    mean_f1 = float(np.mean(fold_f1s))
    std_f1 = float(np.std(fold_f1s))

    print("\n================ LOSO SUMMARY ================")
    print(f"Test Acc: mean={mean_acc:.4f} std={std_acc:.4f}")
    print(f"Test macro-F1: mean={mean_f1:.4f} std={std_f1:.4f}")

    out_dir = "loso_analysis"
    save_loso_report(fold_subjects, fold_accs, fold_f1s, fold_cms, output_dir=out_dir)
    print(f"Saved LOSO report to: {out_dir}")

    return fold_subjects, fold_accs, fold_f1s


if __name__ == "__main__":
    base_path = r"C:\Users\xinji\Desktop\archive\seed_iv\eeg_feature_smooth"
    run_loso_eegnet(base_path)
