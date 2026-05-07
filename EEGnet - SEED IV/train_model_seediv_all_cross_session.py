import os
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix

from load_data import SEEDIVLoader


class EEGNet_SEEDIV(nn.Module):
    def __init__(self, nb_classes=4, chans=62, samples=5, dropout_rate=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 3), padding=(0, 1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.depthwise_conv = nn.Conv2d(16, 32, (chans, 1), groups=16, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.activation = nn.ELU()
        self.separable_conv = nn.Conv2d(32, 32, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(dropout_rate)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 1 * samples, nb_classes)

    def forward(self, x):
        x = self.batchnorm1(self.conv1(x))
        x = self.activation(self.batchnorm2(self.depthwise_conv(x)))
        x = self.dropout(x)
        x = self.activation(self.batchnorm3(self.separable_conv(x)))
        x = self.dropout(x)
        x = self.flatten(x)
        return self.fc(x)


def slice_trials(features_list, labels, target_time=42, trial_id_offset=0):
    """
    Convert trial list to window-level tensors and keep trial ids.
    features_list: list of (62, T, 5)
    labels: shape (n_trials,)
    """
    if len(features_list) == 0:
        raise ValueError("features_list is empty.")

    x_list, y_list, trial_ids = [], [], []
    for i, (trial, lab) in enumerate(zip(features_list, labels)):
        if trial.shape[1] > target_time:
            trial = trial[:, :target_time, :]
        elif trial.shape[1] < target_time:
            pad = np.repeat(trial[:, -1:, :], target_time - trial.shape[1], axis=1)
            trial = np.concatenate([trial, pad], axis=1)

        # (62, T, 5) -> (T, 62, 5)
        trial_windows = trial.transpose(1, 0, 2)
        x_list.append(trial_windows)
        y_list.append(np.full(target_time, int(lab), dtype=np.int64))
        trial_ids.append(np.full(target_time, i + trial_id_offset, dtype=np.int64))

    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    tids = np.concatenate(trial_ids, axis=0)
    return torch.FloatTensor(x).unsqueeze(1), torch.LongTensor(y), torch.LongTensor(tids)


def _find_subject_session_paths(base_path, subject_id):
    paths = []
    for session_id in [1, 2, 3]:
        session_dir = os.path.join(base_path, str(session_id))
        files = sorted(
            [
                os.path.join(session_dir, f)
                for f in os.listdir(session_dir)
                if f.startswith(f"{subject_id}_") and f.endswith(".mat")
            ]
        )
        if len(files) == 0:
            return None
        paths.append(files[0])
    return paths


def load_population_raw(base_path):
    loader = SEEDIVLoader()
    subject_ids = range(1, 16)

    s1_trials, s1_labels = [], []
    s2_trials, s2_labels = [], []
    s3_trials, s3_labels = [], []
    s1_subjects, s2_subjects, s3_subjects = [], [], []

    print("Loading raw session data...")
    for sid in tqdm(subject_ids):
        paths = _find_subject_session_paths(base_path, sid)
        if paths is None:
            print(f"[Skip] Subject {sid}: missing one or more session files.")
            continue

        s1_f, s1_l = loader.load_single_session(paths[0], session_id=1)
        s2_f, s2_l = loader.load_single_session(paths[1], session_id=2)
        s3_f, s3_l = loader.load_single_session(paths[2], session_id=3)

        s1_trials.extend(s1_f)
        s1_labels.extend(s1_l)
        s1_subjects.extend([sid] * len(s1_f))
        s2_trials.extend(s2_f)
        s2_labels.extend(s2_l)
        s2_subjects.extend([sid] * len(s2_f))
        s3_trials.extend(s3_f)
        s3_labels.extend(s3_l)
        s3_subjects.extend([sid] * len(s3_f))

    return (
        s1_trials,
        np.array(s1_labels),
        np.array(s1_subjects),
        s2_trials,
        np.array(s2_labels),
        np.array(s2_subjects),
        s3_trials,
        np.array(s3_labels),
        np.array(s3_subjects),
    )


def normalize_by_subject(target_trials, target_subjects, ref_trials, ref_subjects):
    """
    Per-subject z-score using statistics fitted on ref_trials of each subject.
    target_trials/ref_trials: list of (62, T, 5)
    target_subjects/ref_subjects: ndarray of subject ids
    """
    stats = {}
    for sid in np.unique(ref_subjects):
        sid = int(sid)
        sid_ref = [trial for trial, sub in zip(ref_trials, ref_subjects) if int(sub) == sid]
        flat = np.concatenate(
            [trial.transpose(1, 0, 2).reshape(-1, 5) for trial in sid_ref],
            axis=0,
        )
        mu = np.mean(flat, axis=0)
        std = np.std(flat, axis=0) + 1e-7
        stats[sid] = (mu, std)

    normalized = []
    for trial, sid in zip(target_trials, target_subjects):
        sid = int(sid)
        if sid not in stats:
            raise ValueError(f"Subject {sid} missing in reference set for normalization.")
        mu, std = stats[sid]
        trial_n = ((trial.transpose(1, 0, 2) - mu) / std).transpose(1, 0, 2)
        normalized.append(trial_n)

    return normalized


def evaluate_with_trial_aggregation(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    win_true, win_pred = [], []
    trial_sum_logits = {}
    trial_count = {}
    trial_true = {}

    with torch.no_grad():
        for batch_x, batch_y, batch_tid in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)

            loss = criterion(logits, batch_y)
            total_loss += loss.item() * batch_x.size(0)

            pred = torch.argmax(logits, dim=1)
            correct += (pred == batch_y).sum().item()
            total += batch_y.size(0)

            logits_np = logits.cpu().numpy()
            pred_np = pred.cpu().numpy()
            y_np = batch_y.cpu().numpy()
            tid_np = batch_tid.cpu().numpy()

            win_true.append(y_np)
            win_pred.append(pred_np)

            for i in range(len(tid_np)):
                tid = int(tid_np[i])
                if tid not in trial_sum_logits:
                    trial_sum_logits[tid] = logits_np[i].copy()
                    trial_count[tid] = 1
                    trial_true[tid] = int(y_np[i])
                else:
                    trial_sum_logits[tid] += logits_np[i]
                    trial_count[tid] += 1

    y_true_win = np.concatenate(win_true)
    y_pred_win = np.concatenate(win_pred)

    y_true_trial, y_pred_trial = [], []
    for tid in sorted(trial_sum_logits.keys()):
        mean_logits = trial_sum_logits[tid] / trial_count[tid]
        y_true_trial.append(trial_true[tid])
        y_pred_trial.append(int(np.argmax(mean_logits)))

    y_true_trial = np.array(y_true_trial)
    y_pred_trial = np.array(y_pred_trial)

    return {
        "window_loss": total_loss / max(total, 1),
        "window_acc": (correct / max(total, 1)),
        "window_f1": f1_score(y_true_win, y_pred_win, average="macro"),
        "trial_acc": float(np.mean(y_true_trial == y_pred_trial)),
        "trial_f1": f1_score(y_true_trial, y_pred_trial, average="macro"),
        "window_cm": confusion_matrix(y_true_win, y_pred_win, labels=[0, 1, 2, 3]),
        "trial_cm": confusion_matrix(y_true_trial, y_pred_trial, labels=[0, 1, 2, 3]),
        "y_true_trial": y_true_trial,
        "y_pred_trial": y_pred_trial,
    }


def plot_stage1_curves(history, save_path):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss(S2)", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage1 Loss Curves (S1 Train / S2 Val)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_confusion_matrix(cm, title, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    classes = ["Neut", "Sad", "Fear", "Happy"]
    plt.xticks(np.arange(4), classes)
    plt.yticks(np.arange(4), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_limitations_figures(output_dir, stage1_history, best_val_trial_f1, test_metrics, trial_subjects):
    os.makedirs(output_dir, exist_ok=True)

    # 1) Stage1 train/val loss curves
    plot_stage1_curves(stage1_history, save_path=os.path.join(output_dir, "fig1_stage1_loss_curve.png"))

    # 2) Chance / Val / Test performance comparison
    chance = 0.25
    names = ["Chance", "Val Trial-F1", "S3 Trial-F1", "S3 Trial-Acc"]
    vals = [chance, best_val_trial_f1, test_metrics["trial_f1"], test_metrics["trial_acc"]]
    plt.figure(figsize=(7, 4))
    plt.bar(names, vals)
    plt.ylim(0, 1)
    plt.title("Cross-Session Performance Gap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_perf_gap.png"), dpi=180)
    plt.close()

    # 3) Window-level vs trial-level
    names = ["Window Acc", "Window F1", "Trial Acc", "Trial F1"]
    vals = [
        test_metrics["window_acc"],
        test_metrics["window_f1"],
        test_metrics["trial_acc"],
        test_metrics["trial_f1"],
    ]
    plt.figure(figsize=(7, 4))
    plt.bar(names, vals)
    plt.ylim(0, 1)
    plt.title("Window-Level vs Trial-Level Metrics (S3)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig3_window_vs_trial.png"), dpi=180)
    plt.close()

    # 4) Trial-level confusion matrix
    plot_confusion_matrix(
        test_metrics["trial_cm"],
        title="Trial-Level Confusion Matrix (S3)",
        save_path=os.path.join(output_dir, "fig4_trial_confusion.png"),
    )

    # 5) Per-class recall and F1 (trial-level)
    y_true = test_metrics["y_true_trial"]
    y_pred = test_metrics["y_pred_trial"]
    per_class_recall = []
    per_class_f1 = []
    for c in range(4):
        tp = np.sum((y_true == c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        fp = np.sum((y_true != c) & (y_pred == c))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    x = np.arange(4)
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, per_class_recall, width, label="Recall")
    plt.bar(x + width / 2, per_class_f1, width, label="F1")
    plt.xticks(x, ["Neut", "Sad", "Fear", "Happy"])
    plt.ylim(0, 1)
    plt.title("Per-Class Recall/F1 (Trial-Level, S3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_per_class_metrics.png"), dpi=180)
    plt.close()

    # 6) Subject-level distribution (trial-level)
    subj_acc, subj_f1 = [], []
    for sid in np.unique(trial_subjects):
        idx = trial_subjects == sid
        yt = y_true[idx]
        yp = y_pred[idx]
        subj_acc.append(float(np.mean(yt == yp)))
        subj_f1.append(float(f1_score(yt, yp, average="macro")))

    plt.figure(figsize=(7, 4))
    plt.boxplot([subj_acc, subj_f1], labels=["Subject Acc", "Subject Macro-F1"])
    plt.ylim(0, 1)
    plt.title("Subject-Level Performance Distribution (S3)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig6_subject_boxplot.png"), dpi=180)
    plt.close()


def plot_subjectwise_cross_session_figures(output_dir, subjects, accs, f1s, y_true_all, y_pred_all):
    os.makedirs(output_dir, exist_ok=True)

    x = np.arange(len(subjects))
    plt.figure(figsize=(10, 4))
    plt.bar(x - 0.2, accs, width=0.4, label="Trial Acc")
    plt.bar(x + 0.2, f1s, width=0.4, label="Trial Macro-F1")
    plt.xticks(x, [str(s) for s in subjects])
    plt.ylim(0, 1)
    plt.xlabel("Subject")
    plt.title("Cross-Session Subject-wise Performance (S1+S2 -> S3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_subjectwise_bar.png"), dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.boxplot([accs, f1s], labels=["Subject Acc", "Subject Macro-F1"])
    plt.ylim(0, 1)
    plt.title("Subject-wise Distribution (Cross-Session)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig2_subjectwise_boxplot.png"), dpi=180)
    plt.close()

    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1, 2, 3])
    plot_confusion_matrix(
        cm,
        title="Aggregated Trial-Level Confusion Matrix (Cross-Session)",
        save_path=os.path.join(output_dir, "fig3_confusion_agg.png"),
    )

    per_class_recall = []
    per_class_f1 = []
    for c in range(4):
        tp = np.sum((y_true_all == c) & (y_pred_all == c))
        fn = np.sum((y_true_all == c) & (y_pred_all != c))
        fp = np.sum((y_true_all != c) & (y_pred_all == c))
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    x = np.arange(4)
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, per_class_recall, width, label="Recall")
    plt.bar(x + width / 2, per_class_f1, width, label="F1")
    plt.xticks(x, ["Neut", "Sad", "Fear", "Happy"])
    plt.ylim(0, 1)
    plt.title("Per-Class Recall/F1 (Trial-Level, Aggregated)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_per_class_metrics.png"), dpi=180)
    plt.close()


def train_with_session_validation(
    model,
    train_loader,
    val_loader,
    device,
    max_epochs=80,
    patience=12,
    lr=1e-3,
    weight_decay=1e-3,
    label_smoothing=0.05,
):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val_trial_f1 = -1.0
    best_epoch = 1
    best_state = None
    no_improve = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_trial_f1": [],
        "val_trial_acc": [],
        "val_window_acc": [],
    }

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            train_correct += (torch.argmax(logits, dim=1) == batch_y).sum().item()
            train_total += batch_y.size(0)

        scheduler.step()

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_metrics = evaluate_with_trial_aggregation(model, val_loader, criterion, device)
        val_trial_f1 = val_metrics["trial_f1"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["window_loss"])
        history["val_trial_f1"].append(val_trial_f1)
        history["val_trial_acc"].append(val_metrics["trial_acc"])
        history["val_window_acc"].append(val_metrics["window_acc"])

        if val_trial_f1 > best_val_trial_f1:
            best_val_trial_f1 = val_trial_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d} | "
                f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} || "
                f"Val Win Acc {val_metrics['window_acc']:.4f} "
                f"Val Trial Acc {val_metrics['trial_acc']:.4f} "
                f"Val Trial F1 {val_trial_f1:.4f}"
            )

        if no_improve >= patience:
            print(f"Early stop at epoch {epoch}. Best epoch: {best_epoch}")
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return model, best_epoch, best_val_trial_f1, history


def train_on_all_train_data(
    train_loader,
    device,
    epochs,
    lr=1e-3,
    weight_decay=1e-3,
    label_smoothing=0.05,
):
    model = EEGNet_SEEDIV().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            running_correct += (torch.argmax(logits, dim=1) == batch_y).sum().item()
            running_total += batch_y.size(0)

        scheduler.step()

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"[Re-train] Epoch {epoch:02d}/{epochs:02d} | "
                f"Loss {running_loss / max(running_total, 1):.4f} "
                f"Acc {running_correct / max(running_total, 1):.4f}"
            )

    return model, criterion


def run_final_training(base_path):
    (
        s1_trials_raw,
        s1_labels,
        s1_subjects,
        s2_trials_raw,
        s2_labels,
        s2_subjects,
        s3_trials_raw,
        s3_labels,
        s3_subjects,
    ) = load_population_raw(base_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unique_subjects = sorted(np.unique(s1_subjects).astype(int).tolist())
    subj_accs, subj_f1s = [], []
    all_true_trial, all_pred_trial = [], []

    for sid in unique_subjects:
        print(f"\n===== Cross-session subject-wise | Subject {sid:02d} =====")
        idx_s1 = np.where(s1_subjects == sid)[0]
        idx_s2 = np.where(s2_subjects == sid)[0]
        idx_s3 = np.where(s3_subjects == sid)[0]

        sid_s1_raw = [s1_trials_raw[i] for i in idx_s1]
        sid_s2_raw = [s2_trials_raw[i] for i in idx_s2]
        sid_s3_raw = [s3_trials_raw[i] for i in idx_s3]
        sid_s1_labels = s1_labels[idx_s1]
        sid_s2_labels = s2_labels[idx_s2]
        sid_s3_labels = s3_labels[idx_s3]
        sid_s1_subjects = s1_subjects[idx_s1]
        sid_s2_subjects = s2_subjects[idx_s2]
        sid_s3_subjects = s3_subjects[idx_s3]

        # Stage1 normalization: fit on S1 and apply to S1/S2 for this subject.
        sid_s1 = normalize_by_subject(sid_s1_raw, sid_s1_subjects, sid_s1_raw, sid_s1_subjects)
        sid_s2 = normalize_by_subject(sid_s2_raw, sid_s2_subjects, sid_s1_raw, sid_s1_subjects)

        # Stage2 normalization: fit on S1+S2 and apply to S1+S2/S3 for this subject.
        sid_s12_raw = sid_s1_raw + sid_s2_raw
        sid_s12_subjects = np.concatenate([sid_s1_subjects, sid_s2_subjects], axis=0)
        sid_s12 = normalize_by_subject(sid_s12_raw, sid_s12_subjects, sid_s12_raw, sid_s12_subjects)
        sid_s3 = normalize_by_subject(sid_s3_raw, sid_s3_subjects, sid_s12_raw, sid_s12_subjects)

        x_s1, y_s1, _ = slice_trials(sid_s1, sid_s1_labels, target_time=42)
        x_s2, y_s2, tid_s2 = slice_trials(sid_s2, sid_s2_labels, target_time=42)
        x_s3, y_s3, tid_s3 = slice_trials(sid_s3, sid_s3_labels, target_time=42)

        sid_s12_labels = np.concatenate([sid_s1_labels, sid_s2_labels], axis=0)
        x_s12, y_s12, _ = slice_trials(sid_s12, sid_s12_labels, target_time=42)

        train_loader_s1 = DataLoader(TensorDataset(x_s1, y_s1), batch_size=128, shuffle=True)
        val_loader_s2 = DataLoader(TensorDataset(x_s2, y_s2, tid_s2), batch_size=128, shuffle=False)
        train_loader_s12 = DataLoader(TensorDataset(x_s12, y_s12), batch_size=128, shuffle=True)
        test_loader_s3 = DataLoader(TensorDataset(x_s3, y_s3, tid_s3), batch_size=128, shuffle=False)

        stage1_model = EEGNet_SEEDIV().to(device)
        stage1_model, best_epoch, best_val_trial_f1, _ = train_with_session_validation(
            model=stage1_model,
            train_loader=train_loader_s1,
            val_loader=val_loader_s2,
            device=device,
            max_epochs=80,
            patience=12,
            lr=1e-3,
            weight_decay=1e-3,
            label_smoothing=0.05,
        )
        print(f"Subject {sid:02d} best epoch={best_epoch}, S2 val trial-F1={best_val_trial_f1:.4f}")

        final_model, final_criterion = train_on_all_train_data(
            train_loader=train_loader_s12,
            device=device,
            epochs=max(best_epoch, 1),
            lr=1e-3,
            weight_decay=1e-3,
            label_smoothing=0.05,
        )
        test_metrics = evaluate_with_trial_aggregation(final_model, test_loader_s3, final_criterion, device)
        subj_accs.append(float(test_metrics["trial_acc"]))
        subj_f1s.append(float(test_metrics["trial_f1"]))
        all_true_trial.append(test_metrics["y_true_trial"])
        all_pred_trial.append(test_metrics["y_pred_trial"])
        print(
            f"Subject {sid:02d} S3 trial-Acc={test_metrics['trial_acc']:.4f}, "
            f"trial-F1={test_metrics['trial_f1']:.4f}"
        )

    all_true_trial = np.concatenate(all_true_trial)
    all_pred_trial = np.concatenate(all_pred_trial)
    mean_acc = float(np.mean(subj_accs))
    std_acc = float(np.std(subj_accs))
    mean_f1 = float(np.mean(subj_f1s))
    std_f1 = float(np.std(subj_f1s))

    out_dir = "analysis_figures"
    plot_subjectwise_cross_session_figures(
        output_dir=out_dir,
        subjects=unique_subjects,
        accs=subj_accs,
        f1s=subj_f1s,
        y_true_all=all_true_trial,
        y_pred_all=all_pred_trial,
    )
    np.savez(
        os.path.join(out_dir, "results_arrays.npz"),
        y_true_trial=all_true_trial,
        y_pred_trial=all_pred_trial,
        subjects=np.array(unique_subjects, dtype=int),
        accs=np.array(subj_accs, dtype=float),
        f1s=np.array(subj_f1s, dtype=float),
    )
    with open(os.path.join(out_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "paradigm": "cross_session_subject_wise",
                "subjects": unique_subjects,
                "per_subject_acc": [float(v) for v in subj_accs],
                "per_subject_f1": [float(v) for v in subj_f1s],
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                # keep compatibility for downstream scripts
                "s3_trial_acc": mean_acc,
                "s3_trial_f1": mean_f1,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Saved analysis figures and metrics to: {out_dir}")

    print("\n" + "=" * 58)
    print("FINAL CROSS-SESSION SUBJECT-WISE SUMMARY (S1+S2 -> S3)")
    print(f"Trial-level Acc: mean={mean_acc:.4f} std={std_acc:.4f}")
    print(f"Trial-level F1 : mean={mean_f1:.4f} std={std_f1:.4f}")
    print("=" * 58)


if __name__ == "__main__":
    base_path = r"C:\Users\xinji\Desktop\archive\seed_iv\eeg_feature_smooth"
    run_final_training(base_path)
