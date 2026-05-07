import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from deap_experiment_utils import (
    normalize_fold,
    load_subject_windows,
    binarize_and_filter,
    make_groupkfold_splits,
    calc_acc_f1,
    save_binary_confusion_figure,
)


class Square(nn.Module):
    def forward(self, x):
        return x * x


class SafeLog(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


class ShallowConvNetDE(nn.Module):
    """
    ShallowNet-style model adapted to DE feature maps with shape [B, 1, chans, bands].
    """

    def __init__(self, chans=32, bands=5, nb_classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.Conv2d(40, 40, kernel_size=(chans, 1), bias=False),
            nn.BatchNorm2d(40),
            Square(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 1)),
            SafeLog(),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            z = torch.zeros(1, 1, chans, bands)
            z = self.features(z)
            flat_dim = int(z.numel())

        self.classifier = nn.Linear(flat_dim, nb_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def _eval_with_loss(model, loader, device, criterion):
    model.eval()
    y_true, y_pred = [], []
    loss_sum, n = 0.0, 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        out = model(bx)
        loss = criterion(out, by)
        pred = torch.argmax(out, dim=1)
        y_true.append(by.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
        loss_sum += float(loss.item()) * bx.size(0)
        n += bx.size(0)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc, f1 = calc_acc_f1(y_true, y_pred)
    return loss_sum / max(n, 1), acc, f1, y_true, y_pred


def train_one_task_subject(X, y_raw_task, groups, device, ambiguous_margin, task_name, max_epochs=80, patience=10):
    y_bin_all, keep = binarize_and_filter(y_raw_task, threshold=5.0, ambiguous_margin=ambiguous_margin)
    X = X[keep]
    y_bin = y_bin_all[keep]
    groups = groups[keep]

    splits = make_groupkfold_splits(X, y_bin, groups, n_splits=5)
    fold_accs, fold_f1s = [], []
    histories = []
    y_true_all, y_pred_all = [], []

    for fold_i, (train_idx, val_idx) in enumerate(splits, start=1):
        x_tr, y_tr = X[train_idx], y_bin[train_idx]
        x_va, y_va = X[val_idx], y_bin[val_idx]
        x_tr, x_va = normalize_fold(x_tr, x_va)

        tr_ds = TensorDataset(torch.FloatTensor(x_tr).unsqueeze(1), torch.LongTensor(y_tr))
        va_ds = TensorDataset(torch.FloatTensor(x_va).unsqueeze(1), torch.LongTensor(y_va))
        tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=64, shuffle=False)

        model = ShallowConvNetDE(chans=x_tr.shape[1], bands=x_tr.shape[2], nb_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

        best_acc = -1.0
        best_state = None
        bad = 0
        hist = {"fold": fold_i, "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, max_epochs + 1):
            model.train()
            train_loss_sum, train_n = 0.0, 0
            train_correct = 0

            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

                pred = torch.argmax(out, dim=1)
                train_correct += int((pred == by).sum().item())
                train_loss_sum += float(loss.item()) * bx.size(0)
                train_n += bx.size(0)

            train_loss = train_loss_sum / max(train_n, 1)
            train_acc = train_correct / max(train_n, 1)

            val_loss, val_acc, _, _, _ = _eval_with_loss(model, va_loader, device, criterion)

            hist["train_loss"].append(train_loss)
            hist["train_acc"].append(train_acc)
            hist["val_loss"].append(val_loss)
            hist["val_acc"].append(val_acc)

            print(
                f"[{task_name}] Fold {fold_i:02d} Epoch {epoch:03d} | "
                f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} || "
                f"Test(Val) Loss {val_loss:.4f} Acc {val_acc:.4f}"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = model.state_dict()
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        _, acc, f1, y_true, y_pred = _eval_with_loss(model, va_loader, device, criterion)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        histories.append(hist)
        y_true_all.append(y_true)
        y_pred_all.append(y_pred)

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

    return {
        "mean_acc": float(np.mean(fold_accs)),
        "std_acc": float(np.std(fold_accs)),
        "mean_f1": float(np.mean(fold_f1s)),
        "std_f1": float(np.std(fold_f1s)),
        "fold_accs": [float(x) for x in fold_accs],
        "fold_f1s": [float(x) for x in fold_f1s],
        "histories": histories,
        "cm": cm.tolist(),
    }


def main():
    data_dir = "C:/Users/xinji/Desktop/data_preprocessed_python"
    files = sorted([f for f in os.listdir(data_dir) if f.startswith("s") and f.endswith(".dat")])
    device = get_device()

    trim_post_sec = 3.0
    ambiguous_margin = 0.5

    result = {
        "paradigm": "subject_dependent",
        "model": "shallownet",
        "trim_post_sec": trim_post_sec,
        "ambiguous_margin": ambiguous_margin,
        "valence": {"accs": [], "f1s": [], "cms": []},
        "arousal": {"accs": [], "f1s": [], "cms": []},
        "subjects": [],
    }
    histories = {}

    for f_name in tqdm(files, desc="Subjects"):
        file_path = os.path.join(data_dir, f_name)
        X, y, groups = load_subject_windows(
            file_path,
            trim_post_sec=trim_post_sec,
            use_ratio_start=0.5,
            use_ratio_end=1.0,
        )
        if X is None or len(X) == 0:
            continue

        print(f"\n===== Subject {f_name} | VALENCE =====")
        v_res = train_one_task_subject(X, y[:, 0], groups, device, ambiguous_margin, "Valence")
        print(f"\n===== Subject {f_name} | AROUSAL =====")
        a_res = train_one_task_subject(X, y[:, 1], groups, device, ambiguous_margin, "Arousal")

        result["subjects"].append(f_name)
        result["valence"]["accs"].append(v_res["mean_acc"])
        result["valence"]["f1s"].append(v_res["mean_f1"])
        result["valence"]["cms"].append(v_res["cm"])
        result["arousal"]["accs"].append(a_res["mean_acc"])
        result["arousal"]["f1s"].append(a_res["mean_f1"])
        result["arousal"]["cms"].append(a_res["cm"])

        histories[f_name] = {"valence": v_res["histories"], "arousal": a_res["histories"]}

        print(
            f"{f_name}: V-Acc {v_res['mean_acc']:.4f} V-F1 {v_res['mean_f1']:.4f} | "
            f"A-Acc {a_res['mean_acc']:.4f} A-F1 {a_res['mean_f1']:.4f}"
        )

    for task in ["valence", "arousal"]:
        arr_acc = np.array(result[task]["accs"], dtype=float)
        arr_f1 = np.array(result[task]["f1s"], dtype=float)
        result[task]["mean_acc"] = float(np.mean(arr_acc)) if len(arr_acc) else 0.0
        result[task]["std_acc"] = float(np.std(arr_acc)) if len(arr_acc) else 0.0
        result[task]["mean_f1"] = float(np.mean(arr_f1)) if len(arr_f1) else 0.0
        result[task]["std_f1"] = float(np.std(arr_f1)) if len(arr_f1) else 0.0

        if len(result[task]["cms"]) > 0:
            cms = np.array(result[task]["cms"], dtype=int)
            result[task]["cm_sum"] = np.sum(cms, axis=0).tolist()
        else:
            result[task]["cm_sum"] = [[0, 0], [0, 0]]

    out_dir = "deap_shallownet_subject_dependent"
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.json")
    history_path = os.path.join(out_dir, "histories.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(histories, f, ensure_ascii=False, indent=2)

    save_binary_confusion_figure(
        result["valence"]["cm_sum"],
        save_path=os.path.join(out_dir, "confusion_valence.png"),
        title="ShallowNet Valence Confusion Matrix (Sum)",
    )
    save_binary_confusion_figure(
        result["arousal"]["cm_sum"],
        save_path=os.path.join(out_dir, "confusion_arousal.png"),
        title="ShallowNet Arousal Confusion Matrix (Sum)",
    )

    print("\n=== ShallowNet Subject-dependent Summary ===")
    print(
        f"Valence: Acc {result['valence']['mean_acc']:.4f}+-{result['valence']['std_acc']:.4f} | "
        f"F1 {result['valence']['mean_f1']:.4f}+-{result['valence']['std_f1']:.4f}"
    )
    print(
        f"Arousal: Acc {result['arousal']['mean_acc']:.4f}+-{result['arousal']['std_acc']:.4f} | "
        f"F1 {result['arousal']['mean_f1']:.4f}+-{result['arousal']['std_f1']:.4f}"
    )
    print(f"Saved: {summary_path}")
    print(f"Saved: {history_path}")


if __name__ == "__main__":
    main()
