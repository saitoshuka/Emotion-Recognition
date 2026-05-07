import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from deap_experiment_utils import (
    DE_Net,
    normalize_fold,
    load_subject_windows,
    binarize_and_filter,
    calc_acc_f1,
    save_binary_confusion_figure,
)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _list_subject_files(data_dir):
    return sorted([f for f in os.listdir(data_dir) if f.startswith("s") and f.endswith(".dat")])


def _eval_with_loss(model, loader, device, criterion):
    model.eval()
    y_true, y_pred = [], []
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            out = model(bx)
            loss = criterion(out, by)
            pred = torch.argmax(out, dim=1)
            y_pred.append(pred.cpu().numpy())
            y_true.append(by.cpu().numpy())
            loss_sum += float(loss.item()) * bx.size(0)
            n += bx.size(0)
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    acc, f1 = calc_acc_f1(y_true, y_pred)
    return loss_sum / max(n, 1), acc, f1, y_true, y_pred


def _train_one_fold(x_train, y_train, x_val, y_val, device, task_name, fold_name, max_epochs=80, patience=10):
    tr_ds = TensorDataset(torch.FloatTensor(x_train).unsqueeze(1), torch.LongTensor(y_train))
    va_ds = TensorDataset(torch.FloatTensor(x_val).unsqueeze(1), torch.LongTensor(y_val))
    tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=64, shuffle=False)

    model = DE_Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_acc = -1.0
    best_state = None
    bad = 0
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

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
            f"[{task_name}] {fold_name} Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} || "
            f"Test(Val) Loss {val_loss:.4f} Acc {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


def _prepare_task_data(X, y_raw_task, ambiguous_margin):
    y_bin_all, keep = binarize_and_filter(y_raw_task, threshold=5.0, ambiguous_margin=ambiguous_margin)
    X = X[keep]
    y_bin = y_bin_all[keep]
    return X, y_bin


def run_loso_task(files, data_dir, device, task_idx, task_name, trim_post_sec, ambiguous_margin):
    subject_names, accs, f1s, cms = [], [], [], []
    history = {}

    for test_file in tqdm(files, desc=f"LOSO {task_name}"):
        train_x_list, train_y_list = [], []

        for f in files:
            if f == test_file:
                continue
            fp = os.path.join(data_dir, f)
            X, y, _ = load_subject_windows(fp, trim_post_sec=trim_post_sec, use_ratio_start=0.5, use_ratio_end=1.0)
            if X is None or len(X) == 0:
                continue
            X_task, y_task = _prepare_task_data(X, y[:, task_idx], ambiguous_margin)
            if len(y_task) == 0:
                continue
            train_x_list.append(X_task)
            train_y_list.append(y_task)

        test_fp = os.path.join(data_dir, test_file)
        X_te, y_te_raw, _ = load_subject_windows(test_fp, trim_post_sec=trim_post_sec, use_ratio_start=0.5, use_ratio_end=1.0)
        if X_te is None or len(X_te) == 0:
            continue
        X_te, y_te = _prepare_task_data(X_te, y_te_raw[:, task_idx], ambiguous_margin)

        if len(train_x_list) == 0 or len(y_te) == 0:
            continue

        X_tr_all = np.concatenate(train_x_list, axis=0)
        y_tr_all = np.concatenate(train_y_list, axis=0)

        if len(np.unique(y_tr_all)) < 2 or len(np.unique(y_te)) < 2:
            continue

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        tr_idx, va_idx = next(sss.split(np.zeros(len(y_tr_all)), y_tr_all))

        x_tr = X_tr_all[tr_idx]
        y_tr = y_tr_all[tr_idx]
        x_va = X_tr_all[va_idx]
        y_va = y_tr_all[va_idx]

        x_tr_n, x_va_n = normalize_fold(x_tr, x_va)
        mu = np.mean(x_tr, axis=0, keepdims=True)
        std = np.std(x_tr, axis=0, keepdims=True) + 1e-8
        x_te_n = (X_te - mu) / std

        model, hist = _train_one_fold(
            x_tr_n,
            y_tr,
            x_va_n,
            y_va,
            device,
            task_name=task_name,
            fold_name=f"TestSubj={test_file}",
        )

        te_ds = TensorDataset(torch.FloatTensor(x_te_n).unsqueeze(1), torch.LongTensor(y_te))
        te_loader = DataLoader(te_ds, batch_size=64, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        _, acc, f1, y_true, y_pred = _eval_with_loss(model, te_loader, device, criterion)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        subject_names.append(test_file)
        accs.append(acc)
        f1s.append(f1)
        cms.append(cm.tolist())
        history[test_file] = hist
        print(f"{test_file}: Acc {acc:.4f} F1 {f1:.4f}")

    cm_sum = np.sum(np.array(cms, dtype=int), axis=0).tolist() if len(cms) else [[0, 0], [0, 0]]

    return {
        "subjects": subject_names,
        "accs": accs,
        "f1s": f1s,
        "cms": cms,
        "cm_sum": cm_sum,
        "mean_acc": float(np.mean(accs)) if len(accs) else 0.0,
        "std_acc": float(np.std(accs)) if len(accs) else 0.0,
        "mean_f1": float(np.mean(f1s)) if len(f1s) else 0.0,
        "std_f1": float(np.std(f1s)) if len(f1s) else 0.0,
        "history": history,
    }


def main():
    data_dir = "C:/Users/xinji/Desktop/data_preprocessed_python"
    files = _list_subject_files(data_dir)
    device = get_device()

    trim_post_sec = 3.0
    ambiguous_margin = 0.5

    result = {
        "paradigm": "loso",
        "trim_post_sec": trim_post_sec,
        "ambiguous_margin": ambiguous_margin,
    }

    result["valence"] = run_loso_task(files, data_dir, device, 0, "Valence", trim_post_sec, ambiguous_margin)
    result["arousal"] = run_loso_task(files, data_dir, device, 1, "Arousal", trim_post_sec, ambiguous_margin)

    out_dir = "deap_eegnet_loso"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.json")
    hist_path = os.path.join(out_dir, "histories.json")

    history_dump = {
        "valence": result["valence"].pop("history"),
        "arousal": result["arousal"].pop("history"),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history_dump, f, ensure_ascii=False, indent=2)


    save_binary_confusion_figure(
        result["valence"]["cm_sum"],
        save_path=os.path.join(out_dir, "confusion_valence.png"),
        title="Valence Confusion Matrix (Sum)",
    )
    save_binary_confusion_figure(
        result["arousal"]["cm_sum"],
        save_path=os.path.join(out_dir, "confusion_arousal.png"),
        title="Arousal Confusion Matrix (Sum)",
    )

    print("\n=== EEGNet LOSO Summary ===")
    print(
        "Valence: Acc "
        f"{result['valence']['mean_acc']:.4f}+-{result['valence']['std_acc']:.4f} | "
        f"F1 {result['valence']['mean_f1']:.4f}+-{result['valence']['std_f1']:.4f}"
    )
    print(
        "Arousal: Acc "
        f"{result['arousal']['mean_acc']:.4f}+-{result['arousal']['std_acc']:.4f} | "
        f"F1 {result['arousal']['mean_f1']:.4f}+-{result['arousal']['std_f1']:.4f}"
    )
    print(f"Saved: {out_path}")
    print(f"Saved: {hist_path}")


if __name__ == "__main__":
    main()
