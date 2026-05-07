import os
import json
import time
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from deap_experiment_utils import (
    load_subject_windows,
    binarize_and_filter,
    calc_acc_f1,
    save_binary_confusion_figure,
)


def _list_subject_files(data_dir):
    return sorted([f for f in os.listdir(data_dir) if f.startswith("s") and f.endswith(".dat")])


def _fit_svm(kernel="linear", c=1.0):
    # LinearSVC is much faster than SVC(kernel='linear') on large LOSO folds.
    if kernel == "linear":
        return make_pipeline(StandardScaler(), LinearSVC(C=c, dual="auto", max_iter=5000))
    return make_pipeline(StandardScaler(), SVC(kernel=kernel, C=c, gamma="scale"))


def _to_binary_task(X, y_raw_task, ambiguous_margin):
    y_bin_all, keep = binarize_and_filter(y_raw_task, threshold=5.0, ambiguous_margin=ambiguous_margin)
    return X[keep], y_bin_all[keep]


def _preload_task_data(files, data_dir, task_idx, trim_post_sec, ambiguous_margin):
    data_map = {}
    for f in tqdm(files, desc=f"Preload task {task_idx}"):
        fp = os.path.join(data_dir, f)
        X, y, _ = load_subject_windows(
            fp,
            trim_post_sec=trim_post_sec,
            use_ratio_start=0.5,
            use_ratio_end=1.0,
        )
        if X is None or len(X) == 0:
            continue
        X_task, y_task = _to_binary_task(X, y[:, task_idx], ambiguous_margin)
        if len(y_task) == 0:
            continue
        data_map[f] = (X_task, y_task)
    return data_map


def run_loso_task(files, data_dir, task_idx, trim_post_sec, ambiguous_margin, kernel, c):
    subject_names, accs, f1s, cms = [], [], [], []

    data_map = _preload_task_data(files, data_dir, task_idx, trim_post_sec, ambiguous_margin)
    test_files = [f for f in files if f in data_map]

    for test_file in tqdm(test_files, desc=f"SVM LOSO task {task_idx}"):
        start = time.time()

        x_te, y_te = data_map[test_file]
        train_x_list, train_y_list = [], []

        for f in test_files:
            if f == test_file:
                continue
            x_f, y_f = data_map[f]
            train_x_list.append(x_f)
            train_y_list.append(y_f)

        if len(train_x_list) == 0 or len(y_te) == 0:
            continue

        x_tr = np.concatenate(train_x_list, axis=0)
        y_tr = np.concatenate(train_y_list, axis=0)

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        model = _fit_svm(kernel=kernel, c=c)
        model.fit(x_tr.reshape(len(x_tr), -1), y_tr)
        y_pred = model.predict(x_te.reshape(len(x_te), -1))

        acc, f1 = calc_acc_f1(y_te, y_pred)
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])

        subject_names.append(test_file)
        accs.append(acc)
        f1s.append(f1)
        cms.append(cm.tolist())

        elapsed = time.time() - start
        print(
            f"{test_file}: Acc {acc:.4f} F1 {f1:.4f} | "
            f"TrainN {len(y_tr)} TestN {len(y_te)} | {elapsed:.1f}s"
        )

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
    }


def main():
    data_dir = "C:/Users/xinji/Desktop/data_preprocessed_python"
    files = _list_subject_files(data_dir)

    trim_post_sec = 3.0
    ambiguous_margin = 0.5
    kernel = "linear"
    c = 1.0

    result = {
        "paradigm": "loso",
        "model": "svm",
        "kernel": kernel,
        "c": c,
        "trim_post_sec": trim_post_sec,
        "ambiguous_margin": ambiguous_margin,
    }

    result["valence"] = run_loso_task(files, data_dir, 0, trim_post_sec, ambiguous_margin, kernel, c)
    result["arousal"] = run_loso_task(files, data_dir, 1, trim_post_sec, ambiguous_margin, kernel, c)

    out_dir = "deap_svm_loso"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


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

    print("\n=== SVM LOSO Summary ===")
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


if __name__ == "__main__":
    main()
