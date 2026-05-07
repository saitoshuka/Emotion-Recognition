import os
import json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
    return make_pipeline(StandardScaler(), SVC(kernel=kernel, C=c, gamma="scale"))


def _to_binary_task(X, y_raw_task, groups, ambiguous_margin):
    y_bin_all, keep = binarize_and_filter(y_raw_task, threshold=5.0, ambiguous_margin=ambiguous_margin)
    return X[keep], y_bin_all[keep], groups[keep]


def run_subject_dependent_task(files, data_dir, task_idx, trim_post_sec, ambiguous_margin, kernel, c, seed=42):
    subject_names, accs, f1s, cms = [], [], [], []

    for i, f in enumerate(tqdm(files, desc=f"SVM SD task {task_idx}")):
        fp = os.path.join(data_dir, f)
        X, y, groups = load_subject_windows(fp, trim_post_sec=trim_post_sec, use_ratio_start=0.5, use_ratio_end=1.0)
        if X is None or len(X) == 0:
            continue

        X, y_bin, groups = _to_binary_task(X, y[:, task_idx], groups, ambiguous_margin)
        if len(y_bin) < 10 or len(np.unique(y_bin)) < 2:
            continue

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + i)
        tr_idx, te_idx = next(gss.split(X, y_bin, groups=groups))

        x_tr, y_tr = X[tr_idx], y_bin[tr_idx]
        x_te, y_te = X[te_idx], y_bin[te_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        model = _fit_svm(kernel=kernel, c=c)
        model.fit(x_tr.reshape(len(x_tr), -1), y_tr)
        y_pred = model.predict(x_te.reshape(len(x_te), -1))

        acc, f1 = calc_acc_f1(y_te, y_pred)
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1])

        subject_names.append(f)
        accs.append(acc)
        f1s.append(f1)
        cms.append(cm.tolist())
        print(f"{f}: Acc {acc:.4f} F1 {f1:.4f}")

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
        "paradigm": "subject_dependent",
        "model": "svm",
        "split": "group_shuffle_by_trial",
        "kernel": kernel,
        "c": c,
        "trim_post_sec": trim_post_sec,
        "ambiguous_margin": ambiguous_margin,
    }

    result["valence"] = run_subject_dependent_task(files, data_dir, 0, trim_post_sec, ambiguous_margin, kernel, c)
    result["arousal"] = run_subject_dependent_task(files, data_dir, 1, trim_post_sec, ambiguous_margin, kernel, c)

    out_dir = "deap_svm_subject_dependent"
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

    print("\n=== SVM Subject-dependent Summary ===")
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
