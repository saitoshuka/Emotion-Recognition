import os
import json
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, confusion_matrix
from tqdm import tqdm

from deap_experiment_utils import load_subject_windows


CLASS_NAMES = ["LVLA", "LVHA", "HVLA", "HVHA"]
# id mapping:
# 0: LVLA (val<=5, aro<=5)
# 1: LVHA (val<=5, aro>5)
# 2: HVLA (val>5, aro<=5)
# 3: HVHA (val>5, aro>5)


def _list_subject_files(data_dir):
    return sorted([f for f in os.listdir(data_dir) if f.startswith("s") and f.endswith(".dat")])


def _fit_svm(kernel="rbf", c=1.0):
    return make_pipeline(StandardScaler(), SVC(kernel=kernel, C=c, gamma="scale"))


def _encode_4class(y_va_raw, y_ar_raw, threshold=5.0, ambiguous_margin=0.0):
    y_v = y_va_raw.astype(np.float32)
    y_a = y_ar_raw.astype(np.float32)

    keep = np.ones(len(y_v), dtype=bool)
    if ambiguous_margin > 0:
        keep_v = np.abs(y_v - threshold) >= ambiguous_margin
        keep_a = np.abs(y_a - threshold) >= ambiguous_margin
        keep = keep_v & keep_a

    v_high = (y_v > threshold).astype(np.int64)
    a_high = (y_a > threshold).astype(np.int64)

    y4 = v_high * 2 + a_high
    return y4, keep


def _calc_acc_macro_f1(y_true, y_pred):
    acc = float(np.mean(y_true == y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return acc, f1


def run_subject_dependent_4class(
    data_dir,
    trim_post_sec=3.0,
    ambiguous_margin=0.5,
    kernel="rbf",
    c=1.0,
    seed=42,
):
    files = _list_subject_files(data_dir)

    subject_names, accs, f1s, cms = [], [], [], []

    for i, f in enumerate(tqdm(files, desc="SVM SD 4-class")):
        fp = os.path.join(data_dir, f)
        X, y_raw, groups = load_subject_windows(
            fp,
            trim_post_sec=trim_post_sec,
            use_ratio_start=0.5,
            use_ratio_end=1.0,
        )
        if X is None or len(X) == 0:
            continue

        y4, keep = _encode_4class(y_raw[:, 0], y_raw[:, 1], threshold=5.0, ambiguous_margin=ambiguous_margin)
        X = X[keep]
        y4 = y4[keep]
        groups = groups[keep]

        if len(y4) < 20 or len(np.unique(y4)) < 2:
            continue

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + i)
        tr_idx, te_idx = next(gss.split(X, y4, groups=groups))

        x_tr, y_tr = X[tr_idx], y4[tr_idx]
        x_te, y_te = X[te_idx], y4[te_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        model = _fit_svm(kernel=kernel, c=c)
        model.fit(x_tr.reshape(len(x_tr), -1), y_tr)
        y_pred = model.predict(x_te.reshape(len(x_te), -1))

        acc, f1 = _calc_acc_macro_f1(y_te, y_pred)
        cm = confusion_matrix(y_te, y_pred, labels=[0, 1, 2, 3])

        subject_names.append(f)
        accs.append(acc)
        f1s.append(f1)
        cms.append(cm)

        print(f"{f}: Acc {acc:.4f} Macro-F1 {f1:.4f}")

    cm_sum = np.sum(np.stack(cms, axis=0), axis=0) if len(cms) else np.zeros((4, 4), dtype=int)

    return {
        "paradigm": "subject_dependent",
        "model": "svm",
        "split": "group_shuffle_by_trial",
        "task": "4class_valence_arousal",
        "class_names": CLASS_NAMES,
        "kernel": kernel,
        "c": c,
        "trim_post_sec": trim_post_sec,
        "ambiguous_margin": ambiguous_margin,
        "subjects": subject_names,
        "accs": accs,
        "f1s": f1s,
        "mean_acc": float(np.mean(accs)) if len(accs) else 0.0,
        "std_acc": float(np.std(accs)) if len(accs) else 0.0,
        "mean_f1": float(np.mean(f1s)) if len(f1s) else 0.0,
        "std_f1": float(np.std(f1s)) if len(f1s) else 0.0,
        "confusion_matrix_sum": cm_sum.tolist(),
    }


def main():
    data_dir = "C:/Users/xinji/Desktop/data_preprocessed_python"

    result = run_subject_dependent_4class(
        data_dir=data_dir,
        trim_post_sec=3.0,
        ambiguous_margin=0.5,
        kernel="rbf",
        c=1.0,
        seed=42,
    )

    out_dir = "deap_svm_subject_dependent_4class"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== SVM Subject-dependent 4-class Summary ===")
    print(f"Acc: {result['mean_acc']:.4f}+-{result['std_acc']:.4f}")
    print(f"Macro-F1: {result['mean_f1']:.4f}+-{result['std_f1']:.4f}")
    print("Class order:", result["class_names"])
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
