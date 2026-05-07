import os
import json
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

from svm_seediv_common import (
    get_loader,
    make_window_dataset,
    aggregate_trial_predictions,
    trial_metrics,
    fit_svm_model,
)


def run_loso_svm(base_path, kernel="rbf", c=1.0):
    loader = get_loader()
    feats, labels, subjects = loader.load_all_subjects(base_path=base_path, subject_ids=None)
    labels = np.asarray(labels).astype(int)
    subjects = np.asarray(subjects).astype(int)

    feats_obj = np.empty(len(feats), dtype=object)
    for i, trial in enumerate(feats):
        feats_obj[i] = trial

    logo = LeaveOneGroupOut()
    fold_subjects, fold_accs, fold_f1s = [], [], []

    for train_idx, test_idx in logo.split(feats_obj, labels, groups=subjects):
        sid = int(subjects[test_idx[0]])
        x_train_trials = [feats_obj[i] for i in train_idx]
        y_train_trials = labels[train_idx]
        x_test_trials = [feats_obj[i] for i in test_idx]
        y_test_trials = labels[test_idx]

        x_tr, y_tr, _ = make_window_dataset(x_train_trials, y_train_trials)
        x_te, y_te, tid_te = make_window_dataset(x_test_trials, y_test_trials)

        model = fit_svm_model(kernel=kernel, c=c)
        model.fit(x_tr, y_tr)
        te_scores = model.decision_function(x_te)
        y_true_trial, y_pred_trial = aggregate_trial_predictions(te_scores, tid_te, y_te)
        acc, f1 = trial_metrics(y_true_trial, y_pred_trial)

        fold_subjects.append(sid)
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"LOSO test subject {sid:02d} | Acc {acc:.4f} F1 {f1:.4f}")

    return {
        "subjects": fold_subjects,
        "accs": fold_accs,
        "f1s": fold_f1s,
        "mean_acc": float(np.mean(fold_accs)),
        "std_acc": float(np.std(fold_accs)),
        "mean_f1": float(np.mean(fold_f1s)),
        "std_f1": float(np.std(fold_f1s)),
    }


def main():
    base_path = r"C:\Users\xinji\Desktop\archive\seed_iv\eeg_feature_smooth"
    out_root = "svm_analysis"
    os.makedirs(out_root, exist_ok=True)

    result = run_loso_svm(base_path, kernel="rbf", c=1.0)
    with open(os.path.join(out_root, "loso_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nLOSO: Acc {result['mean_acc']:.4f}±{result['std_acc']:.4f} "
          f"F1 {result['mean_f1']:.4f}±{result['std_f1']:.4f}")
    print(f"Saved: {os.path.join(out_root, 'loso_summary.json')}")


if __name__ == "__main__":
    main()
