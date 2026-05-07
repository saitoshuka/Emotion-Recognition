import os
import json
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from svm_seediv_common import (
    get_loader,
    make_window_dataset,
    aggregate_trial_predictions,
    trial_metrics,
    fit_svm_model,
)


def run_subject_dependent_svm(base_path, kernel="rbf", c=1.0, seed=42):
    loader = get_loader()
    subject_ids, accs, f1s = [], [], []

    for sid in range(1, 16):
        trials, labels = loader.load_single_subject(sid, base_path=base_path)
        labels = np.asarray(labels).astype(int)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + sid)
        train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))

        x_train_trials = [trials[i] for i in train_idx]
        y_train_trials = labels[train_idx]
        x_test_trials = [trials[i] for i in test_idx]
        y_test_trials = labels[test_idx]

        x_tr, y_tr, _ = make_window_dataset(x_train_trials, y_train_trials)
        x_te, y_te, tid_te = make_window_dataset(x_test_trials, y_test_trials)

        model = fit_svm_model(kernel=kernel, c=c)
        model.fit(x_tr, y_tr)
        te_scores = model.decision_function(x_te)
        y_true_trial, y_pred_trial = aggregate_trial_predictions(te_scores, tid_te, y_te)
        acc, f1 = trial_metrics(y_true_trial, y_pred_trial)

        subject_ids.append(sid)
        accs.append(acc)
        f1s.append(f1)
        print(f"Subject-dependent subject {sid:02d} | Acc {acc:.4f} F1 {f1:.4f}")

    return {
        "subjects": subject_ids,
        "accs": accs,
        "f1s": f1s,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
    }


def main():
    base_path = r"C:\Users\xinji\Desktop\archive\seed_iv\eeg_feature_smooth"
    out_root = "svm_analysis"
    os.makedirs(out_root, exist_ok=True)

    result = run_subject_dependent_svm(base_path, kernel="rbf", c=1.0, seed=42)
    with open(os.path.join(out_root, "subject_dependent_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSubject-dependent: Acc {result['mean_acc']:.4f}±{result['std_acc']:.4f} "
          f"F1 {result['mean_f1']:.4f}±{result['std_f1']:.4f}")
    print(f"Saved: {os.path.join(out_root, 'subject_dependent_summary.json')}")


if __name__ == "__main__":
    main()
