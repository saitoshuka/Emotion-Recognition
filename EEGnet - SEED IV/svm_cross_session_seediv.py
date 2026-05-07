import os
import json
import numpy as np

from svm_seediv_common import (
    get_loader,
    find_subject_session_paths,
    make_window_dataset,
    aggregate_trial_predictions,
    trial_metrics,
    fit_svm_model,
)


def run_cross_session_svm_subject_wise(base_path, kernel="rbf", c=1.0):
    loader = get_loader()
    subject_ids, accs, f1s = [], [], []

    for sid in range(1, 16):
        paths = find_subject_session_paths(base_path, sid)
        if paths is None:
            continue
        s1_f, s1_l = loader.load_single_session(paths[0], session_id=1)
        s2_f, s2_l = loader.load_single_session(paths[1], session_id=2)
        s3_f, s3_l = loader.load_single_session(paths[2], session_id=3)

        train_trials = s1_f + s2_f
        train_labels = np.concatenate([s1_l, s2_l])
        test_trials = s3_f
        test_labels = s3_l

        x_tr, y_tr, _ = make_window_dataset(train_trials, np.array(train_labels))
        x_te, y_te, tid_te = make_window_dataset(test_trials, np.array(test_labels))

        model = fit_svm_model(kernel=kernel, c=c)
        model.fit(x_tr, y_tr)
        te_scores = model.decision_function(x_te)
        y_true_trial, y_pred_trial = aggregate_trial_predictions(te_scores, tid_te, y_te)
        acc, f1 = trial_metrics(y_true_trial, y_pred_trial)

        subject_ids.append(sid)
        accs.append(acc)
        f1s.append(f1)
        print(f"Cross-session subject {sid:02d} | Acc {acc:.4f} F1 {f1:.4f}")

    return {
        "subjects": subject_ids,
        "accs": accs,
        "f1s": f1s,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "trial_acc": float(np.mean(accs)),
        "trial_f1": float(np.mean(f1s)),
    }


def main():
    base_path = r"C:\Users\xinji\Desktop\archive\seed_iv\eeg_feature_smooth"
    out_root = "svm_analysis"
    os.makedirs(out_root, exist_ok=True)

    result = run_cross_session_svm_subject_wise(base_path, kernel="rbf", c=1.0)
    with open(os.path.join(out_root, "cross_session_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nCross-session: Acc {result['mean_acc']:.4f}±{result['std_acc']:.4f} "
          f"F1 {result['mean_f1']:.4f}±{result['std_f1']:.4f}")
    print(f"Saved: {os.path.join(out_root, 'cross_session_summary.json')}")


if __name__ == "__main__":
    main()
