import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from load_data import SEEDIVLoader


TARGET_TIME = 42
N_CLASSES = 4


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
        if not files:
            return None
        paths.append(files[0])
    return paths


def _pad_or_crop_trial(trial, target_time=TARGET_TIME):
    if trial.shape[1] > target_time:
        return trial[:, :target_time, :]
    if trial.shape[1] < target_time:
        pad = np.repeat(trial[:, -1:, :], target_time - trial.shape[1], axis=1)
        return np.concatenate([trial, pad], axis=1)
    return trial


def _trial_to_windows(trial):
    # (62, T, 5) -> (T, 62*5)
    trial = _pad_or_crop_trial(trial, TARGET_TIME)
    w = trial.transpose(1, 0, 2).reshape(TARGET_TIME, -1)
    return w


def _make_window_dataset(trials, labels):
    x_list, y_list, tid_list = [], [], []
    for i, (trial, lab) in enumerate(zip(trials, labels)):
        w = _trial_to_windows(trial)
        x_list.append(w)
        y_list.append(np.full(TARGET_TIME, int(lab), dtype=np.int64))
        tid_list.append(np.full(TARGET_TIME, i, dtype=np.int64))
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    tids = np.concatenate(tid_list, axis=0)
    return x, y, tids


def _aggregate_trial_predictions(scores, tids, y_true_window):
    score_sum = {}
    count = {}
    trial_true = {}
    for i in range(len(tids)):
        tid = int(tids[i])
        if tid not in score_sum:
            score_sum[tid] = scores[i].copy()
            count[tid] = 1
            trial_true[tid] = int(y_true_window[i])
        else:
            score_sum[tid] += scores[i]
            count[tid] += 1

    y_true_trial = []
    y_pred_trial = []
    for tid in sorted(score_sum.keys()):
        mean_score = score_sum[tid] / count[tid]
        y_true_trial.append(trial_true[tid])
        y_pred_trial.append(int(np.argmax(mean_score)))
    y_true_trial = np.array(y_true_trial)
    y_pred_trial = np.array(y_pred_trial)
    return y_true_trial, y_pred_trial


def _trial_metrics(y_true_trial, y_pred_trial):
    acc = float(np.mean(y_true_trial == y_pred_trial))
    f1 = float(f1_score(y_true_trial, y_pred_trial, average="macro", zero_division=0))
    return acc, f1


def _fit_svm_model(kernel="rbf", c=1.0):
    # probability=False keeps it fast; use decision_function for score aggregation.
    return make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=c, gamma="scale", decision_function_shape="ovr"),
    )


def run_cross_session_svm(base_path, kernel="rbf", c=1.0):
    loader = SEEDIVLoader()
    subject_ids, accs, f1s = [], [], []

    for sid in range(1, 16):
        paths = _find_subject_session_paths(base_path, sid)
        if paths is None:
            continue
        s1_f, s1_l = loader.load_single_session(paths[0], session_id=1)
        s2_f, s2_l = loader.load_single_session(paths[1], session_id=2)
        s3_f, s3_l = loader.load_single_session(paths[2], session_id=3)
        train_trials = s1_f + s2_f
        train_labels = np.concatenate([s1_l, s2_l])
        test_trials = s3_f
        test_labels = s3_l

        x_tr, y_tr, _ = _make_window_dataset(train_trials, np.array(train_labels))
        x_te, y_te, tid_te = _make_window_dataset(test_trials, np.array(test_labels))

        model = _fit_svm_model(kernel=kernel, c=c)
        model.fit(x_tr, y_tr)

        te_scores = model.decision_function(x_te)
        y_true_trial, y_pred_trial = _aggregate_trial_predictions(te_scores, tid_te, y_te)
        trial_acc, trial_f1 = _trial_metrics(y_true_trial, y_pred_trial)
        subject_ids.append(sid)
        accs.append(trial_acc)
        f1s.append(trial_f1)
        print(f"Cross-session subject {sid:02d} | Acc {trial_acc:.4f} F1 {trial_f1:.4f}")

    return {
        "subjects": subject_ids,
        "accs": accs,
        "f1s": f1s,
        "mean_acc": float(np.mean(accs)),
        "std_acc": float(np.std(accs)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        # compatibility keys
        "trial_acc": float(np.mean(accs)),
        "trial_f1": float(np.mean(f1s)),
    }


def run_loso_svm(base_path, kernel="rbf", c=1.0):
    loader = SEEDIVLoader()
    feats, labels, subjects = loader.load_all_subjects(base_path=base_path, subject_ids=None)
    labels = np.asarray(labels).astype(int)
    subjects = np.asarray(subjects).astype(int)
    feats_obj = np.empty(len(feats), dtype=object)
    for i, trial in enumerate(feats):
        feats_obj[i] = trial

    logo = LeaveOneGroupOut()
    fold_accs, fold_f1s = [], []
    fold_subjects = []

    for train_idx, test_idx in logo.split(feats_obj, labels, groups=subjects):
        sid = int(subjects[test_idx[0]])
        x_train_trials = [feats_obj[i] for i in train_idx]
        y_train_trials = labels[train_idx]
        x_test_trials = [feats_obj[i] for i in test_idx]
        y_test_trials = labels[test_idx]

        x_tr, y_tr, _ = _make_window_dataset(x_train_trials, y_train_trials)
        x_te, y_te, tid_te = _make_window_dataset(x_test_trials, y_test_trials)

        model = _fit_svm_model(kernel=kernel, c=c)
        model.fit(x_tr, y_tr)
        te_scores = model.decision_function(x_te)
        y_true_trial, y_pred_trial = _aggregate_trial_predictions(te_scores, tid_te, y_te)
        acc, f1 = _trial_metrics(y_true_trial, y_pred_trial)

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


def run_subject_dependent_svm(base_path, kernel="rbf", c=1.0, seed=42):
    loader = SEEDIVLoader()
    subject_accs, subject_f1s = [], []
    subject_ids = []

    for sid in range(1, 16):
        trials, labels = loader.load_single_subject(sid, base_path=base_path)
        labels = np.asarray(labels).astype(int)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + sid)
        train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))

        x_train_trials = [trials[i] for i in train_idx]
        y_train_trials = labels[train_idx]
        x_test_trials = [trials[i] for i in test_idx]
        y_test_trials = labels[test_idx]

        x_tr, y_tr, _ = _make_window_dataset(x_train_trials, y_train_trials)
        x_te, y_te, tid_te = _make_window_dataset(x_test_trials, y_test_trials)

        model = _fit_svm_model(kernel=kernel, c=c)
        model.fit(x_tr, y_tr)
        te_scores = model.decision_function(x_te)
        y_true_trial, y_pred_trial = _aggregate_trial_predictions(te_scores, tid_te, y_te)
        acc, f1 = _trial_metrics(y_true_trial, y_pred_trial)

        subject_ids.append(sid)
        subject_accs.append(acc)
        subject_f1s.append(f1)
        print(f"Subject-dependent subject {sid:02d} | Acc {acc:.4f} F1 {f1:.4f}")

    return {
        "subjects": subject_ids,
        "accs": subject_accs,
        "f1s": subject_f1s,
        "mean_acc": float(np.mean(subject_accs)),
        "std_acc": float(np.std(subject_accs)),
        "mean_f1": float(np.mean(subject_f1s)),
        "std_f1": float(np.std(subject_f1s)),
    }


def _safe_load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_one_figure_comparison(svm_root):
    deep_cross = _safe_load_json(os.path.join("analysis_figures", "summary_metrics.json"))
    deep_loso = _safe_load_json(os.path.join("loso_analysis", "loso_summary.json"))
    deep_sd = _safe_load_json(os.path.join("subject_dependent_analysis", "subject_dependent_summary.json"))

    svm_cross = _safe_load_json(os.path.join(svm_root, "cross_session_summary.json"))
    svm_loso = _safe_load_json(os.path.join(svm_root, "loso_summary.json"))
    svm_sd = _safe_load_json(os.path.join(svm_root, "subject_dependent_summary.json"))

    if None in [deep_cross, deep_loso, deep_sd, svm_cross, svm_loso, svm_sd]:
        print("Skip comparison figure: some summary json files are missing.")
        return

    paradigms = ["Cross-session", "LOSO", "Subject-dependent"]

    deep_cross_acc = deep_cross.get("mean_acc", deep_cross.get("s3_trial_acc"))
    deep_cross_acc_std = deep_cross.get("std_acc", 0.0)
    deep_cross_f1 = deep_cross.get("mean_f1", deep_cross.get("s3_trial_f1"))
    deep_cross_f1_std = deep_cross.get("std_f1", 0.0)

    deep_acc = [
        deep_cross_acc,
        deep_loso["mean_acc"],
        deep_sd["mean_acc"],
    ]
    deep_acc_std = [
        deep_cross_acc_std,
        deep_loso["std_acc"],
        deep_sd["std_acc"],
    ]
    deep_f1 = [
        deep_cross_f1,
        deep_loso["mean_f1"],
        deep_sd["mean_f1"],
    ]
    deep_f1_std = [
        deep_cross_f1_std,
        deep_loso["std_f1"],
        deep_sd["std_f1"],
    ]

    svm_acc = [
        svm_cross["mean_acc"],
        svm_loso["mean_acc"],
        svm_sd["mean_acc"],
    ]
    svm_acc_std = [
        svm_cross["std_acc"],
        svm_loso["std_acc"],
        svm_sd["std_acc"],
    ]
    svm_f1 = [
        svm_cross["mean_f1"],
        svm_loso["mean_f1"],
        svm_sd["mean_f1"],
    ]
    svm_f1_std = [
        svm_cross["std_f1"],
        svm_loso["std_f1"],
        svm_sd["std_f1"],
    ]

    x = np.arange(len(paradigms))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].bar(x - width / 2, deep_acc, width, yerr=deep_acc_std, capsize=3, label="EEGNet")
    axes[0].bar(x + width / 2, svm_acc, width, yerr=svm_acc_std, capsize=3, label="SVM")
    axes[0].set_xticks(x, paradigms)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Trial-level Accuracy")
    axes[0].legend()

    axes[1].bar(x - width / 2, deep_f1, width, yerr=deep_f1_std, capsize=3, label="EEGNet")
    axes[1].bar(x + width / 2, svm_f1, width, yerr=svm_f1_std, capsize=3, label="SVM")
    axes[1].set_xticks(x, paradigms)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Trial-level Macro-F1")
    axes[1].legend()

    fig.suptitle("EEGNet vs SVM Baseline (One Figure)")
    fig.tight_layout()
    fig.savefig(os.path.join(svm_root, "eegnet_vs_svm_one_figure.png"), dpi=180)
    plt.close(fig)


def main():
    base_path = r"C:\Users\xinji\Desktop\archive\seed_iv\eeg_feature_smooth"
    kernel = "rbf"
    c = 1.0

    out_root = "svm_analysis"
    os.makedirs(out_root, exist_ok=True)

    print("Running SVM cross-session baseline...")
    cross = run_cross_session_svm(base_path, kernel=kernel, c=c)
    with open(os.path.join(out_root, "cross_session_summary.json"), "w", encoding="utf-8") as f:
        json.dump(cross, f, ensure_ascii=False, indent=2)
    print(f"Cross-session: Acc {cross['mean_acc']:.4f}±{cross['std_acc']:.4f} F1 {cross['mean_f1']:.4f}±{cross['std_f1']:.4f}")

    print("\nRunning SVM LOSO baseline...")
    loso = run_loso_svm(base_path, kernel=kernel, c=c)
    with open(os.path.join(out_root, "loso_summary.json"), "w", encoding="utf-8") as f:
        json.dump(loso, f, ensure_ascii=False, indent=2)
    print(f"LOSO: Acc {loso['mean_acc']:.4f}±{loso['std_acc']:.4f} F1 {loso['mean_f1']:.4f}±{loso['std_f1']:.4f}")

    print("\nRunning SVM subject-dependent baseline...")
    sd = run_subject_dependent_svm(base_path, kernel=kernel, c=c, seed=42)
    with open(os.path.join(out_root, "subject_dependent_summary.json"), "w", encoding="utf-8") as f:
        json.dump(sd, f, ensure_ascii=False, indent=2)
    print(f"Subject-dependent: Acc {sd['mean_acc']:.4f}±{sd['std_acc']:.4f} F1 {sd['mean_f1']:.4f}±{sd['std_f1']:.4f}")

    plot_one_figure_comparison(out_root)
    print(f"\nSaved SVM summaries and figure to: {out_root}")


if __name__ == "__main__":
    main()
