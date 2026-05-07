import os
import numpy as np
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from load_data import SEEDIVLoader


TARGET_TIME = 42


def find_subject_session_paths(base_path, subject_id):
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


def pad_or_crop_trial(trial, target_time=TARGET_TIME):
    if trial.shape[1] > target_time:
        return trial[:, :target_time, :]
    if trial.shape[1] < target_time:
        pad = np.repeat(trial[:, -1:, :], target_time - trial.shape[1], axis=1)
        return np.concatenate([trial, pad], axis=1)
    return trial


def trial_to_windows(trial):
    trial = pad_or_crop_trial(trial, TARGET_TIME)
    return trial.transpose(1, 0, 2).reshape(TARGET_TIME, -1)  # (T, 62*5)


def make_window_dataset(trials, labels):
    x_list, y_list, tid_list = [], [], []
    for i, (trial, lab) in enumerate(zip(trials, labels)):
        w = trial_to_windows(trial)
        x_list.append(w)
        y_list.append(np.full(TARGET_TIME, int(lab), dtype=np.int64))
        tid_list.append(np.full(TARGET_TIME, i, dtype=np.int64))
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    tids = np.concatenate(tid_list, axis=0)
    return x, y, tids


def aggregate_trial_predictions(scores, tids, y_true_window):
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
    return np.array(y_true_trial), np.array(y_pred_trial)


def trial_metrics(y_true_trial, y_pred_trial):
    acc = float(np.mean(y_true_trial == y_pred_trial))
    f1 = float(f1_score(y_true_trial, y_pred_trial, average="macro", zero_division=0))
    return acc, f1


def fit_svm_model(kernel="rbf", c=1.0):
    return make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=c, gamma="scale", decision_function_shape="ovr"),
    )


def get_loader():
    return SEEDIVLoader()
