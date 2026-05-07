import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

from deap_experiment_utils import load_subject_windows


data_dir = "C:/Users/xinji/Desktop/data_preprocessed_python"
files = sorted([f for f in os.listdir(data_dir) if f.startswith('s') and f.endswith('.dat')])

results = []

print(f"{'Subject':<10} | {'Valence':<10} | {'Arousal':<10}")
print("-" * 40)

for f_name in files:
    file_path = os.path.join(data_dir, f_name)

    X, y, groups = load_subject_windows(
        file_path,
        trim_post_sec=3.0,
        use_ratio_start=0.5,
        use_ratio_end=1.0,
        window_size=4.0,
        step_size=2.0,
        cache_root="deap_prepared_cache",
        use_cache=True,
        auto_write_cache=True,
    )
    if X is None or len(X) == 0:
        continue

    X_flat = X.reshape(X.shape[0], -1)
    y_bin = (y > 5).astype(int)

    gkf = GroupKFold(n_splits=3)
    sub_scores = []

    for task_idx in [0, 1]:
        scores = []
        for train_idx, test_idx in gkf.split(X_flat, y_bin[:, task_idx], groups=groups):
            clf = SVC(kernel='rbf', C=1.0)
            clf.fit(X_flat[train_idx], y_bin[train_idx, task_idx])
            pred = clf.predict(X_flat[test_idx])
            scores.append(accuracy_score(y_bin[test_idx, task_idx], pred))
        sub_scores.append(np.mean(scores))

    print(f"{f_name:<10} | {sub_scores[0]:.4f}   | {sub_scores[1]:.4f}")
    results.append(sub_scores)

if results:
    avg_res = np.mean(results, axis=0)
    print("-" * 40)
    print(f"Average     | {avg_res[0]:.4f}   | {avg_res[1]:.4f}")
