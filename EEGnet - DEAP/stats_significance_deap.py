import os
import json
import numpy as np
from scipy.stats import wilcoxon, ttest_rel


def _load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_metric_map(summary, task, metric_key):
    task_dict = summary[task]
    subjects = task_dict.get("subjects", summary.get("subjects"))
    if subjects is None:
        raise KeyError(f"Missing subjects for task={task} in summary")
    values = task_dict[metric_key]
    if len(subjects) != len(values):
        raise ValueError(
            f"Length mismatch for task={task}, metric={metric_key}: "
            f"subjects={len(subjects)} vs values={len(values)}"
        )
    return dict(zip(subjects, values))


def _align_by_subject(eeg_summary, svm_summary, task, metric_key):
    eeg_map = _build_metric_map(eeg_summary, task, metric_key)
    svm_map = _build_metric_map(svm_summary, task, metric_key)
    common = sorted(set(eeg_map.keys()) & set(svm_map.keys()))
    if len(common) == 0:
        raise ValueError("No common subject IDs between EEGNet and SVM summaries.")
    a = np.array([eeg_map[s] for s in common], dtype=float)
    b = np.array([svm_map[s] for s in common], dtype=float)
    return common, a, b


def _cohens_d_paired(x, y):
    d = x - y
    sd = np.std(d, ddof=1) if len(d) > 1 else 0.0
    if sd == 0:
        return 0.0
    return float(np.mean(d) / sd)


def _holm_correction(pvals):
    pvals = np.array(pvals, dtype=float)
    m = len(pvals)
    idx = np.argsort(pvals)
    sorted_p = pvals[idx]
    adj = np.empty(m, dtype=float)
    running = 0.0
    for i in range(m):
        val = (m - i) * sorted_p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    out = np.empty(m, dtype=float)
    out[idx] = adj
    return out.tolist()


def main():
    eeg_sd = _load_json(os.path.join("deap_eegnet_subject_dependent", "summary.json"))
    eeg_loso = _load_json(os.path.join("deap_eegnet_loso", "summary.json"))
    svm_sd = _load_json(os.path.join("deap_svm_subject_dependent", "summary.json"))
    svm_loso = _load_json(os.path.join("deap_svm_loso", "summary.json"))

    tests = []
    raw_wilcoxon_p = []

    configs = [
        ("Subject-dependent", eeg_sd, svm_sd),
        ("LOSO", eeg_loso, svm_loso),
    ]

    for paradigm, eeg_s, svm_s in configs:
        for task in ["valence", "arousal"]:
            for metric_name, key in [("Accuracy", "accs"), ("Macro-F1", "f1s")]:
                subjects, eeg_arr, svm_arr = _align_by_subject(eeg_s, svm_s, task, key)

                w_p = wilcoxon(eeg_arr, svm_arr, zero_method="wilcox", alternative="two-sided").pvalue
                t_p = ttest_rel(eeg_arr, svm_arr).pvalue
                d = _cohens_d_paired(eeg_arr, svm_arr)

                tests.append(
                    {
                        "paradigm": paradigm,
                        "task": task,
                        "metric": metric_name,
                        "n": len(subjects),
                        "eegnet_mean": float(np.mean(eeg_arr)),
                        "svm_mean": float(np.mean(svm_arr)),
                        "cohens_d": d,
                        "wilcoxon_p": float(w_p),
                        "ttest_p": float(t_p),
                    }
                )
                raw_wilcoxon_p.append(float(w_p))

    holm = _holm_correction(raw_wilcoxon_p)
    for i, p_adj in enumerate(holm):
        tests[i]["wilcoxon_holm_p"] = float(p_adj)

    out_dir = "deap_model_comparison"
    os.makedirs(out_dir, exist_ok=True)

    out_json = os.path.join(out_dir, "significance_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"tests": tests}, f, ensure_ascii=False, indent=2)

    out_txt = os.path.join(out_dir, "significance_results.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("EEGNet vs SVM paired tests (DEAP)\n")
        for t in tests:
            line = (
                f"- {t['paradigm']} | {t['task']} | {t['metric']} | "
                f"EEGNet {t['eegnet_mean']:.4f} vs SVM {t['svm_mean']:.4f} | "
                f"d={t['cohens_d']:.3f} | "
                f"Wilcoxon p={t['wilcoxon_p']:.4g} (Holm={t['wilcoxon_holm_p']:.4g}) | "
                f"t-test p={t['ttest_p']:.4g}\n"
            )
            f.write(line)
            print(line.strip())

    print(f"Saved: {out_json}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
