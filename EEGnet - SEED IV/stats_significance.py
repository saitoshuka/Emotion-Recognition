import os
import json
import math
import numpy as np


def _load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _holm_correction(pvals):
    """
    Holm-Bonferroni adjusted p-values.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(running_max, 1.0)
    return adj


def _cohens_d_paired(x, y):
    d = np.array(x, dtype=float) - np.array(y, dtype=float)
    std = np.std(d, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(d) / std)


def _paired_tests(x, y):
    """
    Return Wilcoxon and paired t-test p-values.
    Requires scipy; if unavailable, returns None for p-values.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    out = {
        "n": int(len(x)),
        "mean_diff": float(np.mean(x - y)),
        "cohens_d_paired": _cohens_d_paired(x, y),
        "wilcoxon_p": None,
        "ttest_p": None,
    }
    try:
        from scipy.stats import wilcoxon, ttest_rel  # type: ignore

        # zero_method='wilcox' ignores exact zero-differences in ranks.
        _, w_p = wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
        _, t_p = ttest_rel(x, y, alternative="two-sided")
        out["wilcoxon_p"] = float(w_p)
        out["ttest_p"] = float(t_p)
    except Exception:
        pass
    return out


def _extract_arrays(deep, svm):
    # cross-session summary may store per_subject_* or accs/f1s
    deep_acc = deep.get("per_subject_acc", deep.get("accs"))
    deep_f1 = deep.get("per_subject_f1", deep.get("f1s"))
    svm_acc = svm.get("accs")
    svm_f1 = svm.get("f1s")
    if deep_acc is None or deep_f1 is None or svm_acc is None or svm_f1 is None:
        raise ValueError("Missing per-subject/fold arrays in one of summary json files.")
    return np.array(deep_acc, dtype=float), np.array(deep_f1, dtype=float), np.array(svm_acc, dtype=float), np.array(svm_f1, dtype=float)


def _extract_deep_arrays_with_fallback(summary_dict, npz_path):
    deep_acc = summary_dict.get("per_subject_acc", summary_dict.get("accs"))
    deep_f1 = summary_dict.get("per_subject_f1", summary_dict.get("f1s"))
    if deep_acc is not None and deep_f1 is not None:
        return np.array(deep_acc, dtype=float), np.array(deep_f1, dtype=float)

    if os.path.exists(npz_path):
        arr = np.load(npz_path, allow_pickle=True)
        if "accs" in arr and "f1s" in arr:
            return np.array(arr["accs"], dtype=float), np.array(arr["f1s"], dtype=float)

    raise ValueError(
        f"Missing per-subject/fold arrays in deep summary and fallback npz: {npz_path}"
    )


def main():
    paths = {
        "deep_cross": os.path.join("analysis_figures", "summary_metrics.json"),
        "deep_loso": os.path.join("loso_analysis", "loso_summary.json"),
        "deep_sd": os.path.join("subject_dependent_analysis", "subject_dependent_summary.json"),
        "svm_cross": os.path.join("svm_analysis", "cross_session_summary.json"),
        "svm_loso": os.path.join("svm_analysis", "loso_summary.json"),
        "svm_sd": os.path.join("svm_analysis", "subject_dependent_summary.json"),
    }

    data = {k: _load_json(v) for k, v in paths.items()}

    deep_cross_acc, deep_cross_f1 = _extract_deep_arrays_with_fallback(
        data["deep_cross"], npz_path=os.path.join("analysis_figures", "results_arrays.npz")
    )
    deep_loso_acc, deep_loso_f1 = _extract_deep_arrays_with_fallback(
        data["deep_loso"], npz_path=os.path.join("loso_analysis", "loso_results.npz")
    )
    deep_sd_acc, deep_sd_f1 = _extract_deep_arrays_with_fallback(
        data["deep_sd"], npz_path=os.path.join("subject_dependent_analysis", "subject_dependent_results.npz")
    )
    svm_cross_acc = np.array(data["svm_cross"].get("accs"), dtype=float)
    svm_cross_f1 = np.array(data["svm_cross"].get("f1s"), dtype=float)
    svm_loso_acc = np.array(data["svm_loso"].get("accs"), dtype=float)
    svm_loso_f1 = np.array(data["svm_loso"].get("f1s"), dtype=float)
    svm_sd_acc = np.array(data["svm_sd"].get("accs"), dtype=float)
    svm_sd_f1 = np.array(data["svm_sd"].get("f1s"), dtype=float)

    # Ensure lengths are aligned for paired tests
    pairs = [
        ("Cross-session", "Accuracy", deep_cross_acc, svm_cross_acc),
        ("Cross-session", "Macro-F1", deep_cross_f1, svm_cross_f1),
        ("LOSO", "Accuracy", deep_loso_acc, svm_loso_acc),
        ("LOSO", "Macro-F1", deep_loso_f1, svm_loso_f1),
        ("Subject-dependent", "Accuracy", deep_sd_acc, svm_sd_acc),
        ("Subject-dependent", "Macro-F1", deep_sd_f1, svm_sd_f1),
    ]

    results = []
    raw_wilcoxon_p = []
    for paradigm, metric, x, y in pairs:
        if len(x) != len(y):
            raise ValueError(f"Length mismatch in {paradigm}/{metric}: {len(x)} vs {len(y)}")
        res = _paired_tests(x, y)
        res.update(
            {
                "paradigm": paradigm,
                "metric": metric,
                "eegnet_mean": float(np.mean(x)),
                "svm_mean": float(np.mean(y)),
                "eegnet_std": float(np.std(x, ddof=1)),
                "svm_std": float(np.std(y, ddof=1)),
            }
        )
        results.append(res)
        raw_wilcoxon_p.append(1.0 if res["wilcoxon_p"] is None else res["wilcoxon_p"])

    holm = _holm_correction(raw_wilcoxon_p)
    for i, r in enumerate(results):
        r["wilcoxon_p_holm"] = float(holm[i]) if r["wilcoxon_p"] is not None else None

    out_dir = "comparison_figures"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "significance_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tests": results}, f, ensure_ascii=False, indent=2)

    print("Significance report saved:", out_path)
    print("\nEEGNet vs SVM paired tests:")
    for r in results:
        wp = "NA" if r["wilcoxon_p"] is None else f"{r['wilcoxon_p']:.4g}"
        wh = "NA" if r["wilcoxon_p_holm"] is None else f"{r['wilcoxon_p_holm']:.4g}"
        tp = "NA" if r["ttest_p"] is None else f"{r['ttest_p']:.4g}"
        print(
            f"- {r['paradigm']} | {r['metric']} | "
            f"EEGNet {r['eegnet_mean']:.4f} vs SVM {r['svm_mean']:.4f} | "
            f"d={r['cohens_d_paired']:.3f} | Wilcoxon p={wp} (Holm={wh}) | t-test p={tp}"
        )

    try:
        import scipy  # type: ignore
        _ = scipy.__version__
    except Exception:
        print("\nNote: scipy not available; p-values are NA. Install scipy to enable significance tests.")


if __name__ == "__main__":
    main()
