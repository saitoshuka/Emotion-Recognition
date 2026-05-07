import os
import json
import numpy as np
import matplotlib.pyplot as plt


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)




def _load_significance_map(path):
    d = _load_json(path)
    if d is None:
        return {}

    m = {}
    for t in d.get("tests", []):
        paradigm = str(t.get("paradigm", ""))
        metric = str(t.get("metric", ""))
        if metric == "Accuracy":
            metric_key = "acc"
        elif metric == "Macro-F1":
            metric_key = "f1"
        else:
            continue

        p = t.get("wilcoxon_p_holm", t.get("wilcoxon_p", None))
        if p is None:
            continue
        m[(paradigm, metric_key)] = float(p)
    return m


def _p_to_stars(p):
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def _annotate_sig(ax, x_left, x_right, y, label):
    h = 0.015
    ax.plot([x_left, x_left, x_right, x_right], [y, y + h, y + h, y], color="black", lw=1.0)
    ax.text((x_left + x_right) / 2, y + h + 0.005, label, ha="center", va="bottom", fontsize=10)

def _pick_mean_std(d, mean_key, std_key, fallback_mean=None, fallback_std=0.0):
    if d is None:
        return None, None
    if mean_key in d:
        mean_val = float(d[mean_key])
    elif fallback_mean is not None and fallback_mean in d:
        mean_val = float(d[fallback_mean])
    else:
        return None, None
    std_val = float(d[std_key]) if std_key in d else float(fallback_std)
    return mean_val, std_val


def plot_comparison(
    out_path="model_comparison_one_figure.png",
    deep_cross_path=os.path.join("analysis_figures", "summary_metrics.json"),
    deep_loso_path=os.path.join("loso_analysis", "loso_summary.json"),
    deep_sd_path=os.path.join("subject_dependent_analysis", "subject_dependent_summary.json"),
    svm_cross_path=os.path.join("svm_analysis", "cross_session_summary.json"),
    svm_loso_path=os.path.join("svm_analysis", "loso_summary.json"),
    svm_sd_path=os.path.join("svm_analysis", "subject_dependent_summary.json"),
    sig_report_path=os.path.join("comparison_figures", "significance_report.json"),
):
    deep_cross = _load_json(deep_cross_path)
    deep_loso = _load_json(deep_loso_path)
    deep_sd = _load_json(deep_sd_path)
    svm_cross = _load_json(svm_cross_path)
    svm_loso = _load_json(svm_loso_path)
    svm_sd = _load_json(svm_sd_path)
    sig_map = _load_significance_map(sig_report_path)

    if None in [deep_cross, deep_loso, deep_sd, svm_cross, svm_loso, svm_sd]:
        missing = [
            p for p, d in [
                (deep_cross_path, deep_cross),
                (deep_loso_path, deep_loso),
                (deep_sd_path, deep_sd),
                (svm_cross_path, svm_cross),
                (svm_loso_path, svm_loso),
                (svm_sd_path, svm_sd),
            ] if d is None
        ]
        raise FileNotFoundError(f"Missing summary json files: {missing}")

    deep_cross_acc, deep_cross_acc_std = _pick_mean_std(
        deep_cross, "mean_acc", "std_acc", fallback_mean="s3_trial_acc", fallback_std=0.0
    )
    deep_cross_f1, deep_cross_f1_std = _pick_mean_std(
        deep_cross, "mean_f1", "std_f1", fallback_mean="s3_trial_f1", fallback_std=0.0
    )
    deep_loso_acc, deep_loso_acc_std = _pick_mean_std(deep_loso, "mean_acc", "std_acc")
    deep_loso_f1, deep_loso_f1_std = _pick_mean_std(deep_loso, "mean_f1", "std_f1")
    deep_sd_acc, deep_sd_acc_std = _pick_mean_std(deep_sd, "mean_acc", "std_acc")
    deep_sd_f1, deep_sd_f1_std = _pick_mean_std(deep_sd, "mean_f1", "std_f1")

    svm_cross_acc, svm_cross_acc_std = _pick_mean_std(
        svm_cross, "mean_acc", "std_acc", fallback_mean="trial_acc", fallback_std=0.0
    )
    svm_cross_f1, svm_cross_f1_std = _pick_mean_std(
        svm_cross, "mean_f1", "std_f1", fallback_mean="trial_f1", fallback_std=0.0
    )
    svm_loso_acc, svm_loso_acc_std = _pick_mean_std(svm_loso, "mean_acc", "std_acc")
    svm_loso_f1, svm_loso_f1_std = _pick_mean_std(svm_loso, "mean_f1", "std_f1")
    svm_sd_acc, svm_sd_acc_std = _pick_mean_std(svm_sd, "mean_acc", "std_acc")
    svm_sd_f1, svm_sd_f1_std = _pick_mean_std(svm_sd, "mean_f1", "std_f1")

    paradigms = ["Cross-session", "LOSO", "Subject-dependent"]
    deep_acc = [deep_cross_acc, deep_loso_acc, deep_sd_acc]
    deep_acc_std = [deep_cross_acc_std, deep_loso_acc_std, deep_sd_acc_std]
    deep_f1 = [deep_cross_f1, deep_loso_f1, deep_sd_f1]
    deep_f1_std = [deep_cross_f1_std, deep_loso_f1_std, deep_sd_f1_std]

    svm_acc = [svm_cross_acc, svm_loso_acc, svm_sd_acc]
    svm_acc_std = [svm_cross_acc_std, svm_loso_acc_std, svm_sd_acc_std]
    svm_f1 = [svm_cross_f1, svm_loso_f1, svm_sd_f1]
    svm_f1_std = [svm_cross_f1_std, svm_loso_f1_std, svm_sd_f1_std]

    x = np.arange(len(paradigms))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(x - width / 2, deep_acc, width, yerr=deep_acc_std, capsize=3, label="EEGNet")
    axes[0].bar(x + width / 2, svm_acc, width, yerr=svm_acc_std, capsize=3, label="SVM")
    axes[0].set_xticks(x, paradigms)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Trial-level Accuracy")
    axes[0].legend()


    for k, paradigm in enumerate(paradigms):
        p = sig_map.get((paradigm, "acc"), None)
        if p is None:
            continue
        y = min(0.96, max(deep_acc[k] + deep_acc_std[k], svm_acc[k] + svm_acc_std[k]) + 0.02)
        _annotate_sig(axes[0], x[k] - width / 2, x[k] + width / 2, y, _p_to_stars(p))

    axes[1].bar(x - width / 2, deep_f1, width, yerr=deep_f1_std, capsize=3, label="EEGNet")
    axes[1].bar(x + width / 2, svm_f1, width, yerr=svm_f1_std, capsize=3, label="SVM")
    axes[1].set_xticks(x, paradigms)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Trial-level Macro-F1")
    axes[1].legend()


    for k, paradigm in enumerate(paradigms):
        p = sig_map.get((paradigm, "f1"), None)
        if p is None:
            continue
        y = min(0.96, max(deep_f1[k] + deep_f1_std[k], svm_f1[k] + svm_f1_std[k]) + 0.02)
        _annotate_sig(axes[1], x[k] - width / 2, x[k] + width / 2, y, _p_to_stars(p))

    fig.suptitle("EEGNet vs SVM (Three Paradigms)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    out = plot_comparison(out_path=os.path.join("comparison_figures", "eegnet_vs_svm_one_figure.png"))
    print(f"Saved comparison figure to: {out}")


if __name__ == "__main__":
    os.makedirs("comparison_figures", exist_ok=True)
    main()
