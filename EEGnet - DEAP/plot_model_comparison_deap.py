import os
import json
import numpy as np
import matplotlib.pyplot as plt


def _load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_mean_std(summary, task, metric):
    return summary[task][f"mean_{metric}"], summary[task][f"std_{metric}"]




def _load_significance_map(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sig_map = {}
    for t in data.get("tests", []):
        paradigm = str(t.get("paradigm", ""))
        task = str(t.get("task", "")).lower()
        metric = str(t.get("metric", ""))
        if metric == "Accuracy":
            metric_key = "acc"
        elif metric == "Macro-F1":
            metric_key = "f1"
        else:
            continue

        p = t.get("wilcoxon_holm_p", t.get("wilcoxon_p", None))
        if p is None:
            continue
        sig_map[(paradigm, task, metric_key)] = float(p)
    return sig_map


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

def _mean_curve(curves):
    if len(curves) == 0:
        return None
    max_len = max(len(c) for c in curves)
    arr = np.full((len(curves), max_len), np.nan, dtype=float)
    for i, c in enumerate(curves):
        arr[i, : len(c)] = c
    return np.nanmean(arr, axis=0)


def _plot_training_curves(sd_hist, loso_hist, out_dir):
    tasks = ["valence", "arousal"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i, task in enumerate(tasks):
        sd_train, sd_val = [], []
        for subj_data in sd_hist.values():
            for fold in subj_data.get(task, []):
                sd_train.append(fold.get("train_loss", []))
                sd_val.append(fold.get("val_loss", []))

        loso_train, loso_val = [], []
        for subj_name in loso_hist.get(task, {}):
            h = loso_hist[task][subj_name]
            loso_train.append(h.get("train_loss", []))
            loso_val.append(h.get("val_loss", []))

        sd_train_m = _mean_curve(sd_train)
        sd_val_m = _mean_curve(sd_val)
        lo_train_m = _mean_curve(loso_train)
        lo_val_m = _mean_curve(loso_val)

        ax1 = axes[i, 0]
        if sd_train_m is not None:
            ax1.plot(np.arange(1, len(sd_train_m) + 1), sd_train_m, label="Train")
        if sd_val_m is not None:
            ax1.plot(np.arange(1, len(sd_val_m) + 1), sd_val_m, label="Val")
        ax1.set_title(f"SD {task.capitalize()} Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        ax2 = axes[i, 1]
        if lo_train_m is not None:
            ax2.plot(np.arange(1, len(lo_train_m) + 1), lo_train_m, label="Train")
        if lo_val_m is not None:
            ax2.plot(np.arange(1, len(lo_val_m) + 1), lo_val_m, label="Val")
        ax2.set_title(f"LOSO {task.capitalize()} Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()

    fig.suptitle("EEGNet Training Curves (Mean over folds/subjects)")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "eegnet_training_curves_deap.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    eeg_sd = _load_json(os.path.join("deap_eegnet_subject_dependent", "summary.json"))
    eeg_loso = _load_json(os.path.join("deap_eegnet_loso", "summary.json"))
    svm_sd = _load_json(os.path.join("deap_svm_subject_dependent", "summary.json"))
    svm_loso = _load_json(os.path.join("deap_svm_loso", "summary.json"))

    paradigms = ["Subject-dependent", "LOSO"]
    tasks = ["valence", "arousal"]
    metrics = ["acc", "f1"]
    sig_map = _load_significance_map(os.path.join("deap_model_comparison", "significance_results.json"))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    width = 0.35
    x = np.arange(len(paradigms))

    for i, task in enumerate(tasks):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]

            eeg_means = [
                _get_mean_std(eeg_sd, task, metric)[0],
                _get_mean_std(eeg_loso, task, metric)[0],
            ]
            eeg_stds = [
                _get_mean_std(eeg_sd, task, metric)[1],
                _get_mean_std(eeg_loso, task, metric)[1],
            ]

            svm_means = [
                _get_mean_std(svm_sd, task, metric)[0],
                _get_mean_std(svm_loso, task, metric)[0],
            ]
            svm_stds = [
                _get_mean_std(svm_sd, task, metric)[1],
                _get_mean_std(svm_loso, task, metric)[1],
            ]

            ax.bar(x - width / 2, eeg_means, width, yerr=eeg_stds, capsize=3, label="EEGNet")
            ax.bar(x + width / 2, svm_means, width, yerr=svm_stds, capsize=3, label="SVM")

            ax.set_xticks(x, paradigms)
            ax.set_ylim(0, 1)
            ax.set_title(f"{task.capitalize()} - {'Accuracy' if metric == 'acc' else 'Macro-F1'}")
            if i == 0 and j == 0:
                ax.legend()

            for k, paradigm in enumerate(paradigms):
                p = sig_map.get((paradigm, task, metric), None)
                if p is None:
                    continue
                top_left = eeg_means[k] + eeg_stds[k]
                top_right = svm_means[k] + svm_stds[k]
                y = min(0.96, max(top_left, top_right) + 0.02)
                _annotate_sig(ax, x[k] - width / 2, x[k] + width / 2, y, _p_to_stars(p))

    fig.suptitle("DEAP: EEGNet vs SVM (Subject-dependent & LOSO)")
    fig.tight_layout()

    out_dir = "deap_model_comparison"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eegnet_vs_svm_deap.png")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")

    sd_hist = _load_json(os.path.join("deap_eegnet_subject_dependent", "histories.json"))
    loso_hist = _load_json(os.path.join("deap_eegnet_loso", "histories.json"))
    _plot_training_curves(sd_hist, loso_hist, out_dir)


if __name__ == "__main__":
    main()
