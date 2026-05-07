import argparse
import json
from pathlib import Path

from deap_experiment_utils import save_binary_confusion_figure


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _plot_from_one_summary(summary_path: Path, out_dir: Path, title_prefix: str):
    data = _load_json(summary_path)

    for task in ["valence", "arousal"]:
        if task not in data or "cm_sum" not in data[task]:
            raise KeyError(f"Missing {task}.cm_sum in {summary_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    save_binary_confusion_figure(
        data["valence"]["cm_sum"],
        save_path=str(out_dir / "confusion_valence.png"),
        title=f"{title_prefix} | Valence Confusion Matrix (Sum)",
    )
    save_binary_confusion_figure(
        data["arousal"]["cm_sum"],
        save_path=str(out_dir / "confusion_arousal.png"),
        title=f"{title_prefix} | Arousal Confusion Matrix (Sum)",
    )

    print(f"Saved: {out_dir / 'confusion_valence.png'}")
    print(f"Saved: {out_dir / 'confusion_arousal.png'}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Plot DEAP binary confusion matrices directly from existing summary.json files"
    )
    p.add_argument(
        "--summary",
        type=str,
        default="",
        help="Path to a single summary.json. If empty, plot for all default DEAP experiment folders.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory (only used with --summary). Default: same folder as summary.json",
    )
    p.add_argument(
        "--title-prefix",
        type=str,
        default="DEAP",
        help="Figure title prefix (only used with --summary)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    if args.summary:
        summary_path = Path(args.summary)
        if not summary_path.is_absolute():
            summary_path = (root / summary_path).resolve()
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)

        out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent
        if not out_dir.is_absolute():
            out_dir = (root / out_dir).resolve()

        _plot_from_one_summary(summary_path, out_dir, args.title_prefix)
        return

    targets = [
        (
            root / "deap_eegnet_subject_dependent" / "summary.json",
            root / "deap_eegnet_subject_dependent",
            "EEGNet Subject-dependent",
        ),
        (
            root / "deap_eegnet_loso" / "summary.json",
            root / "deap_eegnet_loso",
            "EEGNet LOSO",
        ),
        (
            root / "deap_svm_subject_dependent" / "summary.json",
            root / "deap_svm_subject_dependent",
            "SVM Subject-dependent",
        ),
        (
            root / "deap_svm_loso" / "summary.json",
            root / "deap_svm_loso",
            "SVM LOSO",
        ),
    ]

    ok = 0
    for summary_path, out_dir, title_prefix in targets:
        if not summary_path.exists():
            print(f"Skip missing: {summary_path}")
            continue
        try:
            _plot_from_one_summary(summary_path, out_dir, title_prefix)
            ok += 1
        except Exception as e:
            print(f"Failed: {summary_path} | {e}")

    if ok == 0:
        raise RuntimeError("No confusion matrices were generated. Check summary.json files.")


if __name__ == "__main__":
    main()
