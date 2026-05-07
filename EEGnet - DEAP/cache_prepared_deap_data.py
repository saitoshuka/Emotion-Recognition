import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from prepare_data import prepare_dataset_single_subject


def _fmt_float(x):
    s = f"{x:.4f}"
    s = s.rstrip("0").rstrip(".")
    return s if s else "0"


def make_config_key(window_size, step_size, trim_post_sec, use_ratio_start, use_ratio_end):
    return (
        f"w{_fmt_float(window_size)}"
        f"_s{_fmt_float(step_size)}"
        f"_t{_fmt_float(trim_post_sec)}"
        f"_r{_fmt_float(use_ratio_start)}-{_fmt_float(use_ratio_end)}"
    )


def cache_one_subject(file_path, out_npz, window_size, step_size, trim_post_sec, use_ratio_start, use_ratio_end):
    X, y, groups = prepare_dataset_single_subject(
        file_path,
        window_size=window_size,
        step_size=step_size,
        trim_post_sec=trim_post_sec,
        use_ratio_start=use_ratio_start,
        use_ratio_end=use_ratio_end,
    )
    if X is None or y is None or groups is None:
        return None

    np.savez_compressed(out_npz, X=X, y=y, groups=groups)
    return {
        "n_samples": int(X.shape[0]),
        "feature_shape": list(X.shape[1:]),
        "label_shape": list(y.shape),
        "n_trials_after_windowing": int(len(np.unique(groups))),
    }


def build_cache(
    data_dir,
    out_dir,
    window_size=1.0,
    step_size=0.5,
    trim_post_sec=3.0,
    use_ratio_start=0.5,
    use_ratio_end=1.0,
    force=False,
):
    files = sorted([f for f in os.listdir(data_dir) if f.startswith("s") and f.endswith(".dat")])
    if not files:
        raise RuntimeError(f"No subject files found under: {data_dir}")

    cfg_key = make_config_key(window_size, step_size, trim_post_sec, use_ratio_start, use_ratio_end)
    cfg_dir = os.path.join(out_dir, cfg_key)
    os.makedirs(cfg_dir, exist_ok=True)

    index = {
        "data_dir": data_dir,
        "config": {
            "window_size": window_size,
            "step_size": step_size,
            "trim_post_sec": trim_post_sec,
            "use_ratio_start": use_ratio_start,
            "use_ratio_end": use_ratio_end,
        },
        "config_key": cfg_key,
        "subjects": {},
        "cached_files": [],
        "skipped_existing": [],
        "failed": [],
    }

    for name in tqdm(files, desc="Caching DEAP prepared data"):
        src = os.path.join(data_dir, name)
        stem = os.path.splitext(name)[0]
        dst = os.path.join(cfg_dir, f"{stem}.npz")

        if (not force) and os.path.exists(dst):
            index["skipped_existing"].append(name)
            index["cached_files"].append(dst)
            continue

        try:
            meta = cache_one_subject(
                file_path=src,
                out_npz=dst,
                window_size=window_size,
                step_size=step_size,
                trim_post_sec=trim_post_sec,
                use_ratio_start=use_ratio_start,
                use_ratio_end=use_ratio_end,
            )
            if meta is None:
                index["failed"].append(name)
                continue
            index["subjects"][name] = meta
            index["cached_files"].append(dst)
        except Exception as e:
            index["failed"].append({"subject": name, "error": str(e)})

    index_path = os.path.join(cfg_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    print("\n=== Cache Summary ===")
    print(f"Config key: {cfg_key}")
    print(f"Cached files: {len(index['cached_files'])}")
    print(f"Skipped existing: {len(index['skipped_existing'])}")
    print(f"Failed: {len(index['failed'])}")
    print(f"Saved index: {index_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Cache DEAP prepared window features per subject.")
    p.add_argument("--data-dir", type=str, default="C:/Users/xinji/Desktop/data_preprocessed_python")
    p.add_argument("--out-dir", type=str, default="deap_prepared_cache")
    p.add_argument("--window-size", type=float, default=1.0)
    p.add_argument("--step-size", type=float, default=0.5)
    p.add_argument("--trim-post-sec", type=float, default=3.0)
    p.add_argument("--use-ratio-start", type=float, default=0.5)
    p.add_argument("--use-ratio-end", type=float, default=1.0)
    p.add_argument("--force", action="store_true", help="Rebuild cache even if .npz already exists.")
    return p.parse_args()


def main():
    args = parse_args()
    build_cache(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        window_size=args.window_size,
        step_size=args.step_size,
        trim_post_sec=args.trim_post_sec,
        use_ratio_start=args.use_ratio_start,
        use_ratio_end=args.use_ratio_end,
        force=args.force,
    )


if __name__ == "__main__":
    main()
