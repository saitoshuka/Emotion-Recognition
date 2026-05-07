import numpy as np
from scipy.signal import welch
from load_data import load_deapdata, remove_baseline


def compute_de(segment, fs=128):
    """Compute differential entropy features for one EEG segment.
    segment: (32, time_points)
    return: (32, 5)
    """
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 14),
        'beta': (14, 31),
        # fs=128 -> Nyquist=64, keep gamma under Nyquist margin
        'gamma': (33, 45),
    }

    feats = []
    for ch in range(segment.shape[0]):
        freqs, psd = welch(segment[ch], fs, nperseg=fs)
        row = []
        for low, high in bands.values():
            idx = np.logical_and(freqs >= low, freqs <= high)
            power = np.sum(psd[idx])
            row.append(np.log(power + 1e-8))
        feats.append(row)
    return np.asarray(feats, dtype=np.float32)


def smooth_trial_features(trial_features, window_len=5):
    """Temporal smoothing inside one trial.
    trial_features: (T, 32, 5)
    """
    kernel = np.ones(window_len, dtype=np.float32) / float(window_len)
    smoothed = np.zeros_like(trial_features)
    for ch in range(32):
        for band in range(5):
            smoothed[:, ch, band] = np.convolve(trial_features[:, ch, band], kernel, mode='same')
    return smoothed


def prepare_dataset_single_subject(
    file_path,
    window_size=1.0,
    step_size=0.5,
    trim_post_sec=0.0,
    use_ratio_start=0.5,
    use_ratio_end=1.0,
):
    """Load one subject and build DE windows.

    Notes:
    - Baseline is removed with first 3s in load_data.remove_baseline.
    - Additional early stimulus trimming is controlled by trim_post_sec.
    - No global normalization here to avoid leakage; normalize per fold in training.
    """
    print(f"Processing {file_path.split('/')[-1]} (DE + smoothing)...")

    data, labels, _ = load_deapdata(file_path)
    if data is None:
        return None, None, None

    data = remove_baseline(data, fs=128, baseline_sec=3)
    trim_pts = int(trim_post_sec * 128)
    if trim_pts > 0:
        data = data[:, :, trim_pts:]

    fs = 128
    win_pts = int(window_size * fs)
    step_pts = int(step_size * fs)

    x_final, y_final, groups = [], [], []

    for trial_idx in range(data.shape[0]):
        trial_data = data[trial_idx]      # (32, T)
        trial_label = labels[trial_idx]   # (2,)

        trial_de = []
        start = 0
        while start + win_pts <= trial_data.shape[1]:
            seg = trial_data[:, start:start + win_pts]
            trial_de.append(compute_de(seg, fs))
            start += step_pts

        if len(trial_de) == 0:
            continue

        trial_de = np.asarray(trial_de, dtype=np.float32)  # (Tw,32,5)
        trial_de = smooth_trial_features(trial_de, window_len=5)

        total_steps = trial_de.shape[0]
        keep_start = int(total_steps * use_ratio_start)
        keep_end = int(total_steps * use_ratio_end)
        keep_start = max(0, min(keep_start, total_steps))
        keep_end = max(keep_start, min(keep_end, total_steps))

        for t in range(keep_start, keep_end):
            x_final.append(trial_de[t])
            y_final.append(trial_label)
            groups.append(trial_idx)

    X = np.asarray(x_final, dtype=np.float32)  # (N,32,5)
    y = np.asarray(y_final, dtype=np.float32)  # (N,2)
    g = np.asarray(groups, dtype=np.int64)

    print(f"Prepared: N={X.shape[0]}, feature_shape={X.shape[1:]}")
    return X, y, g


if __name__ == "__main__":
    file_path = "C:/Users/xinji/Desktop/data_preprocessed_python/s01.dat"
    X, y, g = prepare_dataset_single_subject(file_path, window_size=1, step_size=0.5, trim_post_sec=3.0)
    if X is not None:
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("groups shape:", g.shape)
