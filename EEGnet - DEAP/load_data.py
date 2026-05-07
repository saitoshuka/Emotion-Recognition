import pickle
import numpy as np


def load_deapdata(file_path, selected_channels=None):
    """Load one DEAP subject file.

    Returns:
      data: (40, 32, 8064) EEG channels only
      labels: (40, 2) [valence, arousal]
      channel_names: list[str]
    """
    try:
        with open(file_path, 'rb') as file:
            data_original = pickle.load(file, encoding='latin1')
    except Exception as e:
        print(f"Error: failed to load {file_path}")
        print(f"Detail: {e}")
        return None, None, None

    raw_data = data_original['data']
    raw_labels = data_original['labels']

    eeg_channels = [
        'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
        'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
        'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
        'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
    ]

    if selected_channels is None:
        data = raw_data[:, :32, :]
        channel_names = eeg_channels
    else:
        try:
            indices = [eeg_channels.index(ch) for ch in selected_channels]
            data = raw_data[:, indices, :]
            channel_names = selected_channels
        except ValueError as e:
            print(f"Channel name error: {e}")
            return None, None, None

    labels = raw_labels[:, :2]
    return data, labels, channel_names


def remove_baseline(data, fs=128, baseline_sec=3):
    """Subtract baseline mean and remove baseline segment.

    data: (trials, channels, samples)
    """
    baseline_pts = int(baseline_sec * fs)
    baseline_values = np.mean(data[:, :, :baseline_pts], axis=2, keepdims=True)
    data_stim = data[:, :, baseline_pts:]
    return data_stim - baseline_values


if __name__ == "__main__":
    test_file = "C:/Users/xinji/Desktop/data_preprocessed_python/s01.dat"
    data, labels, chans = load_deapdata(test_file)
    if data is not None:
        print("Raw shape:", data.shape)
        data_clean = remove_baseline(data)
        print("After baseline removal:", data_clean.shape)
        print("Labels shape:", labels.shape)
        print("First labels:\n", labels[:5])
