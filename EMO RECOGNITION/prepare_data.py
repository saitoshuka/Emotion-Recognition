import numpy as np
from load_data import load_deapdata
from feature_extraction import extract_eeg_features

def prepare_dataset(file_path, selected_channels, fs=128):
    data, labels, channel_names = load_deapdata(file_path,selected_channels=selected_channels)

    X = extract_eeg_features(data)
    X = np.array(X)
    y = labels[:, :2]  # Valence and Arousal

    return X, y

# Example usage
if __name__ == "__main__":
    file_path = './data_preprocessed_python/s01.dat'
    selected_channels = ['Fp1', 'Fp2']  # Example channels
    X, y = prepare_dataset(file_path, selected_channels, fs=128)
    print(X)

    print("Combined features shape:", X.shape)
    print("Labels shape:", y.shape)
