import numpy as np
import scipy.signal as signal
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def bandpower(data, sf, band, window_sec=4, relative=False):
    band = np.asarray(band)
    low, high = band
    nperseg = min(window_sec * sf, len(data))
    if nperseg < 2:
        return 0  # Return 0 if not enough data for bandpower calculation
    freqs, psd = signal.welch(data, sf, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    bp = np.sum(psd[idx_band]) * freq_res
    if relative:
        bp /= np.sum(psd)
    return bp

def bandpower2(data, sf, band, window_sec=8, relative=False):
    band = np.asarray(band)
    low, high = band
    nperseg = min(window_sec * sf, len(data))
    if nperseg < 2:
        return 0

    # ハニング窓を使用し、50%オーバーラップを指定
    freqs, psd = signal.welch(data, sf, nperseg=nperseg, noverlap=nperseg//2, window='hann')
    
    # 周波数分解能の計算
    freq_res = freqs[1] - freqs[0]
    
    # 指定された周波数帯のインデックスを取得
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    
    # バンドパワーの計算
    bp = np.sum(psd[idx_band]) * freq_res
    
    if relative:
        # 相対パワーの計算（ノイズの影響を減らすため、全周波数帯のパワーの合計を使用）
        total_power = np.sum(psd) * freq_res
        bp /= total_power if total_power != 0 else 1
    
    return bp

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    y = signal.filtfilt(b, a, data)
    return y

"""def extract_eeg_features(eeg_data, fs=128, apply_ica=False, apply_filter=False):
    features = []
    for trial in eeg_data:
        if apply_filter:
            # Apply bandpass filter to each channel
            for i in range(trial.shape[0]):
                trial[i, :] = bandpass_filter(trial[i, :], 4, 45.0, fs)  

        if apply_ica:
            ica = FastICA(n_components=trial.shape[0], max_iter=1000, tol=0.01, whiten='True')
            trial = ica.fit_transform(trial.T).T

        bands = {
            "delta": [0.5, 4],
            "theta": [4, 8],
            "alpha": [8, 13],
            "low_beta": [13, 20],
            "high_beta": [20, 30],
            #"gamma": [31, 40]
        }
        trial_features = []
        for channel in trial:
            for band in bands:
                freq = bands[band]
                trial_features.append(bandpower(channel, fs, freq))
            # FFT features
            #fft_values = np.abs(np.fft.fft(channel))[:len(channel) // 2]
            #trial_features.extend(fft_values[:5])  # First 5 FFT components
        
        # Standardize the trial features
        #print(trial_features)
        scaler = StandardScaler()
        trial_features = scaler.fit_transform(np.array(trial_features).reshape(-1, 1)).flatten()
        features.append(trial_features)
    return np.array(features)

"""

def extract_eeg_features(eeg_data, fs=128, apply_ica=False, apply_filter=False):
    features = []
    #print(eeg_data.shape)  # データの形状を出力
    # 各試行（サンプル）を取り出す
    for trial in eeg_data:
        # trialの形状は (脳波のチャンネル数, 時系列データ)
        
        # バンドパスフィルタを適用する場合
        if apply_filter:
            for i in range(trial.shape[0]):
                # 各チャンネルの時系列データに対してバンドパスフィルタを適用
                trial[i, :] = bandpass_filter(trial[i, :], 0.5, 45.0, fs)  

        # ICAを適用する場合
        if apply_ica:
            ica = FastICA(n_components=trial.shape[0], max_iter=1000, tol=0.01, whiten='unit-variance')
            trial = ica.fit_transform(trial.T).T

        # 周波数帯域の定義
        bands = {
            "delta": [0.5, 4],
            "theta": [4, 8],
            "alpha": [8, 13],
            "low_beta": [13, 20],
            "high_beta": [20, 30],
            #"gamma": [31, 40]
        }
        
        trial_features = []
        # 各チャンネルに対してバンドパワーを計算
        for channel in trial:
            # channelの形状は (時系列データの長さ,)
            for band in bands:
                freq = bands[band]
                # 各周波数帯域のバンドパワーを計算
                trial_features.append(bandpower(channel, fs, freq))
        #print(trial_features)
        # 各試行の特徴量を標準化
        
        scaler = StandardScaler()
        #scaler = RobustScaler()
        trial_features = scaler.fit_transform(np.array(trial_features).reshape(-1, 1)).flatten()
        
        # -1から1の範囲に正規化
        #scaler_minmax = MinMaxScaler(feature_range=(-1, 1))
        #trial_features = scaler_minmax.fit_transform(trial_features.reshape(-1, 1)).flatten()
        
        # 各試行の特徴量をfeaturesリストに追加
        features.append(trial_features)
    # 特徴量をNumPy配列に変換
    features = np.array(features)
    
    return features

if __name__ == "__main__":
    # For preprocessed data (no ICA needed)
    preprocessed_eeg_sample = np.random.rand(40, 2, 8064)  # Example 2-channel EEG data with 128 samples
    channel_names = ['Fp1', 'Fp2']
    selected_channels = ['Fp1', 'Fp2']
    features_preprocessed = extract_eeg_features(preprocessed_eeg_sample,  fs=128, apply_ica=False, apply_filter=True)
    print("Extracted features (preprocessed data) shape:", features_preprocessed.shape)
    print(features_preprocessed)

    # For raw data (ICA applied)
    raw_eeg_sample = np.random.rand(40, 2, 8064)  # Example 2-channel raw EEG data with 100 samples
    features_raw = extract_eeg_features(raw_eeg_sample,  fs=128, apply_ica=True, apply_filter=True)
    print("Extracted features (raw data) shape:", features_raw.shape)
    #print(features_raw)