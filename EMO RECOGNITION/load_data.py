import pickle

'''


def load_deap_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    # Here we assume the channel names are as follows for the DEAP dataset
    channel_names = ['Fp1', 'Fp2']  # Update this list based on actual channels used in DEAP
    return data['data'], data['labels'], channel_names


'''

def load_deapdata(file_path, selected_channels=None):
    # データの読み込み
    with open(file_path, 'rb') as file:
        data_original = pickle.load(file, encoding='latin1')

    # EEGデータの抽出
    data = data_original['data']

    all_channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2']
    # チャンネル名のインデックスを取得
    # AF
    if selected_channels is None:
        selected_channels = all_channels
    channel_indices = [all_channels.index(ch) for ch in selected_channels]
    print(f"抽出したチャンネルインデックス:{channel_indices}")

    # 指定されたチャンネルのデータを抽出
    selected_data = data[:, channel_indices, :]

    # ラベルの抽出
    labels = data_original['labels']
    labels = labels[:,:2]
    print(selected_channels)
    
    
    return selected_data, labels, selected_channels  # チャンネル名を追加して返す

# Example usage
if __name__ == "__main__":
    file_path = './data_preprocessed_python/s01.dat'  # Update with the correct path to your data file

    channels = ['Fp1','Fp2']
    data, labels, channel_names = load_deapdata(file_path,selected_channels=channels)

    print("Data shape:", data.shape)
    print("Labels shape:", labels.shape)
    print("Channel names:", channel_names)
