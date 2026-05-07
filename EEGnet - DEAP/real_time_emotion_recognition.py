import numpy as np
import csv
import json
import asyncio
from pylsl import StreamInlet, resolve_stream
from tensorflow.keras.models import load_model
from feature_extraction import extract_eeg_features
import websockets

# Load the trained M2 model
model_path = '/Users/vonralphdaneherbuela/Desktop/EmoEst_EEG_CNNLSTM_PP_2_v1/model.keras'
m2_model = load_model(model_path)

# Define the channels and sampling frequency
selected_channels = ['Fp1', 'Fp2']
channel_names = ['Fp1', 'Fp2']
fs = 100  # Sampling frequency for biosignalsplux

# Define the mapping of valence and arousal to emotions
emotion_map = [
    ("sleepy", 0.01, -1.00),
    ("tired", -0.01, -1.00),
    ("afraid", -0.12, 0.79),
    ("angry", -0.40, 0.79),
    ("calm", 0.78, -0.67),
    ("relaxed", 0.71, -0.65),
    ("content", 0.81, -0.55),
    ("depressed", -0.81, -0.48),
    ("discontent", -0.68, -0.32),
    ("determined", 0.89, 0.35),
    ("happy", 0.89, 0.17),
    ("anxious", -0.72, -0.80),
    ("good", 0.78, 0.35),
    ("pensive", 0.03, -0.60),
    ("impressed", 0.39, 0.06),
    ("frustrated", -0.60, 0.40),
    ("disappointed", -0.80, -0.03),
    ("bored", -0.35, -0.78),
    ("annoyed", -0.44, 0.76),
    ("enraged", -0.18, 0.83),
    ("excited", 0.70, 0.71),
    ("melancholy", -0.65, -0.65),
    ("satisfied", 0.77, -0.63),
    ("distressed", -0.76, 0.83),
    ("uncomfortable", -0.68, -0.37),
    ("worried", -0.07, 0.32),
    ("amused", 0.55, -0.02),
    ("apathetic", -0.20, -0.12),
    ("peaceful", 0.55, -0.60),
    ("contemplative", 0.58, -0.60),
    ("embarrassed", -0.31, -0.61),
    ("sad", -0.81, -0.40),
    ("hopeful", 0.61, 0.40),
    ("pleased", 0.89, -0.10)
]

# Function to map valence and arousal to the closest emotion label
def get_emotion_label(valence, arousal):
    closest_emotion = None
    min_distance = float('inf')
    for emotion, e_valence, e_arousal in emotion_map:
        distance = np.sqrt((valence - e_valence) ** 2 + (arousal - e_arousal) ** 2)
        print(f"Emotion: {emotion}, Valence: {e_valence}, Arousal: {e_arousal}, Distance: {distance}")
        if distance < min_distance:
            min_distance = distance
            closest_emotion = emotion
    print(f"Chosen Emotion: {closest_emotion}, Valence: {valence}, Arousal: {arousal}")
    return closest_emotion

# Function to preprocess and extract features
def preprocess_data(eeg_data, fs, selected_channels, channel_names):
    eeg_features = extract_eeg_features(eeg_data, fs=fs, selected_channels=selected_channels, channel_names=channel_names, apply_ica=True)
    return eeg_features

# Function to predict emotion
def predict_emotion(eeg_data, fs, selected_channels, channel_names):
    try:
        preprocessed_data = preprocess_data(eeg_data, fs, selected_channels, channel_names).reshape(1, -1, 1)
        raw_predictions = m2_model.predict(preprocessed_data)
        valence, arousal = raw_predictions[0, 0], raw_predictions[0, 1]

        # Print predictions for debugging
        print(f"Predictions: Valence={valence}, Arousal={arousal}")

        # Get the emotion label using predicted valence and arousal values
        emotion_label = get_emotion_label(valence, arousal)

        return float(valence), float(arousal), emotion_label  # Convert to native Python float
    except Exception as e:
        print(f"Error in predicting emotion: {e}")
        return None, None, None

# Resolve an available OpenSignals stream
print("# Looking for an available OpenSignals stream...")
os_stream = resolve_stream("name", "OpenSignals")

# Create an inlet to receive signal samples from the stream
inlet = StreamInlet(os_stream[0])

print('inlet.info().name() =', inlet.info().name())
print('inlet.info().type() =', inlet.info().type())
print('inlet.info().channel_count() =', inlet.info().channel_count())
print('inlet.info().nominal_srate() =', inlet.info().nominal_srate())

# Initialize buffers for EEG data
eeg_buffer = []
eeg_min_length = 128

# Open a CSV file to save the results
with open('/Users/vonralphdaneherbuela/Desktop/EmoEst_EEG_CNNLSTM_PP_2_v1/Results/EEG_P01.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Valence', 'Arousal', 'Emotion',
                     'Theta', 'Alpha', 'Low Beta', 'High Beta', 'Delta',
                     'Fp1 FFT Component 1', 'Fp1 FFT Component 2', 'Fp1 FFT Component 3', 'Fp1 FFT Component 4', 'Fp1 FFT Component 5',
                     'Fp2 FFT Component 1', 'Fp2 FFT Component 2', 'Fp2 FFT Component 3', 'Fp2 FFT Component 4', 'Fp2 FFT Component 5',
                     'Fp1 Mean', 'Fp2 Mean', 'Fp1 Std', 'Fp2 Std'])

    async def main():
        async with websockets.connect('ws://localhost:8767') as websocket:
            while True:
                try:
                    sample, timestamp = inlet.pull_sample()
                    eeg_data = sample[:2]

                    eeg_buffer.append(eeg_data)
                    print(f"EEG buffer length: {len(eeg_buffer)}")

                    if len(eeg_buffer) >= eeg_min_length:
                        eeg_buffer_array = np.array(eeg_buffer[-eeg_min_length:]).T

                        valence, arousal, emotion_label = predict_emotion(eeg_buffer_array, fs, selected_channels, channel_names)
                        if valence is not None and arousal is not None:
                            print(f"Timestamp: {timestamp}, Valence: {valence}, Arousal: {arousal}, Emotion: {emotion_label}")
                            features = extract_eeg_features(eeg_buffer_array, fs=fs, selected_channels=selected_channels, channel_names=channel_names, apply_ica=True)
                            writer.writerow([timestamp, valence, arousal, emotion_label] + list(features))

                            # Send data to WebSocket
                            data = {"valence": valence, "arousal": arousal, "emotion": emotion_label, "source": "EEG"}
                            await websocket.send(json.dumps(data))
                            print(f"Sent data to WebSocket: {data}")
                        else:
                            print(f"Timestamp: {timestamp}, Prediction Error")
                    else:
                        print("Not enough data, skipping this sample.")
                except Exception as e:
                    print(f"Error: {e}")

    asyncio.run(main())
