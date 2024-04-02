import tensorflow_hub as hub
import numpy as np
import librosa
import pyaudio

# Load pre-trained audio classification model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/yamnet/1")

# Function to capture live audio from the microphone
def capture_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    stream.stop_stream()
    stream.close()
    audio.terminate()

    return np.concatenate(frames)

# Function to classify audio segment
def classify_audio(audio):
    # Extract features (e.g., Mel spectrogram)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=RATE)
    input_data = mel_spec[np.newaxis, :, :, np.newaxis]

    # Perform inference
    logits, embeddings, spectrogram = model(input_data)
    predictions = np.mean(logits, axis=0)

    # Get top prediction
    top_class = np.argmax(predictions)

    return top_class

# Main loop for real-time audio processing
while True:
    # Capture live audio
    audio_chunk = capture_audio()

    # Classify audio
    class_index = classify_audio(audio_chunk)

    # Check if it's a vehicle sound (specific class index based on your model)
    if class_index == vehicle_class_index:
        trigger_alarm()
