import pyaudio
import wave
import whisper

def record_audio(filename, duration=5, channels=1, sample_rate=44100, chunk=1024):
    audio = pyaudio.PyAudio()
    
    # Open the audio stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)
    
    print("Recording...")
    
    frames = []
    
    # Record audio for the specified duration
    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)
    
    print("Finished recording.")
    
    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    # Write the recorded audio to a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

def transcribe_audio(filename):
    model = whisper.load_model("base")
    result = model.transcribe(filename, fp16=False)
    return result["text"]

if __name__ == "__main__":
    filename = "recorded_audio.wav"
    record_audio(filename)
    print(f"Audio recorded and saved to '{filename}'.")

    transcribed_text = transcribe_audio(filename)
    print("Transcribed text:")
    print(transcribed_text)
