import sounddevice as sd
import numpy as np
import queue
import wave

# Audio queue for recording
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def record_audio(filename="live_audio.wav", duration=10, samplerate=16000):
    print("🎙 Recording...")
    with sd.InputStream(samplerate=samplerate, channels=1, callback=callback):
        audio_data = []
        for _ in range(int(duration * samplerate / 1024)):
            audio_data.append(audio_queue.get())

    audio_array = np.concatenate(audio_data, axis=0)
    wavefile = wave.open(filename, 'wb')
    wavefile.setnchannels(1)
    wavefile.setsampwidth(2)
    wavefile.setframerate(samplerate)
    wavefile.writeframes((audio_array * 32767).astype(np.int16).tobytes())
    wavefile.close()

    print("✅ Recording complete.")
    return filename
