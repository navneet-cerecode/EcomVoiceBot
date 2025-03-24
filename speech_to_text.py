import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load Wav2Vec2 model (once)
print("🔄 Loading speech recognition model...")
processor = Wav2Vec2Processor.from_pretrained("ai4bharat/indicwav2vec-hindi")
model = Wav2Vec2ForCTC.from_pretrained("ai4bharat/indicwav2vec-hindi")
print("✅ Speech recognition model loaded.")

def transcribe_speech(audio_path):
    print(f"📝 Transcribing {audio_path}...")
    waveform, rate = torchaudio.load(audio_path)
    inputs = processor(waveform.squeeze(0), sampling_rate=rate, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    
    print(f"📝 Transcribed Text: {transcription}")
    return transcription
