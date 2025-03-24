from record_audio import record_audio
from speech_to_text import transcribe_speech
from translate import translate_text
from intent_classifier import predict_intent, intent_to_response
from text_to_speech import speak

input("🎤 Press Enter to start recording...")
audio_file = record_audio()
hindi_text = transcribe_speech(audio_file)
translated_text = translate_text(hindi_text)
intent = predict_intent(translated_text)
response = intent_to_response.get(intent, "No response available.")

print(f"💬 Response: {response}")
speak(response)
