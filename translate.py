from googletrans import Translator

# Load translator
translator = Translator()

def translate_text(hindi_text):
    print(f"🌍 Translating: {hindi_text}")
    translated_text = translator.translate(hindi_text, src="hi", dest="en").text
    print(f"✅ Translated Text: {translated_text}")
    return translated_text
