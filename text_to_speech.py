import os

# Suppress pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import time
from gtts import gTTS

def speak(text):
    tts = gTTS(text=text, lang="en", tld="co.in")  # Indian English accent
    filename = "temp_audio.mp3"
    tts.save(filename)  # Save speech to a file

    pygame.mixer.init()  # Initialize pygame without messages
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(0.1)

    pygame.mixer.quit()  # Properly close pygame
    os.remove(filename)  # Delete the file
