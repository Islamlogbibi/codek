# this is the code for ai chat 
# it's ollama ai tiny model with text to speech and speech to text
# it's a simple chatbot that can chat with you

import numpy as np
import requests
import time
import threading
import edge_tts
import asyncio
import sounddevice as sd
from faster_whisper import WhisperModel
import queue
import os
import pygame
import tempfile
from gtts import gTTS

print("Loading Whisper model...")
model = WhisperModel("tiny.en")
print("Whisper model loaded!")


RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 8

is_playing = False
audio_queue = queue.Queue()

def record_audio():
    global is_playing
    if is_playing:
        return None  
    
    print("Say something...")
    recording = sd.rec(int(RATE * RECORD_SECONDS), samplerate=RATE, channels=CHANNELS, dtype='int16')
    sd.wait()
    print("Recording finished.")
    
    audio_data = recording.flatten().astype(np.float32) / 32768.0
    return audio_data

def generate_response(text):
    if "move forward" in text.lower():
        return "Moving forward"
    elif "turn left" in text.lower():
        return "Turning left"
    elif "turn right" in text.lower():
        return "Turning right"
    elif "stop" in text.lower():
        return "Stopping now"
    
    try:
        print("Asking Ollama...")
        response = requests.post('http://localhost:11434/api/generate',
                                 json={
                                     'model': 'llama2',
                                     'prompt': text,
                                     'system': 'You are a helpful robot assistant do not use emojis in your responses. Your name is Codek and you have been built by the HTIC club at Badji Mokhtar Annaba University. Keep responses brief and friendly and do not use emojis.',
                                     'stream': False
                                 })
        if response.status_code == 200:
            return response.json()['response']
        else:
            return "I'm having trouble thinking right now."
    except Exception as e:
        print(f"Ollama error: {e}")
        return "I'm having trouble connecting to my brain."

def speak_response(text):
    global is_playing
    print(f"Speaking: {text}")
    is_playing = True  
    
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_mp3 = fp.name
        tts.save(temp_mp3)
        
        pygame.mixer.init()
        pygame.mixer.music.load(temp_mp3)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
    except Exception as e:
        print(f"Error during TTS: {e}")
    finally:
        is_playing = False  
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)


def main():


    while True:
        try:
            while is_playing:
                time.sleep(0.1)

            audio_data = record_audio()
            if audio_data is None:
                continue

            print("Processing with Whisper...")
            segments, info = model.transcribe(audio_data)
            recognized_text = " ".join(segment.text for segment in segments)
            print(f"Recognized: {recognized_text}")

            if len(recognized_text.strip()) < 2:
                print("Nothing recognized, try again")
                continue

            response = generate_response(recognized_text)
            print(f"Response: {response}")

            asyncio.run(speak_response(response))

        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

# made with mabrouk logbibi and ali bahri
# for a competition of robotics in biskra
# we are the team of badji mokhtar annaba university HTIC club