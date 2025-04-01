import numpy as np
import requests
import time
import threading
import edge_tts
import asyncio
import sounddevice as sd
import faster_whisper
import queue

# Load Faster Whisper model (optimized for English)
print("Loading Whisper model...")
model = faster_whisper.WhisperModel("tiny.en")
print("Whisper model loaded!")

# Constants for audio
RATE = 16000
CHANNELS = 1
RECORD_SECONDS = 5

# Global flag to track if TTS is playing
is_playing = False
audio_queue = queue.Queue()

def record_audio():
    """Record audio from the microphone if not playing TTS."""
    global is_playing
    if is_playing:
        return None  # Skip recording if TTS is active
    
    print("Say something...")
    recording = sd.rec(int(RATE * RECORD_SECONDS), samplerate=RATE, channels=CHANNELS, dtype='int16')
    sd.wait()
    print("Recording finished.")
    
    # Normalize the audio data
    audio_data = recording.flatten().astype(np.float32) / 32768.0
    return audio_data

def generate_response(text):
    """Generate response using Ollama with predefined robot commands."""
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
                                     'system': 'You are a helpful robot assistant. Your name is Codek and you have been built by the HTIC club at Badji Mokhtar Annaba University. Keep responses brief and friendly and do not use emojis.',
                                     'stream': False
                                 })
        if response.status_code == 200:
            return response.json()['response']
        else:
            return "I'm having trouble thinking right now."
    except Exception as e:
        print(f"Ollama error: {e}")
        return "I'm having trouble connecting to my brain."

async def speak_response(text):
    global is_playing
    is_playing = True
    try:
        print(f"Speaking: {text}")
        communicator = edge_tts.Communicate(text, "en-US-AriaNeural")
        # Stream audio chunks and add them to a global queue for playback
        await communicator.stream_async(audio_queue.put)
        await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Error during TTS: {e}")
    finally:
        is_playing = False

def audio_player():
    """Play audio from the queue using sounddevice."""
    import sounddevice as sd
    while True:
        data = audio_queue.get()
        if data is None:
            break
        # Play the audio data (assumed to be in a format acceptable to sounddevice)
        sd.play(data, samplerate=24000)
        sd.wait()

def main():
    print("Test program for robot assistant with faster TTS and Whisper")
    print("Press Ctrl+C to exit")

    player_thread = threading.Thread(target=audio_player, daemon=True)
    player_thread.start()

    while True:
        try:
            # Wait until audio playback is done before recording
            while is_playing:
                time.sleep(0.1)

            audio_data = record_audio()
            if audio_data is None:
                continue

            print("Processing with Whisper...")
            segments, info = model.transcribe(audio_data)
            # Concatenate all segments into one string
            recognized_text = " ".join(segment.text for segment in segments)
            print(f"Recognized: {recognized_text}")

            if len(recognized_text.strip()) < 2:
                print("Nothing recognized, try again")
                continue

            response = generate_response(recognized_text)
            print(f"Response: {response}")

            # Run TTS asynchronously and wait for it to complete
            asyncio.run(speak_response(response))

        except KeyboardInterrupt:
            print("Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
