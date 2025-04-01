import speech_recognition as sr
import requests
from gtts import gTTS
import os

API_URL = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
HEADERS = {"Authorization": "Bearer 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"}

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    os.system("mpv --no-video response.mp3") 
    os.remove("response.mp3") 

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source) 
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Sorry, I couldn't understand.")
        return None
    except sr.RequestError:
        print("Speech recognition service is unavailable.")
        return None

def chat_with_ai(prompt):
    try:
        payload = {"inputs": prompt}
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        if response.ok:
            return response.json()[0]["generated_text"]
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return "Sorry, I couldn't process that."
    except Exception as e:
        print(f"Error in AI chat: {e}")
        return "Sorry, something went wrong."

def main():
    print("AI Assistant is ready. Say 'exit' to end the conversation.")
    while True:
        user_input = speech_to_text()
        if user_input:
            if "exit" == user_input.lower():
                text_to_speech("Goodbye!")
                print("AI: Goodbye!")
                break
            ai_response = chat_with_ai(user_input)
            print(f"AI: {ai_response}")
            text_to_speech(ai_response)
        else:
            print("AI: I didn't catch that. Could you repeat?")

if __name__ == "__main__":
    main()