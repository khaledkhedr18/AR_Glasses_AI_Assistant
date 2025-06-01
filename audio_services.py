import pyttsx3
from vosk import Model, KaldiRecognizer
import json
import sounddevice as sd
import numpy as np
import os
from config import vosk_model_paths

class TextToSpeech:
    def __init__(self, driver_name='sapi5'):

        self.engine = pyttsx3.init(driverName=driver_name)

    def speak(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

class SpeechRecognizer:
    def __init__(self, initial_language_code="en"):
        self.model_path = vosk_model_paths.get(initial_language_code)
        self.model = Model(self.model_path)
        self.recognizer = None 


    def get_audio(self, display_output=None, display_output2=None):
        recognizer = KaldiRecognizer(self.model, 16000)
        recognized_text = None
        stream = None

        def callback(indata, frames, time, status):
            nonlocal recognized_text
            if status:
                pass
            if recognizer.AcceptWaveform(indata[:, 0].tobytes()):
                result = json.loads(recognizer.Result())
                recognized_text = result.get("text", "")
                print(f"You said: {recognized_text}")
                if display_output:
                    display_output(f"You said: {recognized_text}")

        try:
            print("Listening...")
            if display_output:
                display_output("Listening... Speak now")
            
            with sd.InputStream(callback=callback, channels=1, samplerate=16000, dtype=np.int16) as stream:
                while recognized_text is None:
                    sd.sleep(100)
            
            return recognized_text
            
        finally:
            if stream is not None and not stream.closed:
                stream.close()
            del recognizer

    def change_model(self, language_code):
            new_model_path = vosk_model_paths.get(language_code)
            if new_model_path !=self.model_path:
                self.model = Model(new_model_path)
                self.model_path = new_model_path
                print(f"Model changed to {language_code}.")

if __name__ == '__main__':
    tts = TextToSpeech()
    tts.speak("Hello from the Text to Speech service.")

    recognizer = SpeechRecognizer()
    recognized_text = recognizer.get_audio()
    print(f"Recognized text: {recognized_text}")
    recognizer.change_model("en")  # Change to Spanish model