from vosk import Model, KaldiRecognizer
import wave
import json

class KaldiRecognizer:
    def __init__(self, model_path, vocab_path, grammar_path=None):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.grammar_path = grammar_path

    def recognize_text_from_speech(self, audio_file):
        """
        Convert audio file to text using Vosk's KaldiRecognizer
        Args:
            audio_file (str): Path to the audio file
        Returns:
            str: Recognized text or None if recognition fails
        """

        try:
            # Initialize Vosk model (ensure you have the model downloaded)
            model = Model(model_path=r"/home/pi/Desktop/gradproj/vosk-model-small-en-us-0.15")

            # Open the audio file
            wf = wave.open(audio_file, "rb")

            # Create recognizer instance
            recognizer = KaldiRecognizer(model, wf.getframerate())

            # Process audio file
            text = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text += result.get("text", "") + " "

            # Get final result
            final_result = json.loads(recognizer.FinalResult())
            text += final_result.get("text", "")

            # Clean up
            wf.close()

            text = text.strip().lower()
            print(f"Recognized text: {text}")
            return text if text else None

        except Exception as e:
            print(f"Error processing audio file: {e}")
            return None

