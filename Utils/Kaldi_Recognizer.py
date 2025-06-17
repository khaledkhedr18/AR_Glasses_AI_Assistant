from vosk import Model, KaldiRecognizer
import wave
import json
import io


class SpeechRecognizer:
    def __init__(self, model_path):
        """Initialize the recognizer with a Vosk model."""
        self.model = Model(model_path)

    def recognize_text_from_speech(self, wave_data):
        """
        Convert WAV audio data to text using Vosk.
        Args:
            wave_data: BytesIO object containing WAV data or raw wave chunks
        Returns:
            str: Recognized text or None if recognition fails
        """
        try:
            # If input is already a BytesIO/file-like object, use it directly
            if isinstance(wave_data, (io.BytesIO, io.BufferedRandom)):
                wf = wave.open(wave_data, "rb")
            else:
                # If input is raw bytes, wrap it in BytesIO
                buffer = io.BytesIO(wave_data)
                wf = wave.open(buffer, "rb")

            with wf:
                recognizer = KaldiRecognizer(self.model, wf.getframerate())
                text = ""

                # Process audio in chunks
                while True:
                    data = wf.readframes(4000)
                    if len(data) == 0:
                        break
                    if recognizer.AcceptWaveform(data):
                        result = json.loads(recognizer.Result())
                        text += result.get("text", "") + " "

                # Get final recognition result
                final_result = json.loads(recognizer.FinalResult())
                text += final_result.get("text", "")

            text = text.strip().lower()
            return text if text else None

        except Exception as e:
            print(f"Error processing audio: {e}")
            return None