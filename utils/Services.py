from fuzzywuzzy import fuzz
from vosk import Model, KaldiRecognizer
import wave
import json
import io
from utils.Logging import Logger
from utils.Config import SERVICES_CONFIG, IO_CONFIG


class Services:
    def __init__(self):
        self.model = Model(IO_CONFIG['RECOGNIZER_MODEL_PATH'])
        self.logger = Logger()

    def recognize_text_from_speech(self, wave_data, language=None):
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

    def verify_user_input(self, text, items_dict, confidence_threshold = SERVICES_CONFIG['confidence_threshold'] if 'confidence_threshold' in SERVICES_CONFIG else 75):
        """
        Generic fuzzy matching method to verify user input against a dictionary or list of items.

        Args:
            text (str): The text to verify
            items_dict (dict/list): Dictionary {name: code} or list of keywords mapping to a single value
            confidence_threshold (int): Minimum score for match (default: 75)

        Returns:
            str or None: Matched value (language code, mode, etc.) or None if no match
        """
        if not text or not isinstance(text, str):
            self.logger.warning("Invalid input text for verification")
            return None

        text = text.lower().strip()

        # Handle dictionary input (like languages)
        if isinstance(items_dict, dict):
            comparison_dict = items_dict

        # Handle list input with implicit mapping (like modes)
        elif isinstance(items_dict, list):
            # For lists like ["speech", "voice", "audio"] -> "speech"
            # First item is considered the target value
            target_value = items_dict[0]
            comparison_dict = {keyword: target_value for keyword in items_dict}
        else:
            self.logger.error("Invalid items_dict type")
            return None

        # Calculate fuzzy match scores
        best_overall_score = 0
        best_match_value = None

        for item_name, item_value in comparison_dict.items():
            # Calculate different types of fuzzy matches
            partial_score = fuzz.partial_ratio(text, item_name.lower())
            token_sort_score = fuzz.token_sort_ratio(text, item_name.lower())
            token_set_score = fuzz.token_set_ratio(text, item_name.lower())

            # Get best score among different matching methods
            best_score = max(partial_score, token_sort_score, token_set_score)

            if best_score > best_overall_score:
                best_overall_score = best_score
                best_match_value = item_value

            self.logger.debug(f"Match '{item_name}': score={best_score}%")

        if best_overall_score >= confidence_threshold:
            self.logger.info(f"Match found with confidence: {best_overall_score}%")
            return best_match_value
        else:
            self.logger.warning(f"No match found above threshold ({confidence_threshold}%)")
            return None

