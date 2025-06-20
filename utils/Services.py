from fuzzywuzzy import fuzz
from vosk import Model, KaldiRecognizer
from Handlers.LTD_Handler import LTDHandler
import wave
import json
import io
from utils.Logging import Logger
from utils.Config import SERVICES_CONFIG


class Services:
    def __init__(self):
        self.logger = Logger()
        self.recognizer_model_path = SERVICES_CONFIG['recognizer_model_path']
        self.supported_lang_codes = SERVICES_CONFIG.get('supported_lang_codes', [])
        self.languages_map = SERVICES_CONFIG.get('languages_map', {})

    def recognize_text_from_speech(self, wave_data, model_components=None):
        """
        Convert WAV audio data to text using Vosk.

        Args:
            wave_data: BytesIO object containing WAV data or raw wave chunks
            model_components (tuple): (model, success) tuple from LTDHandler

        Returns:
            str: Recognized text or None if recognition fails
        """
        try:
            if not model_components or not model_components[1]:  # Check if model_components is valid
                self.logger.error("Invalid speech model components")
                return None

            speech_model = model_components[0]  # Extract the Vosk model

            # If input is already a BytesIO/file-like object, use it directly
            if isinstance(wave_data, (io.BytesIO, io.BufferedRandom)):
                wf = wave.open(wave_data, "rb")
            else:
                # If input is raw bytes, wrap it in BytesIO
                buffer = io.BytesIO(wave_data)
                wf = wave.open(buffer, "rb")

            with wf:
                recognizer = KaldiRecognizer(speech_model, wf.getframerate())
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
            self.logger.error(f"Error processing audio: {e}")
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

    def get_language_code(self, language):
        """
        Convert language name to standardized code.

        Args:
            language (str): Language name (e.g. 'english') or code (e.g. 'en')

        Returns:
            str: Standardized language code if valid, None otherwise
        """

        if not language:
            self.logger.error("Empty language input")
            return None

        # Convert input to lowercase for consistency
        language = language.lower().strip()

        # Case 1: Input is already a valid language code
        if language in self.supported_lang_codes:
            self.logger.debug(f"Valid language code: {language}")
            return language

        # Case 2: Input is a language name that needs to be mapped to code
        if language in self.languages_map:
            code = self.languages_map[language]
            self.logger.debug(f"Mapped {language} to code: {code}")
            return code

        self.logger.warning(f"Unsupported language: {language}")
        return None

