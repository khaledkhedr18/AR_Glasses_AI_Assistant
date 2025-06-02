from vosk import Model as VoskModel
from transformers import MarianMTModel, MarianTokenizer
import threading

_vosk_models = {}
_translation_models = {}
_lock = threading.Lock()

def get_vosk_model(language_code: str):
    """
    Get the Vosk model for the given language
    Loads it if not already loaded
    """
    global _vosk_models
    with _lock:
        if language_code not in _vosk_models:
            if language_code == "ar":
                model_path = "/home/pi/Desktop/gradproj/vosk-model-ar-mgb2-0.4"
            elif language_code == "en":
                model_path = "/home/pi/Desktop/gradproj/vosk-model-small-en-us-0.15"
            elif language_code == "fr":
                model_path = "/home/pi/Desktop/gradproj/vosk-model-small-fr-0.22"
            else:
                raise ValueError("Unsupported Language Code")
            _vosk_models[language_code] = VoskModel(model_path)

        return _vosk_models[language_code]


def get_translation_model(source: str, target: str):
    """
    Get the Marian translation pipeline for source to target.
    Loads it if not already loaded
    """
    global _translation_models
    with _lock:
        key = f"{source}-{target}"
        if key not in _translation_models:
            model_name = f'Helsinki-NLP/opus-mt-{source}-{target}'
            from transformers import pipeline
            _translation_models[key] = pipeline("translation", model=model_name)
        return _translation_models[key]

