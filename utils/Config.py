# config.py

class CameraConfig:
    # Image capture settings
    RESOLUTION = {
        'WIDTH': 1920,
        'HEIGHT': 1080
    }

    # Image storage settings
    IMAGE = {
        'QUALITY': 90,
        'SAVE_DIRECTORY': r"/tmp/AIAssistant/",
        'FILENAME': "captured_image.jpg"
    }

    # Camera parameters
    PARAMETERS = {
        'EXPOSURE': 10000,
        'GAIN': 1.0,
        'FRAME_RATE': 30,
        'FRAME_DURATION': 33333
    }

    # Camera controls
    CONTROLS = {
        'AWB_ENABLE': True,
        'AE_ENABLE': True,
        'FOCUS_MODE': 'continuous',
        'HFLIP': 1,
        'VFLIP': 1,
        'COLOR_FORMAT': "RGB888"
    }


class IOConfig:
    # Threading settings
    TIMING = {
        'CAMERA_THREAD_TIMEOUT': 0.5,
        'FRAME_INTERVAL': 0.03,
        'AUDIO_RECORD_TIMEOUT': 5
    }

    # Interface settings
    INTERFACE = {
        'WAKE_WORD': "hi david",
        'MAX_ATTEMPTS': 3,
        'WINDOWS': {
            'CAMERA': "camera window",
            'AI': "ai window",
            'USER_INPUT': "user input window"
        }
    }

    # Keywords for commands
    KEYWORDS = {
        'MODE': {
            'SPEECH': ["speech", "voice", "audio", "speak", "one", "1"],
            'IMAGE': ["image", "picture", "photo", "text", "two", "2"],
            'BOTH': ["both", "combined", "all", "three", "3"]
        },
        'COMMANDS': {
            'START': ['start', 'begin', 'launch', 'activate', 'open'],
            'STOP': ['stop', 'end', 'finish', 'quit'],
            'TRANSLATE': ['translate', 'convert', 'change', 'interpret'],
            'EXIT': ['exit', 'quit', 'close', 'leave']
        }
    }


class OCRConfig:
    MODES = {
        'DEFAULT': '--oem 3 --psm 3',
        'ACCURATE': '--oem 3 --psm 6'
    }

    PROCESSING = {
        'LEVEL': 'medium',
        'THRESH_VALUE': 150,
        'KERNEL_SIZE': (1, 1),
        'DENOISE_H': 10,
        'MAX_VALUE': 255
    }

    STORAGE = {
        'SAVE_DIRECTORY': r"/tmp/AIAssistant/",
        'PROCESSED_FRAME_FILENAME': "processed_frame.jpg"
    }


class MLConfig:
    TRANSLATION = {
        'TRANSLATION_MODELS_DIR': './models/translation',
        'MODEL_NAMES': {
            'en-ar': 'Helsinki-NLP/opus-mt-en-ar',  # English to Arabic
            'fr-en': 'Helsinki-NLP/opus-mt-fr-en',  # French to English
            'ar-en': 'Helsinki-NLP/opus-mt-ar-en',  # Arabic to English
        },
        'MODELS_LOAD_TIMEOUT': 300,
        'MODELS_DIR': './models/translation/cache',
        'SUPPORTED_SPEECH_MODELS': ['en'],
        'SUPPORTED_TRANSLATION_PAIRS': [
            ('en', 'ar'),  # English to Arabic
            ('fr', 'en'),  # French to English
            ('ar', 'en')   # Arabic to English
        ],
        'MAX_WORKERS': 2,
        'USE_LOW_MEMORY': True,
        'TORCH_DTYPE': 'float32'
    }

class ServicesConfig:
    LANGUAGES = {
        'SUPPORTED': ['en', 'ar', 'fr'],
        'MAPPING': {
            'english': 'en',
            'arabic': 'ar',
            'french': 'fr'
        }
    }

    RECOGNITION = {
        'VOSK_MODEL_PATH': './models/vosk',
        'VOSK_MODELS': {
            'en': 'vosk-model-small-en-us'
        },
        'RECOGNITION_MODEL_DIR': './models/recognition',
        'FUZZY_CONFIDENCE_THRESHOLD': 75
    }

class LLMConfig:
    PROMPTS = {
        'SUPPORTED': {
            'translate': "tr",
            'extract': "ex"
        }
    }