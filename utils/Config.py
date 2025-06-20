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
        'KERNEL_SIZE': (1, 1)
    }

    STORAGE = {
        'SAVE_DIRECTORY': r"/tmp/AIAssistant/",
        'PROCESSED_FRAME_FILENAME': "processed_frame.jpg"
    }


class LTDConfig:
    TRANSLATION = {
        'TRANSLATION_MODELS_DIR': '/path/to/translation/models/',
        'DEFAULT_LANGUAGE': 'en',
        'MAX_LENGTH': 512,
        'BATCH_SIZE': 8,
        'MODELS_LOAD_TIMEOUT': 30
    }


class ServicesConfig:
    # Recognition settings
    RECOGNITION = {
        'FUZZY_CONFIDENCE_THRESHOLD': 75,
        'RECOGNITION_MODEL_DIR': r"/path/to/kaldi/model",
        'VOSK_MODELS': {
            'en': r"vosk-model-small-en-us-0.15",
            'ar': r"vosk-model-ar-mgb2-0.4",
            'fr': r"vosk-model-small-fr-0.22"
        }
    }

    # Language settings
    LANGUAGES = {
        'SUPPORTED': ['en', 'ar', 'fr'],
        'MAPPING': {
            'arabic': 'ar',
            'english': 'en',
            'french': 'fr'
        }
    }


class LLMConfig:
    PROMPTS = {
        'SUPPORTED': {
            'translate': "tr",
            'extract': "ex"
        }
    }