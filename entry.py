#!/usr/bin/env python3
"""
Entry point for AR Glasses AI Assistant Application
This file handles environment setup and launches the main application
"""

import os
import sys
import subprocess
import time
import gc
from pathlib import Path

os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/5/tessdata/'

def setup_environment():
    """Setup the environment for running the application"""
    print("\n" + "=" * 60)
    print("         AR GLASSES AI ASSISTANT")
    print("=" * 60 + "\n")

    print("Setting up AR Glasses AI Assistant environment...")

    # Ensure we're in the correct directory
    script_dir = Path(__file__).parent.absolute()
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")

    # Check if required directories exist
    required_dirs = [
        'models/vosk',
        'models/translation/cache',
        'logs',
        '/tmp/AIAssistant'
    ]

    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Directory ensured: {dir_path}")

    # Set environment variables for PyQt5 on Raspberry Pi
    env_vars = {
        'QT_QPA_PLATFORM_PLUGIN_PATH': '/usr/lib/python3/dist-packages/PyQt5/Qt/plugins',
        'QT_QPA_PLATFORM': 'eglfs',
        'QT_QPA_EGLFS_ALWAYS_SET_MODE': '1',
        'QT_QPA_EGLFS_WIDTH': '800',
        'QT_QPA_EGLFS_HEIGHT': '600'
    }

    for var, value in env_vars.items():
        os.environ[var] = value
        print(f"Environment variable set: {var}={value}")

    # Clear memory before starting application
    gc.collect()
    print("Environment setup complete!")
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    print("\nChecking dependencies...")

    required_modules = [
        'PyQt5',
        'picamera2',
        'vosk',
        'transformers',
        'opencv-python',
        'numpy',
        'fuzzywuzzy',
        'pytesseract'
    ]

    missing_modules = []

    for module in required_modules:
        try:
            if module == 'opencv-python':
                import cv2
            elif module == 'PyQt5':
                from PyQt5.QtCore import Qt
            elif module == 'picamera2':
                from picamera2 import Picamera2
            elif module == 'vosk':
                import vosk
            elif module == 'transformers':
                import transformers
            elif module == 'numpy':
                import numpy
            elif module == 'fuzzywuzzy':
                from fuzzywuzzy import fuzz
            elif module == 'pytesseract':
                import pytesseract

            print(f"✓ {module} - OK")
        except ImportError:
            print(f"✗ {module} - MISSING")
            missing_modules.append(module)

    if missing_modules:
        print(f"\nMissing dependencies: {', '.join(missing_modules)}")
        print("Please install missing dependencies before running the application.")
        return False

    print("All dependencies are available!")
    return True

def check_models():
    """Check if required model directories are available"""
    print("\nChecking for required models...")

    model_files = [
        'models/vosk/vosk-model-small-en-us-0.15',
        # Add other model files as needed
    ]

    missing_models = []

    for model_path in model_files:
        if os.path.exists(model_path):
            print(f"✓ {model_path} - Found")
        else:
            print(f"✗ {model_path} - Missing")
            missing_models.append(model_path)

    if missing_models:
        print(f"\nMissing models: {', '.join(missing_models)}")
        print("The application will attempt to download or use models from cache.")
        print("This may take some time during first run.")
        # Don't return false, just warn the user
    else:
        print("All required models are available!")

    return True

def launch_application():
    """Launch the main application"""
    print("\n" + "="*50)
    print("Launching AR Glasses AI Assistant...")
    print("="*50)

    try:
        # Import and run the main application
        from main import main

        print("Starting application...")
        success = main()

        if success:
            print("Application finished successfully")
            return 0
        else:
            print("Application finished with errors")
            return 1

    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
        return 0
    except Exception as e:
        print(f"Error launching application: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        # Setup environment
        if not setup_environment():
            print("Failed to setup environment")
            sys.exit(1)

        # Check dependencies
        if not check_dependencies():
            print("Dependency check failed")
            sys.exit(1)

        # Check models
        check_models()  # Just warn about missing models, don't exit

        # Launch application
        exit_code = launch_application()
        sys.exit(exit_code)

    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
