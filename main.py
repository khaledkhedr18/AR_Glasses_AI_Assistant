#!/usr/bin/env python3

import sys
import signal
import threading
import time
import json
import os
import cv2
import numpy as np
from utils.Logging import Logger
from utils.Config import APP_CONFIG, IO_CONFIG, SERVICES_CONFIG
from Managers.IO_Manager import IOManager
from Managers.LLM_Manager import LLMManager
from utils.Socket_Handler import SocketHandler
from utils.WorkerThread import create_worker
import base64


class ARGlassesAssistant:
    """
    Main application class for AR Glasses AI Assistant
    Controls application flow and coordinates between components
    """

    def __init__(self):
        """Initialize the application and its components"""
        # Setup logger
        self.logger = Logger()
        self.logger.info("Starting AR Glasses AI Assistant")

        # Initialize state variables
        self.running = False
        self.io_manager = None
        self.llm_manager = None
        self.socket_handler = None
        self.main_window = None
        self.shutting_down = False
        self.wake_word_detected = False
        self.last_online_status = None

        # User configuration storage
        self.user_config = {
            'operation_type': None,  # 'ocr' or 'translate'
            'input_type': None,      # 'speech' or 'image'
            'source_lang': None,
            'target_lang': None,
            'online_mode': False,
            'use_prompt': False,
            'prompt_text': None,
            'speech_text': None,
            'translation_mode': None
        }

        # Store model references
        self.speech_model = None

    def initialize(self):
        """Initialize all components and subsystems"""
        try:
            # Initialize LLM Manager first (this is lightweight and only loads models on demand)
            self.logger.info("Initializing LLM Manager")
            self.llm_manager = LLMManager()

            # Load speech model for English (needed for initial interaction)
            speech_result = self.llm_manager.get_speech_model("en")
            if not speech_result or not speech_result[0]:
                self.logger.error("Failed to load speech model")
                return False

            self.speech_model = speech_result[0]
            self.logger.info("Speech model loaded successfully")

            # Initialize IO Manager with speech model
            self.logger.info("Initializing IO Manager")
            self.io_manager = IOManager(self.speech_model)

            # Initialize Socket Handler for online mode
            self.socket_handler = SocketHandler()

            # Create main application window
            self.logger.info("Creating main window")
            fullscreen = APP_CONFIG.get('DISPLAY', {}).get('FULLSCREEN', True)
            self.main_window = self.io_manager.create_main_window(
                title="AR Glasses Assistant",
                fullscreen=fullscreen
            )

            if not self.main_window:
                self.logger.error("Failed to create main window")
                return False

            # Create all standard overlay widgets
            self.logger.info("Creating overlay widgets")
            self.widgets = self.io_manager.create_all_overlay_widgets()

            # Update status to show we're ready
            self._setup_status_monitor()
            self._update_system_status()
            self.logger.info("Application initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False

    def start(self):
        """Start the application and begin processing"""
        if self.running:
            self.logger.warning("Application already running")
            return False

        try:
            self.running = True

            # Start camera stream
            self.logger.info("Starting camera stream")
            self.io_manager.start_camera_stream()

            # Setup signal handlers for clean shutdown
            signal.signal(signal.SIGINT, self._handle_interrupt)
            signal.signal(signal.SIGTERM, self._handle_interrupt)

            # Start main processing thread
            self.process_thread = threading.Thread(target=self._processing_loop)
            self.process_thread.daemon = True
            self.process_thread.start()

            # Show initial greeting
            welcome_message = "AR Glasses Assistant Ready. Say 'Hi David' to start..."
            self.io_manager.gui.update_ai_response(welcome_message)
            self.io_manager.interact_with_user(welcome_message, mode="both")

            # Run the GUI main loop (this blocks until window is closed)
            self.logger.info("Entering GUI main loop")
            return_code = self.io_manager.gui.run()

            # Clean shutdown after GUI loop exits
            self._shutdown()
            return return_code == 0

        except Exception as e:
            self.logger.error(f"Error starting application: {str(e)}")
            self.running = False
            return False

    def _setup_status_monitor(self):
        """Start periodic status updates"""
        self.status_timer = threading.Timer(
            interval=5.0,  # Check every 5 seconds
            function=self._status_monitor_loop
        )
        self.status_timer.daemon = True
        self.status_timer.start()
        self.logger.info("Status monitor started")

    def _status_monitor_loop(self):
        """Continuous status monitoring loop"""
        while self.running:
            self._update_system_status()
            time.sleep(5)  # 5 second interval

        self.logger.info("Status monitor stopped")


    def _update_system_status(self):
        """Check and update system status through all layers"""
        try:
            # Check internet connection
            is_online = self.socket_handler.check_internet_connection()

            # Debug log to verify connection status
            self.logger.info(f"Internet connection status: {'Online' if is_online else 'Offline'}")

            # Only update if status changed or this is the first check
            if not hasattr(self, 'last_online_status') or is_online != self.last_online_status:
                self.last_online_status = is_online
                status_text = "System operational"

                # Update the status widget through IO_Manager
                if hasattr(self, 'io_manager') and self.io_manager:
                    # First update the GUI directly
                    self.io_manager.gui.update_status(status_text, is_online)

                    # Then speak the status if it changed
                    status_message = f"You are now {'online' if is_online else 'offline'}"
                    self.io_manager.interact_with_user(status_message, mode="speech")

                return True

        except Exception as e:
            self.logger.error(f"Status update failed: {str(e)}")
            return False

    def _processing_loop(self):
        """Background processing loop that handles user interactions"""
        self.logger.info("Processing loop started")

        # Start by waiting for wake word
        wake_word_thread = create_worker(
            self._wait_for_wake_word,
            on_result=self._on_wake_word_detected,
            task_name="Wake Word Detection"
        )
        wake_word_thread.start()

        while self.running:
            try:
                # Short sleep to prevent CPU hogging
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error in processing loop: {str(e)}")
                time.sleep(1)  # Sleep longer after an error

    def _on_wake_word_detected(self, detected):
        """Callback when wake word is detected"""
        if detected and self.running:
            self.wake_word_detected = True
            self._start_conversation_flow()

    def _wait_for_wake_word(self):
        """Wait for wake word to be spoken before starting conversation"""
        wake_words = IO_CONFIG.get('USER_COMMANDS', {}).get('WAKE_WORD', ["hi david", "hey david"])

        # Ensure wake_words is a list
        if isinstance(wake_words, set):
            wake_words = list(wake_words)

        self.logger.info(f"Waiting for wake word from: {wake_words}")

        while self.running and not self.wake_word_detected:
            # Use real-time speech recognition instead of start-stop approach
            self.logger.info("Listening for wake word...")
            text = self.io_manager.get_user_speech(max_duration=3)

            if text:
                text_lower = text.lower()
                self.logger.info(f"Heard: '{text_lower}'")

                # Check for wake word
                for wake_word in wake_words:
                    if wake_word.lower() in text_lower:
                        self.logger.info(f"Wake word detected: {wake_word}")
                        self.io_manager.gui.update_user_speech(f"You: {text}")
                        return True

            # Sleep briefly before trying again
            time.sleep(0.1)

        return False

    def _start_conversation_flow(self):
        """Start the main conversation flow"""
        try:
            self.io_manager.clear_speech_queue()
            self.logger.info("Starting conversation flow")
            self.io_manager.interact_with_user("How can I assist you today?", mode="both")

            # Reset user configuration
            self._reset_user_config()

            operation_success = self._ask_input_type()
            if not operation_success:
                self.io_manager.interact_with_user("Operation selection failed.", mode="both")
                self._reset_conversation()
                return
            self.io_manager.clear_speech_queue()

            if self.last_online_status:
                online_success = self._ask_online_mode()
            else:
                online_success = False

            self.io_manager.clear_speech_queue()

            if online_success:
                self._handle_online_flow()
            else:
                self._handle_offline_flow()

            # Save configuration and reset for next conversation
            self._save_config()
            self._reset_conversation()

        except Exception as e:
            self.logger.error(f"Error in conversation flow: {str(e)}")
            self.io_manager.interact_with_user("An error occurred.", mode="both")
            self._reset_conversation()

    def _handle_online_flow(self):
        """Handle the online mode workflow"""
        if self.user_config['input_type'] == 'speech':
            self.user_config['use_prompt'] = True
            self._handle_online_speech_translation()
        else:
            self._handle_online_image_translation()


    def _handle_offline_flow(self):
        """Handle the offline mode workflow"""
        # Ask for operation type (OCR or Translate)
        operation_success = self._ask_operation_type()
        if not operation_success:
            self.io_manager.interact_with_user("Operation selection failed.", mode="both")
            return
        self.io_manager.clear_speech_queue()

        # Handle the selected operation
        if self.user_config['operation_type'] == 'translate':
            self._handle_offline_translation_flow()
        elif self.user_config['operation_type'] == 'ocr':
            self._handle_ocr_flow()

    def _handle_online_speech_translation(self):
        """Handle online speech translation workflow"""
        # Ask if they want to send a prompt
        prompt_audio_path = None
        if self.user_config['use_prompt']:
            # Get the prompt text
            prompt_audio_path = self.io_manager.get_user_audio_prompt()
            if not prompt_audio_path:
                return

        # Send request to server
        self._send_online_speech_request(prompt_audio_path=prompt_audio_path)

    def _handle_online_image_translation(self):
        """Handle online image translation workflow"""
        # Ask if they want to send a prompt
        if not self._ask_prompt_option():
            return

        if self.user_config['use_prompt']:
            # Get the prompt text
            prompt_audio_path = self.io_manager.get_user_audio_prompt()
            if not prompt_audio_path:
                return

        # Capture image
        if not self._ask_take_picture():
            return

        self.io_manager.interact_with_user("Capturing image...", mode="display")
        image_path = self.io_manager.get_image()

        if not image_path:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Failed to capture image", mode="both")
            return

        # Send to server
        self._send_online_request(image_path, prompt_audio_path=prompt_audio_path)

    def _handle_offline_translation_flow(self):
        """Handle offline translation workflow"""
                # Get source language
        if not self._ask_source_language():
            return None

        # Get target language
        if not self._ask_target_language():
            return None

        # Handle the selected input type
        if self.user_config['input_type'] == 'image':
            self._handle_offline_image_translation()
        elif self.user_config['input_type'] == 'speech':
            self._handle_speech_translation()

    def _send_online_speech_request(self, prompt_audio_path=None):
        """Send speech request to online server"""
        try:
            # Connect to server
            self.io_manager.interact_with_user("Connecting to server...", mode="display")
            if not self.socket_handler.connect_to_server():
                self.io_manager.interact_with_user("Could not connect to server", mode="both")
                return

            # Prepare request
            self.io_manager.interact_with_user("Sending request to server...", mode="display")

            # Send appropriate request type
            if self.user_config['use_prompt'] and prompt_audio_path:
                # Send combined request with prompt
                response = self.socket_handler.send_and_receive(
                    "audio",
                    "en",
                    "en",
                    audio_data=prompt_audio_path,
                )
            else:
                # Send speech only
                response = self.socket_handler.send_and_receive(
                    "text",
                    "en",  # Default source language for online
                    "en",  # Default target language for online
                    text=self.user_config.get('prompt_text', '')
                )

            # Process response
            if response:
                self.io_manager.gui.update_ai_response(f"Server response: {response}")
                self.io_manager.interact_with_user(f"Online processing completed response is: {response}", mode="both")
                time.sleep(5)
            else:
                self.io_manager.interact_with_user("No response from server", mode="both")

        except Exception as e:
            self.logger.error(f"Error in online request: {str(e)}")
            self.io_manager.interact_with_user("Online processing failed due to an error", mode="both")
        finally:
            # Always disconnect from server
            try:
                self.socket_handler.disconnect_from_server()
            except:
                pass


    def _reset_user_config(self):
        """Reset user configuration to defaults"""
        self.user_config = {
            'operation_type': None,
            'input_type': None,
            'source_lang': None,
            'target_lang': None,
            'online_mode': False,
            'use_prompt': False,
            'prompt_text': None,
            'speech_text': None
        }

    def _ask_operation_type(self):
        """Ask user to choose between OCR or Translate"""
        max_attempts = 3
        attempts = 0

        if self.user_config['input_type'] == 'speech':
            self.user_config['operation_type'] = 'translate'
            return True

        while attempts < max_attempts and self.running:
            self.io_manager.clear_speech_queue()

            self.io_manager.interact_with_user("What would you like to do? Say 'OCR' or 'Translate'", mode="both")

            # Use real-time recognition
            text = self.io_manager.get_user_speech(max_duration=5)

            if text:
                self.io_manager.gui.update_user_speech(f"You: {text}")
                text_lower = text.lower()

                if 'extract' in text_lower:
                    self.io_manager.clear_speech_queue()
                    self.user_config['operation_type'] = 'ocr'
                    self.io_manager.interact_with_user("OCR selected", mode="both")
                    return True
                elif 'translate' in text_lower or 'translation' in text_lower:
                    self.io_manager.clear_speech_queue()
                    self.user_config['operation_type'] = 'translate'
                    self.io_manager.interact_with_user("Translation selected", mode="both")
                    return True
                else:
                    self.io_manager.clear_speech_queue()
                    self.io_manager.interact_with_user("I didn't understand. Please say 'OCR' or 'Translate'", mode="both")
            else:
                self.io_manager.clear_speech_queue()
                self.io_manager.interact_with_user("I didn't hear anything. Please say 'OCR' or 'Translate'", mode="both")

            attempts += 1

        if attempts >= max_attempts:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Too many failed attempts. Please try again later.", mode="both")

        return False

    def _handle_translation_flow(self):
        """Handle the complete translation workflow"""
        # Ask for input type (speech or image)
        if not self._ask_input_type():
            return

        if self.user_config['input_type'] == 'image':
            self._handle_image_translation()
        elif self.user_config['input_type'] == 'speech':
            self._handle_speech_translation()

    def _ask_input_type(self):
        """Ask user to choose between speech or image input"""
        max_attempts = 3
        attempts = 0

        while attempts < max_attempts and self.running:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("What type of input? Say 'Speech' or 'Image'", mode="both")

            text = self.io_manager.get_user_speech(max_duration=5)
            if text:
                self.io_manager.gui.update_user_speech(f"You: {text}")
                text_lower = text.lower()

                if 'speech' in text_lower or 'voice' in text_lower:
                    self.io_manager.clear_speech_queue()
                    self.user_config['input_type'] = 'speech'
                    self.io_manager.interact_with_user("Speech input selected", mode="both")
                    return True
                elif 'image' in text_lower or 'picture' in text_lower or 'photo' in text_lower:
                    self.io_manager.clear_speech_queue()
                    self.user_config['input_type'] = 'image'
                    self.io_manager.interact_with_user("Image input selected", mode="both")
                    return True
                else:
                    self.io_manager.clear_speech_queue()
                    self.io_manager.interact_with_user("I didn't understand. Please say 'Speech' or 'Image'", mode="both")
            else:
                self.io_manager.clear_speech_queue()
                self.io_manager.interact_with_user("I didn't hear anything. Please say 'Speech' or 'Image'", mode="both")


            attempts += 1

        if attempts >= max_attempts:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Too many failed attempts. Please try again later.", mode="both")

        return False

    def _handle_image_translation(self):
        """Handle image translation workflow"""
        # Get source language
        if not self._ask_source_language():
            return

        # Get target language
        if not self._ask_target_language():
            return

        # Ask about online mode
        if not self._ask_online_mode():
            return

        if self.user_config['online_mode']:
            self._handle_online_image_translation()
        else:
            self._handle_offline_image_translation()

    def _handle_speech_translation(self):
        """Handle speech translation workflow with proper configuration"""
        # Initialize config with defaults
        self.user_config.update({
            'operation_type': 'translate',
            'input_type': 'speech',
            'online_mode': False,
            'use_prompt': False,
            'translation_mode': 'text'
        })

        # Get speech input
        if not self._get_speech_input():
            return None

        # Process translation with validated config
        translation_result = self._process_speech_translation()

        if not translation_result or not translation_result.get('success'):
            error = translation_result.get('error', 'Translation failed') if translation_result else 'Unknown error'
            self.io_manager.interact_with_user(f"Translation failed: {error}", mode="both")
            return None

        return translation_result

    def _handle_ocr_flow(self):
        """Handle OCR workflow"""
        # Get source language
        if not self._ask_source_language():
            self.io_manager.interact_with_user("Language selection failed.", mode="both")
            return

        # Ask user to take picture
        if not self._ask_take_picture():
            return

        # Process OCR
        self._process_ocr()

    def _ask_source_language(self):
        """Ask for source language"""
        max_attempts = 3
        attempts = 0

        # Get supported languages
        lang_mapping = SERVICES_CONFIG.get('LANGUAGES', {}).get('MAPPING', {
            'english': 'en',
            'arabic': 'ar',
            'french': 'fr'
        })
        supported_languages = ", ".join([f"'{lang}'" for lang in lang_mapping.keys()])

        while attempts < max_attempts and self.running:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user(
                f"What is the source language? Say {supported_languages}",
                mode="both"
            )

            text = self.io_manager.get_user_speech(max_duration=5)

            if text:
                self.io_manager.gui.update_user_speech(f"You: {text}")

                # Get language code using service
                lang_code = self.io_manager.service.get_language_code(text)

                if lang_code:
                    self.io_manager.clear_speech_queue()
                    self.user_config['source_lang'] = lang_code
                    self.io_manager.interact_with_user(f"Source language set to {text}", mode="both")
                    return True
                else:
                    self.io_manager.clear_speech_queue()
                    self.io_manager.interact_with_user(
                        f"Language not recognized. Please say one of {supported_languages}",
                        mode="both"
                    )
            else:
                self.io_manager.clear_speech_queue()
                self.io_manager.interact_with_user(
                    f"I didn't hear anything. Please say one of {supported_languages}",
                    mode="both"
                )
            attempts += 1

        if attempts >= max_attempts:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Too many failed attempts. Please try again later.", mode="both")

        return False

    def _ask_target_language(self):
        """Ask for target language"""
        max_attempts = 3
        attempts = 0

        # Get supported languages
        lang_mapping = SERVICES_CONFIG.get('LANGUAGES', {}).get('MAPPING', {
            'english': 'en',
            'arabic': 'ar',
            'french': 'fr'
        })
        supported_languages = ", ".join([f"'{lang}'" for lang in lang_mapping.keys()])

        while attempts < max_attempts and self.running:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user(
                f"What is the target language? Say {supported_languages}",
                mode="both"
            )

            text = self.io_manager.get_user_speech(max_duration=3)

            if text:
                self.io_manager.gui.update_user_speech(f"You: {text}")

                # Get language code using service
                lang_code = self.io_manager.service.get_language_code(text)

                if lang_code:
                    self.io_manager.clear_speech_queue()
                    self.user_config['target_lang'] = lang_code
                    self.io_manager.interact_with_user(f"Target language set to {text}", mode="both")
                    return True
                else:
                    self.io_manager.clear_speech_queue()
                    self.io_manager.interact_with_user(
                        f"Language not recognized. Please say one of {supported_languages}",
                        mode="both"
                    )
            else:
                self.io_manager.clear_speech_queue()
                self.io_manager.interact_with_user(
                    f"I didn't hear anything. Please say one of {supported_languages}",
                    mode="both"
                )
            attempts += 1

        if attempts >= max_attempts:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Too many failed attempts. Please try again later.", mode="both")

        return False

    def _ask_online_mode(self):
        """Ask if user wants to use online mode"""
        max_attempts = 3
        attempts = 0

        while attempts < max_attempts and self.running:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Do you want to continue in online mode? Say 'Yes' or 'No'", mode="both")

            text = self.io_manager.get_user_speech(max_duration=3)
            if text:
                self.io_manager.clear_speech_queue()
                self.io_manager.gui.update_user_speech(f"You: {text}")
                text_lower = text.lower()

                if 'yes' in text_lower or 'yeah' in text_lower:
                    self.io_manager.clear_speech_queue()
                    self.user_config['online_mode'] = True
                    self.io_manager.interact_with_user("Online mode enabled", mode="both")
                    # Ask if they want to send a prompt
                    return True
                elif 'no' in text_lower or 'nope' in text_lower:
                    self.io_manager.clear_speech_queue()
                    self.user_config['online_mode'] = False
                    self.io_manager.interact_with_user("Offline mode selected", mode="both")
                    return False
                else:
                    self.io_manager.clear_speech_queue()
                    self.io_manager.interact_with_user("Please say 'Yes' or 'No'", mode="both")
            else:
                self.io_manager.clear_speech_queue()
                self.io_manager.interact_with_user("I didn't hear anything. Please say 'Yes' or 'No'", mode="both")

            attempts += 1

        if attempts >= max_attempts:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Too many failed attempts. Using offline mode by default.", mode="both")
            self.user_config['online_mode'] = False

        return True

    def _ask_prompt_option(self):
        """Ask if user wants to send a prompt"""
        max_attempts = 3
        attempts = 0

        while attempts < max_attempts and self.running:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Do you want to send a prompt? Say 'Yes' or 'No'", mode="both")

            text = self.io_manager.get_user_speech(max_duration=3)

            if text:
                self.io_manager.gui.update_user_speech(f"You: {text}")
                text_lower = text.lower()

                if 'yes' in text_lower or 'yeah' in text_lower:
                    self.io_manager.clear_speech_queue()
                    self.user_config['use_prompt'] = True
                    return True
                elif 'no' in text_lower or 'nope' in text_lower:
                    self.io_manager.clear_speech_queue()
                    self.user_config['use_prompt'] = False
                    self.io_manager.interact_with_user("No prompt will be sent", mode="both")
                    return True
                else:
                    self.io_manager.clear_speech_queue()
                    self.io_manager.interact_with_user("Please say 'Yes' or 'No'", mode="both")
            else:
                self.io_manager.clear_speech_queue()
                self.io_manager.interact_with_user("I didn't hear anything. Please say 'Yes' or 'No'", mode="both")
            attempts += 1

        if attempts >= max_attempts:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Too many failed attempts. No prompt will be sent.", mode="both")
            self.user_config['use_prompt'] = False

        return True

    def _get_prompt_text(self):
        """Get the prompt text from user"""
        max_attempts = 3
        attempts = 0

        while attempts < max_attempts and self.running:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Please say your prompt now", mode="both")
            text = self.io_manager.get_user_speech(max_duration=10)
            if text:
                self.io_manager.gui.update_user_speech(f"You: {text}")
                self.user_config['prompt_text'] = text
                self.io_manager.interact_with_user(f"Prompt received: {text}", mode="both")
                return True
            else:
                self.io_manager.clear_speech_queue()
                self.io_manager.interact_with_user("I didn't hear any prompt. Please speak again.", mode="both")
            attempts += 1

        if attempts >= max_attempts:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Too many failed attempts. No prompt will be used.", mode="both")
            self.user_config['use_prompt'] = False

        return True

    def _get_speech_input(self):
        """Get speech input for translation"""
        max_attempts = 3
        attempts = 0

        while attempts < max_attempts and self.running:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Please say the text you want to translate", mode="both")

            text = self.io_manager.get_user_speech(max_duration=5)

            if text:
                self.io_manager.gui.update_user_speech(f"You: {text}")
                self.user_config['speech_text'] = text
                self.io_manager.interact_with_user(f"Speech input received: {text}", mode="both")
                return True
            else:
                self.io_manager.clear_speech_queue()
                self.io_manager.interact_with_user("I didn't hear anything. Please speak again.", mode="both")
            attempts += 1

        if attempts >= max_attempts:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Too many failed attempts. Cancelling speech translation.", mode="both")

        return False

    def _ask_take_picture(self):
        """Ask user to take a picture"""
        max_attempts = 3
        attempts = 0

        while attempts < max_attempts and self.running:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Say 'take' when you're ready to capture the image", mode="both")

            text = self.io_manager.get_user_speech(max_duration=3)

            if text:
                self.io_manager.gui.update_user_speech(f"You: {text}")
                text_lower = text.lower()

                if 'take' in text_lower or 'capture' in text_lower or 'photo' in text_lower:
                    self.io_manager.clear_speech_queue()
                    return True
                else:
                    self.io_manager.clear_speech_queue()
                    self.io_manager.interact_with_user("Say 'take' to capture the image", mode="both")
            else:
                self.io_manager.clear_speech_queue()
                self.io_manager.interact_with_user("I didn't hear anything. Say 'take' to capture the image", mode="both")
            attempts += 1

        if attempts >= max_attempts:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Too many failed attempts. Cancelling picture capture.", mode="both")

        return False

    def _handle_offline_image_translation(self):
        """Handle offline image translation"""
        if not self._ask_take_picture():
            return

        # Capture image
        self.io_manager.interact_with_user("Capturing image...", mode="display")
        image_path = self.io_manager.get_image()

        if not image_path:
            self.io_manager.clear_speech_queue()
            self.io_manager.interact_with_user("Failed to capture image", mode="both")
            return

        # Process offline
        self._process_offline_image_translation(image_path)

    def _process_speech_translation(self):
        try:
            self.logger.info("Starting speech translation process")

            # Validate configuration
            if not all(k in self.user_config for k in ['source_lang', 'target_lang', 'speech_text']):
                self.logger.error("Missing required translation parameters")
                return {'success': False, 'error': 'Invalid configuration'}

            text = self.user_config['speech_text']
            self.logger.info(f"Processing translation for text: {text}")

            # Get translation model
            translation_result = self.llm_manager.get_translation_model(
                self.user_config['source_lang'],
                self.user_config['target_lang']
            )

            if not translation_result or not translation_result[2]:
                self.logger.error("Failed to load translation model")
                return {'success': False, 'error': 'Model loading failed'}

            # Perform translation
            self.logger.info("Executing translation...")
            translated_text = self.llm_manager.translation_handler.translate_text(
                text,
                translation_result[0]  # pipeline object
            )

            if not translated_text:
                self.logger.error("Translation returned no result")
                return {'success': False, 'error': 'Translation failed'}

            self.logger.info(f"Final translation: {translated_text}")
            return {
                'success': True,
                'translated_text': translated_text,
                'original_text': text
            }

        except Exception as e:
            self.logger.error(f"Translation process error: {str(e)}")
            return {'success': False, 'error': str(e)}

    def _validate_translation_config(self):
        """Validate the current translation configuration"""
        required_keys = [
            'operation_type', 'input_type',
            'source_lang', 'target_lang',
            'translation_mode', 'speech_text' if self.user_config['input_type'] == 'speech' else None
        ]
        required_keys = [key for key in required_keys if key is not None]


        missing = [key for key in required_keys if key not in self.user_config or not self.user_config[key]]
        if missing:
            self.logger.error(f"Missing configuration keys: {missing}")
            return False

        if self.user_config['operation_type'] == 'translate' and not self.user_config['translation_mode']:
            self.logger.error("Missing translation mode")
            return False

        return True
    def _process_offline_image_translation(self, image_path):
        """Process offline image translation using LLM Manager"""
        try:
            # Read image data
            image = cv2.imread(image_path)
            if image is None:
                self.io_manager.interact_with_user("Could not read the captured image", mode="both")
                return

            # Prepare configuration for LLM Manager
            llm_config = {
                'translation_mode': 'image',
                'source_lang': self.user_config['source_lang'],
                'target_lang': self.user_config['target_lang']
            }

            # Show processing message
            self.io_manager.interact_with_user("Processing image, please wait...", mode="display")

            # Process with LLM Manager
            result = self.llm_manager.process_user_inputs(llm_config, image)

            if result and result.get('success'):
                original_text = result.get('original_text', '')
                translated_text = result.get('translated_text', '')

                if translated_text:
                    result_text = f"Extracted: {original_text}\nTranslated: {translated_text}"
                    self.io_manager.gui.update_ai_response(result_text)
                    self.io_manager.interact_with_user("Translation complete", mode="speech")
                elif original_text:
                    self.io_manager.gui.update_ai_response(f"Extracted: {original_text}")
                    self.io_manager.interact_with_user("Text extracted but translation failed", mode="both")
                else:
                    self.io_manager.interact_with_user("No text found in image", mode="both")
            else:
                error = result.get('error', 'Unknown error') if result else 'Image processing failed'
                self.io_manager.interact_with_user(f"Image processing failed: {error}", mode="both")

        except Exception as e:
            self.logger.error(f"Error in offline image translation: {str(e)}")
            self.io_manager.interact_with_user("Image translation failed due to an error", mode="both")

    def _process_ocr(self):
        """Process OCR operation"""
        try:
            # Capture image
            self.io_manager.interact_with_user("Capturing image...", mode="display")
            image_path = self.io_manager.get_image()

            if not image_path:
                self.io_manager.interact_with_user("Could not capture image", mode="both")
                return

            # Read image data
            image = cv2.imread(image_path)
            if image is None:
                self.io_manager.interact_with_user("Could not read the captured image", mode="both")
                return

            # Prepare LLM config for OCR processing
            llm_config = {
                'translation_mode': 'ocr',
                'source_lang': self.user_config['source_lang']
            }

            # Show processing message
            self.io_manager.interact_with_user("Extracting text, please wait...", mode="display")

            # Process with LLM Manager
            result = self.llm_manager.process_user_inputs(llm_config, image)

            if result and result.get('success') and result.get('original_text'):
                extracted_text = result.get('original_text', '')

                # Save to file
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                text_file = f"ocr_result_{timestamp}.txt"

                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"Extracted Text:\n{extracted_text}\n")
                    f.write(f"Source Language: {self.user_config['source_lang']}\n")
                    f.write(f"Image Path: {image_path}\n")

                # Display result
                self.io_manager.gui.update_ai_response(f"Extracted: {extracted_text}")
                self.io_manager.interact_with_user(f"OCR complete. Text saved to {text_file}", mode="both")
            else:
                self.io_manager.interact_with_user("Could not extract text from image", mode="both")

        except Exception as e:
            self.logger.error(f"Error in OCR processing: {str(e)}")
            self.io_manager.interact_with_user("OCR failed due to an error", mode="both")

    def _send_online_request(self, image_path, prompt_audio_path=None):
        """Send request to online server"""
        try:
            # Connect to server
            self.io_manager.interact_with_user("Connecting to server...", mode="display")
            if not self.socket_handler.connect_to_server():
                self.io_manager.interact_with_user("Could not connect to server", mode="both")
                return

            # Prepare request
            self.io_manager.interact_with_user("Sending request to server...", mode="display")

            # Convert source and target languages to codes
            source_code = self.user_config['source_lang']  # Should already be a code
            target_code = self.user_config['target_lang']  # Should already be a code
            with open(image_path, "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # Send appropriate request type
            if self.user_config['use_prompt'] and prompt_audio_path:
                # Send combined request with prompt
                response = self.socket_handler.send_and_receive(
                    "audio_and_image",
                    "en",
                    "en",
                    image_data=image_data,
                    audio_data=prompt_audio_path
                )
            else:
                # Send image only
                response = self.socket_handler.send_and_receive(
                    "image",
                    source_code,
                    target_code,
                    image_data=image_path
                )

            # Process response
            if response:
                self.io_manager.gui.update_ai_response(f"Server response: {response}")
                self.io_manager.interact_with_user(f"Online translation completed response is: {response}", mode="both")
                time.sleep(5)
            else:
                self.io_manager.interact_with_user("No response from server", mode="both")

        except Exception as e:
            self.logger.error(f"Error in online request: {str(e)}")
            self.io_manager.interact_with_user("Online translation failed due to an error", mode="both")
        finally:
            # Always disconnect from server
            try:
                self.socket_handler.disconnect_from_server()
            except:
                pass

    def _save_config(self):
        """Save current configuration to JSON file"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            config_file = f"session_config_{timestamp}.json"

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_config, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Configuration saved to {config_file}")

        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")

    def _reset_conversation(self):
        """Reset conversation state and prepare for next interaction"""
        try:
            # Tell the user we're ready for the next command
            self.io_manager.interact_with_user("Say 'Hi David' when you're ready for the next command.", mode="both")

            # Reset wake word detection flag
            self.wake_word_detected = False

            # Reset user config
            self._reset_user_config()

            # Start wake word detection again
            wake_word_thread = create_worker(
                self._wait_for_wake_word,
                on_result=self._on_wake_word_detected,
                task_name="Wake Word Detection"
            )
            wake_word_thread.start()

        except Exception as e:
            self.logger.error(f"Error resetting conversation: {str(e)}")

    def _handle_interrupt(self, signum, frame):
        """Handle interrupt signals (Ctrl+C)"""
        self.logger.info(f"Received signal {signum}, shutting down")
        self.running = False
        self._shutdown()

        # Force exit if needed
        if signum == signal.SIGINT:
            sys.exit(0)

    def _shutdown(self):
        """Clean shutdown of all components"""
        if not hasattr(self, 'shutting_down') or not self.shutting_down:
            self.shutting_down = True
            self.logger.info("Shutting down application")

            # Stop processing loop
            self.running = False
            if hasattr(self, 'status_timer'):
                self.status_timer.cancel()
            self.wake_word_detected = False

            # Stop camera stream
            try:
                if hasattr(self, 'io_manager') and self.io_manager:
                    self.logger.info("Stopping camera stream")
                    self.io_manager.stop_camera_stream()
            except Exception as e:
                self.logger.error(f"Error stopping camera: {str(e)}")

            # Clean up LLM Manager
            try:
                if hasattr(self, 'llm_manager') and self.llm_manager:
                    self.logger.info("Cleaning up LLM Manager")
                    self.llm_manager.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up LLM Manager: {str(e)}")

            # Disconnect from server
            try:
                if hasattr(self, 'socket_handler') and self.socket_handler:
                    self.logger.info("Disconnecting from server")
                    self.socket_handler.disconnect_from_server()
            except Exception as e:
                self.logger.error(f"Error disconnecting from server: {str(e)}")

            self.logger.info("Shutdown complete")


def main():
    """Main entry point for the application"""
    app = ARGlassesAssistant()
    if app.initialize():
        return app.start()
    else:
        print("Failed to initialize application")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
