from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QProgressBar
from .widgets.camera_widget import CameraWidget
from .widgets.overlay_widget import StatusLabel, SpeechLabel
from .threads.worker_thread import WorkerThread
from .threads.audio_thread import AudioThread
from core.services.translation import TranslationService
from core.models.base import ModelFactory
from config.settings import Settings
from core.utils.logging import Logger

class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.logger = Logger()
        self.settings = Settings()
        self.translation_service = TranslationService()
        self.setup_ui()
        self.setup_connections()
        self.showFullScreen()

    def setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("AR AI Assistant")

        # Create central widget
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Create layout
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Create camera widget
        self.camera_widget = CameraWidget(self.central_widget)
        self.layout.addWidget(self.camera_widget)

        # Create overlay labels
        self.status_label = StatusLabel(self.central_widget)
        self.speech_label = SpeechLabel(self.central_widget)
        self.camera_widget.add_overlay(self.status_label)
        self.camera_widget.add_overlay(self.speech_label)

        # Create progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background: transparent;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
            }
        """)
        self.layout.addWidget(self.progress_bar)

        # Position labels
        self.update_label_positions()

        # Create status check timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Check every 1s

        # Initialize audio thread
        self.audio_thread = AudioThread()
        self.audio_thread.text_received.connect(self.handle_user_speech)
        self.audio_thread.start()

    def setup_connections(self):
        """Setup signal connections"""
        self.audio_thread.text_received.connect(self.handle_user_speech)
        self.translation_service.progress.progress_updated.connect(self.update_progress)
        self.resizeEvent = self.handle_resize

    def update_label_positions(self):
        """Update the positions of overlay labels"""
        self.status_label.move(
            self.width() - self.status_label.width() - 20,
            20
        )
        self.speech_label.move(
            (self.width() - self.speech_label.width()) // 2,
            self.height() - self.speech_label.height() - 20
        )

    def handle_resize(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        if hasattr(self, 'camera_widget'):
            self.camera_widget.setGeometry(0, 0, self.width(), self.height())
        self.update_label_positions()

    def update_status(self):
        """Update the connection status"""
        if self.translation_service.network.connected:
            self.status_label.setText("Connected")
            self.status_label.setStyleSheet("color: #4CAF50;")
        else:
            self.status_label.setText("Disconnected")
            self.status_label.setStyleSheet("color: #F44336;")

    def update_progress(self, value: int, status: str):
        """Update progress bar and status"""
        self.progress_bar.setValue(value)
        if status:
            self.status_label.setText(status)

    def handle_user_speech(self, text: str):
        """Handle user speech input"""
        if not text:
            return

        self.speech_label.setText(text)

        if "hi david" in text.lower():
            self.start_translation_process()

    def start_translation_process(self):
        """Start the translation process"""
        if hasattr(self, 'is_processing') and self.is_processing:
            return

        self.is_processing = True
        self.audio_thread.pause()
        self.speech_label.setText("Assistant: Initializing...")

        # Start tool detection
        self.tool_detection_worker = WorkerThread(
            self.detect_input_type,
            self.audio_thread.speech_model
        )
        self.tool_detection_worker.result.connect(self.handle_input_type)
        self.tool_detection_worker.start()

    def detect_input_type(self, model):
        """Detect the type of input (speech or image)"""
        try:
            # Check if there's any text in the speech label
            current_text = self.speech_label.text().lower()

            # Check for image-related keywords
            image_keywords = ['image', 'picture', 'photo', 'scan', 'camera']
            if any(keyword in current_text for keyword in image_keywords):
                return "image"

            # Check for speech-related keywords
            speech_keywords = ['speak', 'talk', 'say', 'tell', 'voice']
            if any(keyword in current_text for keyword in speech_keywords):
                return "speech"

            # Default to speech if no specific keywords found
            return "speech"

        except Exception as e:
            self.logger.error(f"Error detecting input type: {e}")
            return None

    def handle_input_type(self, input_type: str):
        """Handle the detected input type"""
        if not input_type:
            self.speech_label.setText("Could not detect input type. Please try again.")
            self.is_processing = False
            self.audio_thread.resume()
            return

        self.input_type = input_type
        self.speech_label.setText(f"Detected {input_type} input")
        self.start_language_selection()

    def start_language_selection(self):
        """Start the language selection process"""
        self.source_lang_worker = WorkerThread(
            self.get_source_language,
            self.audio_thread.speech_model
        )
        self.source_lang_worker.result.connect(self.handle_source_language)
        self.source_lang_worker.start()

    def get_source_language(self, model):
        """Get the source language from user input"""
        try:
            # Get current text from speech label
            text = self.speech_label.text().lower()

            # Map common language names to language codes
            language_map = {
                'english': 'en',
                'french': 'fr',
                'arabic': 'ar',
                'spanish': 'es',
                'german': 'de',
                'italian': 'it',
                'portuguese': 'pt',
                'russian': 'ru',
                'chinese': 'zh',
                'japanese': 'ja',
                'korean': 'ko'
            }

            # Check for language names in the text
            for lang_name, lang_code in language_map.items():
                if lang_name in text:
                    return lang_code

            # If no language specified, try to detect from the text
            if model:
                detected_lang = model.detect_language(text)
                if detected_lang:
                    return detected_lang

            # Default to English if no language detected
            return 'en'

        except Exception as e:
            self.logger.error(f"Error getting source language: {e}")
            return None

    def handle_source_language(self, source_lang: str):
        """Handle the selected source language"""
        if not source_lang:
            self.speech_label.setText("Invalid source language. Please try again.")
            self.is_processing = False
            return

        self.source_lang = source_lang
        self.speech_label.setText(f"Source language: {source_lang}")

        # Start target language selection
        self.target_lang_worker = WorkerThread(
            self.get_target_language,
            self.audio_thread.speech_model
        )
        self.target_lang_worker.result.connect(self.handle_target_language)
        self.target_lang_worker.start()

    def get_target_language(self, model):
        """Get the target language from user input"""
        try:
            # Get current text from speech label
            text = self.speech_label.text().lower()

            # Map common language names to language codes
            language_map = {
                'english': 'en',
                'french': 'fr',
                'arabic': 'ar',
                'spanish': 'es',
                'german': 'de',
                'italian': 'it',
                'portuguese': 'pt',
                'russian': 'ru',
                'chinese': 'zh',
                'japanese': 'ja',
                'korean': 'ko'
            }

            # Check for language names in the text
            for lang_name, lang_code in language_map.items():
                if lang_name in text:
                    # Don't allow same source and target language
                    if lang_code != self.source_lang:
                        return lang_code

            # If no language specified, suggest a different language
            available_langs = list(language_map.values())
            if self.source_lang in available_langs:
                available_langs.remove(self.source_lang)

            # Return first available language or default to English
            return available_langs[0] if available_langs else 'en'

        except Exception as e:
            self.logger.error(f"Error getting target language: {e}")
            return None

    def handle_target_language(self, target_lang: str):
        """Handle the selected target language"""
        if not target_lang:
            self.speech_label.setText("Invalid target language. Please try again.")
            self.is_processing = False
            return

        self.target_lang = target_lang
        self.speech_label.setText(f"Target language: {target_lang}")
        self.start_translation()

    def start_translation(self):
        """Start the translation process"""
        try:
            self.translation_service.initialize_models(
                self.source_lang,
                self.target_lang
            )

            if self.input_type == "speech":
                self.translation_worker = WorkerThread(
                    self.translation_service.translate_text,
                    self.speech_label.text(),
                    self.source_lang,
                    self.target_lang
                )
            else:  # image
                self.translation_worker = WorkerThread(
                    self.translation_service.translate_image,
                    self.source_lang,
                    self.target_lang
                )

            self.translation_worker.result.connect(self.handle_translation)
            self.translation_worker.start()

        except Exception as e:
            self.speech_label.setText(f"Error: {str(e)}")
            self.is_processing = False

    def handle_translation(self, result):
        """Handle the translation result"""
        if isinstance(result, tuple):
            success, text = result
            if not success:
                self.speech_label.setText(f"Error: {text}")
                self.is_processing = False
                return
        else:
            text = result

        self.speech_label.setText(f"Assistant: {text}")
        self.is_processing = False
        self.audio_thread.resume()

    def closeEvent(self, event):
        """Handle window close events"""
        self.audio_thread.stop()
        self.audio_thread.wait()
        self.translation_service.cleanup()
        event.accept()
