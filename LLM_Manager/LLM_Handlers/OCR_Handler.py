import os
import cv2
import numpy as np
import pytesseract
from PyQt5.QtCore import QObject
from IO_Manager.IO_Handlers.Audio_Handler import AudioHandler

class OCR_Handler(QObject):
    """
    Handler class for Optical Character Recognition operations.

    This class provides methods for processing images and extracting text,
    with support for multiple languages and image preprocessing techniques.
    """

    def __init__(self, audio_handler=None):
        """
        Initialize the OCR_Handler.

        Args:
            audio_handler (AudioHandler, optional): An instance of AudioHandler for speech feedback.
                If None, a new instance will be created.
        """
        super().__init__()
        self.audio_handler = audio_handler or AudioHandler()

        # Configure Tesseract path if needed (especially for Windows)
        if os.name == 'nt':  # Windows
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        # Language mappings for Tesseract
        self.tesseract_languages = {
            "en": "eng",
            "ar": "ara",
            "fr": "fra"
        }

        # Advanced OCR configurations
        self.ocr_configs = {
            "default": r"--oem 3 --psm 11",
            "document": r"--oem 3 --psm 1",
            "table": r"--oem 3 --psm 6",
            "single_line": r"--oem 3 --psm 7"
        }

        # Processing parameters
        self.preprocessing_level = "medium"  # Can be "low", "medium", "high"

    def recognize_text(self, image_path, lang="eng", save_processed=True, mode="default"):
        """
        Recognize text in an image using Tesseract OCR with advanced preprocessing.

        Args:
            image_path (str): The path to the image file.
            lang (str, optional): The language of the text in the image. Defaults to "eng".
            save_processed (bool, optional): Whether to save the processed image. Defaults to True.
            mode (str, optional): OCR mode - "default", "document", "table", or "single_line". Defaults to "default".

        Returns:
            str: The recognized text if successful, otherwise an empty string.
        """
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                self.audio_handler.speak("Failed to open image file")
                return ""

            # Process image based on preprocessing level
            processed = self._preprocess_image(img)

            # Save processed image if requested
            if save_processed:
                processed_path = os.path.splitext(image_path)[0] + "_processed.jpg"
                cv2.imwrite(processed_path, processed)
                print(f"Saved preprocessed image to: {processed_path}")

            # Get OCR config
            custom_config = self.ocr_configs.get(mode, self.ocr_configs["default"])

            # Perform OCR
            raw_text = pytesseract.image_to_string(processed, lang=lang, config=custom_config)
            text = raw_text.strip()

            if not text:
                self.audio_handler.speak("No text found in the image")

            return text

        except Exception as e:
            self.audio_handler.speak(f"OCR processing error")
            print(f"OCR Error: {e}")
            return ""

    def _preprocess_image(self, img):
        """
        Preprocess image for better OCR results based on current preprocessing level.

        Args:
            img: OpenCV image object

        Returns:
            Processed OpenCV image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.preprocessing_level == "low":
            # Minimal processing
            return gray

        # Medium processing (default)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        _, thresh = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if self.preprocessing_level == "medium":
            return thresh

        # High processing - additional steps for challenging images
        if self.preprocessing_level == "high":
            processed = cv2.dilate(thresh, np.ones((1, 1), np.uint8), iterations=1)
            processed = cv2.erode(processed, np.ones((1, 1), np.uint8), iterations=1)

            # Edge enhancement
            edges = cv2.Canny(processed, 50, 150)
            processed = cv2.addWeighted(processed, 0.8, edges, 0.2, 0)

            return processed

        # Default fallback
        return thresh

    def set_preprocessing_level(self, level):
        """
        Set image preprocessing level.

        Args:
            level (str): "low", "medium", or "high"

        Returns:
            bool: True if successful, False otherwise
        """
        if level in ["low", "medium", "high"]:
            self.preprocessing_level = level
            return True
        return False

    def get_text_from_image(self, image_path, language_code):
        """
        Extract text from an image using the appropriate language setting.

        Args:
            image_path (str): Path to the image file
            language_code (str): Two-letter language code ("en", "ar", "fr")

        Returns:
            str: Recognized text
        """
        # Convert language code to Tesseract format
        tesseract_lang = self.get_tesseract_language(language_code)

        # Perform OCR
        return self.recognize_text(image_path, lang=tesseract_lang)

    def get_tesseract_language(self, language_code):
        """
        Convert a two-letter language code to the Tesseract language format.

        Args:
            language_code (str): Two-letter language code ("en", "ar", "fr")

        Returns:
            str: Three-letter Tesseract language code ("eng", "ara", "fra")
        """
        return self.tesseract_languages.get(language_code, "eng")

    def capture_and_recognize(self, camera_widget, language_code, max_attempts=3):
        """
        Capture an image from camera and recognize text in one operation.

        Args:
            camera_widget: Camera widget to capture image from
            language_code (str): Two-letter language code
            max_attempts (int): Maximum number of capture attempts

        Returns:
            tuple: (success (bool), text (str))
        """
        # Try to capture image
        for attempt in range(max_attempts):
            self.audio_handler.speak(f"Capturing image, attempt {attempt + 1}")
            filename = camera_widget.capture_image(f"ocr_capture_{attempt}.jpg")

            if not filename or not os.path.exists(filename):
                continue

            # Get tesseract language and perform OCR
            tesseract_lang = self.get_tesseract_language(language_code)
            text = self.recognize_text(filename, lang=tesseract_lang)

            if text:
                return True, text

            self.audio_handler.speak("No text found, trying again")

        self.audio_handler.speak("Could not detect text after multiple attempts")
        return False, ""
