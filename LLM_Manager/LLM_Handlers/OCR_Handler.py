import os
import cv2
import numpy as np
import pytesseract
from utils.logging import Logger

class OCR_Handler:
    """
    Handler class for Optical Character Recognition operations.
    Works independently of other handlers, communicating only through LLM_Manager.
    """

    def __init__(self):
        """Initialize the OCR_Handler."""
        self.logger = Logger()
        self.logger.info("Initializing OCR_Handler")

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
            image_path (str): Path to the image file
            lang (str): The language code for OCR
            save_processed (bool): Whether to save the processed image
            mode (str): OCR mode (default, document, etc.)

        Returns:
            str: Recognized text or empty string if failed
        """
        try:
            # Load and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                self.logger.error(f"Failed to open image file: {image_path}")
                return ""

            # Process image based on preprocessing level
            processed = self._preprocess_image(img)

            # Save processed image if requested
            if save_processed:
                processed_path = os.path.splitext(image_path)[0] + "_processed.jpg"
                cv2.imwrite(processed_path, processed)
                self.logger.debug(f"Saved preprocessed image to: {processed_path}")

            # Get OCR config
            custom_config = self.ocr_configs.get(mode, self.ocr_configs["default"])

            # Perform OCR
            raw_text = pytesseract.image_to_string(processed, lang=lang, config=custom_config)
            text = raw_text.strip()

            if not text:
                self.logger.warning("No text found in the image")

            return text

        except Exception as e:
            self.logger.error(f"OCR Error: {e}")
            return ""

    def _preprocess_image(self, img):
        """
        Preprocess image for better OCR results.

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
            language_code (str): Two-letter language code

        Returns:
            str: Recognized text
        """
        # Convert language code to Tesseract format
        tesseract_lang = self.get_tesseract_language(language_code)

        # Perform OCR
        return self.recognize_text(image_path, lang=tesseract_lang)

    def get_tesseract_language(self, language_code):
        """
        Convert a language code to Tesseract format.

        Args:
            language_code (str): Two-letter language code

        Returns:
            str: Three-letter Tesseract language code
        """
        return self.tesseract_languages.get(language_code, "eng")
