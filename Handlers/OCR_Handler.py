import os
import cv2
import numpy as np
import pytesseract
from utils.Config import OCR_CONFIG
from utils.Logging import Logger

class OCRHandler:
    def __init__(self):
        self.logger = Logger()
        self.logger.info("Initializing OCR Handler")

        # Initialize OCR configurations with dictionary access
        self.__load_config()

    def extract_text_from_frame(self, frame, lang="en", save_processed=False, mode=None):
        """
        Extract text from camera frame array using OCR.
        """
        try:
            if frame is None:
                self.logger.error("Invalid frame array")
                return ""

            mode = mode or self.default_mode
            processed = self.__preprocess_image(frame)

            if save_processed:
                save_path = os.path.join(self.save_directory, self.processed_frame_filename)
                cv2.imwrite(save_path, processed)
                self.logger.debug(f"Saved preprocessed frame to: {save_path}")

            custom_config = self.ocr_configs[mode]
            text = pytesseract.image_to_string(processed, lang=lang, config=custom_config)
            text = text.strip()

            if not text:
                self.logger.warning("No text found in frame")

            return text

        except Exception as e:
            self.logger.log_error_with_traceback("OCR Error", e)
            return ""

    def set_preprocessing_level(self, level):
        """Set image preprocessing level."""
        if level in ["low", "medium", "high"]:
            self.preprocessing_level = level
            return True
        return False

    def __preprocess_image(self, frame):
        """Preprocess image array for better OCR results."""
        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            if self.preprocessing_level == "low":
                return gray

            # Use config values with .get() for safety
            processing_config = OCR_CONFIG.get('PROCESSING', {})
            denoised = cv2.fastNlMeansDenoising(
                gray,
                h=processing_config.get('DENOISE_H', 10)
            )

            _, thresh = cv2.threshold(
                denoised,
                processing_config.get('THRESH_VALUE', 150),
                processing_config.get('MAX_VALUE', 255),
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            if self.preprocessing_level == "medium":
                return thresh

            if self.preprocessing_level == "high":
                kernel = np.ones(
                    processing_config.get('KERNEL_SIZE', (1, 1)),
                    np.uint8
                )
                processed = cv2.dilate(thresh, kernel, iterations=1)
                processed = cv2.erode(processed, kernel, iterations=1)
                return processed

            return thresh

        except Exception as e:
            self.logger.log_error_with_traceback("Error preprocessing image", e)
            return None

    def __load_config(self):
        """Load configuration settings for OCR."""
        # Load processing settings with defaults
        processing_config = OCR_CONFIG.get('PROCESSING', {})
        self.preprocessing_level = processing_config.get('LEVEL', 'medium')
        self.thresh_value = processing_config.get('THRESH_VALUE', 150)
        self.kernel_size = processing_config.get('KERNEL_SIZE', (1, 1))
        self.denoise_h = processing_config.get('DENOISE_H', 10)
        self.max_value = processing_config.get('MAX_VALUE', 255)

        # Load storage settings with defaults
        storage_config = OCR_CONFIG.get('STORAGE', {})
        self.save_directory = storage_config.get('SAVE_DIRECTORY', '/tmp/AIAssistant/')
        self.processed_frame_filename = storage_config.get('PROCESSED_FRAME_FILENAME', 'processed_frame.jpg')

        # Load OCR modes with defaults
        self.ocr_configs = OCR_CONFIG.get('MODES', {
            'DEFAULT': '--oem 3 --psm 3',
            'ACCURATE': '--oem 3 --psm 6'
        })
        self.default_mode = 'DEFAULT'

        # Create save directory if it doesn't exist
        os.makedirs(self.save_directory, exist_ok=True)
