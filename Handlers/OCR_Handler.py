import os
import cv2
import numpy as np
import pytesseract
from utils.config import OCR_CONFIG
from utils.logging import Logger

class OCRHandler:
    def __init__(self):
        self.logger = Logger()
        self.logger.info("Initializing OCR Handler")

        # Initialize OCR configurations with dictionary access
        self.preprocessing_level = OCR_CONFIG['PROCESSING']['LEVEL']
        self.default_mode = 'DEFAULT'
        self.save_directory = OCR_CONFIG['STORAGE']['SAVE_DIRECTORY']
        self.processed_frame_filename = OCR_CONFIG['STORAGE']['PROCESSED_FRAME_FILENAME']
        self.ocr_configs = OCR_CONFIG['MODES']
        os.makedirs(self.save_directory, exist_ok=True)

    def extract_text_from_frame(self, frame, lang="en", save_processed=False, mode=None):
        """
        Extract text from camera frame array using OCR.
        """
        try:
            if frame is None:
                self.logger.error("Invalid frame array")
                return ""

            mode = mode or self.default_mode
            processed = self._preprocess_image(frame)

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

    def _preprocess_image(self, frame):
        """
        Preprocess image array for better OCR results.
        """
        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            if self.preprocessing_level == "low":
                return gray

            # Updated to use dictionary access
            denoised = cv2.fastNlMeansDenoising(gray, h=OCR_CONFIG['PROCESSING']['DENOISE_H'])
            _, thresh = cv2.threshold(denoised, OCR_CONFIG['PROCESSING']['THRESH_VALUE'],
                                    OCR_CONFIG['PROCESSING']['MAX_VALUE'],
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if self.preprocessing_level == "medium":
                return thresh

            if self.preprocessing_level == "high":
                # Updated to use dictionary access
                kernel = np.ones(OCR_CONFIG['PROCESSING']['KERNEL_SIZE'], np.uint8)
                processed = cv2.dilate(thresh, kernel, iterations=1)
                processed = cv2.erode(processed, kernel, iterations=1)
                return processed

            return thresh

        except Exception as e:
            self.logger.log_error_with_traceback("Error preprocessing image", e)
            return None
