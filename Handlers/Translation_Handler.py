from utils.Logging import Logger

class TranslationHandler:
    def __init__(self):
        self.logger = Logger()
        self.logger.info("Initializing Translation Handler")

    def translate_text(self, text, model_components=None):
        """
        Translate text using provided model components.

        Args:
            text (str): Text to translate
            model_components (tuple): (model, tokenizer, success) from LTDHandler

        Returns:
            str: Translated text or None if translation fails
        """
        try:
            if not text or not model_components:
                self.logger.error("Missing text or model components")
                return None

            model, tokenizer, success = model_components

            if not success:
                self.logger.error("Model components not valid")
                return None

            # Use the provided model and tokenizer
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            translated = model.generate(**inputs)
            translation = tokenizer.decode(translated[0], skip_special_tokens=True)

            return translation

        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}")
            return None

    def __batch_translate_text(self, texts, model_components):
        """Batch translate multiple texts"""
        if not texts or not model_components:
            return []

        model, tokenizer, success = model_components
        if not success:
            return []

        try:
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            translations = model.generate(**inputs)
            return [tokenizer.decode(t, skip_special_tokens=True) for t in translations]
        except Exception as e:
            self.logger.error(f"Batch translation error: {str(e)}")
            return []
