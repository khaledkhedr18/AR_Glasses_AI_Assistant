from utils.Logging import Logger

class TranslationHandler:
    def __init__(self):
        self.logger = Logger()
        self.logger.info("Initializing Translation Handler")

    def translate_text(self, text, model_components=None):
        try:
            self.logger.info(f"Translation input: {text}")

            # Get the pipeline object
            if hasattr(model_components, '__call__'):
                pipeline_obj = model_components
            elif isinstance(model_components, dict):
                pipeline_obj = model_components.get('translation_pipeline')
            else:
                self.logger.error("Invalid model components format")
                return None

            if not pipeline_obj:
                self.logger.error("No translation pipeline available")
                return None

            # Process translation with safeguards
            result = pipeline_obj(
                text,
                max_length=50,
                num_beams=2,
                early_stopping=True
            )

            if not result:
                self.logger.error("Empty translation result")
                return None

            translation = result[0].get('translation_text', '').strip()
            self.logger.info(f"Raw translation output: {translation}")

            return translation

        except Exception as e:
            self.logger.error(f"Translation error: {str(e)}", exc_info=True)
            return None
