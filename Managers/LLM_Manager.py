class LLM_Manager:
    def __init__(self, llm):
        self.llm = llm

    # def generate_response(self, prompt):
    #     response = self.llm.generate(prompt)
    #     return response
    #
    # def set_model(self, model_name):
    #     self.llm.set_model(model_name)
    #
    # def get_available_models(self):
    #     return self.llm.get_available_models()

    def recognize_model_path(self, language_name):
        """
        Returns the file path to the Vosk speech recognition model based on the provided language code.

        Args:
            language_name (str): Language code ('ar' for Arabic, 'en' for English, 'fr' for French).

        Returns:
            str: The file path to the corresponding Vosk model for the given language.
        """

        if (language_name == "ar"):
            model_path = r"/home/pi/Desktop/gradproj/vosk-model-ar-mgb2-0.4"
        elif (language_name == "en"):
            model_path = r"/home/pi/Desktop/gradproj/vosk-model-small-en-us-0.15"
        elif (language_name == "fr"):
            model_path = r"/home/pi/Desktop/gradproj/vosk-model-small-fr-0.22"

        return model_path