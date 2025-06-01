import time
import os
import platform
from config import vosk_model_paths,vosk_languages ,DEFAULT_USERNAME
# Import services
from audio_services import SpeechRecognizer, TextToSpeech
from language_services import TranslationService
from vision_services import OCRService, CameraManager
from api_service import APIService

class InteractionManager:
    def __init__(self, ui_manager, speech_recognizer: SpeechRecognizer, tts_engine: TextToSpeech,
                 translation_service: TranslationService, ocr_service: OCRService,
                 api_service: APIService, camera_manager: CameraManager,
                 initial_username=DEFAULT_USERNAME):
        self.ui = ui_manager
        self.speech_recognizer = speech_recognizer
        self.tts = tts_engine
        self.translator = translation_service
        self.ocr = ocr_service
        self.api = api_service
        self.camera = camera_manager

        self.username = initial_username
        self.current_vosk_lang_code = "en"
        
        self.app_running = True
        self.image_capture_path = "captured_image.jpg"
        self.current_interaction_mode = "idle"
        print("InteractionManager initialized.")

    def _get_audio_input(self, prompt_message=None, update_interaction=True):
        if prompt_message and update_interaction:
            self.ui.display_output(prompt_message)
        
        text = self.speech_recognizer.get_audio(
            display_output=self.ui.display_output,
            display_output2=self.ui.display_output2, 
        )
        return text.strip().lower() if text else ""

    def _speak_and_update_ui(self, text_to_speak, interaction_text_to_display=None):
        display_text = interaction_text_to_display if interaction_text_to_display is not None else text_to_speak
        self.ui.display_output2(display_text)
        self.tts.speak(text_to_speak)

    def _reset_vosk_model_to_english(self):
        """Resets the Vosk speech recognition model to English if it's not already."""
        if self.current_vosk_lang_code != "en":
            print("Resetting Vosk model to English...")
            if self.speech_recognizer.change_model("en"):
                self.current_vosk_lang_code = "en"
                print("Vosk model successfully reset to English.")
                return True
            else:
                print("Failed to reset Vosk model to English.")
                self._speak_and_update_ui("Error: Could not switch back to English speech model.")
                return False
        else:
            return True # Already English
    def _handle_shutdown(self):
        """Handles the application shutdown sequence."""
        self._speak_and_update_ui("Shutting down AR EyeConic. Goodbye!", "Shutting down...")
        self.app_running = False # Stop the main loop
        # Perform OS-specific shutdown
        system = platform.system()
        print("Shutdown command received. OS-level shutdown initiated.")
        try:
            if system == "Windows":
                os.system("shutdown /s /t 1")
            elif system == "Linux" or system == "Darwin":
                os.system("sudo shutdown now")
            else:
                print(f"Unsupported OS for automated shutdown: {system}")
                self._speak_and_update_ui(f"Automated shutdown not supported on {system}.")
        except Exception as e:
            print(f"Error during OS shutdown command: {e}")
            self._speak_and_update_ui("Error trying to shut down the system.")


    def _select_language(self, prompt_message, for_source=True):
        self._speak_and_update_ui(prompt_message)
        
        self._reset_vosk_model_to_english()

        while self.app_running:

            lang_name_input = self._get_audio_input(update_interaction=False)

            if not lang_name_input:
                self._speak_and_update_ui("I didn't catch that. Please try again.")
            else:
                break
                return lang_name_input

        normalized_input = lang_name_input.strip().title()
        
        selected_lang_code = None

        if normalized_input in vosk_model_paths:
             selected_lang_code = normalized_input
        elif normalized_input in vosk_languages:
            selected_lang_code = vosk_languages[normalized_input]
        else: 
            for lang_name, lang_code in vosk_languages.items():
                if normalized_input in lang_name or lang_name in normalized_input : 
                    selected_lang_code = lang_code
                    self._speak_and_update_ui(f"Selected language: {lang_name}")
                    break
        
        if selected_lang_code:

            if for_source:
                if self.current_vosk_lang_code != selected_lang_code:
                    self.ui.show_loading_screen(f"Loading {normalized_input} speech model...")
                    if self.speech_recognizer.change_model(selected_lang_code):
                        self.current_vosk_lang_code = selected_lang_code
                        self.ui.hide_loading_screen()
                        return selected_lang_code
                    else:
                        self.ui.hide_loading_screen()
                        self._speak_and_update_ui(f"Sorry, I couldn't load the speech model for {normalized_input}.")
                        self._reset_vosk_model_to_english() # Revert to English on failure
                        return None
                else: # Already the correct model
                    return selected_lang_code
            else:
                return selected_lang_code
        else:
            self._speak_and_update_ui(f"I couldn't recognize the language '{lang_name_input}'. Please try again.")
            return None

    def _handle_speech_translation(self):

        self.current_interaction_mode = "speech_translation"
        self._speak_and_update_ui("Entering speech translation mode.")
        
        source_lang = self._select_language("What language will you be speaking in?", for_source=True)
        if not source_lang:
            self._reset_vosk_model_to_english()
            return

        target_lang = self._select_language("What language do you want to translate to?", for_source=False)
        if not target_lang:
            self._reset_vosk_model_to_english()
            return

        if source_lang == target_lang:
            self._speak_and_update_ui("Source and target languages are the same. No translation needed.")
            self._reset_vosk_model_to_english()
            return

        self._speak_and_update_ui(f"Okay, I will translate from {source_lang} to {target_lang}. Speak when ready.")
        self.ui.display_output("") # Clear previous translation

        while self.app_running and self.current_interaction_mode == "speech_translation":
            user_speech = self._get_audio_input(f"Speak in {source_lang} (or say 'get out' to exit):", update_interaction=True)

            if not user_speech:
                continue

            english_equivalent = ""
            if source_lang != "en":
                english_equivalent = self.translator.translate_to_english(user_speech)
                print(f"DBG: Spoken '{user_speech}' ({source_lang}) -> English equiv: '{english_equivalent}'")

            if "get out" in user_speech.lower() or (english_equivalent and "get out" in english_equivalent.lower().strip("!?.")):
                self._speak_and_update_ui("Exiting speech translation mode.")
                self.ui.display_output("")
                break 

            self.ui.show_loading_screen("Translating...")
            translated_text = self.translator.translate_text(user_speech, target_language=target_lang, source_language=source_lang)
            self.ui.hide_loading_screen()

            if translated_text:
                self.ui.display_output(translated_text)
                self.tts.speak(translated_text)
            else:
                self._speak_and_update_ui("Sorry, I couldn't translate that.")
        
        self._reset_vosk_model_to_english()
        self.current_interaction_mode = "idle"


    def _handle_image_translation(self):
        """Manages the image translation mode."""
        self.current_interaction_mode = "image_translation"
        self._speak_and_update_ui("Entering image translation mode.")

        source_lang_ocr = self._select_language("What is the language of the text in the image?", for_source=False) 

        target_lang_translate = self._select_language("What language do you want to translate the image text to?", for_source=False)
        
        tesseract_lang_code = self.ocr.get_tesseract_lang_code(source_lang_ocr)
        self._speak_and_update_ui(f"Okay, I will look for {source_lang_ocr} text and translate to {target_lang_translate}.")
        self.ui.display_output("")
        self.ui.display_output("")

        while self.app_running and self.current_interaction_mode == "image_translation":
            self._reset_vosk_model_to_english()
            
            command = self._get_audio_input("Say 'capture' to take a picture, or 'get out' to exit:")
            if not command:
                continue

            if "capture" in command:
                self.ui.display_output("Capturing image...")
                captured_image_file = self.camera.capture_image(self.image_capture_path)
                if captured_image_file:
                    self._speak_and_update_ui("Image captured.", f"Image captured: {os.path.basename(captured_image_file)}")
                    
                    self.ui.show_loading_screen("Recognizing text from image...")
                    recognized_text = self.ocr.recognize_text_from_image(captured_image_file, lang_code=tesseract_lang_code) 
                    self.ui.hide_loading_screen()

                    if "Error:" in recognized_text or "No text detected" in recognized_text :
                        self._speak_and_update_ui(recognized_text)
                        self.ui.display_output(recognized_text)
                    else:
                        self.ui.display_output(recognized_text) 
                        
                        if source_lang_ocr == target_lang_translate:
                            self._speak_and_update_ui(f"Recognized text: {recognized_text}")
                            self.ui.display_output("Source and target languages are the same.")
                        else:
                            self.ui.show_loading_screen(f"Translating text to {target_lang_translate}...")
                            translated_ocr_text = self.translator.translate_text(recognized_text, target_language=target_lang_translate, source_language=source_lang_ocr)
                            self.ui.hide_loading_screen()
                            if translated_ocr_text:
                                self.ui.display_output(translated_ocr_text)
                                self._speak_and_update_ui(translated_ocr_text) # Speak the translation
                            else:
                                self._speak_and_update_ui("Sorry, I couldn't translate the text from the image.")
                else:
                    self._speak_and_update_ui("Failed to capture image.")

            elif "get out" in command:
                self._speak_and_update_ui("Exiting image translation mode.")
                self.ui.display_output("")
                self.ui.display_output("")
                break
        
        self._reset_vosk_model_to_english()
        self.current_interaction_mode = "idle"

    def _get_tool_selection(self):
        """Asks user if they want to use speech or image for translation."""
        self._reset_vosk_model_to_english()
        while self.app_running:
            self._speak_and_update_ui("What is the data you will use for your translation: speech or image?",
                                      interaction_text_to_display="Select data type: speech or image?")
            choice = self._get_audio_input(update_interaction=False)
            if "speech" in choice:
                return "speech"
            elif "image" in choice:
                return "image"
            elif "get out" in choice:
                return None
            else:
                self._speak_and_update_ui("I didn't understand that. Please say 'speech' or 'image'.")
        return None

    def _get_continue_confirmation(self):
        """Asks user if they want to continue with the current operation (e.g., image translation)."""
        self._reset_vosk_model_to_english()
        while self.app_running:
            self.ui.display_output("Do you want to continue?")
            self._speak_and_update_ui("Do you want to continue?", interaction_text_to_display="Continue? (yes/no)")
            choice = self._get_audio_input(update_interaction=False)
            if "yes" in choice:
                return True
            elif "no" in choice:
                return False
            else:
                self._speak_and_update_ui("I didn't understand. Please say 'yes' or 'no'.")
        return False 

    def _handle_iconic_sub_assistant(self, initial_image_path=None):
        """Handles the nested 'Iconic' assistant that uses send_to_api."""
        self.current_interaction_mode = "iconic_sub_chat"
        self._speak_and_update_ui("Hello, I am Iconic, your personal assistant. Tell me how I can help you.",
                                  interaction_text_to_display="Iconic: How can I help?")
        self.ui.clear_all_text_outputs()
        
        
        current_image_for_api = initial_image_path

        while self.app_running and self.current_interaction_mode == "iconic_sub_chat":
            self._reset_vosk_model_to_english()
            
            prompt_text = "Iconic: Your query?"
            if current_image_for_api:
                prompt_text = f"Iconic: Query about {os.path.basename(current_image_for_api)} or new query?"
            
            user_query = self._get_audio_input(prompt_text)

            if not user_query:
                continue

            if "get out" in user_query:
                self._speak_and_update_ui("Exiting Iconic sub-assistant.")
                break

            if "capture" in user_query: 
                self.ui.display_output("Capturing image for Iconic...")
                captured_file = self.camera.capture_image(self.image_capture_path)
                if captured_file:
                    self._speak_and_update_ui(f"Image {os.path.basename(captured_file)} captured for Iconic.",
                                              interaction_text_to_display=f"Iconic: Image {os.path.basename(captured_file)} captured.")
                    current_image_for_api = captured_file 
                    continue 
                else:
                    self._speak_and_update_ui("Failed to capture image for Iconic.")
                    current_image_for_api = None 
                    continue
            
            self.ui.show_loading_screen("Iconic is thinking...")
            api_response = self.api.send_chat_to_api(user_query, image_path=current_image_for_api)
            self.ui.hide_loading_screen()

            if "Error:" in api_response:
                self._speak_and_update_ui(f"Iconic Error: {api_response}")
            else:
                self._speak_and_update_ui(api_response)
            

            current_image_for_api = None
            self.ui.display_output("") 

        self.current_interaction_mode = "idle" 
        self.ui.clear_all_text_outputs()


    def _handle_online_mode(self):
        self.current_interaction_mode = "online_david"
        self._speak_and_update_ui("David Online: You are connected. You can say 'capture', 'Iconic', or 'get out'.",
                                  interaction_text_to_display="David Online: 'capture', 'Iconic', or 'get out'.")
        
        last_captured_image = None

        while self.app_running and self.current_interaction_mode == "online_david":
            self._reset_vosk_model_to_english()
            command = self._get_audio_input(update_interaction=False)

            if not command:
                continue

            if "capture" in command:
                self.ui.display_output2("Capturing image...")
                captured_file = self.camera.capture_image(self.image_capture_path)
                if captured_file:
                    self._speak_and_update_ui(f"Image {os.path.basename(captured_file)} captured.",
                                              interaction_text_to_display=f"Image captured: {os.path.basename(captured_file)}")
                    last_captured_image = captured_file 
                else:
                    self._speak_and_update_ui("Failed to capture image.")
                continue 

            elif "iconic" in command:
                self._handle_iconic_sub_assistant(initial_image_path=last_captured_image)
                last_captured_image = None # Clear after Iconic has potentially used it
                # After Iconic sub-assistant finishes, prompt for next online_david command
                self._speak_and_update_ui("David Online: 'capture', 'Iconic', or 'get out'.",
                                          interaction_text_to_display="David Online: 'capture', 'Iconic', or 'get out'.")
                continue

            elif "get out" in command:
                self._speak_and_update_ui("Exiting David Online mode.")
                break
            
            elif "shut down" in command:
                self._handle_shutdown()
                return # Exit immediately

            else:
                if command:
                    self._speak_and_update_ui(f"David Online: Unknown command '{command}'. Try 'capture', 'Iconic', or 'get out'.")
        
        self.current_interaction_mode = "idle"


    def _handle_offline_mode(self):
        """Handles the offline mode when 'Hi David' is said and internet is NOT connected."""
        self.current_interaction_mode = "offline_david"
        self._speak_and_update_ui(f"Hello {self.username}, I am David, your offline assistant. How are you?",
                                  interaction_text_to_display=f"David Offline: Hello {self.username}!")
        
        tool_choice = self._get_tool_selection()

        if tool_choice == "speech":
            self._handle_speech_translation()
        elif tool_choice == "image":
            self._handle_image_translation()
        
        self.current_interaction_mode = "idle"


    def start_interaction_loop(self):
        while self.app_running:
            self.ui.root.update() 
            self._reset_vosk_model_to_english() 


            self.ui.display_output2("Say 'Hi David' to begin, or 'Shut down' to exit.")


            command = self._get_audio_input(update_interaction=False)

            if not command and self.app_running:
                time.sleep(0.1)
                continue
            
            if not self.app_running: break 

            if "shut down" in command:
                self._handle_shutdown()
                break

            elif "hi david" in command:
                self._speak_and_update_ui("Hello, I am David, your personal assistant.") # Original greeting
                
                if self.api.is_connected():
                    self._speak_and_update_ui("You are connected do you want to use our online assistant or offline assistant?",
                                              interaction_text_to_display="Connected: Online or Offline assistant?")
                while self.app_running and self.current_interaction_mode == "idle":
                    command = self._get_audio_input(update_interaction=False)
                    if "online" in command.lower():
                        self._speak_and_update_ui("You chose online assistant. Loading David Online...")
                        self._handle_online_mode()
                    elif "offline" in command.lower():
                        self._speak_and_update_ui("You chose offline assistant. Loading David Offline...")
                        self._handle_offline_mode()
                    else:
                        self._speak_and_update_ui("I didn't understand that. Please say 'online' or 'offline'.")
                        continue
                else:
                    self._handle_offline_mode()
                
                self.ui.display_output("") 
                self.ui.display_output("")
                self.ui.display_output("")

        
        print("Interaction loop ended.")