import os
import socket
import select
import time
from vosk import Model, KaldiRecognizer
import json
import base64
from transformers import MarianMTModel, MarianTokenizer, pipeline
import pytesseract
import subprocess
import numpy as np
import sounddevice as sd
import time
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QTimer
import threading
from fuzzywuzzy import fuzz
from model_loader import get_translation_model
import datetime
import cv2

SERVER_IP = '192.168.1.65'
SERVER_PORT = 4040
audio_lock = threading.Lock()
tts_threads = []

class CommunicationSignals(QObject):
    update_output = pyqtSignal(str)
    update_ai_speech = pyqtSignal(str)
    update_user_speech = pyqtSignal(str)

class TTSThread(QThread):
    finished = pyqtSignal()

    def __init__(self, text):
        """
        Initialize a TTSThread with given text.

        This constructor takes a text string as an argument and stores it
        as an instance variable. The text is played as speech when the
        thread is started.

        :param text: The text to play as speech.
        """
        super().__init__()
        self.text = text

    def run(self):
        """
        Runs the thread to play the given text as speech.

        This method is called automatically when the thread is started. It
        runs the flite command to play the given text as speech and then
        emits a finished signal.

        :return: None
        """
        subprocess.run(["flite", self.text])
        self.finished.emit()


class SocketManager:
    def __init__(self):
        """
        Initializes a SocketManager object.

        This method initializes a SocketManager object by setting the
        socket, file object, and connected flag to None, None, and False,
        respectively. It also sets the script directory and image counter
        to the current directory and 0, respectively.

        :return: None
        """
        self.sock = None
        self.file_obj = None
        self.connected = False
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.image_counter = 0

    def connect(self):
        """
        Establishes a connection to the server.

        This method attempts to connect to the server with the specified
        IP and port. If the connection is successful, it sets the connected
        flag to True. If the connection fails, it prints an error message and
        retries after 2 seconds. If an unexpected error occurs, it returns
        False.

        Returns:
            bool: True if the connection is established successfully, False otherwise.
        """
        while True:
            try:
                print(f"[{datetime.datetime.now().isoformat()}] Connecting to {SERVER_IP}:{SERVER_PORT}...")
                self.sock = socket.create_connection((SERVER_IP, SERVER_PORT))
                self.file_obj = self.sock.makefile('r')
                self.connected = True
                print(f"[{datetime.datetime.now().isoformat()}] Connection established")
                return True
            except (ConnectionRefusedError, OSError) as e:
                print(f"Connection error: {e}. Retrying in 2 seconds...")
                time.sleep(2)
            except Exception as e:
                print(f"Unexpected error: {e}")
                return False

    def send_image(self, image_path, source_lang, target_lang):
        """
        Sends an image to the server.

        This method reads the given image file as bytes, encodes it using
        Base64, and sends it to the server as a JSON message. The message
        contains the image data, source language, target language, and a
        timestamp. If the connection is not established or an error occurs
        during the sending process, it returns False.

        Args:
            image_path (str): The path to the image file to be sent.
            source_lang (str): The source language of the text to be translated.
            target_lang (str): The target language of the text to be translated.

        Returns:
            bool: True if the image is sent successfully, False otherwise.
        """
        if not self.connected or not self.sock:
            print("Not connected to server")
            return False

        try:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()

            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            message = {
                "type": "image",
                "data": base64_image,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "timestamp": datetime.datetime.now().isoformat()
            }
            data = json.dumps(message) + "\n"
            self.sock.sendall(data.encode('utf-8'))
            print(f"[{datetime.datetime.now().isoformat()}] Image sent successfully")
            return True
        except Exception as e:
            print(f"Error sending image: {e}")
            self.connected = False
            return False

    def send_text(self, text, source_lang, target_lang):
        """
        Sends a text message to the server.

        This method constructs a message containing the specified text,
        source and target languages, and sends it to the server over an
        established socket connection. In case the connection is not
        established or an error occurs during the sending process, it
        returns False.

        :param text: The text to send to the server.
        :param source_lang: The source language of the text.
        :param target_lang: The target language of the text.
        :return: True if the message is sent successfully, False otherwise.
        """
        if not self.connected or not self.sock:
            print("Not connected to server")
            return False

        try:
            message = {
                "type": "text",
                "data": text,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "timestamp": datetime.datetime.now().isoformat()
            }
            data = json.dumps(message) + "\n"
            self.sock.sendall(data.encode('utf-8'))
            print(f"[{datetime.datetime.now().isoformat()}] Text sent successfully")
            return True
        except Exception as e:
            print(f"Error sending text: {e}")
            self.connected = False
            return False

    def send_combined(self, text, image_data, source_lang, target_lang):
        """
        Sends a combined text and image data message to the server.

        This method constructs a message containing both text and
        image data, along with the specified source and target
        languages, and sends it to the server over an established
        socket connection. In case the connection is not established
        or an error occurs during the sending process, it returns
        False.

        Args:
            text (str): Text to be sent as part of the message.
            image_data (str): Base64 encoded image data to be sent.
            source_lang (str): The source language code.
            target_lang (str): The target language code.

        Returns:
            bool: True if the data is sent successfully, otherwise False.
        """

        if not self.connected:
            return False

        try:
            message = {
                "type": "text_and_image",
                "text": text if text else "",
                "image": image_data,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "timestamp": datetime.datetime.now().isoformat()
            }
            data = json.dumps(message) + "\n"
            self.sock.sendall(data.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Error sending combined data: {e}")
            return False



    def receive_response(self, timeout=10):
        """
        Receives a response from the server.

        This method waits for a response from the server within the given
        timeout period, and returns the received data as a string. If no
        response is received within the timeout period, it returns None. If
        an error occurs during the receiving process, it prints the error
        and returns None.

        Args:
            timeout (int): The timeout period in seconds.

        Returns:
            str: The received data as a string, or None if no response is
            received or an error occurs.
        """
        self.sock.setblocking(0)
        ready = select.select([self.sock], [], [], timeout)
        if ready[0]:
            try:
                data = self.sock.recv(4096).decode('utf-8')
                if data:
                    try:
                        message = json.loads(data)
                        return message.get("data")
                    except json.JSONDecodeError:
                        print("Received invalid JSON.")
                        return None
                else:
                    print("No data received.")
                    return None
            except socket.error as e:
                print(f"Socket error: {e}")
                return None
        else:
            print("No response received within timeout period.")
            return None

    def close(self):
        """
        Closes the socket and sets the connected flag to False.

        This method is a safe way to close the socket, as it ignores any
        exceptions that may occur during the closing process. It also
        sets the connected flag to False, the file object to None, and
        the socket to None.

        :return: None
        """
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.connected = False
        self.file_obj = None
        self.sock = None

# Initialize socket manager as a global instance
socket_manager = SocketManager()

# Modified send_and_receive function
def send_and_receive(data_type, source_lang, target_lang, image_data=None, text=None):
    """
    Sends a request to the server and receives the response.

    This function sends a request to the server using the given data type,
    source language, target language, and optional image data or text. It
    then receives the response from the server and prints it. If a
    communication error occurs, it closes the connection and returns None.

    Args:
        data_type (str): The type of data to be sent, either "image", "text",
            or "text_and_image".
        source_lang (str): The source language of the text to be translated.
        target_lang (str): The target language of the text to be translated.
        image_data (str): The base64 encoded image data, or None if not provided.
        text (str): The text to be translated, or None if not provided.

    Returns:
        str: The response from the server, or None if a communication error
        occurs.
    """
    if not socket_manager.connected:
        if not socket_manager.connect():
            return None

    try:
        success = False
        if data_type == "image":
            success = socket_manager.send_image(image_data, source_lang, target_lang)
        elif data_type == "text":
            success = socket_manager.send_text(text, source_lang, target_lang)
        elif data_type == "text_and_image":
            success = socket_manager.send_combined(text or "", image_data, source_lang, target_lang)

        if not success:
            return None
        response = socket_manager.receive_response()
        print(response)
        return response
    except Exception as e:
        print(f"Communication error: {e}")
        socket_manager.close()
        return None

def translate_model(source_language="en", target_language="en"):
    """
    Create a translation model that translates from source_language to target_language.

    Args:
        source_language (str): The language to translate from. Defaults to "en".
        target_language (str): The language to translate to. Defaults to "en".

    Returns:
        HuggingFace pipeline object for translation.
    """
    model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language}'
    pipe = pipeline("translation", model=model_name)
    return pipe


def check_wifi_connection():
    """
    Check if the device is connected to the internet via WiFi.

    This function attempts to connect to www.google.com on port 80 with a timeout of 2
    seconds. If the connection is successful, the function returns True. If the
    connection times out or an OSError is raised, the function returns False.

    Returns
    -------
    bool
        True if the device is connected to the internet via WiFi, False otherwise.
    """
    try:
        sock = socket.create_connection(("www.google.com", 80), timeout=2)
        sock.close()
        return True
    except (socket.error, OSError):
        return False


def capture_image_from_camera(camera_widget, model):
    """
    Captures an image using the camera and saves it to a file.

    This method uses voice commands to capture an image. It will
    prompt the user to say 'capture' to take a picture. If the
    command is recognized successfully, it will save the image
    to a file named 'captured_image.jpg' and return the path to
    the saved image.

    If the command is not recognized or the image is not saved
    successfully, it will prompt the user to try again, up to
    a maximum of 3 attempts. After the maximum attempts is
    reached, it will return None.

    Parameters:
    camera_widget (CameraWidget): The instance of CameraWidget
                                   to use for capturing the image.
    model (vosk.Model): The Vosk model to use for speech recognition.

    Returns:
    str or None: The path to the saved image if successful, or None
                if the image is not saved successfully.
    """
    try:
        print("Say 'take' to take a picture.")
        filename = "captured_image.jpg"
        # max_attempts = 3
        attempts = 0
        audio_command = ""

        while "take" not in audio_command.strip().lower() or "get out" not in audio_command.strip().lower():
            speak("Please say 'take' to take a picture")
            print("Please say 'take' to take a picture")
            audio_command = get_audio(model)
            if audio_command and "take" in audio_command.strip().lower():
                saved_path = camera_widget.capture_image(filename)
                if saved_path and os.path.exists(saved_path):
                    return saved_path
                else:
                    speak("Failed to capture image, please try again")
                    attempts += 1
                    print(f"Attempt No.: {attempts}")
            else:
                attempts += 1
                print(f"Attempt No.: {attempts}")

        speak("Maximum attempts reached")
        return None

    except Exception as e:
        print(f"Camera error: {e}")
        speak("Camera error occurred")
        return None


def recognize_text(image_path, lang="eng"):
    """
    Recognize text in an image using Tesseract OCR.

    Args:
        image_path (str): The path to the image file.
        lang (str, optional): The language of the text in the image. Defaults to "eng".

    Returns:
        str: The recognized text if successful, otherwise an error message if an exception occurs.
    """
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        _, thresh = cv2.threshold(denoised, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        processed = cv2.dilate(thresh, np.ones((1, 1), np.uint8), iterations=1)
        # Before OCR processing in recognize_text()
        processed_path = os.path.splitext(image_path)[0] + "_processed.jpg"
        cv2.imwrite(processed_path, processed)
        print(f"Saved preprocessed image to: {processed_path}")
        custom_config = r"--oem 3 --psm 11"
        raw_text = pytesseract.image_to_string(processed, lang=lang, config=custom_config)
        return raw_text.strip() if raw_text else ""

    except Exception as e:
        print(f"OCR Erorr: {e}")
        return ""


def filepath(filename):
    """
    Returns the absolute path of the given filename.

    Args:
        filename (str): The name of the file for which the absolute path is required.

    Returns:'
        str: The absolute path of the file if successful, otherwise None if the filename is invalid or an error occurs.
    """

    if not filename:
        return None
    try:
        absolute_path = os.path.abspath(filename)
        return absolute_path
    except Exception as e:
        print(f"Path error: {e}")
        speak(f"Path error: {e}")
        return None



def translate_en():
    """
    Initializes and returns a MarianMTModel and a MarianTokenizer
    for multilingual to English translation.

    This function loads the "Helsinki-NLP/opus-mt-mul-en" model, which is
    capable of translating from multiple languages into English. It returns
    both the model and tokenizer instances needed for performing translations.

    Returns:
        tuple: A tuple containing the MarianMTModel and MarianTokenizer
        instances.
    """

    model_name = "Helsinki-NLP/opus-mt-mul-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer


def translate_to_english(text, model, tokenizer):
    """
    Translates the given text into English using the specified model and tokenizer.

    Args:
        text (str): The text to be translated.
        model (transformers.PreTrainedModel): The model used for translation.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to process the text.

    Returns:
        str: The translated text in English.
    """

    inputs = tokenizer(text, return_tensors="pt",
                       padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

def translate_text(text, pipe):
    """
    Translate text using the provided pipeline.

    Args:
        text (str): The text to translate.
        pipe (transformers.TranslationPipeline): The translation pipeline.

    Returns:
        str or None: The translated text if successful, otherwise None.
    """
    try:
        translated_text = pipe(text)[0]['translation_text']
        return translated_text
    except Exception:
        print("You donnot have this model please check your network connection or i cannot recognize the text ")
        return None


def get_lang1(model):
    """
    Prompts the user to specify the source language for translation.

    The function asks the user to specify the source language by speaking
    either "English", "Arabic", or "French". It uses fuzzy matching to interpret
    the user's response and returns the corresponding language code ("en", "ar",
    "fr") if a match is found with a confidence score above 70. If no match is
    identified after 3 attempts, it returns None.

    Args:
        model: The speech recognition model used to capture user input.

    Returns:
        str or None: The language code for the source language if a match is
        found, otherwise None.
    """

    speak("What is the source language? (English or Arabic or French)")
    print("What is the source language? (English/Arabic/French)")
    for _ in range(5):  # Max 5 attempts
        x = get_audio(model)
        if not x:
            continue
        # Fuzzy match
        scores = {
            "en": fuzz.partial_ratio(x.lower(), "english"),
            "ar": fuzz.partial_ratio(x.lower(), "arabic"),
            "fr": fuzz.partial_ratio(x.lower(), "french")
        }
        if max(scores.values()) > 70:
            return max(scores, key=scores.get)
    return None


def get_lang2(model):
    """
    Ask the user for the target language and return the language code.
    Fuzzy matches the user's response with "english", "arabic", or "french".
    Returns None if no match is found after 3 attempts.
    """
    speak("What is the target language? (English or Arabic or French)")
    print("What is the target language? (English or Arabic or French)")
    for _ in range(5):  # Max 5 attempts
            x = get_audio(model)
            if not x:
                continue
            # Fuzzy match
            scores = {
                "en": fuzz.partial_ratio(x.lower(), "english"),
                "ar": fuzz.partial_ratio(x.lower(), "arabic"),
                "fr": fuzz.partial_ratio(x.lower(), "french")
            }
            if max(scores.values()) > 70:
                return max(scores, key=scores.get)
    return None


def speak(text):
    """
    Speak the given text using the flite text-to-speech engine.

    Args:
        text (str): The text to be spoken.

    Returns:
        None
    """
    clean_text = " ".join(str(text).splitlines()).strip()
    clean_text = clean_text.replace('"', '').replace("'", "")
    tts_thread = TTSThread(clean_text)

    def cleanup():
        tts_threads.remove(tts_thread)

    tts_thread.finished.connect(cleanup)
    tts_threads.append(tts_thread)
    tts_thread.start()

def recognition_model(l1):
    """
    Returns the file path to the Vosk speech recognition model based on the provided language code.

    Args:
        l1 (str): Language code ('ar' for Arabic, 'en' for English, 'fr' for French).

    Returns:
        str: The file path to the corresponding Vosk model for the given language.
    """

    if (l1 == "ar"):
        model_path = r"/home/pi/Desktop/gradproj/vosk-model-ar-mgb2-0.4"
    elif (l1 == "en"):
        model_path = r"/home/pi/Desktop/gradproj/vosk-model-small-en-us-0.15"
    elif (l1 == "fr"):
        model_path = r"/home/pi/Desktop/gradproj/vosk-model-small-fr-0.22"
    return model_path

def get_audio(model, period=10):
    """
    Records audio from the default microphone and attempts to recognize spoken text.
    Returns the recognized text, or None if no audio is detected.
    """
    with audio_lock:
        recognizer = KaldiRecognizer(model, 16000)
        recognized_text = ""
        start_time = time.time()

        def callback(indata, frames, time, status):
            nonlocal recognized_text
            if recognizer.AcceptWaveform(indata.tobytes()):
                result = json.loads(recognizer.Result())
                recognized_text = result.get("text", "")
                print(f"You said: {recognized_text}")

        print("Listening...")
        with sd.InputStream(callback=callback, channels=1, samplerate=16000, dtype=np.int16):
            while (time.time() - start_time) < period:
                if recognized_text:
                    return recognized_text
                sd.sleep(100)
        recognizer.Reset()
        return recognized_text


def tool_detection(model):
    """
    Determines the input type for translation based on user speech input.

    Continuously prompts the user to specify if the input will be 'speech' or 'image'.
    Uses voice commands or numbers ('one' or 'two') to detect the user's choice.
    Recursively calls itself until valid input is recognized.

    Parameters:
    model (vosk.Model): The voice recognition model used to capture audio input.

    Returns:
    str: Returns "speech" if the user selects speech input, "image" if the user selects image input.
    """

    speak("Speech or image input?")
    print("Speech or image input?")
    max_attempts = 3
    attempts = 0
    while attempts < max_attempts:
        t = get_audio(model)
        if t:
            if "speech" in t.lower() or "one" in t.lower():
                return "speech"
            elif "image" in t.lower() or "two" in t.lower():
                return "image"
            attempts += 1
    return None  # Let GUI handle invalid input

def img_lang_det(l1):
    """
    Convert a language abbreviation (e.g. "en", "fr", "ar") to a 3-letter language code (e.g. "eng", "fra", "ara").

    Parameters
    ----------
    l1 : str
        The language abbreviation to be converted.

    Returns
    -------
    str
        The converted 3-letter language code.
    """
    if l1 == "ar":
        lang = "ara"
    elif l1 == "en":
        lang = "eng"
    elif l1 == "fr":
        lang = "fra"
    return lang


def initialize_audio_output(card_index):
    """
    Set the default audio output device.

    Args:
        card_index (int): The index of the audio device to set as default.
    """
    os.environ["PULSE_SINK"] = str(card_index)



def should_stop_translation(text, model, tokenizer):
    """
    Check if the given text is a translation stop phrase.

    Args:
        text (str): The text to be translated.
        model (MarianMTModel): The translation model used.
        tokenizer (MarianTokenizer): The tokenizer used for translation.

    Returns:
        bool: True if the text is a translation stop phrase, False otherwise.
    """
    stop_phrases = {"stop", "exit", "quit", "end"}
    english_text = translate_to_english(text, model, tokenizer).lower()
    return any(phrase in english_text for phrase in stop_phrases)

def handle_online_translation(text, signals, l1, l2):
    """
    Handle online translation flow.

    This method sends the given text to the translation server and displays the
    translated text. If the translation service is unavailable, it shows an
    appropriate error message.

    :param text: The text to be translated.
    :param signals: The communication signals object.
    :param l1: The source language.
    :param l2: The target language.
    :return: None
    """
    signals.update_ai_speech.emit("Processing online translation...")
    response = send_and_receive("text", l1, l2, text=text)

    if response:
        # Show translation for 2 seconds before clearing
        signals.update_ai_speech.emit(response)
        QTimer.singleShot(2000, lambda: signals.update_ai_speech.emit("Listening"))
        speak(response)
    else:
        signals.update_ai_speech.emit("Translation service unavailable")
        speak("Online translation failed")
        print("Online translation failed")


def handle_offline_translation(text, pipeline, signals):
    """
    Handle offline translation flow.

    This method translates the given text using the offline translation pipeline and displays the
    translated text. If the translation pipeline is unavailable, it shows an appropriate error
    message.

    :param text: The text to be translated.
    :param pipeline: The offline translation pipeline.
    :param signals: The communication signals object.
    :return: None
    """
    signals.update_ai_speech.emit("Processing offline translation...")
    if not pipeline:
        signals.update_ai_speech.emit("Offline model unavailable")
        speak("Translation resources missing")
        print("Translation resources missing")
        return

    translated_text = translate_text(text, pipeline)
    if translated_text:
        # Show translation for 2 seconds before clearing
        signals.update_ai_speech.emit(translated_text)
        QTimer.singleShot(2000, lambda: signals.update_ai_speech.emit("Listening"))
        signals.update_output.emit(f"Translated: {translated_text}")
        speak(translated_text)
        print(translated_text)
    else:
        signals.update_ai_speech.emit("Offline translation failed")
        speak("Could not translate text")
        print("Could not translate text")


def process_speech(model, l1, l2, signals):
    """
    Handles speech-to-speech translation with consistent application style.
    Features:
    - Retry mechanism for audio capture
    - Unified online/offline handling
    - Signal-based status updates
    - Clean exit conditions
    """
    online = check_wifi_connection()
    max_attempts = 3
    attempts = 0
    translation_pipe = None

    try:
        # Initialize translation resources
        if not online:
            translation_pipe = get_translation_model(l1, l2)
        translating_model, tokenizer = translate_en()

        signals.update_ai_speech.emit("Ready for speech input")

        while attempts < max_attempts:
            speak("Please say the words you want to translate")
            print("Please say the words you want to translate")

            # Get audio input with timeout
            text = get_audio(model, 15)
            if not text:
                attempts += 1
                signals.update_ai_speech.emit(f"Retrying ({attempts}/{max_attempts})")
                continue

            # Check for stop command
            if should_stop_translation(text, translating_model, tokenizer):
                signals.update_ai_speech.emit("Ending translation session")
                speak("I will be happy to help you. Goodbye.")
                print("I will be happy to help you. Goodbye.")
                break

            # Process translation
            if online:
                handle_online_translation(text, signals, l1, l2)
            else:
                handle_offline_translation(text, translation_pipe, signals)

            attempts = 0  # Reset attempts on success

    except Exception as e:
        signals.update_ai_speech.emit(f"Error: {str(e)}")
        speak("A processing error occurred")
        print("A processing error occurred")
    finally:
        signals.update_ai_speech.emit("Speech processing completed")


def handle_online_image(camera_widget, model, signals, l1, l2):
    """
    Handle online image translation flow.

    This method sends the given image to the translation server and displays the
    translated text. If the translation service is unavailable, it shows an
    appropriate error message.

    :param camera_widget: The camera widget used to capture images.
    :param model: The Vosk model used for speech recognition.
    :param signals: The communication signals object.
    :param l1: The source language.
    :param l2: The target language.
    :return: None
    """
    signals.update_ai_speech.emit("Starting online processing")

    # Get user confirmation
    if not get_confirmation("online mode", model, signals):
        return False

    # Capture image
    signals.update_ai_speech.emit("Capturing image...")
    filename = capture_with_retry(camera_widget, model, signals)
    if not filename:
        return False

    # Get optional prompt
    prompt = get_optional_prompt(model, signals) or ""

    # Prepare and send data
    with open(filename, "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    signals.update_ai_speech.emit("Sending to server...")
    response = send_and_receive(
        "text_and_image",
        l1,
        l2,
        text=prompt.strip(),
        image_data=image_data
    )

    # Handle response
    if response:
        signals.update_ai_speech.emit(response)
        signals.update_output.emit(f"Server response: {response}")
        speak(response)
        # print(response)
        return True

    signals.update_ai_speech.emit("No server response")
    return False

def handle_offline_image(camera_widget, l1, l2, pipeline, signals):
    """
    Handle offline image translation flow.

    This method captures an image, recognizes text within it, and translates it
    using the offline translation pipeline. If the translation pipeline is unavailable,
    it shows an appropriate error message.

    :param camera_widget: The camera widget used to capture images.
    :param l1: The source language.
    :param l2: The target language.
    :param pipeline: The offline translation pipeline.
    :param signals: The communication signals object.
    :return: True if translation was successful, False otherwise.
    """
    signals.update_ai_speech.emit("Starting offline processing")

    # Capture image
    filename = capture_with_retry(camera_widget, None, signals)
    if not filename:
        return False

    # OCR processing
    signals.update_ai_speech.emit("Recognizing text...")
    lang = img_lang_det(l1)
    text = recognize_text(filename, lang)

    if not text:
        signals.update_ai_speech.emit("No text recognized")
        speak("No text found in image")
        print("No text found in image")
        return False

    # Translation
    signals.update_ai_speech.emit("Translating...")
    translated = translate_text(text, pipeline)

    if translated:
        signals.update_ai_speech.emit(translated)
        signals.update_output.emit(f"Translated: {translated}")
        speak(translated)
        print(translated)
        return True

    signals.update_ai_speech.emit("Translation failed")
    return False

def capture_with_retry(camera_widget, model, signals, max_attempts=3):
    """
    Captures an image from the camera widget with retry mechanism.

    This method captures an image from the camera widget and saves it to a file.
    If the capture is successful, the filename is returned; otherwise, it retries
    up to a maximum of 'max_attempts' with a 1-second delay between attempts.
    If all attempts fail, it returns None.

    :param camera_widget: The camera widget used to capture images.
    :param model: The Vosk model to use for speech recognition (not used).
    :param signals: The communication signals object.
    :param max_attempts: The maximum number of attempts (default: 3).
    :return: The filename if the capture is successful, or None otherwise.
    """
    for attempt in range(1, max_attempts+1):
        signals.update_ai_speech.emit(f"Capture attempt {attempt}/{max_attempts}")
        filename = camera_widget.capture_image()
        if filename and os.path.exists(filename):
            return filename
        speak("Capture failed, please try again")
        print("Capture failed, please try again")
    return None

def get_optional_prompt(model, signals):
    """
    Asks the user if they want to add a prompt, and if so, records audio and returns the text.
    If the user declines or the recording fails, an empty string is returned.

    :param model: The Vosk model to use for speech recognition.
    :param signals: The communication signals object.
    :return: The user's prompt if given, or an empty string otherwise.
    """
    signals.update_ai_speech.emit("Prompt check")
    if get_confirmation("add a prompt", model, signals):
        signals.update_ai_speech.emit("Listening for prompt...")
        response = get_audio(model, 15)
        return response.strip() if response else ""
    return ""

def get_confirmation(prompt, model, signals, max_attempts=3):
    """
    Asks the user for a confirmation with a prompt, and returns a boolean accordingly.

    This method asks the user for a confirmation by speaking a prompt and waiting
    for a response. It will retry up to 'max_attempts' times if the user does not
    respond with either "yes" or "no". If the user confirms, it returns True;
    otherwise, it returns False.

    :param prompt: The prompt to ask the user.
    :param model: The Vosk model to use for speech recognition.
    :param signals: The communication signals object.
    :param max_attempts: The maximum number of attempts (default: 3).
    :return: A boolean indicating whether the user confirmed (True) or not (False).
    """
    for _ in range(max_attempts):
        speak(f"Do you want to {prompt}?")
        print(f"Do you want to {prompt}?")
        response = get_audio(model)
        if "yes" in response.lower():
            return True
        if "no" in response.lower():
            return False
        speak("Please say yes or no")
        print("Please say yes or no")
    return False

def ask_continue(signals, model, max_attempts=2):
    """
    Asks the user if they want to process another image and returns a boolean.

    This function prompts the user to decide if they would like to process
    another image. It uses speech recognition to interpret the user's response
    and returns True if the user responds with "yes", and False if they respond
    with "no". The function will retry prompting the user up to 'max_attempts'
    times if neither "yes" nor "no" is detected.

    :param signals: The communication signals object.
    :param model: The Vosk model to use for speech recognition.
    :param max_attempts: The maximum number of attempts (default: 2).
    :return: A boolean indicating whether the user wants to continue (True) or not (False).
    """

    for _ in range(max_attempts):
        speak("Would you like to process another image?")
        print("Would you like to process another image?")
        response = get_audio(model)
        if "yes" in response.lower():
            return True
        if "no" in response.lower():
            return False
    return False

def process_image(camera_widget, model, l1, l2, signals):
    """
    Handles image translation workflow with consistent application style.
    Features:
    - Unified online/offline handling
    - Signal-based progress updates
    - Retry mechanisms
    - Clean resource management
    """
    online = check_wifi_connection()
    max_attempts = 3
    attempts = 0
    translation_pipe = None

    try:
        # Initialize translation resources
        if not online:
            translation_pipe = get_translation_model(l1, l2)
            signals.update_ai_speech.emit("Initialized offline translation")

        signals.update_ai_speech.emit("Starting image processing")

        while attempts < max_attempts:
            if online:
                result = handle_online_image(camera_widget, model, signals, l1, l2)
            else:
                result = handle_offline_image(camera_widget, l1, l2, translation_pipe, signals)

            if result:
                attempts = 0  # Reset on success
                if not ask_continue(signals, model):
                    break
            else:
                attempts += 1
                signals.update_ai_speech.emit(f"Attempt {attempts}/{max_attempts}")

        if attempts >= max_attempts:
            signals.update_ai_speech.emit("Max attempts reached")
            speak("Maximum processing attempts reached")
            print("Maximum processing attempts reached")

    except Exception as e:
        signals.update_ai_speech.emit(f"Error: {str(e)}")
        speak("An image processing error occurred")
        print("An image processing error occurred")
    finally:
        signals.update_ai_speech.emit("Image processing completed")
