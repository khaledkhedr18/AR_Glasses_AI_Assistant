import os
import time
#import pyttsx3
from vosk import Model, KaldiRecognizer
import json
from transformers import MarianMTModel, MarianTokenizer, pipeline
import cv2
import pytesseract
from PIL import Image
import sounddevice as sd
import numpy as np
from dic import *
import socket
import time
import threading
import tkinter as tk
import subprocess

def connected_func(model, display_output=None, display_output2=None):
    if is_connected():
        display_output2("hello you are connected do you want to use our online assistant or offline assistant")
        speak(f"hello you are connected do you want to use our online assistant or offline assistant")
        x = get_audio(model, display_output, display_output2).strip().lower()
        if "online" in x:
            return True
        elif "offline" in x:
            return False
        else:
            display_output2("i cannot recognize your answer please try again")
            speak("i cannot recognize your answer please try again")
            return connected_func(model, display_output, display_output2)


def is_connected():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False



def capture_image_from_camera(picam2, model, display_output=None, display_output2=None):
    while True:
        print("Listening for capture command")
        if display_output2:
            display_output2("Listening for capture command")
        
        command = get_audio(model, display_output, display_output2)
        
        if command and "capture" in command.lower():
            filename = "captured_image.jpg"
            time.sleep(0.5)
            
            # Capture image using picam2
            picam2.capture_file(filename)
            
            # Load image using OpenCV
            image = cv2.imread(filename)
            
            # Convert image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Optionally save grayscale image (for debug)
            cv2.imwrite("captured_image_grey.jpg", gray_image)
            
            # If your model needs the grayscale image, you can pass gray_image to it here
            # result = model.predict(gray_image)  <-- Example usage

            if display_output2:
                display_output2(f"Image captured and converted to grayscale: {filename}")
            
            return True    

        elif command and "get out" in command.lower():
            print("Exiting capture mode")
            if display_output2:
                display_output2("Exiting capture mode")
            return False
def recognize_text(image_path, lang="eng"):
    try:
        pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img,lang)
        if not text:
            return "No text detected"
            display_output2("No text detected")
        os.replace(image_path, image_path)
        return text
    except Exception as e:
        return f"An error occurred: {e}"
def filepath(filename):
    absolute_path = os.path.abspath(filename)
    return absolute_path
def translate_en():
    model_name = "Helsinki-NLP/opus-mt-mul-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return model, tokenizer
def translate_to_english(text,model,tokenizer):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)
def translate_model(source_language="en",target_language="ar", display_output2=None):
    try:
        model_name = f'Helsinki-NLP/opus-mt-{source_language}-{target_language}'
        pipe = pipeline("translation", model=model_name)
        return pipe
    except:
        display_output2("this model is not available")
        speak ("this model is not available")
    
def translate_text(text,pipe, display_output2=None):
    try:
        translated_text = pipe(text)[0]['translation_text'] 
        return translated_text
    except:
        display_output2("You donnot have this model please check your network connection or i cannot recognize the text or the model is not available")
        
        return None
def get_lang1(model, display_output=None, display_output2=None):
    while True:
        display_output2("What is the language you will be using")
        speak("What is the language you will be using")
        x = get_audio(model, display_output, display_output2).strip().lower()

        normalized_keys = {key.lower(): key for key in vosk_languages}

        if x in normalized_keys:
            original_key = normalized_keys[x]
            l1 = vosk_languages[original_key]
            break
        else:
            display_output2("I cannot recognize the language you're trying to use. Try again, please.")
            speak("I cannot recognize the language you're trying to use. Try again, please.")
    return l1


def get_lang2(model, display_output=None, display_output2=None):
    while True:
        display_output2("What is the language you will be trying to translate to")
        speak("What is the language you will be trying to translate to")
        s = get_audio(model, display_output, display_output2).strip().lower()

        normalized_keys = {key.lower(): key for key in vosk_languages}

        if s in normalized_keys:
            original_key = normalized_keys[s]
            l2 = vosk_languages[original_key]
            break
        else:
            display_output2("I cannot recognize the language you're trying to translate to. Try again, please.")
            speak("I cannot recognize the language you're trying to translate to. Try again, please.")
    return l2

def get_lang3(model, display_output=None, display_output2=None):
    while True:
        display_output2("What is the language you are trying to detect")
        speak("What is the language you are trying to detect")
        l = get_audio(model, display_output, display_output2).strip().lower()

        normalized_keys = {key.lower(): key for key in vosk_languages}

        if l in normalized_keys:
            original_key = normalized_keys[l]
            l3 = vosk_languages[original_key]
            break
        else:
            display_output2("I cannot recognize the language you're trying to detect. Try again, please.")
            speak("I cannot recognize the language you're trying to detect. Try again, please.")
    return l3

def speak(text):
    try:
        subprocess.run(['flite','-t',text],check=True)
    except FileNotFoundError:
            print("x")
    except subprocess.CalledProcessError as e:
            print("y")
            

def recognition_model(l1):
    model_path = vosk_model_paths.get(l1)
    if model_path:
        return model_path
    else:
        raise ValueError(f"No model path defined for language code: {l1}")


def get_audio(model, display_output=None, display_output2=None):
    recognizer = KaldiRecognizer(model, 16000)
    recognized_text = None
    stream = None

    def callback(indata, frames, time, status):
        nonlocal recognized_text
        if status:
            pass
        if recognizer.AcceptWaveform(indata[:, 0].tobytes()):
            result = json.loads(recognizer.Result())
            recognized_text = result.get("text", "")
            print(f"You said: {recognized_text}")
            # Update display if display functions are provided
            if display_output:
                display_output(f"You said: {recognized_text}")

    try:
        print("Listening...")
        if display_output:
            display_output("Listening... Speak now")
        
        with sd.InputStream(callback=callback, channels=1, samplerate=16000, dtype=np.int16) as stream:
            while recognized_text is None:
                sd.sleep(100)  # Sleep in smaller chunks to be more responsive
        
        return recognized_text
        
    finally:
        if stream is not None and not stream.closed:
            stream.close()
        del recognizer



def tool_detection(model, display_output=None, display_output2=None):
    while True:
        r = ""
        display_output2("what is the data you will use for your translation  speech or image ")
        speak("what is the data you will use for your translation  speech or image ")
        t = get_audio(model, display_output, display_output2)
        if "speech" in t:
            r = "speech"
            break
        elif "image" in t:
            r = "image"
            break
        else:
            display_output2("i cannot recognize the data type   please try again ")
            speak("i cannot recognize the data type   please try again ")
    return r
def img_lang_det(l1):
    return vosk_to_tesseract.get(l1, "eng")
def reset(model_path):
    if model_path!=r"/home/pi/Desktop/gradproj/vosk-model-small-en-us-0.15":
        model_path=r"/home/pi/Desktop/gradproj/vosk-model-small-en-us-0.15"
    else:
        pass
    return model_path

def detect_continue(model, display_output,display_output2):
    while True:
        display_output2("Do you want to continue?")
        speak("Do you want to continue?")
        x = get_audio(model, display_output, display_output2).strip().lower()
        if "yes" in x:
            return True
        elif "no" in x:
            return False
        else:
            display_output2("I cannot recognize your answer. Please try again.")
            speak("I cannot recognize your answer. Please try again.")

