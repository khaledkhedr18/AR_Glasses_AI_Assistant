import os
import requests
import socket
import mimetypes
from config import API_CHAT_ENDPOINT, API_TRANSCRIBE_ENDPOINT

class APIService:
    def __init__(self):
        pass

    def is_connected(self):
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True
        except OSError:
            return False
        

    def send_audio_to_api(self, audio_path="recording.wav"):
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return {"error": "Audio file not found"}

        try:
            with open(audio_path, 'rb') as audio_file:
                files = {'audio': (os.path.basename(audio_path), audio_file, 'audio/wav')}
                response = requests.post(API_TRANSCRIBE_ENDPOINT, files=files, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            print("❌ API request timed out (send_audio_to_api).")
            return {"error": "API request timed out"}
        except requests.RequestException as e:
            print(f"❌ Failed to send audio to API: {e}")
            return {"error": str(e)}

    def send_chat_to_api(self, prompt, image_path=None):
        data = {'prompt': prompt}
        files = {}

        if image_path and os.path.exists(image_path):
            mime_type, _ = mimetypes.guess_type(image_path)
            try:
                image_file_opened = open(image_path, 'rb')
                files['image'] = (
                    os.path.basename(image_path),
                    image_file_opened,
                    mime_type or 'application/octet-stream'
                )
            except IOError as e:
                print(f"❌ Error opening image file {image_path}: {e}")
                pass


        try:
            response = requests.post(API_CHAT_ENDPOINT, data=data, files=files if files else None, timeout=20) # Increased timeout for potentially larger uploads/processing
            response.raise_for_status()
            result = response.json()
            return result.get('response', '').strip()
        except requests.Timeout:
            print("❌ API request timed out (send_chat_to_api).")
            return "Error: API request timed out. Please try again."
        except requests.RequestException as e:
            print(f"❌ Error sending chat to API: {e}")
            return f'Error: Could not connect to the chat service. {str(e)}'
        finally:
            if 'image' in files and files['image'][1]:
                files['image'][1].close()

if __name__ == '__main__':
    service = APIService()

    # Test connectivity
    if service.is_connected():
        print("🌐 Internet connection available.")
        chat_response_no_image = service.send_chat_to_api("What is the weather like today?")
        print(f"Chat API (no image) Response: {chat_response_no_image}")

    else:
        print("❌ No internet connection.")