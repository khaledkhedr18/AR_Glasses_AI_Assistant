
from Managers.LLM_Manager import LLM_Manager
from configrations import config
import requests
import os
import mimetypes


class GpmRequestHandler(LLM_Manager):
    def __init__(self, purpose, prompt, image_path=config.IMAGE_PATH):
        super().__init__(purpose=purpose)
        self.prompt = prompt 
        self.image_path = image_path 
        self.__url=config.API_CHAT_ENDPOINT


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
            response = requests.post(url=self.__url, data=data, files=files if files else None, timeout=20) # Increased timeout for potentially larger uploads/processing
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
