from Managers.LLM_Manager import LLM_Manager
from configrations import config
import requests
import os
class AudioTranscriptionRequestHandler(LLM_Manager):
    def __init__(self, purpose=None,audio_path=config.AUDIO_PATH):
        super().__init__(purpose=purpose)
        self.__transcribe_url = config.API_TRANSCRIBE_ENDPOINT
        self.audio_path = audio_path

    def send_audio_to_api(self, audio_path="recording.wav"):
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return {"error": "Audio file not found"}

        try:
            with open(audio_path, 'rb') as audio_file:
                files = {'audio': (os.path.basename(audio_path), audio_file, 'audio/wav')}
                response = requests.post(self.__transcribe_url, files=files, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.Timeout:
            print("❌ API request timed out (send_audio_to_api).")
            return {"error": "API request timed out"}
        except requests.RequestException as e:
            print(f"❌ Failed to send audio to API: {e}")
            return {"error": str(e)}