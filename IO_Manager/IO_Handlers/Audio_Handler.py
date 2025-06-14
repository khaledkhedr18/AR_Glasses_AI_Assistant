from PyQt5.QtCore import QThread, pyqtSignal
import subprocess
import sounddevice as sd
import numpy as np
import time
import json
import threading
from vosk import KaldiRecognizer
from utils.logging import Logger
from utils.WorkerThread import create_worker

class AudioHandler:
    """
    Handles audio I/O operations including speech recognition and text-to-speech
    in a framework-agnostic manner.
    """

    def __init__(self):
        """Initialize the AudioHandler."""
        self.logger = Logger()
        self.active_tasks = []
        self.audio_lock = threading.Lock()
        self.listening = False

    def start_recording(self, model, duration=10, on_result=None):
        """
        Start recording audio for speech recognition.

        Args:
            model: The Vosk speech recognition model to use
            duration (int): Maximum recording time in seconds
            on_result: Callback for when speech is recognized

        Returns:
            WorkerThread: The worker thread handling the recording
        """
        def audio_task():
            return self._record_audio(model, duration)

        worker = create_worker(
            audio_task,
            on_result=on_result,
            on_finished=lambda: self._remove_task(worker),
            task_name="audio_recording"
        )

        self.active_tasks.append(worker)
        worker.start()
        return worker

    def _record_audio(self, model, period=10):
        """Records audio and attempts to recognize spoken text."""
        with self.audio_lock:
            try:
                self.listening = True
                recognizer = KaldiRecognizer(model, 16000)
                recognized_text = ""
                start_time = time.time()

                def callback(indata, frames, time, status):
                    nonlocal recognized_text
                    if status:
                        self.logger.warning(f"Audio input error: {status}")

                    # Process audio data
                    if recognizer.AcceptWaveform(indata.tobytes()):
                        result = json.loads(recognizer.Result())
                        text = result.get("text", "")
                        if text.strip():  # Only update if we got non-empty text
                            recognized_text = text
                            self.logger.info(f"Speech recognized: {recognized_text}")

                self.logger.info("Listening for speech...")
                with sd.InputStream(callback=callback, channels=1, samplerate=16000,
                                    dtype=np.int16, blocksize=8000):
                    # Wait until we either get text or timeout
                    while (time.time() - start_time) < period:
                        if recognized_text:
                            self.listening = False
                            return recognized_text
                        sd.sleep(100)

                # Get any partial results
                final_result = json.loads(recognizer.FinalResult())
                final_text = final_result.get("text", "")
                if final_text and not recognized_text:
                    recognized_text = final_text

                return recognized_text

            except Exception as e:
                self.logger.error(f"Error recording audio: {e}")
                return ""
            finally:
                self.listening = False
                if 'recognizer' in locals():
                    recognizer.Reset()

    def stop_recording(self):
        """Stop any ongoing recording sessions."""
        if self.listening:
            for task in list(self.active_tasks):
                if task.task_name == "audio_recording":
                    task.stop()
            self.listening = False
            return True
        return False

    def output_speech(self, text, on_finished=None):
        """
        Convert text to speech and play it through the system's audio output.

        Args:
            text (str): The text to be spoken
            on_finished: Optional callback when speech completes

        Returns:
            WorkerThread: The worker thread handling the TTS
        """
        def speak_task():
            clean_text = " ".join(str(text).splitlines()).strip()
            clean_text = clean_text.replace('"', '').replace("'", "")
            subprocess.run(["flite", clean_text])
            return True

        worker = create_worker(
            speak_task,
            on_finished=lambda: self._handle_speech_finished(worker, on_finished),
            task_name="text_to_speech"
        )

        self.active_tasks.append(worker)
        worker.start()
        return worker

    def _handle_speech_finished(self, worker, callback=None):
        """Handle completion of a speech task"""
        self._remove_task(worker)
        if callback:
            callback()

    def _remove_task(self, task):
        """Remove a task from the active tasks list"""
        if task in self.active_tasks:
            self.active_tasks.remove(task)

    def mute_speech(self):
        """Stop any ongoing speech output."""
        stopped_count = 0
        for task in list(self.active_tasks):
            if task.task_name == "text_to_speech":
                task.stop()
                # Additionally kill any running flite processes
                try:
                    subprocess.run(["pkill", "-f", "flite"], stderr=subprocess.DEVNULL)
                    stopped_count += 1
                except Exception as e:
                    self.logger.error(f"Error stopping speech: {e}")

        return stopped_count

    def is_listening(self):
        """Check if the handler is currently listening for audio input."""
        return self.listening

    def get_user_audio(self, model, duration=10):
        """Synchronously record and recognize user speech."""
        return self._record_audio(model, duration)

    def cleanup(self):
        """Clean up resources used by the AudioHandler."""
        # Stop all active tasks
        for task in list(self.active_tasks):
            task.stop()

        # Kill any lingering flite processes
        try:
            subprocess.run(["pkill", "-f", "flite"], stderr=subprocess.DEVNULL)
        except:
            pass

        self.active_tasks.clear()
