import subprocess
import sounddevice as sd
import numpy as np
import threading
import os
import scipy.io.wavfile as wavfile
import json
import time
from datetime import datetime
from utils.Logging import Logger
from utils.WorkerThread import create_worker
from vosk import KaldiRecognizer

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

        # Audio recording properties
        self.sample_rate = 16000  # 16kHz
        self.channels = 1  # Mono
        self.audio_chunks = []
        self.recording_thread = None
        self.stop_recording_event = None
        self.recording_mode = "offline"
        self.audio_save_dir = os.path.join(os.path.expanduser('~'), 'ar_glasses_audio')

        # Create audio save directory if it doesn't exist
        if not os.path.exists(self.audio_save_dir):
            os.makedirs(self.audio_save_dir)

    def record_and_recognize(self, model, max_duration=10):
        """
        Records audio and performs real-time speech recognition using VOSK model.

        Args:
            model: The Vosk model to use for recognition
            max_duration (int): Maximum recording duration in seconds

        Returns:
            str: The recognized text, or None if no text was recognized
        """
        with self.audio_lock:
            if self.listening:
                self.logger.warning("Recording already in progress")
                return None

            self.listening = True
            self.logger.info(f"Starting real-time speech recognition (max {max_duration}s)...")

            # Initialize recognizer with the speech model
            recognizer = KaldiRecognizer(model, self.sample_rate)
            recognized_text = ""
            stop_event = threading.Event()

            def audio_callback(indata, frames, time_info, status):
                if status:
                    self.logger.warning(f"Audio input error: {status}")

                # Process audio in real-time with VOSK
                if recognizer.AcceptWaveform(indata.tobytes()):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        nonlocal recognized_text
                        recognized_text += text + " "
                        self.logger.debug(f"Partial recognition: {text}")

            try:
                start_time = time.time()

                # Start streaming with callback
                with sd.InputStream(
                    callback=audio_callback,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    dtype=np.int16,
                    blocksize=4000
                ):
                    # Continue until timeout or external stop
                    while not stop_event.is_set() and (time.time() - start_time) < max_duration:
                        sd.sleep(100)  # Sleep to reduce CPU usage

                # Get final result from recognizer
                final_result = json.loads(recognizer.FinalResult())
                final_text = final_result.get("text", "")
                if final_text:
                    recognized_text += final_text

                recognized_text = recognized_text.strip()

                if recognized_text:
                    self.logger.info(f"Recognized text: '{recognized_text}'")
                    return recognized_text
                else:
                    self.logger.info("No speech recognized")
                    return None

            except Exception as e:
                self.logger.error(f"Error during speech recognition: {e}")
                return None
            finally:
                self.listening = False
                recognizer = None  # Help garbage collection

    def start_recording(self, mode="offline"):
        """
        Start recording audio without processing.

        Args:
            mode (str): "online" to save to file, "offline" to keep in memory

        Returns:
            bool: True if recording started successfully
        """
        with self.audio_lock:
            if self.listening:
                self.logger.warning("Recording already in progress - skipping start")
                return False

            self.logger.info(f"Starting audio recording in {mode} mode")
            self.audio_chunks = []  # Reset chunks
            self.stop_recording_event = threading.Event()
            self.listening = True
            self.recording_mode = mode

            # Start recording in a separate thread
            self.recording_thread = threading.Thread(
                target=self._record_audio_raw,
                args=(self.stop_recording_event,)
            )
            self.recording_thread.daemon = True
            self.recording_thread.start()

            return True

    def stop_recording(self):
        """
        Stop recording and return the recorded audio based on mode.

        Returns:
            For offline mode: numpy array of audio data
            For online mode: Path to saved audio file
            None if recording failed
        """
        with self.audio_lock:
            if not self.listening:
                self.logger.warning("No recording in progress - skipping stop")
                return None

            self.logger.info("Stopping audio recording")

            # Signal recording to stop and wait for thread to finish
            if self.stop_recording_event:
                self.stop_recording_event.set()

            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=2.0)

            # Process the recorded audio
            if not self.audio_chunks:
                self.logger.warning("No audio data recorded")
                self.listening = False  # Ensure we reset state even if no data
                return None

            try:
                # Combine all audio chunks
                audio_data = np.concatenate(self.audio_chunks)

                # Handle based on recording mode
                if self.recording_mode == "online":
                    # Save to file for online mode
                    file_path = self._save_audio_to_file(audio_data)
                    self.logger.info(f"Audio saved to: {file_path}")
                    result = file_path
                else:
                    # Return raw data for offline mode
                    self.logger.info(f"Returning raw audio data ({len(audio_data)} samples)")
                    result = audio_data

                return result

            except Exception as e:
                self.logger.error(f"Error processing recorded audio: {str(e)}")
                return None
            finally:
                # Reset recording state
                self.listening = False
                self.stop_recording_event = None
                self.recording_thread = None

    def get_audio(self, file_path):
        """
        Read audio file and return the audio data.

        Args:
            file_path (str): Path to the audio file

        Returns:
            numpy.ndarray: Audio data from the file
        """
        if not os.path.exists(file_path):
            self.logger.error(f"Audio file not found: {file_path}")
            return None

        try:
            # Handle different file types
            if file_path.endswith('.mp3'):
                # Convert MP3 to WAV using ffmpeg
                temp_wav = os.path.join(self.audio_save_dir, "temp_convert.wav")
                subprocess.run([
                    "ffmpeg", "-y", "-i", file_path, temp_wav
                ], stderr=subprocess.DEVNULL)

                # Read the WAV file
                sample_rate, audio_data = wavfile.read(temp_wav)
                os.remove(temp_wav)
                return audio_data
            elif file_path.endswith('.wav'):
                sample_rate, audio_data = wavfile.read(file_path)
                return audio_data
            else:
                self.logger.error(f"Unsupported audio format: {file_path}")
                return None

        except Exception as e:
            self.logger.error(f"Error reading audio file: {e}")
            return None

    def is_listening(self):
        """Check if the handler is currently recording audio."""
        return self.listening

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
            try:
                clean_text = " ".join(str(text).splitlines()).strip()
                clean_text = clean_text.replace('"', '').replace("'", "")

                # Direct output to default audio device instead of stdout
                self.logger.info(f"Speaking: {clean_text[:50]}...")
                subprocess.run(["flite", "-t", clean_text], check=True)
                return True
            except Exception as e:
                self.logger.error(f"Speech output error: {str(e)}")
                return False

        worker = create_worker(
            speak_task,
            on_finished=lambda: self._handle_speech_finished(worker, on_finished),
            task_name="text_to_speech"
        )

        self.active_tasks.append(worker)
        worker.start()
        return worker

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

    def cleanup(self):
        """Clean up resources used by the AudioHandler."""
        # Stop recording if active
        if self.listening:
            self.stop_recording()

        # Stop all active tasks
        for task in list(self.active_tasks):
            task.stop()

        # Kill any lingering flite processes
        try:
            subprocess.run(["pkill", "-f", "flite"], stderr=subprocess.DEVNULL)
        except:
            pass

        self.active_tasks.clear()

    def _record_audio_raw(self, stop_event):
        """
        Record raw audio data until stop_event is set.

        Args:
            stop_event (threading.Event): Event to signal recording should stop
        """
        try:
            def audio_callback(indata, frames, time, status):
                if status:
                    self.logger.warning(f"Audio input error: {status}")

                # Store the audio chunk
                self.audio_chunks.append(indata.copy())

            # Start the input stream
            with sd.InputStream(
                callback=audio_callback,
                channels=self.channels,
                samplerate=self.sample_rate,
                dtype=np.int16,
                blocksize=8000
            ):
                self.logger.info("Recording audio...")
                while not stop_event.is_set():
                    sd.sleep(100)  # Sleep to reduce CPU usage

        except Exception as e:
            self.logger.error(f"Error during audio recording: {e}")
        finally:
            self.listening = False

    def _save_audio_to_file(self, audio_data):
        """
        Save audio data to file in WAV and MP3 formats.

        Args:
            audio_data (numpy.ndarray): Audio data to save

        Returns:
            str: Path to the saved MP3 file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wav_path = os.path.join(self.audio_save_dir, f"recording_{timestamp}.wav")
        mp3_path = os.path.join(self.audio_save_dir, f"recording_{timestamp}.mp3")

        # Save as WAV first
        try:
            wavfile.write(wav_path, self.sample_rate, audio_data)

            # Convert to MP3 using ffmpeg
            subprocess.run([
                "ffmpeg", "-y", "-i", wav_path,
                "-acodec", "libmp3lame", "-ab", "128k", mp3_path
            ], stderr=subprocess.DEVNULL)

            # Remove the temporary WAV file
            os.remove(wav_path)
            return mp3_path

        except Exception as e:
            self.logger.error(f"Error saving audio file: {e}")
            return None

    def _handle_speech_finished(self, worker, callback=None):
        """Handle completion of a speech task"""
        self._remove_task(worker)
        if callback:
            callback()

    def _remove_task(self, task):
        """Remove a task from the active tasks list"""
        if task in self.active_tasks:
            self.active_tasks.remove(task)
