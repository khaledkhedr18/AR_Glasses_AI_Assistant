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
        self.speech_queue = []
        self.current_speech_worker = None
        self.speech_active = threading.Event()
        self.speech_thread = None
        self.speech_lock = threading.Lock()
        self.stop_speech_event = threading.Event()

        # Create audio save directory if it doesn't exist
        if not os.path.exists(self.audio_save_dir):
            os.makedirs(self.audio_save_dir)

        self._start_speech_processor()

    def _start_speech_processor(self):
        """Start the dedicated speech processing thread."""
        def speech_processor():
            while not self.stop_speech_event.is_set():
                try:
                    # Wait for speech items
                    with self.speech_lock:
                        if not self.speech_queue:
                            time.sleep(0.1)
                            continue

                        text, on_finished = self.speech_queue.pop(0)

                    # Process the speech item
                    self._speak(text)

                    # Call completion callback if provided
                    if on_finished:
                        try:
                            on_finished()
                        except Exception as e:
                            self.logger.error(f"Speech completion callback error: {e}")

                except Exception as e:
                    self.logger.error(f"Speech processor error: {e}")
                    time.sleep(0.5)

        self.speech_thread = threading.Thread(target=speech_processor, daemon=True)
        self.speech_thread.start()

    def _speak(self, text):
        """Internal method to actually speak text (blocking) with proper file descriptor handling"""
        clean_text = " ".join(str(text).splitlines()).strip()
        clean_text = clean_text.replace('"', '').replace("'", "")

        self.logger.info(f"Speaking: {clean_text[:50]}...")
        self.speech_active.set()

        try:
            # Use Popen with proper file descriptor handling
            self.current_speech_process = subprocess.Popen(
                ["flite", "-t", clean_text],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,  # Add stdin pipe
                close_fds=True  # Ensure proper file descriptor handling
            )

            # Wait for process to complete with timeout
            try:
                stdout, stderr = self.current_speech_process.communicate(timeout=len(clean_text.split())*0.5 + 5)
                if stderr:
                    self.logger.warning(f"Speech output warnings: {stderr.decode('utf-8')}")
            except subprocess.TimeoutExpired:
                self.logger.warning("Speech output timed out - terminating")
                self.current_speech_process.terminate()
                try:
                    self.current_speech_process.communicate(timeout=1)
                except:
                    pass
            except Exception as e:
                self.logger.error(f"Speech communication error: {e}")

        except Exception as e:
            self.logger.error(f"Speech output error: {e}")
        finally:
            self.current_speech_process = None
            self.speech_active.clear()
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
        Queue text for speech output (non-blocking).

        Args:
            text (str): Text to speak
            on_finished (callable): Optional callback when speech completes
        """
        with self.speech_lock:
            self.speech_queue.append((text, on_finished))

    def stop_current_speech(self):
        """Stop any currently playing speech immediately."""
        if self.current_speech_process:
            try:
                self.current_speech_process.terminate()
                self.current_speech_process.communicate(timeout=1)
            except:
                pass
            self.current_speech_process = None
            self.speech_active.clear()

    def clear_speech_queue(self):
        """Clear all pending speech items."""
        with self.speech_lock:
            self.speech_queue.clear()

    def is_speaking(self):
        """Check if speech is currently playing or queued."""
        return self.speech_active.is_set() or (len(self.speech_queue) > 0)

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
        """Clean up all resources."""
        # Stop recording if active
        if self.listening:
            self.stop_recording()

        # Stop speech processing
        self.stop_speech_event.set()
        self.stop_current_speech()
        self.clear_speech_queue()

        # Wait for speech thread to finish
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join(timeout=1)

        # Kill any lingering flite processes
        try:
            subprocess.run(["pkill", "-f", "flite"],
                         stderr=subprocess.DEVNULL,
                         timeout=1)
        except:
            pass

    def clear_queue(self):
        """Clear all pending speech items and stop current speech."""
        with self.speech_lock:
            self.speech_queue.clear()
            if self.current_speech_process:
                try:
                    self.current_speech_process.terminate()
                    self.current_speech_process.communicate(timeout=0.5)
                except:
                    pass
                self.current_speech_process = None
            self.speech_active.clear()

    def is_queue_empty(self):
        """Check if speech queue is empty."""
        with self.speech_lock:
            return len(self.speech_queue) == 0 and not self.speech_active.is_set()

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

        self.current_speech_worker = None
        if callback:
            callback()

    def _remove_task(self, task):
        """Remove a task from the active tasks list"""
        with self.audio_lock:
            if task in self.active_tasks:
                self.active_tasks.remove(task)
