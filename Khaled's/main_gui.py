import sys
import numpy as np
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QEventLoop
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QWidget, QSizePolicy
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont
from picamera2 import Picamera2, Controls
from functions import *
import threading
import libcamera
import cv2
from model_loader import get_vosk_model, get_translation_model

MODEL_PATH_DEFAULT = r"/home/pi/Desktop/gradproj/vosk-model-small-en-us-0.15"

class CameraWidget(QWidget):
    def __init__(self, parent=None):
        """
        Initialize the CameraWidget.

        :param parent: Parent QWidget.
        """
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.picam2 = None
        self.pixmap = QPixmap()
        self.pixmap.fill(Qt.black)
        self.camera_lock = threading.Lock()
        self.initialize_camera()

    def initialize_camera(self):
        """
        Initialize the camera with native sensor resolution and proper color format.

        This method configures the camera with the native sensor resolution and
        RGB888 color format. It also sets the autofocus mode to continuous and
        starts the camera. The capture dimensions are extracted from the
        configuration and stored as instance variables. The `QTimer` is started
        to update the frame at a rate of 30ms (~33 FPS).

        If there is an error during initialization, the `pixmap` is filled with
        red color to indicate the error.
        """
        try:
            if self.picam2 is None:
                self.picam2 = Picamera2()
                # Use sensor resolution and proper color format
                config = self.picam2.create_preview_configuration(
                    main={
                        "size": (1920, 1080),  # Native sensor mode for better FPS
                        "format": "RGB888"},
                    controls={
                        "AwbEnable": True,
                        "AeEnable": True,
                        "ExposureTime": 10000,  # Adjust based on lighting
                        "AnalogueGain": 1.0,
                        "FrameDurationLimits": (33333, 33333)  # 30fps
                    },
                    transform=libcamera.Transform(hflip=1, vflip=1)  # Adjust based on mounting
                )
                self.picam2.configure(config)

                # Set autofocus controls
                controls = {"AfMode": libcamera.controls.AfModeEnum.Continuous}
                self.picam2.set_controls(controls)
                self.picam2.start()

                # Get actual capture dimensions
                self.capture_width = config["main"]["size"][0]
                self.capture_height = config["main"]["size"][1]

                self.timer = QTimer()
                self.timer.timeout.connect(self.update_frame)
                self.timer.start(30)  # ~33 FPS

        except Exception as e:
            print(f"Camera Initialization Error: {e}")
            self.pixmap = QPixmap(640, 480)
            self.pixmap.fill(Qt.red)

    def update_frame(self):
        """
        Capture a frame from the camera, convert it to RGB, resize it to the widget size,
        and update the widget with the new image.

        This method is called by a QTimer to continuously update the frame.
        """
        try:
            # Capture full resolution frame
            buffer = self.picam2.capture_array("main")

            # Convert color space and resize
            rgb_buffer = cv2.cvtColor(buffer, cv2.COLOR_BGR2RGB)
            resized_buffer = cv2.resize(rgb_buffer, (self.width(), self.height()), interpolation=cv2.INTER_LANCZOS4)

            # Create QImage with correct dimensions
            image = QImage(resized_buffer.data, self.width(), self.height(),
                         QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(image)
            self.update()

        except Exception as e:
            print(f"Frame update error: {e}")

    def paintEvent(self, event):
        """
        Reimplemented from QWidget.

        Paints the current pixmap onto the widget.

        If the pixmap is null, nothing is painted.

        :param event: The paint event.
        """
        if self.pixmap.isNull():
            return

        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap)

    def capture_image(self, filename="captured_image.jpg"):
        """
        Capture an image from the camera and save it to a file.

        This method locks the camera, captures a full resolution image,
        converts it to RGB, and saves it to the given filename.

        If the capture is successful, the filename is returned; otherwise,
        None is returned.

        :param filename: The filename to save the image to (default: "captured_image.jpg")
        :return: The filename if the capture is successful, or None otherwise
        """
        with self.camera_lock:
            try:
                # Capture full resolution image
                buffer = self.picam2.capture_array("main")
                cv2.imwrite(filename, buffer, [cv2.IMWRITE_JPEG_QUALITY, 90])
                print(f"Image saved as {filename}")
                return filename
            except Exception as e:
                print(f"Error capturing image: {e}")
                return None

    def resizeEvent(self, a0):
        """
        Reimplemented from QWidget.

        Called when the widget is resized.

        The pixmap is updated when the widget is resized to ensure the image
        is displayed correctly.

        :param a0: The resize event.
        """
        self.update()

class WorkerThread(QThread):
    finished = pyqtSignal()
    result = pyqtSignal(object)
    def __init__(self, task_func, *args):
        """
        Initialize a WorkerThread with a task function and arguments.

        This constructor takes a function and its arguments, and runs the
        function in a separate thread when the `run` method is called.

        :param task_func: The function to run in the separate thread
        :param args: The arguments to pass to the function
        """
        super().__init__()
        self.task_func = task_func
        self.args = args
        self._is_running = True

    def run(self):
        """
        Run the task function in a separate thread.

        This method is called automatically when the thread is started. It
        runs the task function with the given arguments, emits the result
        of the function as a signal, and then emits a finished signal.

        :return: None
        """
        if self._is_running:
            result = self.task_func(*self.args)
            self.result.emit(result)
            self.finished.emit()

    def stop(self):
            self._is_running = False
            self.quit()
            self.wait()

class AIAssistantGUI(QMainWindow):
    def __init__(self):
        """
        Initialize the AIAssistantGUI window.

        This method creates a new QMainWindow with a title of "AR AI Assistant" and
        shows it full screen. It creates a central widget and adds a CameraWidget to it.
        It then creates several overlay labels and sets up a timer to update the status
        every second. Finally, it creates an AudioThread and starts it, connecting its
        text_received signal to the update_user_speech slot.

        :return: None
        """
        super().__init__()
        self.setWindowTitle("AR AI Assistant")
        self.showFullScreen()

        # self.setGeometry(100, 100, 640, 480)
        self.central_container = QWidget(self)
        self.setCentralWidget(self.central_container)

        self.camera_widget = CameraWidget(self.central_container)
        self.camera_widget.setGeometry(0, 0, self.width(), self.height())

        # self.centralWidget(self.central_container)
        self.create_overlays()

        self.status_label.raise_()
        self.user_speech_label.raise_()
        self.ai_response_label.raise_()

        # Camera background
        # self.camera_widget = CameraWidget()

        # Overlay elements
        self.status_check_timer = QTimer()
        self.status_check_timer.timeout.connect(self.update_status)
        self.status_check_timer.start(1000)  # Check every 1s

        # Audio thread
        self.model = get_vosk_model("en")
        self.audio_thread = AudioThread(self.model)
        self.audio_thread.text_received.connect(self.update_user_speech)
        self.audio_thread.start()
        self.signals = CommunicationSignals()
        self.signals.update_ai_speech.connect(self.update_ai_response)
        self.signals.update_user_speech.connect(self.update_user_speech)
        self.is_processing = False
        self.data_type = None
        self.lang1 = None
        self.lang2 = None
        # self.tool_loop = None

    def create_overlays(self):
        # Status indicator
        """
        Create and configure the overlay labels.

        This method creates and configures the labels that will be used to display the AI's
        response, the user's speech, and the connection status. It sets the text color, background
        color, border radius, padding, and font size for each label.

        :return: None
        """
        self.status_label = QLabel(self.central_container)
        self.status_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0,0,0,200);
                border-radius: 5px;
                padding: 5px;
                font: bold 14px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedSize(200, 40)
        self.status_label.move((self.width() - 200) // 2, 20)

        # User speech
        self.user_speech_label = QLabel(self.central_container)
        self.user_speech_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0,0,0,200);
                border-radius: 5px;
                padding: 8px;
                font: 12px;
            }
        """)
        self.user_speech_label.setFixedSize(300, 100)
        self.user_speech_label.move(self.width() - 320, self.height() - 120)
        self.user_speech_label.setWordWrap(True)

        # AI response
        self.ai_response_label = QLabel(self.central_container)
        self.ai_response_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(0,0,0,200);
                border-radius: 5px;
                padding: 8px;
                font: 12px;
            }
        """)
        self.ai_response_label.setFixedSize(300, 100)
        self.ai_response_label.move(20, self.height() - 120)
        self.ai_response_label.setWordWrap(True)

        # Make labels ignore mouse events
        for label in [self.status_label, self.user_speech_label, self.ai_response_label]:
            label.setAttribute(Qt.WA_TransparentForMouseEvents)
            label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    # def create_overlays(self):
    #     # Status indicator
    #     self.status_label = QLabel(self)
    #     self.status_label.setStyleSheet("color: white; background-color: rgba(0,0,0,150);")
    #     self.status_label.setAlignment(Qt.AlignCenter)
    #     self.status_label.setGeometry(260, 20, 120, 30)

    #     # User speech
    #     self.user_speech_label = QLabel(self)
    #     self.user_speech_label.setStyleSheet("color: white; background-color: rgba(0,0,0,150);")
    #     self.user_speech_label.setGeometry(400, 400, 220, 60)
    #     self.user_speech_label.setWordWrap(True)

    #     # AI response
    #     self.ai_response_label = QLabel(self)
    #     self.ai_response_label.setStyleSheet("color: white; background-color: rgba(0,0,0,150);")
    #     self.ai_response_label.setGeometry(20, 400, 220, 60)
    #     self.ai_response_label.setWordWrap(True)
    def resizeEvent(self, event):
        # Add null check for camera_widget
        """
        This method is called whenever the window is resized. It updates the size and
        position of the camera widget and overlay labels accordingly.

        :param event: The QResizeEvent that triggered this method.
        :return: None
        """
        if hasattr(self, 'camera_widget') and self.camera_widget:
            self.camera_widget.setGeometry(0, 0, self.width(), self.height())

        # Update overlay positions
        if hasattr(self, 'status_label'):
            self.status_label.move((self.width() - 200) // 2, 20)
        if hasattr(self, 'user_speech_label'):
            self.user_speech_label.move(self.width() - 320, self.height() - 120)
        if hasattr(self, 'ai_response_label'):
            self.ai_response_label.move(20, self.height() - 120)

        super().resizeEvent(event)

    def update_status(self):
        """
        This method is called whenever the status timer times out. It checks the
        current WiFi connection status and updates the status label accordingly.

        :return: None
        """
        status = "ONLINE" if check_wifi_connection() else "OFFLINE"
        self.status_label.setText(f"Status: {status}")

    def update_user_speech(self, text):
        """
        This slot is connected to the text_received signal of the AudioThread. It
        updates the user speech label with the received text and checks if the text
        contains the wake word. If it does, it calls the respond_to_user method.

        :param text: The received text from the AudioThread.
        :return: None
        """
        self.user_speech_label.setText(f"You: {text}")
        if not self.is_processing:
            if "hi david" in text.lower():
                self.respond_to_user()

    def respond_to_user(self):
        """
        This method is called when the user speech contains the wake word. It stops
        the audio thread, updates the AI response label, and starts a new thread
        to run the tool detection function.

        :return: None
        """
        if self.is_processing:
            return
        self.audio_thread.pause()
        self.is_processing = True
        self.update_ai_response('Initializing...')
        USER_NAME = "Khaled"
        self.update_ai_response(f"Hello {USER_NAME}, how are you?")
        online_status = check_wifi_connection()
        if online_status:
            self.update_ai_response("I am online.")
        else:
            self.update_ai_response("Working offline.")

        # self.tool_loop = QEventLoop()
        self.tool_detection_worker = WorkerThread(tool_detection, self.model)
        self.tool_detection_worker.result.connect(self.handle_data_type)
        # self.tool_detection_worker.finished.connect(self.tool_loop.quit)
        # self.tool_detection_worker.finished.connect(self.tool_detection_worker.deleteLater)
        print("Starting Tool Detection")
        self.tool_detection_worker.start()
        # self.tool_loop.exec_()

        # if self.tool_detection_worker.isRunning():
        #     self.tool_detection_worker.quit()
        #     self.tool_detection_worker.wait()


    def handle_data_type(self, data_type):
        """
        This method is called when the tool detection function returns a result. It
        checks if the result is valid and updates the AI response label. If the
        result is valid, it starts the language selection process.

        :param data_type: The detected input type (speech or image).
        :return: None
        """
        if not data_type:
            self.update_ai_response("Could not detect input type. Please try again.")
            print("Could not detect input type. Please try again.")
            self.is_processing = False
            self.audio_thread.resume()
            return
        self.data_type = data_type
        self.update_ai_response(f"Detected {data_type} input")
        # self.tool_loop.exit()
        self.start_language_selection()

    def start_language_selection(self):
        """
        Initiate the language selection process.

        This method starts a worker thread to determine the source language
        by calling the `get_lang1` function. The result of this function is
        connected to the `handle_lang1` method for further processing.

        :return: None
        """

        self.lang1_worker = WorkerThread(get_lang1, self.model)
        self.lang1_worker.result.connect(self.handle_lang1)
        self.lang1_worker.start()

    def handle_lang1(self, lang1):
        """
        This method is invoked after the source language selection process is
        completed. If the language is invalid, it updates the AI response label
        and stops processing. Otherwise, it updates the AI response label with
        the selected source language and initiates the target language selection
        process by starting a worker thread.

        :param lang1: The source language selected by the user.
        :return: None
        """
        if not lang1:
            self.update_ai_response("Invalid source language. Please try again.")
            self.is_processing = False
            return
        self.lang1 = lang1
        self.update_ai_response(f"Source language: {lang1}")
        self.lang2_worker = WorkerThread(get_lang2, self.model)
        self.lang2_worker.result.connect(self.handle_lang2)
        self.lang2_worker.start()

    def handle_lang2(self, lang2):
        """
        This method is called when the target language selection process is
        finished. If the language is invalid, it updates the AI response label
        and stops the processing. Otherwise, it updates the AI response label
        and starts the processing worker.

        :param lang2: The target language selected by the user.
        :return: None
        """
        if not lang2:
            self.update_ai_response("Invalid target language. Please try again.")
            self.is_processing = False
            return
        self.lang2 = lang2
        self.update_ai_response(f"Target language: {lang2}")
        self.start_processing()

    def start_processing(self):
        """
        Initiates the processing of data based on the selected data type.

        This method initializes the necessary model for language recognition
        based on the selected source language. Depending on the data type
        (speech or image), it starts a worker thread to handle the processing
        using the appropriate function. Connects the worker's finished signal
        to the `on_processing_finished` method for cleanup and further actions.
        Handles exceptions by updating the AI response and stopping the processing.

        :return: None
        """

        try:
            # model_path = recognition_model(self.lang1)
            model = get_vosk_model(self.lang1)

            if self.data_type == "speech":
                self.worker_process = WorkerThread(
                    process_speech, model, self.lang1, self.lang2, self.signals
                )
            elif self.data_type == "image":
                self.worker_process = WorkerThread(
                    process_image, self.camera_widget, model, self.lang1, self.lang2, self.signals
                )

            self.worker_process.finished.connect(self.on_processing_finished)
            self.worker_process.start()
        except Exception as e:
            self.update_ai_response(f"Error: {str(e)}")
            self.is_processing = False

    def on_processing_finished(self):
        """
        Handles the completion of data processing.

        This method is called when the processing of data is finished. It resets the
        processing state, resumes the audio thread, updates the AI response to indicate
        readiness for new commands, and cleans up the worker thread by deleting it.

        :return: None
        """
        self.is_processing = False
        self.audio_thread.resume()
        self.update_ai_response("Ready for commands")
        self.sender().deleteLater()  # Clean up worker

    def update_ai_response(self, text):
        self.ai_response_label.setText(f"Assistant: {text}")

    def closeEvent(self, event):
        self.audio_thread.stop()
        self.audio_thread.wait()
        event.accept()

class AudioThread(QThread):
    text_received = pyqtSignal(str)
    def __init__(self, model):
        """
        Initialize the AudioThread with a model.

        This constructor takes a model as an argument, and runs the
        run method in a separate thread when the `start` method is called.

        :param model: The model to use for speech recognition.
        """
        super().__init__()
        self.model = model
        self._is_paused = False
        self._is_running = True

    def pause(self):
        """
        Pause the audio thread.

        This method is used to pause the audio thread. While the thread is paused,
        it will not process any audio data. This is useful when you want to temporarily
        pause the audio processing and then resume it later.

        :return: None
        """
        self._is_paused = True

    def resume(self):
        """
        Resume the audio thread if it was previously paused.

        This method is used to resume the audio thread after it was paused by calling
        the `pause` method. This is useful when you want to temporarily pause the
        audio processing and then resume it later.

        Note: This method does not start the thread, it only resumes it if it was
        previously paused. To start the thread, call the `start` method.

        :return: None
        """
        self._is_paused = False

    def stop(self):
        self._is_running = False

    def run(self):
        """
        The main loop of the thread.

        This method is called when the `start` method is called. It enters an
        infinite loop and continuously checks if the thread is paused or not.
        If the thread is not paused, it calls the `get_audio` method to get the
        recognized text from the microphone. If the `get_audio` method returns a
        non-empty string, the recognized text is emitted as a signal.

        :return: None
        """
        while self._is_running:
            if not self._is_paused:
                text = get_audio(self.model)
                if text:
                    self.text_received.emit(text)
            self.msleep(100)

if __name__ == "__main__":
    get_translation_model("en", "fr")
    get_translation_model("en", "ar")
    app = QApplication(sys.argv)
    window = AIAssistantGUI()
    window.show()
    sys.exit(app.exec_())
