import tkinter as tk
import threading
import time
import cv2
from PIL import Image, ImageTk

# Configuration and Services
from config import DEFAULT_USERNAME, vosk_languages # vosk_languages needed by InteractionManager._select_language
from ui_manager import UIManager
from vision_services import CameraManager, HandGestureRecognizer, OCRService
from audio_services import TextToSpeech, SpeechRecognizer
from language_services import TranslationService
from api_service import APIService
from interaction_manager import InteractionManager

class Application:
    def __init__(self, root_tk):
        self.root = root_tk
        self.app_running = True
        print("Initializing services...")
        self.ui_manager = UIManager(self.root, app_title="AR EyeConic", logo_path="logo.png") 

        # Vision Services
        try:
            self.camera_manager = CameraManager(camera_index=0) # Or allow selection
        except IOError as e:
            print(f"Fatal Error: Could not initialize CameraManager: {e}")
            self.ui_manager._display_output(f"Error: Camera not found or cannot be opened. Please check connection. ({e})")
            # Potentially disable features requiring camera or exit
            self.camera_manager = None # Ensure it's None so other parts can check
            # self.root.destroy() # Or a more graceful shutdown
            # return # Stop further initialization if camera is critical

        self.hand_gesture_recognizer = HandGestureRecognizer() if self.camera_manager else None
        self.ocr_service = OCRService()

        # Audio Services
        self.tts_engine = TextToSpeech()
        try:
            self.speech_recognizer = SpeechRecognizer(initial_language_code="en")
        except Exception as e: # Catch broader exceptions from Vosk model loading
            print(f"Fatal Error: Could not initialize SpeechRecognizer: {e}")
            self.ui_manager.update_interaction_text(f"Error: Speech recognition model failed to load. ({e})")
            self.speech_recognizer = None
            # self.root.destroy()
            # return


        # Language and API Services
        self.translation_service = TranslationService()
        self.api_service = APIService()

        # Interaction Manager (The Core Logic Unit)
        if self.camera_manager and self.speech_recognizer:
            self.interaction_manager = InteractionManager(
                ui_manager=self.ui_manager,
                speech_recognizer=self.speech_recognizer,
                tts_engine=self.tts_engine,
                translation_service=self.translation_service,
                ocr_service=self.ocr_service,
                api_service=self.api_service,
                camera_manager=self.camera_manager,
                initial_username=DEFAULT_USERNAME
            )
        else:
            self.interaction_manager = None
            print("InteractionManager not initialized due to missing critical services (Camera or Speech).")
            self.ui_manager.update_response_text("Critical services failed to load. Application functionality will be limited.")


        # Bind close window event
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        print("Application services initialized.")

    def _camera_and_gesture_thread(self):
        """Handles camera frame updates and gesture detection."""
        if not self.camera_manager or not self.hand_gesture_recognizer:
            print("Camera or Gesture Recognizer not available. Camera thread exiting.")
            return

        frame_counter = 0
        while self.app_running and self.camera_manager.cap.isOpened():
            ret, frame_bgr = self.camera_manager.get_frame()
            if not ret or frame_bgr is None:
                print("Camera thread: Failed to get frame.")
                time.sleep(0.1) # Wait a bit before retrying
                continue

            # Process gestures every few frames for performance
            gesture_event = None
            new_zoom_factor_from_gesture = None
            if frame_counter % 3 == 0: 
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                gesture_event, new_zoom_factor_from_gesture = self.hand_gesture_recognizer.detect_gestures(frame_rgb, frame_bgr) 

            if new_zoom_factor_from_gesture is not None:
                 self.camera_manager.set_zoom_factor(new_zoom_factor_from_gesture)


            # Handle gesture events
            if gesture_event == "SHOW_LABELS":
                self.ui_manager.show_labels()
            elif gesture_event == "HIDE_LABELS":
                self.ui_manager.hide_labels()
            try:
                target_w = self.ui_manager.video_label.winfo_width()
                target_h = self.ui_manager.video_label.winfo_height()

                if target_w > 0 and target_h > 0:
                    display_frame = cv2.resize(frame_bgr, (target_w, target_h))
                else: # Fallback if UI dimensions not ready
                    display_frame = frame_bgr # Or a default size resize

                img_rgb_for_tk = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb_for_tk)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.ui_manager.update_video_frame(img_tk)
            except Exception as e:
                pass 

            frame_counter += 1
            time.sleep(0.03) # Aim for ~30 FPS for camera feed

        print("Camera and gesture thread stopped.")


    def _interaction_thread(self):
        """Handles the main voice/command interaction loop."""
        if self.interaction_manager:
            print("Starting interaction loop...")
            self.interaction_manager.start_interaction_loop() # This is a blocking call
        else:
            print("InteractionManager not available. Interaction thread exiting.")
        
        self.app_running = False 
        print("Interaction thread stopped.")


    def run(self):
        """Starts the application threads and the Tkinter main loop."""
        if not self.camera_manager or not self.speech_recognizer:
            print("Cannot run application due to missing critical services (Camera or Speech).")
            self.root.mainloop()
            return

        print("Starting application threads...")
        # Camera and Gesture Thread
        self.cam_thread = threading.Thread(target=self._camera_and_gesture_thread, daemon=True)
        self.cam_thread.start()

        # Interaction Logic Thread
        self.inter_thread = threading.Thread(target=self._interaction_thread, daemon=True)
        self.inter_thread.start()
        
        print("Starting Tkinter main loop...")
        self.root.mainloop() # This blocks until the window is closed

        # Cleanup after mainloop exits
        print("Tkinter main loop ended. Cleaning up...")
        self.app_running = False # Ensure threads know to stop
        
        if self.cam_thread.is_alive():
            self.cam_thread.join(timeout=1.0) # Wait for camera thread
        if self.inter_thread.is_alive():
            self.inter_thread.join(timeout=1.0) # Wait for interaction thread

        if self.camera_manager:
            self.camera_manager.release()
        if self.hand_gesture_recognizer:
            self.hand_gesture_recognizer.close() # If it has a close method
        
        print("Application shutdown complete.")


    def _on_closing(self):
        """Handles the event when the window's close button is pressed."""
        print("Window close button pressed. Initiating shutdown...")
        self.app_running = False # Signal threads to stop

        # Give threads a moment to stop based on app_running flag
        # InteractionManager's loop should break, CameraManager's loop should break.
        
        # Call InteractionManager's shutdown if it has specific cleanup beyond stopping loop
        if self.interaction_manager and hasattr(self.interaction_manager, '_handle_shutdown_os_command_only'):
             # A method that only does os.system("shutdown") without UI/speech
             # self.interaction_manager._handle_shutdown_os_command_only() 
             pass # For now, let the main loop exit handle OS shutdown if desired.
        
        # For a cleaner exit, explicitly join threads here after setting app_running to False
        # However, daemon threads will exit when the main program exits.
        # If non-daemon, they must be joined.

        if self.camera_manager:
            self.camera_manager.release()
        
        print("Destroying Tkinter root window.")
        self.root.destroy()


if __name__ == "__main__":
    print("Starting AR EyeConic Application...")
    main_root = tk.Tk()
    app = Application(main_root)
    try:
        app.run()
    except Exception as e:
        print(f"Unhandled exception in Application.run(): {e}")
    finally:
        print("Application has exited.")