import cv2
import mediapipe as mp
import pytesseract
from PIL import Image
import os
import time
import math
from config import TESSERACT_CMD, vosk_to_tesseract


try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
except Exception as e:
    print(f"Warning: Could not set Tesseract command path or verify Tesseract: {e}. OCR might not work.")
    print(f"Ensure Tesseract is installed and TESSERACT_CMD in config.py is correct: '{TESSERACT_CMD}'")


class CameraManager:
    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open webcam at index {camera_index}")
        self.zoom_factor = 1.0
        print("CameraManager initialized.")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None # Return None for frame if not successful

        # Apply zoom
        if self.zoom_factor != 1.0:
            h, w = frame.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # Calculate new dimensions based on zoom factor
            # Ensure scaled_w and scaled_h are positive
            scaled_w = max(1, int(w / self.zoom_factor))
            scaled_h = max(1, int(h / self.zoom_factor))

            # Calculate top-left corner of the zoomed region
            x1 = max(0, center_x - scaled_w // 2)
            y1 = max(0, center_y - scaled_h // 2)

            # Calculate bottom-right corner, ensuring it doesn't exceed frame dimensions
            x2 = min(w, x1 + scaled_w)
            y2 = min(h, y1 + scaled_h)
            
            # Adjust x1, y1 again if x2 or y2 were capped, to maintain aspect ratio as much as possible
            # This part can be complex if perfect aspect ratio preservation during crop is needed.
            # For simplicity, we'll use the calculated crop.
            
            cropped_frame = frame[y1:y2, x1:x2]
            
            # Resize cropped frame back to original dimensions
            if cropped_frame.size > 0 : # Check if cropped_frame is not empty
                 frame = cv2.resize(cropped_frame, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                # print("Warning: Cropped frame is empty, cannot resize. Zoom factor might be too extreme.")
                # Keep original frame or handle as an error
                pass


        return ret, frame

    def set_zoom_factor(self, factor):
        # Add reasonable limits to zoom factor
        self.zoom_factor = max(0.1, min(factor, 5.0)) 

    def capture_image(self, filename="captured_image.jpg"):
        ret, frame = self.get_frame() # Use get_frame to include zoom
        if ret and frame is not None:
            try:
                cv2.imwrite(filename, frame)
                print(f"Image captured: {filename}")
                return filename
            except Exception as e:
                print(f"Error saving image {filename}: {e}")
                return None
        else:
            print("Error: Could not read frame from camera for capture.")
            return None

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
        print("CameraManager released.")

class HandGestureRecognizer:
    def __init__(self, max_num_hands=1, gesture_cooldown=1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_gesture_time = 0
        self.gesture_cooldown = gesture_cooldown # seconds
        self.current_gesture = None # e.g., "FIVE_FINGERS", "FIST", "PINCH_ZOOM"
        
        # Zoom related state
        self.zooming_active = False
        self.last_zoom_pinch_distance = 0
        self.base_pinch_distance = 0.15 # Normalized distance, adjust based on typical hand size in frame
        self.max_zoom_out_factor = 0.3
        self.max_zoom_in_factor = 3.0
        self.new_zoom_factor = 1.0 # Stores the calculated zoom factor from pinch

        print("HandGestureRecognizer initialized.")

    def detect_gestures(self, frame_rgb, frame_bgr_draw):
        results = self.hands.process(frame_rgb)
        gesture_event = None
        self.new_zoom_factor = None # Reset for this frame

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                # Get finger landmarks
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

                
                thumb_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_MCP]
                index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
                middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                ring_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
                pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]


                thumb_extended = thumb_tip.x > hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x if hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x < thumb_tip.x else thumb_tip.x < hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP].x # Basic L/R hand check
                index_extended = index_tip.y < index_mcp.y
                middle_extended = middle_tip.y < middle_mcp.y
                ring_extended = ring_tip.y < ring_mcp.y
                pinky_extended = pinky_tip.y < pinky_mcp.y
                
                extended_fingers = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
                current_time = time.time()

                # Pinch gesture for zoom (thumb and index finger)
                is_pinch_gesture = thumb_extended and index_extended and not (middle_extended or ring_extended or pinky_extended)

                if is_pinch_gesture:
                    current_pinch_distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
                    
                    if not self.zooming_active:
                        self.zooming_active = True
                        self.last_zoom_pinch_distance = current_pinch_distance
                        gesture_event = "ZOOM_ACTIVE" 

                    

                    if current_pinch_distance < 0.01:
                        self.new_zoom_factor = self.max_zoom_out_factor
                    else:
                        scale = current_pinch_distance / self.base_pinch_distance
                        if scale < 1.0: 
                            self.new_zoom_factor = self.max_zoom_out_factor + (1.0 - self.max_zoom_out_factor) * scale
                        else: # Pinching wider than base
                            self.new_zoom_factor = 1.0 + (self.max_zoom_in_factor - 1.0) * min((scale - 1.0), 1.0) # Cap growth

                        self.new_zoom_factor = max(self.max_zoom_out_factor, min(self.new_zoom_factor, self.max_zoom_in_factor))
                    
                    # Display zoom level
                    zoom_color = (0, 255, 0) if self.new_zoom_factor >= 1.0 else (0, 0, 255)
                    cv2.putText(frame_bgr_draw, f"Zoom: {self.new_zoom_factor:.1f}x", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, zoom_color, 2)

                elif self.zooming_active: # Pinch ended
                    self.zooming_active = False
                    gesture_event = "ZOOM_ENDED"


                if not self.zooming_active and (current_time - self.last_gesture_time) > self.gesture_cooldown:
                    if extended_fingers == 5: # Open hand
                        gesture_event = "SHOW_LABELS"
                        self.last_gesture_time = current_time
                    elif extended_fingers == 0: # Fist
                        gesture_event = "HIDE_LABELS"
                        self.last_gesture_time = current_time
            
            return gesture_event, self.new_zoom_factor


        if not results.multi_hand_landmarks and self.zooming_active:
            self.zooming_active = False
            gesture_event = "ZOOM_ENDED"
        return gesture_event, self.new_zoom_factor


    def close(self):
        self.hands.close()
        print("HandGestureRecognizer closed.")


class OCRService:
    def __init__(self):
        print("OCRService initialized.")

    def recognize_text_from_image(self, image_path, lang_code="en"):
        if not os.path.exists(image_path):
            return "Error: Image file not found."

        tesseract_lang = vosk_to_tesseract.get(lang_code, "eng")

        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang=tesseract_lang)
            if not text.strip():
                return "No text detected in the image."
            return text.strip()
        except pytesseract.TesseractNotFoundError:
            print(f"ERROR: Tesseract is not installed or not in your PATH. OCR will not work.")
            print(f"Please install Tesseract and ensure '{TESSERACT_CMD}' in config.py is correct.")
            return "Error: Tesseract not found. Please check installation."
        except Exception as e:
            return f"OCR Error: {e}"

    def get_tesseract_lang_code(self, vosk_lang_code):
        return vosk_to_tesseract.get(vosk_lang_code, "eng")


if __name__ == '__main__':
    # Test CameraManager
    try:
        cam_manager = CameraManager(camera_index=0) # Use a different index if 0 is not your webcam
        ret, frame = cam_manager.get_frame()
        if ret and frame is not None:
            cv2.imshow("Camera Test", frame)
            print("Camera test: Press any key to close.")
            cv2.waitKey(0)
            
            # Test capture
            # captured_file = cam_manager.capture_image("test_capture.jpg")
            # if captured_file:
            #     print(f"Image captured to {captured_file}")
                # os.remove(captured_file) # Clean up

            # Test zoom
            # cam_manager.set_zoom_factor(2.0)
            # ret_zoom, frame_zoom = cam_manager.get_frame()
            # if ret_zoom and frame_zoom is not None:
            #     cv2.imshow("Zoomed Camera Test (2x)", frame_zoom)
            #     cv2.waitKey(0)
            # cam_manager.set_zoom_factor(1.0) # Reset zoom


        else:
            print("Failed to get frame from camera.")
        cam_manager.release()
        cv2.destroyAllWindows()
    except IOError as e:
        print(f"CameraManager test failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during CameraManager test: {e}")


    # Test HandGestureRecognizer (requires a live camera feed)
    # This test is more involved as it needs a loop.
    # print("\nTesting HandGestureRecognizer...")
    # try:
    #     cam_for_gesture = CameraManager(0)
    #     gesture_recognizer = HandGestureRecognizer()
    #     print("Move your hand in front of the camera. Press 'q' to quit.")
    #     while True:
    #         ret_g, frame_g_bgr = cam_for_gesture.get_frame()
    #         if not ret_g or frame_g_bgr is None:
    #             print("Failed to get frame for gesture recognition.")
    #             break
            
    #         frame_g_rgb = cv2.cvtColor(frame_g_bgr, cv2.COLOR_BGR2RGB)
    #         gesture_event, new_zoom = gesture_recognizer.detect_gestures(frame_g_rgb, frame_g_bgr)

    #         if gesture_event:
    #             print(f"Gesture Event: {gesture_event}, New Zoom: {new_zoom}")
    #             if new_zoom is not None:
    #                 cam_for_gesture.set_zoom_factor(new_zoom)


    #         cv2.imshow("Hand Gesture Test", frame_g_bgr)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     gesture_recognizer.close()
    #     cam_for_gesture.release()
    #     cv2.destroyAllWindows()
    # except IOError as e:
    #     print(f"HandGestureRecognizer test failed (camera issue): {e}")
    # except Exception as e:
    #     print(f"An unexpected error occurred during HandGestureRecognizer test: {e}")


    # Test OCRService
    # Create a dummy image with text for testing
    print("\nTesting OCRService...")
    try:
        from PIL import Image, ImageDraw, ImageFont
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except IOError:
            font = ImageFont.load_default() # Fallback font
        
        img_ocr = Image.new('RGB', (400, 100), color = (255, 255, 255))
        d = ImageDraw.Draw(img_ocr)
        d.text((10,10), "Hello Tesseract OCR", fill=(0,0,0), font=font)
        test_ocr_image_path = "test_ocr_image.png"
        img_ocr.save(test_ocr_image_path)

        ocr_service = OCRService()
        # Test English OCR
        text_en_ocr = ocr_service.recognize_text_from_image(test_ocr_image_path, lang_code="en")
        print(f"OCR (en) from '{test_ocr_image_path}': {text_en_ocr}")

        # Test with another language if you have the language pack for Tesseract (e.g., Arabic 'ar')
        # d_ar = ImageDraw.Draw(img_ocr) # Re-draw for Arabic text
        # img_ocr.paste( (255,255,255), [0,0,img_ocr.size[0],img_ocr.size[1]]) # Clear image
        # d_ar.text((10,10), "مرحبا بالعالم", fill=(0,0,0), font=font) # This needs an Arabic-supporting font
        # test_ocr_image_ar_path = "test_ocr_image_ar.png"
        # img_ocr.save(test_ocr_image_ar_path)
        # text_ar_ocr = ocr_service.recognize_text_from_image(test_ocr_image_ar_path, lang_code="ar")
        # print(f"OCR (ar) from '{test_ocr_image_ar_path}': {text_ar_ocr}")
        # if os.path.exists(test_ocr_image_ar_path): os.remove(test_ocr_image_ar_path)

        if os.path.exists(test_ocr_image_path):
            os.remove(test_ocr_image_path) # Clean up
            
    except ImportError:
        print("Pillow (PIL) is not installed. Skipping OCRService image creation for test.")
    except pytesseract.TesseractNotFoundError:
        print("Tesseract not found during OCR test. Ensure it's installed and configured.")
    except Exception as e:
        print(f"An unexpected error occurred during OCRService test: {e}")