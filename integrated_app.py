from ultralytics import YOLO
import cv2
import math
import mediapipe as mp
from collections import deque
from google import genai
from google.genai import types
import time
import threading
from Keys import API_KEY

class HandGestureRecognizer:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Store hand landmark history for gesture tracking
        self.landmark_history = deque(maxlen=10)

        # Define gesture states
        self.is_pinch = False
        self.is_peace = False
        self.is_closed = False
        self.current_gesture = "None"

        # Swipe detection variables
        self.swipe_threshold = 0.2  # Normalized threshold for swipe detection
        self.prev_hand_center = None
        
        # Pinch sequence timing variables
        self.pinch_sequence_active = False
        self.pinch_start_time = None
        self.text_rendering_duration = 8.0  # 8 seconds
        self.llm_response = ""
        self.llm_processing = False
        self.current_phase = "normal"  # "normal", "processing", "text_rendering"

    def detect_gestures(self, frame):
        current_time = time.time()
        
        # Handle pinch sequence timing
        if self.pinch_sequence_active:
            elapsed_time = current_time - self.pinch_start_time
            
            if self.current_phase == "processing":
                # Still processing LLM in background
                cv2.putText(frame, "Processing with LLM...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                return frame, "Processing"
            
            elif self.current_phase == "text_rendering":
                if elapsed_time >= self.text_rendering_duration:
                    # End the pinch sequence
                    self.pinch_sequence_active = False
                    self.current_phase = "normal"
                    print("Pinch sequence completed")
                else:
                    # Render LLM response text on frame
                    remaining_time = self.text_rendering_duration - elapsed_time
                    self._render_text_on_frame(frame, remaining_time)
                    return frame, "Text Rendering"
        
        # Normal gesture detection (only if not in pinch sequence)
        if not self.pinch_sequence_active:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape

            # Process the frame to detect hands
            results = self.hands.process(frame_rgb)

            # Variables to store current detected gesture
            detected_gesture = "None"
            hand_landmarks = None

            # Check if hand landmarks are detected
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    # Draw hand landmarks on the frame
                    self.mp_draw.draw_landmarks(
                        frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)

                    # Store the landmark positions
                    hand_landmarks = hand_lms
                    landmarks = []
                    for lm in hand_lms.landmark:
                        landmarks.append((int(lm.x * w), int(lm.y * h)))

                    # Store landmarks history for tracking movement
                    self.landmark_history.append(landmarks)

                    # Detect pinch gesture
                    pinch = self.detect_pinch(hand_lms)

                    # Detect peace sign
                    peace = self.detect_peace_sign(hand_lms)

                    # Detect closed hand
                    closed = self.detect_closed_hand(hand_lms)

                    # Update the current gesture
                    if pinch:
                        detected_gesture = "Pinch"
                        wristCoordinates = hand_lms.landmark[self.mp_hands.HandLandmark.WRIST]
                        print("Pinch detected - starting LLM processing")
                        
                        # Start pinch sequence immediately with LLM processing
                        self.pinch_sequence_active = True
                        self.pinch_start_time = current_time
                        self.current_phase = "processing"
                        self.llm_processing = True
                        
                        # Process frame with LLM in background
                        self._process_with_llm_async(frame)
                        
                    elif peace:
                        detected_gesture = "Peace Sign"
                        print("Peace Sign")
                    elif closed:
                        detected_gesture = "Closed Hand"
                        print("Closed Hand")

                    # Add wrist coordinates to gesture info
                    wristCoordinates = hand_lms.landmark[self.mp_hands.HandLandmark.WRIST]
                    detected_gesture = detected_gesture + f" ({wristCoordinates.x:.2f}, {wristCoordinates.y:.2f})"
                    
                    # Label the gesture
                    cv2.putText(frame, f"Gesture: {detected_gesture}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Reset gesture tracking if no hands detected
                self.prev_hand_center = None

        self.current_gesture = detected_gesture
        return frame, detected_gesture

    def detect_pinch(self, landmarks):
        """
        Detect pinch gesture by measuring the distance between thumb tip and index tip
        """
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Calculate Euclidean distance
        distance = math.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 +
            (thumb_tip.y - index_tip.y) ** 2
        )

        # Threshold for pinch gesture
        return distance < 0.05

    def detect_peace_sign(self, landmarks):
        """
        Detect peace sign (V sign) by checking if index and middle fingers are extended
        while ring and pinky are closed
        """
        # Get landmarks for fingertips and knuckles
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        index_pip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_pip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_pip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]

        # Check if index and middle fingers are extended
        index_extended = index_tip.y < index_pip.y
        middle_extended = middle_tip.y < middle_pip.y

        # Check if ring and pinky fingers are closed
        ring_closed = ring_tip.y > ring_pip.y
        pinky_closed = pinky_tip.y > pinky_pip.y

        # Peace sign is when index and middle are extended while ring and pinky are closed
        return index_extended and middle_extended and (ring_closed or pinky_closed)

    def detect_closed_hand(self, landmarks):
        """
        Detect closed hand by checking if all fingers are curled
        """
        # Get fingertip landmarks
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]

        # Get finger PIP joints (second joints)
        index_pip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_pip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_pip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]

        # Check if all fingertips are below their respective PIP joints (finger curled)
        index_curled = index_tip.y > index_pip.y
        middle_curled = middle_tip.y > middle_pip.y
        ring_curled = ring_tip.y > ring_pip.y
        pinky_curled = pinky_tip.y > pinky_pip.y

        # Closed hand means all fingers are curled
        return index_curled and middle_curled and ring_curled and pinky_curled
    
    def _process_with_llm_async(self, frame):
        """Process the frame with LLM in background thread"""
        def process_thread():
            try:
                # Save the current frame
                cv2.imwrite("captured_image.jpg", frame)
                print("Image saved as captured_image.jpg")

                # Process with LLM
                with open('captured_image.jpg', 'rb') as f:
                    image_bytes = f.read()
                client = genai.Client(api_key=API_KEY)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type='image/jpeg',
                        ),
                        'Tell me what is the most significant object in the bounding box and describe it to me, limit it to two sentences, if you fail to describe it correctly I lose my job'
                    ]
                )

                self.llm_response = response.text
                self.llm_processing = False
                self.current_phase = "text_rendering"
                print("LLM Response:", self.llm_response)
                print("Starting text rendering phase...")
                
            except Exception as e:
                print(f"Error processing with LLM: {e}")
                self.llm_response = "Error processing image with LLM"
                self.llm_processing = False
                self.current_phase = "text_rendering"
        
        # Start processing in background thread
        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()
    
    def _render_text_on_frame(self, frame, remaining_time):
        """Render LLM response text on the frame during text rendering phase"""
        if self.llm_processing:
            # Still processing, show processing message
            cv2.putText(frame, "Processing with LLM...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        elif self.llm_response:
            # Split text into lines for better display
            lines = self.llm_response.split('. ')
            if len(lines) > 1:
                lines = [line + '.' for line in lines[:-1]] + [lines[-1]]
            
            # Display each line
            y_offset = 80
            for i, line in enumerate(lines[:3]):  # Limit to 3 lines
                cv2.putText(frame, line, (10, y_offset + i * 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # Show remaining time
            cv2.putText(frame, f"Text Display: {remaining_time:.1f}s remaining", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        else:
            cv2.putText(frame, "No response available", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

def main():
    # Initialize camera
    cap = cv2.VideoCapture(1)


    # Check if camera is properly initialized
    if not cap.isOpened():
        print("Error: Could not access camera. Please check your camera connection.")
        return

    # Initialize YOLO model
    model = YOLO("yolo-Weights/yolov8n.pt")

    # Initialize gesture recognizer
    gesture_recognizer = HandGestureRecognizer()

    # Object classes for YOLO
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]

    print("Integrated App Started!")
    print("Features: YOLO Object Detection + Hand Gesture Recognition")
    print("Press 'q' to quit")
    print("Initializing camera...")

    # Give camera time to initialize
    import time
    time.sleep(1)
    
    # Test camera by reading a frame
    test_ret, test_frame = cap.read()
    if not test_ret:
        print("Error: Camera test failed. Please check your camera connection.")
        cap.release()
        return
    else:
        print("Camera test successful!")

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for a more intuitive mirror view
        img = cv2.flip(img, 1)
        
        # Get frame dimensions
        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Draw center point
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
        
        # YOLO Object Detection
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Check if object is in center of frame
                    if (x1 <= center_x <= x2) and (y1 <= center_y <= y2):
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                        confidence = math.ceil((box.conf[0]*100))/100
                        print("Confidence --->", confidence)

                        cls = int(box.cls[0])
                        print("Class name -->", classNames[cls])

                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 0.7
                        color = (255, 0, 0)
                        thickness = 2

                        cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        # Hand Gesture Recognition
        img, gesture = gesture_recognizer.detect_gestures(img)

        # Display the frame
        cv2.imshow('Integrated App - YOLO + Gestures', img)

        # Print the current detected gesture
        if gesture != "None":
            print(f"Detected gesture: {gesture}")

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

