from __future__ import annotations

import argparse
import os
import threading
import time
from collections import deque
from typing import Iterable

import cv2
import math
import mediapipe as mp
from google import genai
from google.genai import types
from ultralytics import YOLO

from Keys import API_KEY


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Integrated WearableDisplay demo combining YOLO detection, hand gesture "
            "recognition, and Gemini responses."
        )
    )
    parser.add_argument(
        "--camera",
        default=None,
        help=(
            "Explicit camera index, preset, or URL (e.g. http://localhost:8000/stream). "
            "If omitted, the app will try the localhost stream first, then fall back "
            "to available USB cameras."
        ),
    )
    parser.add_argument(
        "--weights",
        default=os.getenv("YOLO_WEIGHTS", "yolov8n.pt"),
        help="Path to YOLO weights (default: yolov8n.pt; downloads automatically if missing).",
    )
    parser.add_argument(
        "--mirror/--no-mirror",
        dest="mirror",
        default=True,
        help="Mirror frames horizontally for intuitive display (default: enabled).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("GOOGLE_API_KEY", API_KEY),
        help="Gemini API key. Defaults to GOOGLE_API_KEY env or Keys.API_KEY when omitted.",
    )
    return parser.parse_args()


def _resolve_camera_source(selection: str) -> int | str:
    try:
        return int(selection)
    except ValueError:
        return selection


def _class_names() -> Iterable[str]:
    return [
        "person",
        "bicycle",
        "car",
        "motorbike",
        "aeroplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "sofa",
        "pottedplant",
        "bed",
        "diningtable",
        "toilet",
        "tvmonitor",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]


def _camera_candidates(args: argparse.Namespace) -> list[int | str]:
    """
    Build an ordered list of candidate camera sources.

    Priority:
    1. CLI-provided value.
    2. INTEGRATED_CAMERA environment variable.
    3. Localhost stream URL (defaults to http://localhost:8000/stream).
    4. Preferred USB camera index (DEFAULT_CAMERA_INDEX env or 0).
    5. Alternate USB camera index (the opposite of preferred).
    """
    preferred_env = os.getenv("INTEGRATED_CAMERA")
    localhost_stream = os.getenv("LOCAL_STREAM_URL", "http://localhost:8000/stream")

    candidates: list[int | str] = []

    raw_sources = [
        args.camera,
        preferred_env,
        localhost_stream,
        os.getenv("DEFAULT_CAMERA_INDEX"),
        "0",
        "1",
    ]

    processed: list[int | str] = []
    for source in raw_sources:
        if source is None:
            continue
        resolved = _resolve_camera_source(str(source))
        if resolved not in processed:
            processed.append(resolved)

    # Move the alternate USB index to be the opposite of the preferred when possible.
    usb_indices: list[int] = []
    for item in processed:
        if isinstance(item, int):
            usb_indices.append(item)
    if usb_indices:
        preferred_usb = usb_indices[0]
        alternate_usb = 1 if preferred_usb == 0 else 0
        if alternate_usb not in processed:
            processed.append(alternate_usb)

    candidates.extend(processed)
    return candidates


def _open_camera_with_fallback(candidates: list[int | str]) -> tuple[cv2.VideoCapture | None, int | str | None]:
    """
    Try to open each camera candidate in order, returning the first successful capture.
    """
    for source in candidates:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            cap.release()
            continue

        # Give the capture a moment to start streaming frames.
        if not _warm_up_capture(cap):
            cap.release()
            continue

        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        return cap, source

    return None, None


def _warm_up_capture(cap: cv2.VideoCapture, attempts: int = 10, delay: float = 0.1) -> bool:
    """
    Attempt to read a few frames to confirm the capture is yielding images.
    """
    for _ in range(attempts):
        ret, _ = cap.read()
        if ret:
            return True
        time.sleep(delay)
    return False


class HandGestureRecognizer:
    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
        api_key: str | None = None,
    ):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.api_key = api_key or API_KEY

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
        self.text_rendering_duration = 15.0  # 15 seconds
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
                    prev_pinch, prev_peace, prev_closed = (
                        self.is_pinch,
                        self.is_peace,
                        self.is_closed,
                    )

                    self.is_pinch = pinch
                    self.is_peace = peace
                    self.is_closed = closed

                    if pinch:
                        detected_gesture = "Pinch"
                        if not prev_pinch:
                            wristCoordinates = hand_lms.landmark[self.mp_hands.HandLandmark.WRIST]
                            print("Pinch detected - starting LLM processing")

                            # Start pinch sequence immediately with LLM processing
                            self.pinch_sequence_active = True
                            self.pinch_start_time = current_time
                            self.current_phase = "processing"
                            self.llm_processing = True
                            self.llm_response = ""

                            # Process frame with LLM in background
                            self._process_with_llm_async(frame.copy())
                    elif peace:
                        detected_gesture = "Peace Sign"
                        if not prev_peace:
                            print("Peace Sign")
                    elif closed:
                        detected_gesture = "Closed Hand"
                        if not prev_closed:
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
    
    def _process_with_llm_async(self, frame: cv2.Mat) -> None:
        """Process the frame with LLM in background thread"""
        if not self.api_key:
            print("No API key configured; skipping LLM processing.")
            self.llm_response = "LLM disabled (missing API key)."
            self.llm_processing = False
            self.current_phase = "text_rendering"
            return

        frame_to_save = frame.copy()

        def process_thread():
            try:
                # Save the current frame
                cv2.imwrite("captured_image.jpg", frame_to_save)
                print("Image saved as captured_image.jpg")

                # Process with LLM
                with open("captured_image.jpg", "rb") as f:
                    image_bytes = f.read()
                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/jpeg",
                        ),
                        (
                            "Tell me what is the most significant object in the bounding box and describe it to me, "
                            "limit it to two sentences."
                        ),
                    ],
                )

                self.llm_response = response.text if response else ""
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

def main() -> None:
    args = _parse_args()

    candidates = _camera_candidates(args)
    cap, camera_source = _open_camera_with_fallback(candidates)

    if cap is None or camera_source is None:
        print(
            "Error: Unable to access any camera source. "
            "Confirm the localhost stream is active or plug in a USB camera."
        )
        return

    model = YOLO(args.weights)
    gesture_recognizer = HandGestureRecognizer(api_key=args.api_key)
    class_names = list(_class_names())

    print("Integrated App Started!")
    print("Features: YOLO Object Detection + Hand Gesture Recognition")
    print("Press 'q' to quit")
    print(f"Camera source: {camera_source!r}")

    if args.mirror:
        print("Mirroring frames for display.")

    if not gesture_recognizer.api_key:
        print(
            "Warning: Google API key not provided. Pinch snapshots will be saved "
            "locally but Gemini responses will be skipped."
        )

    last_announced_gesture = None

    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame; ending capture loop.")
            break

        if args.mirror:
            img = cv2.flip(img, 1)

        height, width = img.shape[:2]
        center_x, center_y = width // 2, height // 2
        cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)

        results = model(img, stream=True)

        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                if (x1 <= center_x <= x2) and (y1 <= center_y <= y2):
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    confidence = math.ceil((box.conf[0] * 100)) / 100
                    print("Confidence --->", confidence)

                    cls = int(box.cls[0])
                    label = class_names[cls] if 0 <= cls < len(class_names) else f"class_{cls}"
                    print("Class name -->", label)

                    org = (x1, y1)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    color = (255, 0, 0)
                    thickness = 2

                    cv2.putText(img, label, org, font, font_scale, color, thickness)

        img, gesture = gesture_recognizer.detect_gestures(img)

        cv2.imshow("Integrated App - YOLO + Gestures", img)

        if gesture != last_announced_gesture and gesture != "None":
            print(f"Detected gesture: {gesture}")
            last_announced_gesture = gesture
        elif gesture == "None":
            last_announced_gesture = "None"

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
