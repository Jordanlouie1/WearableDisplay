"""
Utility entry point to interact with camera streams using the CameraConnector
while running YOLO object detection on the incoming frames.

This script can run locally or on a headless server, selecting the camera source
through predefined presets, CLI arguments, or environment variables. It is
designed to support the ComeFriend (GestureCam) workflow where remote clients
may publish camera streams that the backend needs to process hands-free.
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Optional

import cv2
from ultralytics import YOLO

from camera_connector import CameraConnector, CameraPreset, build_default_connector

# Object categories supported by the YOLO v8n model.
YOLO_CLASS_NAMES = [
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


def _resolve_camera(connector: CameraConnector, selection: str) -> str:
    """
    Return the preset key for the requested selection.

    The selection may already be a preset, a numeric device index, or a URL.
    When the selection is not an existing preset, we register a transient preset
    on the connector, enabling flexible remote camera configuration.
    """
    if connector.has_preset(selection):
        return selection

    try:
        index = int(selection)
        preset = CameraPreset(
            key=f"cli-device-{index}",
            source=index,
            description=f"CLI provided device index {index}.",
        )
    except ValueError:
        preset = CameraPreset(
            key="cli-stream",
            source=selection,
            description="CLI provided remote stream URL.",
        )

    connector.register_preset(preset)
    return preset.key


def _configure_writer(
    cam: cv2.VideoCapture, output_path: Optional[str]
) -> Optional[cv2.VideoWriter]:
    """
    Create a VideoWriter when output_path is provided, matching the capture resolution.
    """
    if not output_path:
        return None

    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    return cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a camera stream from predefined presets for the ComeFriend project "
            "and run YOLO object detection."
        )
    )
    parser.add_argument(
        "--camera",
        help="Camera preset key, device index, or stream URL. Defaults to CAMERA_PRESET env or 'local-default'.",
    )
    parser.add_argument(
        "--output",
        default="output.mp4",
        help="Path to save the captured stream (mp4). Use --no-output to disable recording.",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show the frames in a window (disabled by default for server environments).",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available camera presets and exit.",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Probe for local camera indices and exit.",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Disable writing the captured stream to disk.",
    )
    parser.add_argument(
        "--max-probe",
        type=int,
        default=8,
        help="Maximum number of device indices to probe when --probe is used.",
    )
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="Path to the YOLO model weights (default downloads via Ultralytics).",
    )
    return parser.parse_args()


def _draw_center_point(frame: cv2.Mat) -> tuple[int, int]:
    """
    Draw the center point on the provided frame and return its coordinates.
    """
    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2
    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
    return center_x, center_y


def _process_detections(
    frame: cv2.Mat,
    center_x: int,
    center_y: int,
    results,
) -> None:
    """
    Draw detection results and emit basic logging for any box covering the frame center.
    """
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            if (x1 <= center_x <= x2) and (y1 <= center_y <= y2):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                confidence = math.ceil((box.conf[0] * 100)) / 100
                print("Confidence --->", confidence)

                cls = int(box.cls[0])
                if 0 <= cls < len(YOLO_CLASS_NAMES):
                    label = YOLO_CLASS_NAMES[cls]
                else:
                    label = f"class_{cls}"
                print("Class name -->", label)

                org = (x1, y1)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(frame, label, org, font, font_scale, color, thickness)


def main() -> None:
    args = parse_args()

    connector = build_default_connector()

    if args.list_presets:
        for preset in connector.list_presets():
            print(f"{preset.key}: source={preset.source} ({preset.description})")
        return

    if args.probe:
        availability = connector.probe_local_devices(max_devices=args.max_probe)
        for index, is_available in availability.items():
            status = "available" if is_available else "unavailable"
            print(f"Device {index}: {status}")
        return

    selection = (
        args.camera
        or os.getenv("CAMERA_PRESET")
        or ("env-override" if os.getenv("CAMERA_SOURCE") else "local-default")
    )
    preset_key = _resolve_camera(connector, selection)

    cam = connector.open(preset_key)
    # Match the behaviour from the main branch implementation.
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    output_path = None if args.no_output else args.output
    writer = _configure_writer(cam, output_path)

    model = YOLO(args.weights)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to read frame; terminating capture loop.")
                break

            center_x, center_y = _draw_center_point(frame)
            results = model(frame, stream=True)
            _process_detections(frame, center_x, center_y, results)

            if writer:
                writer.write(frame)

            if args.display:
                cv2.imshow("Camera", frame)
                if cv2.waitKey(1) == ord("q"):
                    break
    finally:
        cam.release()
        if writer:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
