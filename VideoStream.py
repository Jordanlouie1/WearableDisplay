"""
Utility entry point to interact with camera streams using the CameraConnector.

This script can run locally or on a headless server, selecting the camera source
through predefined presets, CLI arguments, or environment variables. It is
designed to support the ComeFriend (GestureCam) workflow where remote clients
may publish camera streams that the backend needs to process hands-free.
"""
from __future__ import annotations

import argparse
import os
from typing import Optional

import cv2

from camera_connector import CameraConnector, CameraPreset, build_default_connector


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


def _configure_writer(cam: cv2.VideoCapture, output_path: Optional[str]) -> Optional[cv2.VideoWriter]:
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
        description="Open a camera stream from predefined presets for the ComeFriend project."
    )
    parser.add_argument(
        "--camera",
        help="Camera preset key, device index, or stream URL. Defaults to CAMERA_PRESET env or 'local-default'.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to save the captured stream (mp4).",
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
        "--max-probe",
        type=int,
        default=8,
        help="Maximum number of device indices to probe when --probe is used.",
    )
    return parser.parse_args()


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
    writer = _configure_writer(cam, args.output)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                print("Failed to read frame; terminating capture loop.")
                break

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
