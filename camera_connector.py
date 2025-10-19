"""
Camera connector utilities for the ComeFriend (GestureCam) project.

This module centralises camera selection so that services can rely on
pre-defined presets when opening local or remote video feeds. It aims
to satisfy the cross-platform needs (Linux, Windows, macOS) described in
the product definition while remaining simple to use from other modules
such as `VideoStream.py`.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Union

import cv2

# Map host platforms to the OpenCV backend that typically offers
# the best compatibility. Each preset can override this when needed.
_DEFAULT_BACKENDS = {
    "Windows": cv2.CAP_DSHOW,
    "Darwin": cv2.CAP_AVFOUNDATION,
    "Linux": cv2.CAP_V4L2,
}


def _detect_platform_backend() -> Optional[int]:
    """Return the OpenCV backend that matches the current platform."""
    system = platform.system()
    return _DEFAULT_BACKENDS.get(system)


@dataclass
class CameraPreset:
    """
    Represents a pre-defined way to connect to a camera feed.

    `source` can be either an integer (device index) for local cameras or
    a string (URL) for remote streams (RTSP, HTTP, etc.).
    """

    key: str
    source: Union[int, str]
    description: str = ""
    backend: Optional[int] = None
    metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)

    def resolve_backend(self) -> Optional[int]:
        """
        Determine which backend to use for this preset.

        Local cameras default to the per-platform backend unless explicitly
        overridden. Remote streams typically ignore the backend.
        """
        if self.backend is not None:
            return self.backend
        if isinstance(self.source, int):
            return _detect_platform_backend()
        return None


class CameraConnector:
    """
    Handles opening camera feeds based on named presets.

    Presets can cover local devices or remote network streams, making it
    easier to host the service on a server and accept remote camera inputs.
    """

    def __init__(self, presets: Iterable[CameraPreset]):
        self._presets: Dict[str, CameraPreset] = {preset.key: preset for preset in presets}

    def register_preset(self, preset: CameraPreset) -> None:
        """Add or replace a preset."""
        self._presets[preset.key] = preset

    def has_preset(self, key: str) -> bool:
        """Return True when the preset is registered."""
        return key in self._presets

    def list_presets(self) -> Iterable[CameraPreset]:
        """Return the known presets."""
        return self._presets.values()

    def get_preset(self, key: str) -> CameraPreset:
        """Retrieve a preset by key, raising KeyError when missing."""
        return self._presets[key]

    def open(self, key: str) -> cv2.VideoCapture:
        """
        Open a cv2.VideoCapture based on the preset.

        Raises:
            KeyError: if the preset is unknown.
            RuntimeError: if the camera cannot be opened.
        """
        preset = self.get_preset(key)
        backend = preset.resolve_backend()

        # cv2.VideoCapture accepts backend flags only for integer sources. Passing a backend for
        # URLs can lead to incorrect behaviour, so we only include it when relevant.
        if backend is not None and isinstance(preset.source, int):
            capture = cv2.VideoCapture(preset.source, backend)
        else:
            capture = cv2.VideoCapture(preset.source)

        if not capture.isOpened():
            raise RuntimeError(f"Unable to open camera preset '{key}' (source={preset.source}).")

        return capture

    def probe_local_devices(self, max_devices: int = 8) -> Dict[int, bool]:
        """
        Attempt to detect available local camera indices.

        Returns a dictionary mapping device indices to a boolean indicating whether the device
        could be opened. This is useful to generate new presets or verify hardware availability.
        """
        availability: Dict[int, bool] = {}
        backend = _detect_platform_backend()

        for index in range(max_devices):
            if backend is not None:
                cap = cv2.VideoCapture(index, backend)
            else:
                cap = cv2.VideoCapture(index)

            opened = cap.isOpened()
            availability[index] = opened
            cap.release()

        return availability


def build_default_connector() -> CameraConnector:
    """
    Helper that initialises a connector with sensible defaults.

    The default preset prefers the `DEFAULT_CAMERA_INDEX` (defaults to 1) and keeps
    a fallback entry that points to the opposite common index. An overridden
    `CAMERA_SOURCE` environment variable takes precedence:
    - If `CAMERA_SOURCE` is an integer, it is treated as a local device index.
    - Otherwise, it is used as a URL/stream path for remote clients.
    """
    try:
        default_index = int(os.getenv("DEFAULT_CAMERA_INDEX", "1"))
    except ValueError:
        default_index = 1
    presets = [
        CameraPreset(
            key="local-default",
            source=default_index,
            description=f"Preferred local camera device (index {default_index})."
        ),
    ]

    fallback_index = 0 if default_index != 0 else 1
    presets.append(
        CameraPreset(
            key="local-fallback",
            source=fallback_index,
            description=f"Secondary local camera device (index {fallback_index}).",
        )
    )

    env_source = os.getenv("CAMERA_SOURCE")
    if env_source:
        # Allow explicit override through environment to support remote clients.
        try:
            index = int(env_source)
            presets.append(
                CameraPreset(
                    key="env-override",
                    source=index,
                    description=f"Local camera index {index} specified via CAMERA_SOURCE.",
                )
            )
        except ValueError:
            presets.append(
                CameraPreset(
                    key="env-override",
                    source=env_source,
                    description="Remote camera stream provided via CAMERA_SOURCE.",
                )
            )

    return CameraConnector(presets)


__all__ = [
    "CameraConnector",
    "CameraPreset",
    "build_default_connector",
]
