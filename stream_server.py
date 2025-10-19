"""
HTTP server that exposes the WearableDisplay camera stream on port 8000.

Devices connect from a browser, grant camera permission, and publish frames
via WebSocket. The backend relays the latest frame to `/stream` as an MJPEG
feed. Local OpenCV presets remain available when an explicit `camera`
parameter (or environment override) requests a directly attached device.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import os
from typing import AsyncGenerator, Dict, Generator, Iterable, Optional, Tuple

import cv2
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

from camera_connector import CameraConnector, CameraPreset, build_default_connector

app = FastAPI(title="WearableDisplay Stream Server")

BOUNDARY = "frame"
BOUNDARY_BYTES = BOUNDARY.encode()
MAX_FRAME_BYTES = 3 * 1024 * 1024  # Guard against oversized uploads (≈3 MiB).

# Build the default connector once during startup. The connector is lightweight
# and simply stores preset metadata until a capture handle is requested.
connector: CameraConnector = build_default_connector()

# Map raw selection inputs (preset names, indices, URLs) to the registered
# preset key. This avoids creating duplicate transient presets for the same
# selection.
_dynamic_presets: Dict[str, str] = {}

INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>WearableDisplay Stream</title>
  <style>
    :root { color-scheme: dark; }
    body { font-family: system-ui, sans-serif; margin: 2rem; background: #0b1116; color: #f8fafc; }
    h1 { margin-bottom: 0.5rem; }
    p { max-width: 48rem; }
    video { display: block; width: min(100%, 640px); background: #111827; border-radius: 0.75rem; margin: 1rem 0; }
    .controls { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin: 0.75rem 0; }
    label { font-weight: 600; }
    select { padding: 0.5rem 0.75rem; border-radius: 0.5rem; border: 1px solid #1f2937; background: #111827; color: inherit; }
    button { margin-right: 0.5rem; padding: 0.6rem 1.2rem; border: none; border-radius: 0.5rem; font-size: 1rem; cursor: pointer; }
    button.primary { background: #2563eb; color: #f9fafb; }
    button.secondary { background: #1f2937; color: #e5e7eb; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    #status { margin-top: 1rem; font-weight: 600; }
  </style>
</head>
<body>
  <h1>WearableDisplay Stream</h1>
  <p>Allow camera access and click <strong>Start streaming</strong> to publish this device&apos;s camera to <code>/stream</code> on this server.</p>
  <div class="controls">
    <label for="camera">Camera</label>
    <select id="camera"></select>
    <button id="refresh" class="secondary" type="button">Refresh</button>
  </div>
  <video id="preview" autoplay playsinline muted></video>
  <div class="controls">
    <button id="start" class="primary" type="button">Start streaming</button>
    <button id="stop" class="secondary" type="button" disabled>Stop</button>
  </div>
  <p id="status">Idle</p>
  <script>
    (function () {
      const startBtn = document.getElementById('start');
      const stopBtn = document.getElementById('stop');
      const refreshBtn = document.getElementById('refresh');
      const cameraSelect = document.getElementById('camera');
      const statusEl = document.getElementById('status');
      const videoEl = document.getElementById('preview');
      let mediaStream = null;
      let socket = null;
      let intervalHandle = null;
      let busy = false;

      function setStatus(text) {
        statusEl.textContent = text;
      }

      async function populateCameras() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
          cameraSelect.innerHTML = '<option value=\"\">Camera enumeration not supported</option>';
          cameraSelect.disabled = true;
          refreshBtn.disabled = true;
          return;
        }
        try {
          const devices = await navigator.mediaDevices.enumerateDevices();
          const videoDevices = devices.filter((device) => device.kind === 'videoinput');
          const previousValue = cameraSelect.value;
          cameraSelect.innerHTML = '';
          const autoOption = document.createElement('option');
          autoOption.value = '';
          autoOption.textContent = videoDevices.length ? 'Auto select (system default)' : 'No cameras detected';
          cameraSelect.appendChild(autoOption);
          videoDevices.forEach((device, index) => {
            if (!device.deviceId) {
              return;
            }
            const option = document.createElement('option');
            option.value = device.deviceId;
            option.textContent = device.label || `Camera ${index + 1}`;
            cameraSelect.appendChild(option);
          });
          const restored = Array.from(cameraSelect.options).some((option) => option.value === previousValue);
          if (restored) {
            cameraSelect.value = previousValue;
          }
          const hasChoices = cameraSelect.options.length > 1;
          cameraSelect.disabled = !hasChoices;
          refreshBtn.disabled = false;
        } catch (error) {
          console.error('Failed to enumerate cameras', error);
          setStatus('Could not list cameras.');
        }
      }

      function stopStreaming() {
        if (intervalHandle) {
          window.clearInterval(intervalHandle);
          intervalHandle = null;
        }
        if (socket) {
          socket.close();
          socket = null;
        }
        if (mediaStream) {
          mediaStream.getTracks().forEach((track) => track.stop());
          mediaStream = null;
        }
        videoEl.srcObject = null;
        busy = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        setStatus('Idle');
      }

      async function startStreaming() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          setStatus('Camera API not supported in this browser.');
          return;
        }
        if (socket && socket.readyState === WebSocket.OPEN) {
          return;
        }
        setStatus('Requesting camera...');
        startBtn.disabled = true;

        const selectedId = cameraSelect.disabled ? '' : cameraSelect.value;
        const videoConstraints = selectedId
          ? { deviceId: { exact: selectedId } }
          : { facingMode: 'environment' };

        try {
          mediaStream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
        } catch (error) {
          console.error('Camera error', error);
          setStatus('Camera permission denied or unavailable.');
          startBtn.disabled = false;
          return;
        }

        videoEl.srcObject = mediaStream;

        const track = mediaStream.getVideoTracks()[0];
        const activeSettings = track.getSettings();

        await populateCameras();
        if (activeSettings.deviceId) {
          cameraSelect.value = activeSettings.deviceId;
          cameraSelect.disabled = false;
        }

        const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
        socket = new WebSocket(protocol + window.location.host + '/ingest');
        socket.binaryType = 'arraybuffer';

        socket.addEventListener('open', () => {
          setStatus('Streaming...');
          stopBtn.disabled = false;
        });

        socket.addEventListener('error', (event) => {
          console.error('WebSocket error', event);
          setStatus('WebSocket connection error.');
        });

        socket.addEventListener('close', () => {
          stopStreaming();
        });

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d', { alpha: false });

        function syncCanvasSize() {
          const width = videoEl.videoWidth || activeSettings.width || 640;
          const height = videoEl.videoHeight || activeSettings.height || 480;
          if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
          }
        }

        intervalHandle = window.setInterval(() => {
          if (!socket || socket.readyState !== WebSocket.OPEN) {
            return;
          }
          if (!mediaStream || busy) {
            return;
          }
          syncCanvasSize();
          busy = true;
          ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
          canvas.toBlob((blob) => {
            busy = false;
            if (!blob || !socket || socket.readyState !== WebSocket.OPEN) {
              return;
            }
            blob.arrayBuffer().then((buffer) => {
              if (socket && socket.readyState === WebSocket.OPEN) {
                socket.send(buffer);
              }
            }).catch((err) => {
              console.error('Failed to serialise frame', err);
            });
          }, 'image/jpeg', 0.7);
        }, 100);
      }

      startBtn.addEventListener('click', startStreaming);
      stopBtn.addEventListener('click', stopStreaming);
      refreshBtn.addEventListener('click', (event) => {
        event.preventDefault();
        populateCameras();
      });
      window.addEventListener('beforeunload', stopStreaming);

      populateCameras();
      if (navigator.mediaDevices) {
        if (typeof navigator.mediaDevices.addEventListener === 'function') {
          navigator.mediaDevices.addEventListener('devicechange', populateCameras);
        } else {
          navigator.mediaDevices.ondevicechange = populateCameras;
        }
      }
    })();
  </script>
</body>
</html>
"""


class StreamBuffer:
    """
    In-memory buffer that stores the most recent JPEG frame published by a client.
    """

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._frame: Optional[bytes] = None
        self._version = 0
        self._publishers = 0

    async def publish(self, frame: bytes) -> None:
        async with self._condition:
            self._frame = frame
            self._version += 1
            self._condition.notify_all()

    async def wait_for_frame(self, last_version: int, timeout: float) -> Tuple[int, bytes]:
        async with self._condition:
            await asyncio.wait_for(
                self._condition.wait_for(
                    lambda: self._version != last_version and self._frame is not None
                ),
                timeout,
            )
            assert self._frame is not None
            return self._version, self._frame

    def has_frame(self) -> bool:
        return self._frame is not None

    @property
    def publisher_count(self) -> int:
        return self._publishers

    def publisher_connected(self) -> None:
        self._publishers += 1

    def publisher_disconnected(self) -> None:
        if self._publishers:
            self._publishers -= 1


stream_buffer = StreamBuffer()


def _encode_mjpeg_chunk(frame: bytes) -> bytes:
    return (
        b"--" + BOUNDARY_BYTES + b"\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
    )


def _ensure_preset(selection: str) -> str:
    """
    Resolve the connector preset key for a given selection string.

    The selection may be an existing preset, a numeric camera index, or a URL.
    New presets are registered for selections that are not currently known.
    """
    if connector.has_preset(selection):
        return selection

    if selection in _dynamic_presets:
        return _dynamic_presets[selection]

    try:
        index = int(selection)
    except ValueError:
        preset = _register_stream_preset(selection)
    else:
        preset = _register_device_preset(index)

    _dynamic_presets[selection] = preset.key
    return preset.key


def _register_device_preset(index: int) -> CameraPreset:
    key = f"server-device-{index}"
    preset = CameraPreset(
        key=key,
        source=index,
        description=f"HTTP request camera index {index}.",
    )
    connector.register_preset(preset)
    return preset


def _register_stream_preset(url: str) -> CameraPreset:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    key = f"server-stream-{digest}"
    preset = CameraPreset(
        key=key,
        source=url,
        description=f"HTTP request stream URL ({url}).",
    )
    connector.register_preset(preset)
    return preset


def _select_preset(camera: Optional[str]) -> str:
    """
    Determine which preset to use for the request.

    Priority order:
    - `camera` query parameter
    - `CAMERA_PRESET` environment variable
    - Automatically registered `env-override` when `CAMERA_SOURCE` is set
    - The built-in `local-default` preset (first local camera device)
    """
    if camera:
        return _ensure_preset(camera)

    env_preset = os.getenv("CAMERA_PRESET")
    if env_preset:
        return _ensure_preset(env_preset)

    if os.getenv("CAMERA_SOURCE"):
        # `build_default_connector` already added/overrode this preset.
        return "env-override"

    return "local-default"


def _frame_stream(capture: cv2.VideoCapture) -> Generator[bytes, None, None]:
    """
    Yield encoded MJPEG frames from an open VideoCapture object.
    """
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            success, encoded = cv2.imencode(".jpg", frame)
            if not success:
                continue

            yield _encode_mjpeg_chunk(encoded.tobytes())
    finally:
        capture.release()


def _iter_local_candidates(preferred_key: Optional[str]) -> Iterable[str]:
    seen = set()

    def register(key: Optional[str]) -> Optional[str]:
        if not key:
            return None
        if not connector.has_preset(key) or key in seen:
            return None
        preset = connector.get_preset(key)
        if not isinstance(preset.source, int):
            return None
        seen.add(key)
        return key

    primary = register(preferred_key)
    if primary:
        yield primary

    for key in ("local-default", "local-fallback"):
        candidate = register(key)
        if candidate:
            yield candidate

    for preset in connector.list_presets():
        if preset.key in seen:
            continue
        if isinstance(preset.source, int):
            seen.add(preset.key)
            yield preset.key


def _open_capture_with_fallback(preset_key: str, allow_fallback: bool) -> Tuple[str, cv2.VideoCapture]:
    primary = connector.get_preset(preset_key)
    candidates = []
    seen = set()

    def append(key: str) -> None:
        if key in seen or not connector.has_preset(key):
            return
        seen.add(key)
        candidates.append(key)

    if allow_fallback and isinstance(primary.source, int):
        for key in _iter_local_candidates(preset_key):
            append(key)
    else:
        append(preset_key)

    if not candidates:
        append(preset_key)

    errors = []
    for key in candidates:
        try:
            capture = connector.open(key)
            return key, capture
        except RuntimeError as exc:
            preset = connector.get_preset(key)
            errors.append(f"{key} (source={preset.source}): {exc}")
            continue

    raise RuntimeError("; ".join(errors) if errors else f"Unable to open camera preset '{preset_key}'.")


async def _remote_frame_generator(initial_version: int, initial_frame: bytes) -> AsyncGenerator[bytes, None]:
    version = initial_version
    frame = initial_frame
    yield _encode_mjpeg_chunk(frame)

    while True:
        try:
            version, frame = await stream_buffer.wait_for_frame(version, timeout=30.0)
        except asyncio.TimeoutError:
            continue
        yield _encode_mjpeg_chunk(frame)


@app.get("/healthz", tags=["system"])
async def healthcheck() -> Dict[str, object]:
    return {
        "status": "ok",
        "publishers": stream_buffer.publisher_count,
        "has_frame": stream_buffer.has_frame(),
    }


@app.get("/presets", tags=["camera"])
def list_presets() -> Iterable[Dict[str, str]]:
    return [
        {
            "key": preset.key,
            "source": str(preset.source),
            "description": preset.description,
        }
        for preset in connector.list_presets()
    ]


@app.websocket("/ingest")
async def ingest_camera(websocket: WebSocket) -> None:
    await websocket.accept()
    stream_buffer.publisher_connected()

    try:
        while True:
            message = await websocket.receive()

            frame_bytes: Optional[bytes] = None
            data = message.get("bytes")
            if data is not None:
                frame_bytes = data
            else:
                text = message.get("text")
                if text is None:
                    continue

                if text == "ping":
                    await websocket.send_text("pong")
                    continue

                if text.startswith("data:image"):
                    try:
                        encoded = text.split(",", 1)[1]
                        frame_bytes = base64.b64decode(encoded)
                    except (IndexError, ValueError, base64.binascii.Error):
                        continue

            if frame_bytes is None:
                continue

            if len(frame_bytes) > MAX_FRAME_BYTES:
                # Ignore oversized frames to protect the server.
                continue

            await stream_buffer.publish(frame_bytes)
    except WebSocketDisconnect:
        pass
    finally:
        stream_buffer.publisher_disconnected()


@app.get("/stream", tags=["camera"])
async def stream_camera(
    camera: Optional[str] = Query(
        default=None,
        description="Preset key, device index, or remote stream URL.",
    )
) -> StreamingResponse:
    # Use the browser-provided stream unless an explicit local preset is requested.
    should_use_local_camera = bool(
        camera or os.getenv("CAMERA_PRESET") or os.getenv("CAMERA_SOURCE")
    )

    media_type = f"multipart/x-mixed-replace; boundary={BOUNDARY}"

    if should_use_local_camera:
        preset_key = _select_preset(camera)
        try:
            _, capture = _open_capture_with_fallback(preset_key, allow_fallback=not bool(camera))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        generator = _frame_stream(capture)
        return StreamingResponse(generator, media_type=media_type)

    try:
        version, frame = await stream_buffer.wait_for_frame(-1, timeout=5.0)
    except asyncio.TimeoutError:
        try:
            _, capture = _open_capture_with_fallback("local-default", allow_fallback=True)
        except (KeyError, RuntimeError) as exc:
            raise HTTPException(
                status_code=503,
                detail=f"No browser publisher and local camera unavailable ({exc}); ensure a camera is connected or start streaming from the UI.",
            ) from exc

        generator = _frame_stream(capture)
        return StreamingResponse(generator, media_type=media_type)

    generator = _remote_frame_generator(version, frame)
    return StreamingResponse(generator, media_type=media_type)


@app.get("/", tags=["system"], response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("stream_server:app", host=host, port=port, reload=False)
