# WearableDisplay

Prototype tooling for gesture-driven camera capture and streaming.

## Prerequisites

- Python 3.9 or newer (3.12 recommended).
- [uv](https://docs.astral.sh/uv/) for dependency management, or any Python virtual
  environment workflow.
- A webcam (USB or Continuity Camera) and optionally a browser that can publish video to
  `http://localhost:8000`.

## Setup

Create and populate a virtual environment with the project dependencies:

```bash
# Preferred: uv handles the virtual environment automatically
uv sync

# Or using plain Python tooling
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All Ultralytics YOLO weights are downloaded automatically the first time you run either
script (no model files are stored in the repository).

## Run the HTTP Stream Server

Launch the server that receives frames from a browser and rebroadcasts them over HTTP:

```bash
uv run uvicorn stream_server:app --host 0.0.0.0 --port 8000
```

Then:

1. Open `http://localhost:8000` in a browser on the device whose camera you want to share.
2. Allow camera access, choose the desired device, and press **Start streaming**.
3. View the feed from any machine on the network at `http://localhost:8000/stream`.
   Append `?camera=<preset|index|url>` if you prefer the server to open an OpenCV source
   directly instead of ingesting from the browser.

Environment tips:

- Set `DEFAULT_CAMERA_INDEX` to change the fallback USB camera index when the browser is
  not publishing frames.
- Keep the ingest tab open while streaming; `/stream` returns `503` if nothing is pushing
  frames.

## Run the Integrated Vision App

`integrated_app.py` combines YOLO object detection, MediaPipe hand gestures, and Gemini
responses. It prefers the localhost stream first; if no stream is active it falls back
to available USB cameras.

```bash
uv run python integrated_app.py
```

Useful options:

- `--camera` – Force a specific source (device index or URL).
- `--weights` – Point to a custom YOLO weights file (default `yolov8n.pt`; downloaded on
  demand).
- `--no-mirror` – Disable the mirrored preview.
- `GOOGLE_API_KEY` – Export this environment variable so pinch gestures can send captured
  frames to Gemini and print a short description.

Press `q` to exit the preview window.

## Run the CLI Camera Tool

`VideoStream.py` exposes a lightweight CLI for camera capture with YOLO overlays. Like
the integrated app it will download `yolov8n.pt` automatically if it is missing.

```bash
uv run python VideoStream.py --display
```

Flags of note:

- `--camera` – Preset name, device index, or URL handled by the camera connector.
- `--weights` – Alternate YOLO weights path.
- `--output` / `--no-output` – Record the stream to disk (default `output.mp4`).
- `--list-presets` and `--probe` – Discover connector presets or local device indices.

## Continuity Camera on iOS

To use an iPhone as a webcam on macOS, enable **Settings → General → AirPlay & Continuity
→ Continuity Camera**, then select the phone as the camera source in the browser or
integrated app.
