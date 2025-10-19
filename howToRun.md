# How to Run the WearableDisplay Camera Stream

Follow the steps below to prepare the Python environment with `uv` and launch the camera streaming utility (`VideoStream.py`).

## 1. Prerequisites

- Python 3.9 or newer (3.12 recommended).
- [`uv`](https://docs.astral.sh/uv/) command-line tool. Install it once with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- macOS, Linux, or Windows host with a webcam (or reachable RTSP/HTTP stream).
- Ensure the camera is not in use by other software.

## 2. Install Dependencies (one time per machine)

From the repository root:

```bash
uv sync
```

This command:

- Creates a virtual environment in `.venv/`.
- Installs the pinned dependencies listed in `pyproject.toml` / `uv.lock` (`opencv-python` and its requirements).
- Installs the project itself (`wearable-display`) in editable mode so local changes are immediately available.

If you are re-running the setup and want to enforce the current lock file without refreshing it, use:

```bash
uv sync --frozen
```

## 3. Activate the Virtual Environment

After the initial `uv sync`, activate the environment before running scripts:

- **Linux/macOS**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (PowerShell)**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```

To leave the environment, run `deactivate`.

> Tip: You can also prefix commands with `uv run ...` to execute them inside the synced environment without activating it manually.

## 4. Running the HTTP Stream Server

Launch the FastAPI server to expose the camera feed on port `8000`:

```bash
uv run uvicorn stream_server:app --host 0.0.0.0 --port 8000
```

Key workflow:

- Visit `http://localhost:8000` from the device that should provide the camera
  feed. Allow camera access, pick the desired camera from the dropdown (use
  **Refresh** if you plugged in a new one), then press **Start streaming** to
  push frames to the server over WebSocket.
- Open `http://localhost:8000/stream` (from any device on the same network) to
  view the MJPEG stream sourced from the last publisher.
- If no browser is pushing frames, `/stream` returns `503` until a publisher is
  connected, ensuring all capture flows originate from the localhost UI.
- Optional: append `?camera=<preset|index|url>` to `/stream` when you want the
  server itself to read from a local/remote OpenCV source instead of using a
  browser publisher. Known presets are listed at `/presets`.
- `GET /healthz` – Lightweight health check for deployment scripts.

Stop the server with `Ctrl+C`. You can also run `python stream_server.py` to
start the server with the same defaults when you prefer a Python entry point.

## 5. Running `VideoStream.py`

Basic invocation (assumes an available local webcam at index `0`):

```bash
uv run python VideoStream.py
```

Key options:

- `--list-presets` – Print the known camera presets and exit.
- `--probe` (with optional `--max-probe N`) – Check which local camera indices are available.
- `--camera <preset|index|url>` – Force a specific preset, device index, or remote stream URL.
- `--output path/to/file.mp4` – Record the stream to an MP4 file at 20 FPS (defaults to `output.mp4`).
- `--no-output` – Disable video recording entirely.
- `--display` / `--no-display` – Show (or suppress) a window preview of the frames. Default is `--no-display` for headless environments.

Example: show a preview and record to disk:

```bash
uv run python VideoStream.py --display --output captures/session.mp4
```

## 6. Environment Variable Overrides

- `CAMERA_PRESET` – Set a preset key before launching to select a predefined camera.
- `CAMERA_SOURCE` – Override the capture source entirely. Use an integer (e.g., `1`) for a local device or a URL for a remote stream. When set, `VideoStream.py` automatically creates an `env-override` preset.
- `DEFAULT_CAMERA_INDEX` – Preferred local camera index for `local-default` (defaults to `1`). The connector automatically adds `local-fallback` to try the opposite index when the preferred device is unavailable.

Examples:

```bash
CAMERA_SOURCE=1 uv run python VideoStream.py
CAMERA_SOURCE="rtsp://192.168.1.50/stream" uv run python VideoStream.py
```

## 7. Troubleshooting

- **No frame output / “Unable to open camera”**: Verify the device index (`--probe`), confirm permissions, and ensure the camera is not used elsewhere.
- **High CPU usage**: Reduce resolution via camera driver controls or switch to a lower FPS stream.
- **GUI window not opening**: Some headless servers require `--display` to be omitted or running under a virtual display (`xvfb-run` on Linux).

You're ready to stream video for the WearableDisplay prototype!
