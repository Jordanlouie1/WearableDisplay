# WearableDisplay

Turn on Continuity Camera on Iphone in settings>General>Airplay&Continuity> Continuity camera

## HTTP Stream Server

Expose the active camera feed over HTTP with:

```bash
uv run uvicorn stream_server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` on the device whose camera you want to share,
grant permission, choose the camera from the dropdown (use **Refresh** after
plugging in a new one), and click **Start streaming**. Viewers can then load
`http://localhost:8000/stream` to watch the MJPEG feed (optionally append
`?camera=...` to force a local OpenCV preset instead).

Set `DEFAULT_CAMERA_INDEX` when you need the server-side presets to default to
something other than camera index `1`; the service falls back to the opposite
index automatically when the preferred device is unavailable. The `/stream`
endpoint now serves frames only from browsers connected through the localhost
UI, so keep the ingest tab open while streaming.
