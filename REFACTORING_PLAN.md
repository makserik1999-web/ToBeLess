# app.py Refactoring Plan

Current state: **1 675-line monolith**.
Goal: split into focused modules while keeping `app.py` as a thin entry point.

---

## Proposed File Layout

```
ToBeLess/
├── app.py                   # Entry point: create_app() factory + __main__
├── config.py                # All constants and tunable parameters
├── state.py                 # Shared mutable globals + threading locks
├── core/
│   ├── __init__.py
│   ├── pipeline.py          # processing_loop() — the main video loop
│   ├── stream_manager.py    # VideoCapture lifecycle (open / close / reconnect)
│   └── alerts.py            # Telegram alert helpers + cooldown logic
└── routes/
    ├── __init__.py
    ├── stream_routes.py     # /start_stream  /stop_stream  /video_feed  /stats_stream
    ├── face_routes.py       # /add_face  /reload_faces  /toggle_face_blur  /toggle_face_recognition  /feature_status
    ├── analytics_routes.py  # /analytics  /heatmap  /hotspots  /detection_events  /clear_events
    ├── report_routes.py     # /generate_report  /download_report  /list_reports
    ├── chatbot_routes.py    # /chatbot  /chatbot/send
    └── settings_routes.py   # /settings  /toggle_*  /detection_status  /profiling_stats  /toggle_profiling
```

---

## Step-by-Step Migration

### Step 1 — Extract `config.py`

Move every top-level constant out of `app.py`.
No imports needed other than `pathlib.Path`.

```python
# config.py
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
FACES_DIR  = Path("faces/images")
REPORTS_DIR = Path("reports")

# ── Stream / detection ───────────────────────────────────────────────────────
ALERT_COOLDOWN          = 8      # seconds between Telegram alerts
ANALYTICS_SNAPSHOT_SIZE = 300   # items returned by /analytics
SKIP_FRAMES             = 2      # YOLO runs every Nth frame
RESIZE_WIDTH            = 416    # frame width fed to YOLO
SSE_INTERVAL            = 0.5   # seconds between /stats_stream pushes

# ── Profiler ─────────────────────────────────────────────────────────────────
PROFILING_ENABLED = True
PROFILE_INTERVAL  = 30          # print stats every N frames

# ── Detector defaults ────────────────────────────────────────────────────────
BODY_PROXIMITY_THRESHOLD = 70.0
LIMB_PROXIMITY_THRESHOLD = 25.0
FIGHT_HOLD_DURATION      = 60

# ── Feature toggles (runtime state lives in state.py) ───────────────────────
DEFAULT_WEAPON_DETECTION = False
DEFAULT_FALL_DETECTION   = False
DEFAULT_SCREAM_DETECTION = False
```

**Migration**: search `app.py` for every bare constant and replace with `from config import …`.

---

### Step 2 — Extract `state.py`

Move all global mutable variables and their locks into one place.
Every module that needs shared state imports from here — no more `global` keyword spread across the file.

```python
# state.py
import threading
from collections import deque
from config import ANALYTICS_SNAPSHOT_SIZE

# ── Detector instances ────────────────────────────────────────────────────────
detector        = None   # HybridFightDetector
weapon_detector = None
fall_detector   = None
scream_detector = None

# ── Video / stream ────────────────────────────────────────────────────────────
video_cap     = None
proc_thread   = None
stream_active = False
current_frame = None
frame_lock    = threading.Lock()

# ── Feature toggles ───────────────────────────────────────────────────────────
face_blur_enabled        = False
face_recognition_enabled = False
weapon_detection_enabled = False
fall_detection_enabled   = False
scream_detection_enabled = False

# ── Analytics ─────────────────────────────────────────────────────────────────
analytics_buffer   = deque(maxlen=4000)
latest_stats       = {
    'people': 0, 'fights': 0, 'weapons': 0,
    'falls': 0, 'screams': 0, 'fps': 0,
    'confidence': 0.0, 'timestamp': None,
}
latest_stats_lock  = threading.Lock()

# ── Alert cooldowns ───────────────────────────────────────────────────────────
last_alert_time        = 0.0
last_weapon_alert_time = 0.0
last_fall_alert_time   = 0.0
last_scream_alert_time = 0.0

# ── Batch jobs ────────────────────────────────────────────────────────────────
JOBS      = {}
JOBS_LOCK = threading.Lock()

# ── Events (for report generation) ───────────────────────────────────────────
detection_events      = []
detection_events_lock = threading.Lock()
```

**Migration**:
1. Delete the corresponding lines from `app.py`.
2. Add `import state` (or `from state import …`) at the top of every module that uses them.
3. Replace `global foo; foo = x` with `state.foo = x`.

---

### Step 3 — Extract `core/alerts.py`

The `_send_alert_nonblocking` helper currently lives inside `app.py`.
Move it alongside the Telegram bot import.

```python
# core/alerts.py
import threading

try:
    from bot import send_alert, send_photo
except Exception:
    def send_alert(*a, **k): pass
    def send_photo(*a, **k): pass


def send_nonblocking(text: str, frame_path: str | None = None, caption: str | None = None):
    """Fire-and-forget Telegram alert (spawns a daemon thread)."""
    def _job():
        try:
            send_alert(text)
        except Exception:
            pass
        if frame_path:
            try:
                send_photo(frame_path, caption or "")
            except Exception:
                pass
    threading.Thread(target=_job, daemon=True).start()
```

**Migration**: replace every call to `_send_alert_nonblocking(...)` in `pipeline.py` with `alerts.send_nonblocking(...)`.

---

### Step 4 — Extract `core/pipeline.py`

Move `processing_loop()`, `frame_generator()`, `_create_loading_frame()`, and the `PerformanceProfiler` / `TimingContext` classes.

```python
# core/pipeline.py
"""Main video processing loop and MJPEG frame generator."""

import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

import state
from config import (
    SKIP_FRAMES, RESIZE_WIDTH, ALERT_COOLDOWN,
    PROFILING_ENABLED, PROFILE_INTERVAL,
)
from core.alerts import send_nonblocking


class PerformanceProfiler:
    # … (move verbatim from app.py) …
    pass


class TimingContext:
    # … (move verbatim from app.py) …
    pass


profiler = PerformanceProfiler()


def _create_loading_frame() -> np.ndarray:
    # … (move verbatim from app.py) …
    pass


_loading_frame = _create_loading_frame()


def frame_generator():
    """MJPEG generator: yields JPEG boundary frames forever."""
    while True:
        with state.frame_lock:
            frm = state.current_frame.copy() if state.current_frame is not None else None
        if frm is None:
            if state.stream_active:
                frm = _loading_frame
            else:
                time.sleep(0.1)
                continue
        try:
            _, buf = cv2.imencode(".jpg", frm)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        except Exception:
            time.sleep(0.03)


def processing_loop(source_is_file: bool = False, job_id: str | None = None):
    """Main loop: read frames → detect → alert → push to buffer."""
    # … move the existing processing_loop() body here …
    # Replace all bare global reads/writes with state.xxx
    pass
```

Key transformation rules when moving the loop body:
- `global detector, video_cap, …` → delete; access via `state.detector`, `state.video_cap`, etc.
- `_send_alert_nonblocking(…)` → `alerts.send_nonblocking(…)`
- `SKIP_FRAMES`, `ALERT_COOLDOWN`, etc. → imported from `config`

---

### Step 5 — Extract `core/stream_manager.py`

The "open a video source" logic currently sits inside `start_stream()`.
Isolating it makes it testable and reusable.

```python
# core/stream_manager.py
"""Open, validate, and close video capture sources."""

import cv2
from video_source import RealTimeCapture


def open_source(source: str | int, source_is_file: bool) -> cv2.VideoCapture:
    """
    Return an opened VideoCapture-compatible object.

    Parameters
    ----------
    source : str or int
        File path, RTSP URL, or webcam index.
    source_is_file : bool
        True → plain cv2.VideoCapture (supports loop-back).
        False → RealTimeCapture with auto-reconnect.

    Raises
    ------
    RuntimeError if the source cannot be opened.
    """
    if source_is_file:
        cap = cv2.VideoCapture(str(source))
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
    else:
        src = int(source) if isinstance(source, str) and source.isdigit() else source
        cap = RealTimeCapture(src, reconnect=True)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source!r}")
    return cap


def release(cap) -> None:
    """Safely release a capture object."""
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
```

---

### Step 6 — Extract route Blueprints

Create a `routes/` package. Each file registers a Flask `Blueprint`.
`app.py` imports and registers all blueprints — one line per group.

#### `routes/stream_routes.py` (skeleton)

```python
# routes/stream_routes.py
import time, uuid, threading, traceback
from pathlib import Path

from flask import Blueprint, Response, jsonify, request
from werkzeug.utils import secure_filename

import state
from config import UPLOAD_DIR, SSE_INTERVAL, SKIP_FRAMES
from core import pipeline, stream_manager

bp = Blueprint("stream", __name__)


@bp.route("/start_stream", methods=["POST"])
def start_stream():
    # … move body from app.py verbatim, replacing globals with state.xxx …
    pass


@bp.route("/stop_stream", methods=["POST"])
def stop_stream():
    # … move body verbatim …
    pass


@bp.route("/video_feed")
def video_feed():
    return Response(
        pipeline.frame_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@bp.route("/stats_stream")
def stats_stream():
    import json, time
    def gen():
        while True:
            with state.latest_stats_lock:
                payload = dict(state.latest_stats)
            yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(SSE_INTERVAL)
    return Response(gen(), mimetype="text/event-stream")
```

#### `routes/face_routes.py` (skeleton)

```python
# routes/face_routes.py
import time
from pathlib import Path

import cv2
from flask import Blueprint, jsonify, request
from werkzeug.utils import secure_filename

import state
from config import UPLOAD_DIR

bp = Blueprint("face", __name__)


def _get_face_rec():
    """Helper: navigate the detector hierarchy to the FaceRecognizer."""
    d = state.detector
    if d is None:
        return None
    return (
        d.pose_detector.face_rec
        if hasattr(d, "pose_detector")
        else getattr(d, "face_rec", None)
    )


@bp.route("/add_face", methods=["POST"])
def add_face():
    # … move body verbatim …
    pass


@bp.route("/reload_faces", methods=["POST"])
def reload_faces():
    # … move body verbatim …
    pass


@bp.route("/toggle_face_blur", methods=["POST"])
def toggle_face_blur():
    data = request.get_json(silent=True) or request.form or request.values
    state.face_blur_enabled = str(data.get("enabled", "false")).lower() == "true"
    return jsonify({"success": True, "face_blur_enabled": state.face_blur_enabled})


@bp.route("/toggle_face_recognition", methods=["POST"])
def toggle_face_recognition():
    data = request.get_json(silent=True) or request.form or request.values
    state.face_recognition_enabled = str(data.get("enabled", "false")).lower() == "true"
    return jsonify({"success": True, "face_recognition_enabled": state.face_recognition_enabled})


@bp.route("/feature_status")
def feature_status():
    return jsonify({
        "success": True,
        "face_blur_enabled": state.face_blur_enabled,
        "face_recognition_enabled": state.face_recognition_enabled,
    })
```

#### Remaining blueprints follow the same pattern:
- `routes/analytics_routes.py` → `/analytics`, `/heatmap`, `/hotspots`, `/detection_events`, `/clear_events`
- `routes/report_routes.py` → `/generate_report`, `/download_report`, `/list_reports`
- `routes/chatbot_routes.py` → `/chatbot`, `/chatbot/send`
- `routes/settings_routes.py` → `/settings`, `/toggle_*`, `/detection_status`, `/profiling_stats`, `/toggle_profiling`

---

### Step 7 — Slim down `app.py`

After the above steps `app.py` becomes a thin factory:

```python
# app.py  (final — ~50 lines)
from pathlib import Path

from flask import Flask, render_template
from flask_cors import CORS

from config import UPLOAD_DIR, FACES_DIR, REPORTS_DIR
from routes.stream_routes    import bp as stream_bp
from routes.face_routes      import bp as face_bp
from routes.analytics_routes import bp as analytics_bp
from routes.report_routes    import bp as report_bp
from routes.chatbot_routes   import bp as chatbot_bp
from routes.settings_routes  import bp as settings_bp


def create_app() -> Flask:
    # Ensure directories exist
    for d in (UPLOAD_DIR, FACES_DIR, REPORTS_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)

    app = Flask(__name__, static_folder="static", template_folder="templates")

    CORS(app, resources={
        r"/*": {
            "origins": ["http://localhost:5173", "http://localhost:3000"],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"],
            "supports_credentials": True,
        }
    })

    # Page routes (kept here — they're trivial)
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/detection")
    def detection():
        return render_template("detection.html")

    # Register API blueprints
    app.register_blueprint(stream_bp)
    app.register_blueprint(face_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(report_bp)
    app.register_blueprint(chatbot_bp)
    app.register_blueprint(settings_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8080, debug=True)
```

---

## Recommended Execution Order

| # | Action | Risk |
|---|--------|------|
| 1 | Create `config.py`, update imports in `app.py` | Zero — constants only |
| 2 | Create `state.py`, update `global` → `state.xxx` in `app.py` | Low |
| 3 | Extract `core/alerts.py` | Low |
| 4 | Extract `core/stream_manager.py` | Low |
| 5 | Extract `core/pipeline.py` (loop + frame generator) | Medium — most complex |
| 6 | Extract route blueprints one at a time, test after each | Medium |
| 7 | Replace `app.py` body with `create_app()` factory | Low (final cleanup) |

Always run `python app.py` and hit `/detection` after each step to confirm nothing broke before moving to the next.
