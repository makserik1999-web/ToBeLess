# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ToBeLess AI is a real-time violence detection system that combines pose estimation, face recognition, and Telegram notifications. The system can process webcam streams, video files, and RTSP feeds to detect fights and identify people involved.

**Tech Stack**: Flask web server, YOLO v8 (pose & face detection), OpenCV, PyTorch (CPU)

## Development Commands

### Running the Application
```bash
python app.py
# Server runs on http://0.0.0.0:8080
# Main UI: http://localhost:8080/detection
```

### Dependencies
```bash
pip install -r requirements.txt
# Key packages: Flask, ultralytics, opencv-python, torch, numpy, pandas
```

### Testing Individual Components
```bash
# Test face detection/recognition
python test_detector.py

# Test face identification
python test_identify.py

# Diagnostic tool
python diagnostic_tool.py
```

### Face Database Management
Place face images in `faces/images/` folder with filename format: `PersonName_1.jpg` → "PersonName"
The system auto-loads faces on startup from this folder into `faces/embeddings.json`.

## Architecture

### Core Processing Pipeline

**Entry Point**: `app.py` (989 lines)
- Flask server manages multiple concurrent routes
- Background thread runs `processing_loop()` (lines 604-717)
- Thread-safe frame buffer with locks for concurrent access

**Processing Flow**:
1. `VideoCapture.read()` → Raw frame
2. `FightDetector.process_frame()` (lines 328-558)
   - YOLO pose inference (17 keypoints per person)
   - Face recognition via `FaceRecognizer.identify()`
   - Optional face blur via `blur_faces()`
   - Fight detection algorithm with confidence scoring
   - Skeleton drawing and annotation
3. Push metrics to `analytics_buffer` (deque, maxlen=4000)
4. Send Telegram alerts if fight detected (cooldown: 8 seconds)
5. Update `current_frame` for MJPEG streaming

### Fight Detection Algorithm (FightDetector class, lines 75-558)

The detector uses pose estimation to identify violent interactions:

**Key Metrics**:
- **Body Proximity**: Distance between torso centers (threshold: 120px)
- **Limb Crossings**: Skeleton line intersections using CCW test
- **Close Contacts**: Limb proximity (wrists, elbows, knees, ankles < 50px)

**Confidence Scoring** (0-100%):
- Body distance: 40 points max
- Limb crossings: 30 points max (20 pts per crossing)
- Close contacts: 30 points max (10 pts per contact)

**Pose Smoothing**:
- Uses exponential moving average across last 3 frames
- Filters poses with <8 visible keypoints (occlusion handling)
- Minimum pose confidence: 0.5

**Fight State Management**:
- `fight_hold_duration`: 60 frames (prevents flickering)
- Maintains fight state for 60 frames after detection ends
- Stores fight events in `analytics` dict

### Face Recognition Module (face_recognizer.py, 306 lines)

**Detection Strategy** (cascading fallback):
1. YOLO v8n-face (primary, most accurate)
2. cv2.dnn Caffe SSD (secondary)
3. Haar Cascade (fallback)
- NMS with IoU threshold: 0.4

**Embedding Pipeline**:
- Resize face to 112×112 grayscale
- Apply CLAHE (clipLimit=2.0, tileGrid=(8,8))
- Gaussian blur for noise reduction
- L2-normalize to unit vector

**Recognition**:
- Cosine distance matching (threshold: 0.48)
- Temporal tracking with exponential smoothing
- Auto-cleanup of stale tracking data (>30 frames)
- Database: JSON-based (`faces/embeddings.json`) + in-memory cache

### Real-time Streaming

**MJPEG Stream** (`/video_feed`):
- `frame_generator()` yields JPEG frames with boundary markers
- MIME type: `multipart/x-mixed-replace; boundary=frame`

**Server-Sent Events** (`/stats_stream`):
- Pushes stats JSON every 0.5 seconds
- Fields: `people`, `fights`, `fps`, `confidence`, `timestamp`

**Analytics** (`/analytics`):
- Returns recent 300 snapshots from `analytics_buffer`
- Includes cumulative detector analytics
- Latest stats snapshot

## Important Configuration

### Detector Parameters (app.py lines 43-53, 115-117)
```python
ALERT_COOLDOWN = 8              # Telegram alert rate limit (seconds)
ANALYTICS_SNAPSHOT_SIZE = 300   # Max recent analytics stored
SKIP_FRAMES = 1                 # Process every frame (1 = no skipping)
RESIZE_WIDTH = 640              # Input frame width for YOLO
SSE_INTERVAL = 0.5              # Stats push interval (seconds)

body_proximity_threshold = 120.0    # Pixels
limb_proximity_threshold = 50.0     # Pixels
fight_hold_duration = 60            # Frames
min_pose_confidence = 0.5           # Minimum pose confidence
```

Adjustable via `/settings` POST endpoint.

### Face Recognition Parameters (face_recognizer.py)
```python
vec_size = (112, 112)           # Embedding dimensions
threshold = 0.48                # Cosine distance threshold for match
```

### Required Models
Place in project root:
- `yolov8n-pose.pt` (6.5 MB) - Pose estimation (17 COCO keypoints)
- `yolov8n-face.pt` (6.2 MB) - Face detection
- `yolov8n.pt` (6.5 MB) - Object detection (optional)
- `yolov8s.pt` (22.5 MB) - Larger object detector (optional)

## Key API Routes

### Stream Control
- `POST /start_stream` - Start processing (file upload, RTSP, or webcam index)
- `POST /stop_stream` - Stop processing
- `GET /video_feed` - MJPEG stream
- `GET /stats_stream` - SSE stats stream

### Face Management
- `POST /add_face` - Register single face (file + name)
- `POST /reload_faces` - Bulk load from `faces/images/`
- `POST /toggle_face_blur` - Enable/disable face blur
- `POST /toggle_face_recognition` - Enable/disable recognition
- `GET /feature_status` - Check feature states

### Configuration & Data
- `POST /settings` - Update detector thresholds
- `GET /analytics` - Get analytics data
- `GET /job/<job_id>` - Poll batch job status
- `GET /uploads/<filename>` - Serve saved alerts/results

## Thread Safety

**Locks** (app.py lines 569-576):
- `frame_lock` - Protects `current_frame`
- `latest_stats_lock` - Protects `latest_stats` dict
- `JOBS_LOCK` - Protects `JOBS` dict

Always acquire locks before modifying shared state.

## Telegram Integration (bot.py, 61 lines)

Requires environment variables:
- `TG_BOT_TOKEN` - Bot token from @BotFather
- `TG_CHAT_ID` - Target chat/channel ID

Functions:
- `send_alert(text)` - Text message
- `send_photo(photo_bytes, caption)` - Photo upload

Alerts are non-blocking (spawned in daemon threads).

## Directory Structure

```
ToBeLess/
├── app.py                    # Main Flask application
├── face_recognizer.py        # Face detection & recognition
├── face_blur.py              # Face blurring module
├── bot.py                    # Telegram integration
├── templates/
│   ├── index.html            # Landing page
│   └── detection.html        # Main monitoring UI
├── static/
│   ├── js/script.js          # Frontend (FDApp class)
│   └── css/style.css         # Styling
├── faces/
│   ├── images/               # Face photos for registration
│   └── embeddings.json       # Persistent face database
├── uploads/                  # Alerts, results, temp files
└── *.pt                      # YOLO models
```

## Code Patterns

### Adding New Detection Features

When modifying fight detection logic:
1. Update metrics calculation in `detect_fight()` (lines 260-314)
2. Adjust confidence scoring formula (lines 298-304)
3. Consider pose smoothing impact (`_smooth_keypoints()`, lines 140-160)
4. Test with occlusion handling (`_is_heavily_occluded()`, lines 190-201)

### Modifying Face Recognition

When changing face matching:
1. Embedding generation: `_extract_embedding()` in face_recognizer.py
2. Distance threshold: `threshold` parameter in FaceRecognizer constructor
3. Temporal tracking: lines 449-479 in app.py

### Adding API Endpoints

Follow Flask route pattern:
1. Add route decorator with method
2. Use `request.get_json(silent=True) or request.form or request.values` for flexible input
3. Return `jsonify({'success': bool, ...})`
4. Handle exceptions with try/except and return error messages

## Known Issues & Gotchas

- **File Uploads**: Use `secure_filename()` from werkzeug for safety
- **RTSP Streams**: May need buffer size adjustment (`CAP_PROP_BUFFERSIZE = 1`)
- **Face Recognition**: Requires good lighting and frontal faces for accuracy
- **Fight Detection**: Simplified version (recent revert to simpler algorithm per git history)
- **Thread Safety**: Always use locks when modifying shared state
- **Model Loading**: YOLO models auto-download if missing, but pre-download recommended

## Git History Notes

Recent commits show focus on stability:
- `316a516` - Revert fight detection to simpler version (current)
- `e91fa1f` - Complete face detection and fight detection improvements

Current approach prioritizes reliability over complexity.
