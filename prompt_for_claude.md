# ToBeLess AI — Prompt for Claude Code

## Project Structure
- **Backend**: Flask, `C:\Users\ASUS\Desktop\ToBeLess\app.py`
- **Frontend**: React + Tailwind + TypeScript, `C:\Users\ASUS\Desktop\ToBeLess\pp\src\`
- The React app talks to the Flask backend via API calls (base URL configured in `pp/src/api/client.ts`)

---

## Task: Replace Detection Toggles with Two Radio Selector Boxes

### WHERE to edit in the React app

**Primary file**: `pp/src/components/dashboard/LiveDetectionView.tsx`

Currently, when the user clicks the ⚙️ Settings button (line ~267), a `showSettings` panel appears (lines 656–724) with 3 toggle switches: Weapon Detection, Fall Detection, Scream Detection.

**Replace** that entire Settings panel with **two selector card groups** that appear below the stats cards, always visible (not behind a settings button toggle, or you can keep the settings button but show both groups inside it).

---

### Box 1 — "Detection Mode" (pick ONE, radio-style)

**Label**: Detection Mode  
**Default**: `fight`  
**Options**:
- `fight` → 🥊 **Fight Detection**
- `weapon` → 🔫 **Weapon Detection**  
- `scream` → 🔊 **Scream Detection** — show a small note: *"Requires video upload (not live stream)"*

When changed: call `POST /set_detection_mode` with `{ "mode": "fight" | "weapon" | "scream" }`.

State: add `detectionMode: 'fight' | 'weapon' | 'scream'` to component state.  
On mount: call `GET /get_detection_mode` to load current mode (add this endpoint to backend too).

---

### Box 2 — "Privacy Mode" (pick ONE, radio-style)

**Label**: Privacy Mode  
**Default**: `off`  
**Options**:
- `off` → 🚫 **Off**
- `recognition` → 👤 **Face Recognition**
- `blur` → 👁️ **Face Blur**

When changed: call `POST /set_privacy_mode` with `{ "mode": "off" | "recognition" | "blur" }`.

State: add `privacyMode: 'off' | 'recognition' | 'blur'` to component state.  
On mount: call `GET /get_privacy_mode` to load current mode (add this endpoint to backend too).

---

### UI Style for the selector boxes

Use the existing dark card style (`bg-zinc-950 border-zinc-800` for dark, `bg-white border-purple-200` for light).

Each option should be a **pill/tab button**:
- Selected: `bg-purple-600 text-white`
- Unselected: `bg-transparent text-zinc-400 hover:text-white hover:bg-zinc-800`

Example structure for each group:
```tsx
<div className={`p-4 rounded-2xl border-2 ${theme === 'light' ? 'bg-white border-purple-200' : 'bg-zinc-950 border-zinc-800'}`}>
  <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-3">Detection Mode</h3>
  <div className="flex flex-col gap-1">
    {['fight', 'weapon', 'scream'].map(mode => (
      <button
        key={mode}
        onClick={() => handleDetectionModeChange(mode)}
        className={`px-3 py-2 rounded-xl text-sm font-medium text-left transition-all ${
          detectionMode === mode
            ? 'bg-purple-600 text-white'
            : theme === 'light'
            ? 'text-zinc-600 hover:bg-purple-50'
            : 'text-zinc-400 hover:bg-zinc-800'
        }`}
      >
        {modeLabels[mode]}
      </button>
    ))}
  </div>
</div>
```

---

### Also add API functions to `pp/src/api/detection.ts`

```ts
setDetectionMode: async (mode: 'fight' | 'weapon' | 'scream') => {
  return apiClient.post('/set_detection_mode', { mode });
},
getDetectionMode: async () => {
  return apiClient.get('/get_detection_mode');
},
setPrivacyMode: async (mode: 'off' | 'recognition' | 'blur') => {
  return apiClient.post('/set_privacy_mode', { mode });
},
getPrivacyMode: async () => {
  return apiClient.get('/get_privacy_mode');
},
```

---

### Backend changes (`app.py`)

Add global variables (near line 22):
```python
current_detection_mode = 'fight'   # 'fight' | 'weapon' | 'scream'
# face_blur_enabled and face_recognition_enabled already exist
```

Add 4 new Flask endpoints:

```python
@app.route('/set_detection_mode', methods=['POST'])
def set_detection_mode():
    global current_detection_mode
    data = request.get_json(silent=True) or {}
    mode = data.get('mode', 'fight')
    if mode not in ('fight', 'weapon', 'scream'):
        return jsonify({'success': False, 'error': 'Invalid mode'})
    current_detection_mode = mode
    return jsonify({'success': True, 'mode': mode})

@app.route('/get_detection_mode', methods=['GET'])
def get_detection_mode():
    return jsonify({'success': True, 'mode': current_detection_mode})

@app.route('/set_privacy_mode', methods=['POST'])
def set_privacy_mode():
    global face_recognition_enabled, face_blur_enabled
    data = request.get_json(silent=True) or {}
    mode = data.get('mode', 'off')
    if mode not in ('off', 'recognition', 'blur'):
        return jsonify({'success': False, 'error': 'Invalid mode'})
    face_recognition_enabled = (mode == 'recognition')
    face_blur_enabled = (mode == 'blur')
    return jsonify({'success': True, 'mode': mode,
                    'face_recognition_enabled': face_recognition_enabled,
                    'face_blur_enabled': face_blur_enabled})

@app.route('/get_privacy_mode', methods=['GET'])
def get_privacy_mode():
    mode = 'recognition' if face_recognition_enabled else ('blur' if face_blur_enabled else 'off')
    return jsonify({'success': True, 'mode': mode})
```

Also: in `processing_loop()`, read `current_detection_mode` and run only the selected detector:
- `'fight'` → existing hybrid fight detection (no change)
- `'weapon'` → skip fight detection, only run weapon detector
- `'scream'` → for live streams: skip silently; for file uploads: extract MP3 from video using `subprocess.run(['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path, '-y'])` then analyze with `ScreamDetector`

---

### DO NOT modify
- `face_recognizer.py`
- `face_blur.py`  
- `hybrid_fight_detector.py`
- `scream_detector.py`
- `templates/detection.html` (legacy file, not used by React app)
