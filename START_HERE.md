# ToBeLess AI - Quick Start Guide

## FIRST TIME SETUP

### Step 1: Install Backend Dependencies
**Double-click:** `install_venv.bat`

This will install all Python packages in your virtual environment.

---

## RUNNING THE APPLICATION

You need to run **TWO PROGRAMS** at the same time:

### Step 2A: Start Backend (Flask Server)
**Double-click:** `run_backend.bat`

- This starts the AI detection server on port 8080
- Keep this window open while using the app
- You should see: "Running on http://0.0.0.0:8080"

### Step 2B: Start Frontend (React UI)
**Go to the `pp` folder and double-click:** `run_frontend.bat`

- This starts the web interface on port 5173
- A browser window should open automatically
- If not, open: http://localhost:5173

---

## MANUAL COMMANDS (If batch files don't work)

### Backend:
```powershell
cd "C:\Users\Huawei\OneDrive\Рабочий стол\ToBeLess"
venv\Scripts\activate
pip install flask-cors
pip install -r requirements.txt
python app.py
```

### Frontend (in a NEW terminal):
```powershell
cd "C:\Users\Huawei\OneDrive\Рабочий стол\ToBeLess\pp"
npm install
npm run dev
```

---

## TROUBLESHOOTING

### "ModuleNotFoundError: No module named 'flask_cors'"
- Run `install_venv.bat` again
- Or manually: `venv\Scripts\activate` then `pip install flask-cors`

### "Haar cascade path error" (Cyrillic characters)
- This is NOT critical - YOLO face detection is working
- The app will run fine without Haar cascade

### Port already in use
- **Backend (8080):** Stop any other Python apps
- **Frontend (5173):** Stop any other Vite/Node apps

### Cannot connect to backend
- Make sure BOTH backend and frontend are running
- Check that backend shows "Running on http://0.0.0.0:8080"
- Frontend should be on http://localhost:5173

---

## USING THE APP

1. Open browser to: http://localhost:5173
2. You'll see the landing page
3. Click to enter the Dashboard
4. Go to "Monitoring" section
5. Click "Add Camera" button
6. Choose your video source:
   - **Webcam**: Enter "0" for default camera
   - **Video File**: Upload an MP4 file
   - **RTSP**: Enter RTSP stream URL
7. Toggle features:
   - Fight Detection (always on)
   - Face Recognition
   - Face Blur
8. Click "Start Stream"

---

## NOTES

- The Haar cascade encoding error is harmless (YOLO is being used instead)
- Face recognition requires registered faces in `faces/images/` folder
- Telegram alerts require TG_BOT_TOKEN and TG_CHAT_ID in .env file
