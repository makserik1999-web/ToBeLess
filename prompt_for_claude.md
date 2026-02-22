# Prompt for Implementing High-Quality Face Recognition and Project Architecture

Hi Claude! We are developing a video analytics security system called ToBeLess. Currently, our hybrid fight detector (YOLO-Pose + SlowFast) and RTSP stream handling are working well.

Our next step is to implement **high-quality face recognition and face blurring**, as well as to lay down the architectural foundation for future features (like weapon detection, audio analytics, SaaS platform features).

## Task 1: Face Recognition and Blurring
The current implementation in `face_recognizer.py` uses a custom pixel-vectorization method (cv2.dnn / Haar with pixel brightness arrays), which is inefficient and inaccurate. We need to rewrite the `face_recognizer.py` and `face_blur.py` modules using the **InsightFace** library (ArcFace + RetinaFace) or **dlib (face_recognition)** as a fallback.

**Code Requirements:**
1. Use `insightface` (preferred) to get accurate facial embeddings and bounding boxes.
2. The logic must be strictly encapsulated within `face_recognizer.py` and `face_blur.py` (do not clutter `app.py`). `app.py` should only import the utilities and pass the frame to them.
3. Ensure embedding caching (as currently implemented in `embeddings.json`) so we don't recalculate database vectors on every startup.
4. Add a "Privacy Mode" feature: the ability to toggle blurring for unknown faces (faces not in the database) or all faces.

## Task 2: Optimizing app.py (Architecture Refactoring)
Currently, `app.py` is a monolith of over 1600 lines. Before adding new features, please outline a plan and provide the basic code structure for separating the Flask server logic, RTSP streaming logic, and analytics pipeline into different files (e.g., `server.py`, `video_pipeline.py`, `stream_manager.py`).

Expected Output from you:
1. The fully updated code for `face_recognizer.py` (using InsightFace).
2. The updated code for `face_blur.py` (integrating the new detection data for blurring).
3. A step-by-step refactoring instruction/plan for splitting up `app.py`.
