#!/usr/bin/env python3
"""
analyze_and_notify.py
Run nfd.analyze_video_segments on a video and notify your Flask app about found segments.
Usage:
  python analyze_and_notify.py /path/to/video.mp4 \
        --flask-url http://127.0.0.1:8080/external_alert \
        --threshold 0.03
"""
import argparse
import json
import time
from pathlib import Path

# optional dependency
try:
    import requests
except Exception:
    requests = None

# your analyzer
from nfd import analyze_video_segments

# fallback to bot if POST fails and bot.py available
try:
    from bot import send_alert as bot_send_alert, send_photo as bot_send_photo
except Exception:
    bot_send_alert = None
    bot_send_photo = None

def safe_analyze(video_path, **kwargs):
    """Call analyze_video_segments with backward-compatible signature handling."""
    try:
        return analyze_video_segments(video_path, **kwargs)
    except TypeError:
        # some versions of nfd may not accept some kwargs (save_segment_frames, output_dir, etc.)
        minimal = {k: v for k, v in kwargs.items() if k in ("model_id","num_frames","size","window_sec","step_sec","top_k","threshold","labels_file")}
        return analyze_video_segments(video_path, **minimal)

def post_to_flask(flask_url, payload, timeout=8):
    if not requests:
        print("[WARN] requests not installed — cannot POST to Flask")
        return False, "no-requests"
    try:
        r = requests.post(flask_url, json=payload, timeout=timeout)
        try:
            body = r.text
        except Exception:
            body = ""
        print(f"[POST] {flask_url} -> {r.status_code}; resp: {body[:200]}")
        return r.status_code in (200,201), r.text
    except Exception as e:
        print("[ERROR] POST failed:", e)
        return False, str(e)

def notify_via_bot(text, image_path=None):
    if not bot_send_alert:
        print("[WARN] bot.send_alert not available (bot.py missing).")
        return False
    try:
        bot_send_alert(text)
        if image_path and bot_send_photo:
            bot_send_photo(image_path, caption=text)
        return True
    except Exception as e:
        print("[ERROR] bot notify failed:", e)
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="Path to video file")
    ap.add_argument("--flask-url", default="http://127.0.0.1:8080/external_alert",
                    help="Flask endpoint that accepts external alerts (default http://127.0.0.1:8080/external_alert)")
    ap.add_argument("--threshold", type=float, default=0.03, help="Threshold for fight detection (lower -> more sensitive)")
    ap.add_argument("--model", default="a0", help="MoviNet model id")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--window-sec", type=float, default=1.0)
    ap.add_argument("--step-sec", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--output-dir", default="uploads", help="where to save repr frames (if supported by nfd)")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print("File not found:", video_path)
        return

    print("Analyzing:", str(video_path))
    print("This may take time if model not cached (first run downloads TF Hub).")

    analyze_kwargs = dict(
        model_id=args.model,
        num_frames=args.num_frames,
        size=args.size,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        top_k=args.top_k,
        threshold=args.threshold,
        save_segment_frames=True,   # many nfd variants accept this; safe_analyze will fall back if not
        output_dir=args.output_dir
    )

    start = time.time()
    res = safe_analyze(str(video_path), **analyze_kwargs)
    elapsed = time.time() - start
    print(f"Analysis finished in {elapsed:.1f}s")

    if "error" in res:
        print("Analyzer returned error:", res["error"])
        return

    segs = res.get("segments", []) or []
    print("Duration:", res.get("duration"), "s | FPS:", res.get("fps"), " | total_frames:", res.get("total_frames"))
    print("Segments found:", len(segs))

    if not segs:
        print("No segments found — nothing to send.")
        # Optionally: still send summary notification
        summary_payload = {
            "message": f"Analysis finished for {video_path.name}: no fight-like segments found.",
            "video": str(video_path.name),
            "segments_count": 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        ok, resp = post_to_flask(args.flask_url, summary_payload)
        if not ok:
            print("Flask POST failed; trying bot fallback")
            notify_via_bot(summary_payload["message"])
        return

    # Send each segment as a separate alert
    for i, s in enumerate(segs, 1):
        start_sec = float(s.get("start_sec", 0.0))
        end_sec = float(s.get("end_sec", 0.0))
        max_conf = float(s.get("max_conf", 0.0))
        preds = s.get("top_predictions") or s.get("preds") or []
        repr_frame = s.get("repr_frame") or s.get("repr_image")  # optional (may be None)

        message = (f"🚨 Test Alert — fight-like segment #{i} detected in {video_path.name}\n"
                   f"t={start_sec:.1f}s–{end_sec:.1f}s  max_conf={max_conf:.3f}\n"
                   f"top: {preds[:3]}")

        payload = {
            "message": message,
            "video": str(video_path.name),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "max_conf": max_conf,
            "top_predictions": preds,
            "repr_frame": repr_frame,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

        print(f"\nSegment {i}: {start_sec:.1f}–{end_sec:.1f}s  conf={max_conf:.3f}  repr_frame={repr_frame}")
        # try POST to Flask
        ok, resp = post_to_flask(args.flask_url, payload)
        if not ok:
            print("Flask POST failed (see above). Trying bot fallback...")
            if notify_via_bot(message, image_path=repr_frame):
                print("Bot notification sent via bot.py")
            else:
                print("Bot fallback unavailable or failed.")

    # send summary
    try:
        summary = {
            "message": f"Analysis finished for {video_path.name}: {len(segs)} segment(s) reported.",
            "video": str(video_path.name),
            "segments_count": len(segs),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        post_to_flask(args.flask_url, summary)
    except Exception:
        pass

if __name__ == "__main__":
    main()
