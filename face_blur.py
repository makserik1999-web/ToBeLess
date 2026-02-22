"""
face_blur.py - Privacy-aware face blurring.

Privacy modes
─────────────
  'none'    → Pass-through. No faces are blurred.
  'unknown' → Blur only faces whose identity is NOT in the database.
              Known persons remain visible. Useful for recording known staff
              while protecting bystanders.
  'all'     → Blur every detected face regardless of identity.

Usage (class API)
──────────────────
    from face_recognizer import FaceRecognizer
    from face_blur import FaceBlurProcessor

    fr = FaceRecognizer(db_path="faces/embeddings.json")
    fr.bulk_register_from_folder("faces/images")

    processor = FaceBlurProcessor(fr, privacy_mode='unknown', blur_strength=31)

    # In your video loop:
    out_frame, face_data = processor.process(frame)
    for d in face_data:
        print(d['name'], 'blurred:', d['blurred'])

Usage (backwards-compatible function)
───────────────────────────────────────
    from face_blur import blur_faces
    out_frame, boxes = blur_faces(frame, face_recognizer)
    # boxes: [(x1,y1,x2,y2,conf,name,score), ...]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from face_recognizer import FaceRecognizer

# Valid privacy mode strings
PRIVACY_MODES = ("none", "unknown", "all")


class FaceBlurProcessor:
    """
    Applies face blurring according to a configurable privacy mode.

    Parameters
    ----------
    face_recognizer : FaceRecognizer
        Initialised recognizer instance used for detection and identification.
    privacy_mode : str
        One of 'none', 'unknown', 'all'.
    blur_strength : int
        Gaussian kernel size (odd; larger = heavier blur). Default 31.
    expand : float
        Fractional padding around each detected bounding box before blurring.
        0.15 adds 15 % on each side. Default 0.15.
    min_conf : float
        Minimum face-detection confidence to consider a detection. Default 0.5.
    draw_labels : bool
        If True, draw name labels on *known* faces (no-op when privacy_mode='all').
    """

    def __init__(
        self,
        face_recognizer: FaceRecognizer,
        privacy_mode: str = "unknown",
        blur_strength: int = 31,
        expand: float = 0.15,
        min_conf: float = 0.5,
        draw_labels: bool = True,
    ):
        self.face_recognizer = face_recognizer
        self.blur_strength = int(blur_strength)
        self.expand = float(expand)
        self.min_conf = float(min_conf)
        self.draw_labels = draw_labels
        self.set_privacy_mode(privacy_mode)

    # ─── Configuration ────────────────────────────────────────────────────────

    def set_privacy_mode(self, mode: str):
        """Change privacy mode at runtime. Raises ValueError on unknown mode."""
        if mode not in PRIVACY_MODES:
            raise ValueError(f"privacy_mode must be one of {PRIVACY_MODES}, got {mode!r}")
        self.privacy_mode = mode

    # ─── Main entry point ────────────────────────────────────────────────────

    def process(
        self,
        frame_bgr: np.ndarray,
        run_recognition: bool = True,
    ) -> tuple[np.ndarray, list[dict]]:
        """
        Process a single frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            Input BGR frame (not modified in-place).
        run_recognition : bool
            If False, skip identity matching — all faces are treated as Unknown.
            Useful to save compute when privacy_mode='all'.

        Returns
        -------
        out_frame : np.ndarray
            Frame with blurring applied according to privacy_mode.
        face_data : list[dict]
            One entry per detected face::

                {
                    'bbox':    (x1, y1, x2, y2),
                    'conf':    float,
                    'name':    str,          # 'Unknown' when unrecognised
                    'score':   float | None, # cosine distance (lower = better)
                    'blurred': bool,
                }
        """
        if self.privacy_mode == "none":
            return frame_bgr, []

        out = frame_bgr.copy()
        h, w = out.shape[:2]

        # ── Detection + optional identification ──────────────────────────────
        if run_recognition and self.privacy_mode != "all":
            # Single-pass: detect + embed + identify (most efficient with InsightFace/dlib)
            raw_faces = self.face_recognizer.get_face_data(out, min_conf=self.min_conf)
        else:
            # Detection only — identification not needed
            raw_boxes = self.face_recognizer.detect_faces(out, min_conf=self.min_conf)
            raw_faces = [
                {
                    "bbox": b[:4],
                    "conf": b[4],
                    "embedding": None,
                    "name": "Unknown",
                    "score": None,
                }
                for b in raw_boxes
            ]

        # ── Blur + annotate ──────────────────────────────────────────────────
        face_data: list[dict] = []

        for face in raw_faces:
            x1, y1, x2, y2 = face["bbox"]
            name: str = face.get("name", "Unknown") or "Unknown"
            score = face.get("score")

            should_blur = self.privacy_mode == "all" or (
                self.privacy_mode == "unknown" and name == "Unknown"
            )

            if should_blur:
                out = self._blur_region(out, x1, y1, x2, y2, h, w)

            if self.draw_labels and name != "Unknown" and not should_blur:
                self._draw_label(out, name, score, x1, y1, x2, y2)

            face_data.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "conf": face["conf"],
                    "name": name,
                    "score": score,
                    "blurred": should_blur,
                }
            )

        return out, face_data

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _blur_region(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        h: int,
        w: int,
    ) -> np.ndarray:
        """Apply Gaussian blur to the face region with optional bbox expansion."""
        dw = int((x2 - x1) * self.expand)
        dh = int((y2 - y1) * self.expand)
        bx1 = max(0, x1 - dw)
        by1 = max(0, y1 - dh)
        bx2 = min(w - 1, x2 + dw)
        by2 = min(h - 1, y2 + dh)

        roi = frame[by1:by2, bx1:bx2]
        if roi.size == 0:
            return frame

        k = max(3, self.blur_strength | 1)  # Ensure odd kernel
        frame[by1:by2, bx1:bx2] = cv2.GaussianBlur(roi, (k, k), 0)
        return frame

    @staticmethod
    def _draw_label(
        frame: np.ndarray,
        name: str,
        score: float | None,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ):
        """Draw a face bounding box and a semi-transparent name badge above it."""
        label = name if score is None else f"{name} ({1 - score:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.55, 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

        # Face bounding box (green for known persons)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)

        # Badge background above the box
        bx1 = x1
        by1 = max(0, y1 - th - baseline - 6)
        bx2 = x1 + tw + 6
        by2 = y1

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(
            frame, label, (bx1 + 3, by2 - baseline - 2),
            font, scale, (0, 220, 0), thickness, cv2.LINE_AA,
        )


# ─── Backwards-compatible function interface ──────────────────────────────────

def blur_faces(
    frame_bgr: np.ndarray,
    face_recognizer: FaceRecognizer,
    min_conf: float = 0.25,
    blur_strength: int = 25,
    expand: float = 0.1,
    privacy_mode: str = "all",
) -> tuple[np.ndarray, list[tuple]]:
    """
    Backwards-compatible helper function wrapping :class:`FaceBlurProcessor`.

    Parameters match the original ``blur_faces`` signature.

    Returns
    -------
    (out_frame, boxes)
        ``boxes`` is a list of ``(x1, y1, x2, y2, conf, name, score)``.
    """
    processor = FaceBlurProcessor(
        face_recognizer,
        privacy_mode=privacy_mode,
        blur_strength=blur_strength,
        expand=expand,
        min_conf=min_conf,
        draw_labels=False,  # Legacy callers handle their own labels
    )
    out, face_data = processor.process(frame_bgr)
    boxes = [
        (
            d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3],
            d["conf"], d["name"], d["score"],
        )
        for d in face_data
    ]
    return out, boxes
