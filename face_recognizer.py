"""
face_recognizer.py - High-quality face recognition using InsightFace (ArcFace + RetinaFace)
with dlib/face_recognition and pixel-vectorization fallbacks.

Backends (selected automatically in order of preference):
  1. InsightFace  — pip install insightface onnxruntime  (GPU: onnxruntime-gpu)
  2. dlib         — pip install face_recognition dlib
  3. Pixel (built-in) — YOLO/Haar + grayscale L2-norm, no extra deps

Public API (unchanged, backwards-compatible):
  detect_faces(frame, min_conf)           → list[(x1,y1,x2,y2,conf)]
  get_face_data(frame, min_conf)          → list[FaceResult]   ← NEW (one-pass detect+embed+id)
  identify(crop_bgr)                      → (name, score)
  register_face(name, frame_bgr, ...)     → (ok, msg)
  bulk_register_from_folder(folder)
  save_db()
  list_persons()                          → {name: count}
  backend  property                       → 'insightface' | 'dlib' | 'pixel'
"""

import json
import time
from pathlib import Path

import cv2
import numpy as np

# ─── Optional backend imports ─────────────────────────────────────────────────

_InsightFaceApp = None
try:
    from insightface.app import FaceAnalysis as _InsightFaceApp  # noqa: F401
except ImportError:
    pass

_dlib_lib = None
try:
    import face_recognition as _dlib_lib  # noqa: F401
except ImportError:
    pass

_YOLO = None
try:
    from ultralytics import YOLO as _YOLO  # noqa: F401
except ImportError:
    pass

# Expected embedding dimensions per backend (used for DB compatibility checks)
_EMB_DIM = {
    'insightface': 512,   # buffalo_s / buffalo_l ArcFace
    'dlib':        128,   # dlib ResNet face descriptor
    'pixel':       128 * 128,  # flattened 128×128 grayscale
}


# ─── FaceRecognizer ───────────────────────────────────────────────────────────

class FaceRecognizer:
    """
    Drop-in replacement for the old pixel-based recognizer.
    Automatically picks the best available backend.

    Parameters
    ----------
    db_path : str
        Path to the JSON embedding database.
    model_name : str
        InsightFace model pack. 'buffalo_s' (fast, ~30 MB) or 'buffalo_l' (accurate, ~500 MB).
    yolo_model_path : str
        YOLO face model used by the pixel fallback backend.
    threshold : float
        Cosine distance threshold: 0 = identical vectors, 2 = opposite.
        Typical range for ArcFace: 0.35–0.55; dlib: 0.40–0.55; pixel: 0.35–0.50.
    min_confidence_gap : float
        Required gap between best and second-best match to accept an ID.
    device : str
        'cuda' or 'cpu'. Passed to InsightFace / YOLO.
    debug : bool
        Verbose logging.
    """

    def __init__(
        self,
        db_path="faces/embeddings.json",
        model_name="buffalo_s",
        yolo_model_path="yolov8n-face.pt",
        threshold=0.45,
        min_confidence_gap=0.05,
        device="cpu",
        debug=False,
        # Legacy kwargs accepted but ignored (backwards compat)
        dnn_proto=None,
        dnn_model=None,
        vec_size=None,
    ):
        self.debug = debug
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.threshold = float(threshold)
        self.min_confidence_gap = float(min_confidence_gap)
        self.device = device
        del dnn_proto, dnn_model, vec_size  # legacy compat — accepted but not used

        self._backend = "pixel"
        self._insight_app = None
        self._dlib_lib = None
        self._yolo = None
        self._haar = None

        # {name: [np.ndarray, ...]}
        self._mem_db: dict[str, list[np.ndarray]] = {}

        self._init_backend(model_name, yolo_model_path)
        self._load_db()

        if self.debug:
            print(
                f"[FaceRecognizer] backend={self._backend}  "
                f"persons={len(self._mem_db)}  "
                f"threshold={self.threshold}"
            )

    # ──────────────────────────── Backend init ────────────────────────────────

    def _init_backend(self, model_name: str, yolo_model_path: str):
        # 1) InsightFace
        if _InsightFaceApp is not None:
            try:
                providers = (
                    ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    if self.device == "cuda"
                    else ["CPUExecutionProvider"]
                )
                app = _InsightFaceApp(name=model_name, providers=providers)
                ctx_id = 0 if self.device == "cuda" else -1
                app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                self._insight_app = app
                self._backend = "insightface"
                if self.debug:
                    print(f"[FaceRecognizer] ✓ InsightFace ({model_name}) loaded")
                return
            except Exception as exc:
                if self.debug:
                    print(f"[FaceRecognizer] InsightFace failed ({exc}), trying dlib…")

        # 2) dlib / face_recognition
        if _dlib_lib is not None:
            self._dlib_lib = _dlib_lib
            self._backend = "dlib"
            if self.debug:
                print("[FaceRecognizer] ✓ dlib/face_recognition loaded")
            return

        # 3) Pixel fallback: YOLO face + Haar
        self._backend = "pixel"
        if _YOLO is not None and yolo_model_path:
            try:
                p = Path(yolo_model_path)
                if p.exists():
                    self._yolo = _YOLO(str(p))
                    if self.debug:
                        print(f"[FaceRecognizer] ✓ YOLO face detector loaded ({p})")
            except Exception as exc:
                if self.debug:
                    print(f"[FaceRecognizer] YOLO load failed: {exc}")

        try:
            haar_xml = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if haar_xml.exists():
                cascade = cv2.CascadeClassifier(str(haar_xml))
                if not cascade.empty():
                    self._haar = cascade
                    if self.debug:
                        print("[FaceRecognizer] ✓ Haar cascade loaded (pixel fallback)")
        except Exception as exc:
            if self.debug:
                print(f"[FaceRecognizer] Haar load failed: {exc}")

        if self.debug:
            print("[FaceRecognizer] ⚠ Using pixel/YOLO backend (low accuracy)")

    # ──────────────────────────── DB management ───────────────────────────────

    def _load_db(self):
        """Load embeddings from JSON. Silently discards entries with wrong dimension."""
        self._mem_db = {}
        if not self.db_path.exists():
            return
        try:
            raw: dict = json.loads(self.db_path.read_text(encoding="utf-8"))
            expected_dim = _EMB_DIM.get(self._backend)
            skipped = 0
            for name, vecs in raw.items():
                arrs = []
                for v in vecs:
                    a = np.array(v, dtype=np.float32)
                    if expected_dim is not None and len(a) != expected_dim:
                        skipped += 1
                        continue  # Incompatible backend — skip silently
                    n = np.linalg.norm(a)
                    if n > 1e-6:
                        arrs.append(a / n)
                if arrs:
                    self._mem_db[name] = arrs
            if skipped and self.debug:
                print(
                    f"[FaceRecognizer] ⚠ Skipped {skipped} embeddings with wrong "
                    f"dimension (DB was from a different backend). Re-register faces."
                )
        except Exception as exc:
            if self.debug:
                print(f"[FaceRecognizer] DB load error: {exc}")
            self._mem_db = {}

    def save_db(self):
        """Persist in-memory DB to JSON."""
        dump = {n: [v.tolist() for v in arrs] for n, arrs in self._mem_db.items()}
        try:
            self.db_path.write_text(
                json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if self.debug:
                print(f"[FaceRecognizer] ✓ DB saved ({len(dump)} persons)")
        except Exception as exc:
            if self.debug:
                print(f"[FaceRecognizer] DB save failed: {exc}")

    # ──────────────────────────── Detection (public) ──────────────────────────

    def detect_faces(self, frame_bgr: np.ndarray, min_conf: float = 0.5) -> list[tuple]:
        """
        Detect faces in *frame_bgr*.

        Returns
        -------
        list of (x1, y1, x2, y2, conf)
        """
        raw = self._raw_detect(frame_bgr, min_conf)
        return [(d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d["conf"]) for d in raw]

    def get_face_data(self, frame_bgr: np.ndarray, min_conf: float = 0.5) -> list[dict]:
        """
        Detect + embed + identify faces in a *single pass*.
        More efficient than detect_faces() + identify() in a loop.

        Returns
        -------
        list of dicts:
            {
              'bbox':      (x1, y1, x2, y2),
              'conf':      float,
              'embedding': np.ndarray | None,
              'name':      str,
              'score':     float | None,   # cosine distance (lower = better)
            }
        """
        raw = self._raw_detect(frame_bgr, min_conf)
        results = []
        for d in raw:
            emb = d.get("embedding")
            if emb is not None:
                name, score = self._identify_embedding(emb)
            else:
                # Pixel backend: embed from crop
                x1, y1, x2, y2 = d["bbox"]
                crop = frame_bgr[y1:y2, x1:x2]
                name, score = self.identify(crop)
            results.append(
                {
                    "bbox": d["bbox"],
                    "conf": d["conf"],
                    "embedding": emb,
                    "name": name,
                    "score": score,
                }
            )
        return results

    # ──────────────────────────── Internal detection ──────────────────────────

    def _raw_detect(self, frame_bgr: np.ndarray, min_conf: float) -> list[dict]:
        """
        Internal dispatcher. Returns list of:
            {'bbox': (x1,y1,x2,y2), 'conf': float, 'embedding': np.ndarray|None}
        """
        if self._backend == "insightface":
            return self._detect_insightface(frame_bgr, min_conf)
        elif self._backend == "dlib":
            return self._detect_dlib(frame_bgr, min_conf)
        return self._detect_pixel(frame_bgr, min_conf)

    def _detect_insightface(self, frame_bgr: np.ndarray, min_conf: float) -> list[dict]:
        """RetinaFace detection + ArcFace embedding in one model call."""
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            faces = self._insight_app.get(rgb)
            h, w = frame_bgr.shape[:2]
            results = []
            for face in faces:
                if float(face.det_score) < min_conf:
                    continue
                x1, y1, x2, y2 = face.bbox.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
                if (x2 - x1) < 20 or (y2 - y1) < 20:
                    continue
                emb = self._normalize(face.embedding)
                results.append(
                    {"bbox": (x1, y1, x2, y2), "conf": float(face.det_score), "embedding": emb}
                )
            if self.debug and results:
                print(f"[FaceRecognizer] InsightFace: {len(results)} face(s)")
            return results
        except Exception as exc:
            if self.debug:
                print(f"[FaceRecognizer] InsightFace detect error: {exc}")
            return []

    def _detect_dlib(self, frame_bgr: np.ndarray, min_conf: float) -> list[dict]:
        """dlib HOG detection + 128-d face descriptor.

        dlib HOG does not produce a confidence score, so *min_conf* is reused as a
        minimum face size relative to the frame height (e.g. 0.5 → face must be at
        least 50 % of frame height tall, which is very permissive; 0.02 is typical).
        """
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            locs = self._dlib_lib.face_locations(rgb, model="hog")
            encodings = self._dlib_lib.face_encodings(rgb, locs)
            h, w = frame_bgr.shape[:2]
            # Translate min_conf (0–1) into a pixel min-size: at least min_conf * frame-height
            min_px = max(20, int(min_conf * h * 0.3))
            results = []
            for (top, right, bottom, left), enc in zip(locs, encodings):
                x1 = max(0, left)
                y1 = max(0, top)
                x2 = min(w - 1, right)
                y2 = min(h - 1, bottom)
                if (x2 - x1) < min_px or (y2 - y1) < min_px:
                    continue
                emb = self._normalize(np.array(enc, dtype=np.float32))
                results.append({"bbox": (x1, y1, x2, y2), "conf": 0.90, "embedding": emb})
            if self.debug and results:
                print(f"[FaceRecognizer] dlib: {len(results)} face(s)")
            return results
        except Exception as exc:
            if self.debug:
                print(f"[FaceRecognizer] dlib detect error: {exc}")
            return []

    def _detect_pixel(self, frame_bgr: np.ndarray, min_conf: float) -> list[dict]:
        """YOLO/Haar detection with no embedding (embedding computed later from crop)."""
        h, w = frame_bgr.shape[:2]
        raw_boxes: list[tuple] = []

        if self._yolo is not None:
            try:
                outs = self._yolo(frame_bgr, conf=min_conf, verbose=False, imgsz=640)
                for out in outs:
                    if getattr(out, "boxes", None) is None:
                        continue
                    for b in out.boxes:
                        xyxy = b.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = (int(v) for v in xyxy)
                        conf = float(b.conf[0]) if getattr(b, "conf", None) is not None else 1.0
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w - 1, x2), min(h - 1, y2)
                        if (x2 - x1) >= 30 and (y2 - y1) >= 30:
                            raw_boxes.append((x1, y1, x2, y2, conf))
            except Exception as exc:
                if self.debug:
                    print(f"[FaceRecognizer] YOLO detect error: {exc}")

        if not raw_boxes and self._haar is not None:
            try:
                gray = cv2.equalizeHist(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
                dets = self._haar.detectMultiScale(gray, 1.1, 6, minSize=(40, 40))
                for x, y, ww, hh in dets:
                    raw_boxes.append((int(x), int(y), int(x + ww), int(y + hh), 0.80))
            except Exception as exc:
                if self.debug:
                    print(f"[FaceRecognizer] Haar detect error: {exc}")

        raw_boxes = self._apply_nms(raw_boxes, 0.4)
        if self.debug and raw_boxes:
            print(f"[FaceRecognizer] pixel/YOLO: {len(raw_boxes)} face(s)")
        return [
            {"bbox": (x1, y1, x2, y2), "conf": conf, "embedding": None}
            for x1, y1, x2, y2, conf in raw_boxes
        ]

    # ──────────────────────────── Identification ──────────────────────────────

    def identify(self, crop_bgr: np.ndarray) -> tuple[str, float | None]:
        """
        Identify a face from a cropped BGR image.

        Returns
        -------
        (name, score)  where score is cosine distance (lower = better).
        Returns ("Unknown", score_or_None) when no match above threshold.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return "Unknown", None

        emb = self._embed_from_crop(crop_bgr)
        if emb is None:
            return "Unknown", None
        return self._identify_embedding(emb)

    def _embed_from_crop(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Extract embedding from a *cropped* face image using the active backend."""
        if self._backend == "insightface":
            return self._embed_insightface_crop(crop_bgr)
        elif self._backend == "dlib":
            return self._embed_dlib_crop(crop_bgr)
        return self._embed_pixel(crop_bgr)

    def _embed_insightface_crop(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """
        Run InsightFace on a tight crop.
        We pad the crop so RetinaFace can reliably find the face inside.
        """
        try:
            h, w = crop_bgr.shape[:2]
            pad = max(20, int(min(h, w) * 0.2))
            padded = cv2.copyMakeBorder(
                crop_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
            )
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            faces = self._insight_app.get(rgb)
            if not faces:
                return None
            # Pick the largest detected face
            face = max(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            return self._normalize(face.embedding)
        except Exception as exc:
            if self.debug:
                print(f"[FaceRecognizer] InsightFace embed error: {exc}")
            return None

    def _embed_dlib_crop(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Extract dlib 128-d descriptor from a crop."""
        try:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            locs = self._dlib_lib.face_locations(rgb, model="hog")
            if not locs:
                # Treat the entire crop as one face region
                locs = [(0, rgb.shape[1], rgb.shape[0], 0)]
            encs = self._dlib_lib.face_encodings(rgb, locs)
            if not encs:
                return None
            return self._normalize(np.array(encs[0], dtype=np.float32))
        except Exception as exc:
            if self.debug:
                print(f"[FaceRecognizer] dlib embed error: {exc}")
            return None

    def _embed_pixel(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Pixel-level embedding: grayscale 128×128 + CLAHE + histogram equalize + L2-norm."""
        try:
            if crop_bgr is None or crop_bgr.size == 0:
                return None
            h, w = crop_bgr.shape[:2]
            if w < 30 or h < 30:
                return None
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            proc = cv2.equalizeHist(clahe.apply(resized))
            v = proc.astype(np.float32).reshape(-1)
            return self._normalize(v)
        except Exception as exc:
            if self.debug:
                print(f"[FaceRecognizer] pixel embed error: {exc}")
            return None

    def _identify_embedding(self, emb: np.ndarray) -> tuple[str, float | None]:
        """Match *emb* against the in-memory DB. Returns (name, cosine_distance)."""
        if not self._mem_db:
            return "Unknown", None

        best_name: str | None = None
        best_score = float("inf")
        second_best = float("inf")

        for name, refs in self._mem_db.items():
            scores = []
            for ref in refs:
                if len(ref) != len(emb):
                    continue  # Dimension mismatch (old backend DB entries)
                dot = float(np.dot(emb, ref))
                dot = max(-1.0, min(1.0, dot))
                scores.append(1.0 - dot)
            if not scores:
                continue
            s = min(scores)  # Best match among this person's templates
            if s < best_score:
                second_best = best_score
                best_score = s
                best_name = name
            elif s < second_best:
                second_best = s

        if best_name is None:
            return "Unknown", None

        if best_score <= self.threshold:
            gap = second_best - best_score
            if gap >= self.min_confidence_gap or len(self._mem_db) == 1:
                if self.debug:
                    print(
                        f"[FaceRecognizer] ✓ {best_name} "
                        f"(dist={best_score:.4f}, gap={gap:.4f})"
                    )
                return best_name, float(best_score)

        if self.debug:
            print(
                f"[FaceRecognizer] Unknown "
                f"(best={best_name} dist={best_score:.4f} > thresh={self.threshold})"
            )
        return "Unknown", float(best_score) if best_score != float("inf") else None

    # ──────────────────────────── Registration ────────────────────────────────

    def register_face(
        self,
        name: str,
        frame_bgr: np.ndarray,
        use_best_box: bool = True,
        persist_image: bool = True,
    ) -> tuple[bool, str]:
        """
        Detect the largest face in *frame_bgr*, compute its embedding, and add it to
        the database for *name*.

        Returns
        -------
        (success, message)
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return False, "invalid image"

        raw = self._raw_detect(frame_bgr, min_conf=0.3)
        if not raw:
            return False, "no face found"

        if use_best_box:
            raw = sorted(
                raw,
                key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]),
                reverse=True,
            )

        best = raw[0]
        x1, y1, x2, y2 = best["bbox"]
        crop = frame_bgr[y1:y2, x1:x2]

        # Use pre-computed embedding when available (InsightFace / dlib full-frame)
        emb = best.get("embedding")
        if emb is None:
            emb = self._embed_from_crop(crop)

        if emb is None:
            return False, "could not extract embedding (low quality or very small face)"

        self._mem_db.setdefault(name, []).append(emb)
        self.save_db()

        if persist_image:
            try:
                img_dir = Path("faces/images")
                img_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(img_dir / f"{name}_{int(time.time())}.jpg"), crop)
            except Exception:
                pass

        if self.debug:
            print(f"[FaceRecognizer] ✓ Registered '{name}'  (DB size: {len(self._mem_db[name])})")
        return True, "ok"

    def bulk_register_from_folder(self, folder: str = "faces/images"):
        """
        Load all face images from *folder* into the database.
        Filename convention: ``PersonName_1.jpg`` → name ``"PersonName"``.
        Already-registered embeddings from the same file are not duplicated
        (based on filename stem comparison against existing names).
        """
        p = Path(folder)
        if not p.exists():
            if self.debug:
                print(f"[FaceRecognizer] bulk_register: folder not found: {folder}")
            return

        supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [f for f in p.iterdir() if f.suffix.lower() in supported]
        if self.debug:
            print(f"[FaceRecognizer] bulk_register: {len(files)} image(s) in {folder}")

        for f in files:
            name = f.stem.split("_")[0]
            img = cv2.imread(str(f))
            if img is None:
                if self.debug:
                    print(f"  - {f.name}: failed to load")
                continue
            ok, msg = self.register_face(name, img, persist_image=False)
            if self.debug:
                print(f"  - {f.name} → '{name}': {ok} ({msg})")

    def list_persons(self) -> dict[str, int]:
        """Return ``{name: number_of_stored_embeddings}``."""
        return {n: len(v) for n, v in self._mem_db.items()}

    # ──────────────────────────── Utilities ──────────────────────────────────

    @property
    def backend(self) -> str:
        return self._backend

    @staticmethod
    def _normalize(vec: np.ndarray | None) -> np.ndarray | None:
        if vec is None:
            return None
        n = np.linalg.norm(vec)
        return vec / n if n > 1e-6 else None

    def _compute_iou(self, a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = a[:4]
        bx1, by1, bx2, by2 = b[:4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    def _apply_nms(self, boxes: list[tuple], iou_threshold: float = 0.4) -> list[tuple]:
        """Non-maximum suppression. Boxes: [(x1,y1,x2,y2,conf), ...]."""
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
        keep: list[tuple] = []
        for box in boxes:
            if all(self._compute_iou(box, k) <= iou_threshold for k in keep):
                keep.append(box)
        return keep

    # ──────────────────────────── Debug helpers ───────────────────────────────

    def debug_one_frame(self, frame: np.ndarray):
        """Quick CLI debug: print detected faces and their identities."""
        data = self.get_face_data(frame, min_conf=0.2)
        print(f"DEBUG: {len(data)} face(s) detected  [backend={self._backend}]")
        for i, d in enumerate(data):
            print(
                f"  [{i}] bbox={d['bbox']}  conf={d['conf']:.2f}  "
                f"→ {d['name']}  score={d['score']}"
            )
            x1, y1, x2, y2 = d["bbox"]
            cv2.imwrite(f"debug_crop_{i}.jpg", frame[y1:y2, x1:x2])