from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

@dataclass
class FaceKpsBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    kps: np.ndarray

def _estimate_norm_5pt(kps: np.ndarray, out_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
    src = kps.astype(np.float32)
    dst = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
    w, h = out_size
    if (w, h) != (112, 112):
        dst *= np.array([w / 112.0, h / 112.0], dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        M = cv2.getAffineTransform(src[:3], dst[:3])
    return M.astype(np.float32)

def align_face_5pt(frame_bgr: np.ndarray, kps: np.ndarray, out_size: Tuple[int, int] = (112, 112)) -> Tuple[np.ndarray, np.ndarray]:
    M = _estimate_norm_5pt(kps, out_size)
    w, h = out_size
    aligned = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return aligned, M

def _clip_box(b: np.ndarray, W: int, H: int) -> np.ndarray:
    b = b.astype(np.float32).copy()
    b[0] = np.clip(b[0], 0, W - 1)
    b[1] = np.clip(b[1], 0, H - 1)
    b[2] = np.clip(b[2], 0, W - 1)
    b[3] = np.clip(b[3], 0, H - 1)
    return b

def _bbox_from_5pt(kps: np.ndarray, pad_x: float = 0.55, pad_y_top: float = 0.85, pad_y_bot: float = 1.15) -> np.ndarray:
    x1, y1 = np.min(kps, axis=0)
    x2, y2 = np.max(kps, axis=0)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return np.array([x1 - pad_x * w, y1 - pad_y_top * h, x2 + pad_x * w, y2 + pad_y_bot * h], dtype=np.float32)

def _ema(prev: Optional[np.ndarray], cur: np.ndarray, alpha: float) -> np.ndarray:
    if prev is None:
        return cur.astype(np.float32)
    return (alpha * prev + (1.0 - alpha) * cur).astype(np.float32)

def _kps_span_ok(kps: np.ndarray, min_eye_dist: float = 10.0) -> bool:
    le, re, no, lm, rm = kps.astype(np.float32)
    if np.linalg.norm(re - le) < min_eye_dist:
        return False
    return lm[1] > no[1] and rm[1] > no[1]

class Haar5ptDetector:
    def __init__(
        self,
        haar_xml: Optional[str] = None,
        min_size: Tuple[int, int] = (60, 60),
        smooth_alpha: float = 0.80,
        debug: bool = True,
        model_path: str = "models/face_landmarker.task",
    ):
        self.debug = debug
        self.min_size = tuple(map(int, min_size))
        self.smooth_alpha = float(smooth_alpha)
        if haar_xml is None:
            haar_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(haar_xml)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_xml}")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self.IDX = [33, 263, 1, 61, 291]
        self._prev_box = None
        self._prev_kps = None
        self._timestamp_ms = 0

    def _haar_faces(self, gray: np.ndarray) -> np.ndarray:
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            flags=cv2.CASCADE_SCALE_IMAGE,
            minSize=self.min_size,
        )
        return np.asarray(faces, dtype=np.int32).reshape(-1, 4) if len(faces) else np.zeros((0, 4), dtype=np.int32)

    def _facemesh_5pt(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp_ms += 33
        result = self.face_landmarker.detect_for_video(image, self._timestamp_ms)
        if not result.face_landmarks:
            return None
        lm = result.face_landmarks[0]
        kps = np.array([[lm[i].x * W, lm[i].y * H] for i in self.IDX], dtype=np.float32)
        if kps[0, 0] > kps[1, 0]:
            kps[[0, 1]] = kps[[1, 0]]
        if kps[3, 0] > kps[4, 0]:
            kps[[3, 4]] = kps[[4, 3]]
        return kps

    def detect(self, frame_bgr: np.ndarray, max_faces: int = 1) -> List[FaceKpsBox]:
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar_faces(gray)
        if faces.shape[0] == 0:
            return []
        areas = faces[:, 2] * faces[:, 3]
        x, y, w, h = faces[int(np.argmax(areas))].tolist()
        kps = self._facemesh_5pt(frame_bgr)
        if kps is None:
            if self.debug:
                print("[haar_5pt] Haar found a face but MediaPipe found none")
            return []
        margin = 0.35
        x1m, y1m = x - margin * w, y - margin * h
        x2m, y2m = x + (1 + margin) * w, y + (1 + margin) * h
        inside = (kps[:, 0] >= x1m) & (kps[:, 0] <= x2m) & (kps[:, 1] >= y1m) & (kps[:, 1] <= y2m)
        if inside.mean() < 0.60:
            if self.debug:
                print("[haar_5pt] MediaPipe points do not match Haar box")
            return []
        if not _kps_span_ok(kps):
            if self.debug:
                print("[haar_5pt] Landmark geometry check failed")
            return []
        box = _clip_box(_bbox_from_5pt(kps), W, H)
        box = _ema(self._prev_box, box, self.smooth_alpha)
        kps = _ema(self._prev_kps, kps, self.smooth_alpha)
        self._prev_box = box.copy()
        self._prev_kps = kps.copy()
        x1, y1, x2, y2 = box.tolist()
        return [FaceKpsBox(int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)), 1.0, kps.astype(np.float32))][:max_faces]

    def close(self):
        if self.face_landmarker is not None:
            self.face_landmarker.close()
            self.face_landmarker = None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not opened.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    det = Haar5ptDetector(min_size=(70, 70), smooth_alpha=0.80, debug=True)
    print("Haar + 5pt test. Press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame.")
                break
            faces = det.detect(frame)
            vis = frame.copy()
            if faces:
                f = faces[0]
                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), (0, 255, 0), 2)
                for x, y in f.kps.astype(int):
                    cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
                cv2.putText(vis, "OK", (f.x1, max(20, f.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "NO FACE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("haar_5pt", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        det.close()

if __name__ == "__main__":
    main()
