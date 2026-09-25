# src/face_signals.py

from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# MediaPipe landmark indices
# ============================================================

LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)

# Five important landmarks
NOSE_TIP = 1
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

LIP_TOP = 13
LIP_BOTTOM = 14

FACE_LEFT = 234
FACE_RIGHT = 454


# ============================================================
# Math helpers
# ============================================================

def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(
    points: np.ndarray,
    idx: Tuple[int, ...],
) -> float:
    p1, p2, p3, p4, p5, p6 = (points[i] for i in idx)

    width = max(
        distance(p1, p4),
        1e-6,
    )

    return (
        distance(p2, p6) + distance(p3, p5)
    ) / (2.0 * width)


# ============================================================
# Returned facial signals
# ============================================================

@dataclass
class FaceSignals:
    # Eyes
    ear: float
    blink: bool
    eyes_closed: bool

    # Smile
    smile_score: float
    smiling: bool

    # Additional expressions
    frown_score: float
    frowning: bool

    sad_score: float
    sad: bool

    grimace_score: float
    grimacing: bool

    # Nose position relative to frame center
    nose_x: float
    nose_y: float

    nose_offset_x: float
    nose_offset_y: float

    nose_distance: float
    nose_direction: str


# ============================================================
# Facial signal extractor
# ============================================================

class FaceSignalExtractor:

    def __init__(
        self,
        ear_threshold: float = 0.21,
        blink_min_frames: int = 2,
        blink_max_frames: int = 7,
        closed_frames: int = 8,

        smile_on: float = 0.38,
        smile_off: float = 0.35,

        # Frown thresholds
        frown_on: float = 0.018,
        frown_off: float = 0.012,

        # Sadness thresholds
        sad_on: float = 0.022,
        sad_off: float = 0.016,

        # Grimace thresholds
        grimace_on: float = 0.055,
        grimace_off: float = 0.045,
    ):
        # -----------------------------------------------------
        # Eye parameters
        # -----------------------------------------------------

        self.ear_threshold = ear_threshold
        self.blink_min_frames = blink_min_frames
        self.blink_max_frames = blink_max_frames
        self.closed_frames = closed_frames

        self.low_ear_frames = 0

        # -----------------------------------------------------
        # Smile parameters
        # -----------------------------------------------------

        self.smile_on = smile_on
        self.smile_off = smile_off
        self.smiling = False

        # -----------------------------------------------------
        # Frown parameters
        # -----------------------------------------------------

        self.frown_on = frown_on
        self.frown_off = frown_off
        self.frowning = False

        # -----------------------------------------------------
        # Sad parameters
        # -----------------------------------------------------

        self.sad_on = sad_on
        self.sad_off = sad_off
        self.sad = False

        # -----------------------------------------------------
        # Grimace parameters
        # -----------------------------------------------------

        self.grimace_on = grimace_on
        self.grimace_off = grimace_off
        self.grimacing = False

        # -----------------------------------------------------
        # MediaPipe Tasks API
        # -----------------------------------------------------

        model_path = Path("models/face_landmarker.task")

        if not model_path.exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found: {model_path}"
            )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path.resolve())
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.mesh = (
            mp.tasks.vision.FaceLandmarker.create_from_options(
                options
            )
        )

        # MediaPipe VIDEO mode requires increasing timestamps.
        self.timestamp_ms = 0

    # ========================================================
    # Reset
    # ========================================================

    def reset(self) -> None:
        self.low_ear_frames = 0

        self.smiling = False
        self.frowning = False
        self.sad = False
        self.grimacing = False

    # ========================================================
    # Close MediaPipe
    # ========================================================

    def close(self) -> None:
        self.mesh.close()

    # ========================================================
    # Timestamp
    # ========================================================

    def _next_timestamp_ms(self) -> int:
        """
        Generate strictly increasing timestamps
        for MediaPipe VIDEO mode.
        """

        current = int(time.monotonic() * 1000)

        if current <= self.timestamp_ms:
            current = self.timestamp_ms + 1

        self.timestamp_ms = current

        return self.timestamp_ms

    # ========================================================
    # Main analysis
    # ========================================================

    def analyze(
        self,
        frame: np.ndarray,
        bbox,
    ) -> Optional[FaceSignals]:

        h, w = frame.shape[:2]

        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1

        pad_x = int(0.12 * bw)
        pad_y = int(0.18 * bh)

        rx1 = max(0, x1 - pad_x)
        ry1 = max(0, y1 - pad_y)

        rx2 = min(w, x2 + pad_x)
        ry2 = min(h, y2 + pad_y)

        roi = frame[ry1:ry2, rx1:rx2]

        if roi.size == 0:
            return None

        # ====================================================
        # BGR -> RGB
        # ====================================================

        roi_rgb = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2RGB,
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=roi_rgb,
        )

        # ====================================================
        # MediaPipe Face Landmarker
        # ====================================================

        result = self.mesh.detect_for_video(
            mp_image,
            self._next_timestamp_ms(),
        )

        if not result.face_landmarks:
            return None

        rh, rw = roi.shape[:2]

        lm = result.face_landmarks[0]

        # Convert normalized landmarks to full-frame pixels.
        points = np.array(
            [
                [
                    p.x * rw + rx1,
                    p.y * rh + ry1,
                ]
                for p in lm
            ],
            dtype=np.float32,
        )

        # ====================================================
        # 1. EYE ASPECT RATIO
        # ====================================================

        left_ear = eye_aspect_ratio(
            points,
            LEFT_EYE,
        )

        right_ear = eye_aspect_ratio(
            points,
            RIGHT_EYE,
        )

        ear = 0.5 * (left_ear + right_ear)

        # ====================================================
        # 2. BLINK DETECTION
        # ====================================================

        blink = False

        if ear < self.ear_threshold:

            self.low_ear_frames += 1

        else:

            if (
                self.blink_min_frames
                <= self.low_ear_frames
                <= self.blink_max_frames
            ):
                blink = True

            self.low_ear_frames = 0

        # ====================================================
        # 3. EYES CLOSED
        # ====================================================

        eyes_closed = (
            self.low_ear_frames >= self.closed_frames
        )

        # ====================================================
        # 4. FACE WIDTH
        # ====================================================

        face_width = max(
            distance(
                points[FACE_LEFT],
                points[FACE_RIGHT],
            ),
            1e-6,
        )

        # ====================================================
        # 5. MOUTH GEOMETRY
        # ====================================================

        mouth_left = points[MOUTH_LEFT]
        mouth_right = points[MOUTH_RIGHT]

        lip_top = points[LIP_TOP]
        lip_bottom = points[LIP_BOTTOM]

        nose = points[NOSE_TIP]

        mouth_center = (
            mouth_left + mouth_right
        ) / 2.0

        mouth_width = distance(
            mouth_left,
            mouth_right,
        )

        mouth_height = distance(
            lip_top,
            lip_bottom,
        )

        # ====================================================
        # 6. SMILE
        # ====================================================

        smile_score = (
            mouth_width / face_width
        )

        if self.smiling:

            self.smiling = (
                smile_score >= self.smile_off
            )

        else:

            self.smiling = (
                smile_score >= self.smile_on
            )

        # ====================================================
        # 7. FROWN
        # ====================================================
        #
        # Image coordinates:
        #     smaller Y = higher
        #     larger Y  = lower
        #
        # A frown-like configuration tends to place the
        # mouth corners lower relative to the mouth center.
        #
        # Normalize by face width so the value is less
        # dependent on camera distance.
        # ====================================================

        corner_y = (
            mouth_left[1] + mouth_right[1]
        ) / 2.0

        frown_score = (
            corner_y - mouth_center[1]
        ) / face_width

        # The raw expression geometry can be noisy, so use
        # hysteresis just like the smile detector.

        if self.frowning:

            self.frowning = (
                frown_score >= self.frown_off
            )

        else:

            self.frowning = (
                frown_score >= self.frown_on
            )

        # ====================================================
        # 8. SADNESS-LIKE CONFIGURATION
        # ====================================================
        #
        # This is a facial-geometry estimate, NOT a clinical
        # or psychological determination of sadness.
        #
        # We combine:
        #   - downward mouth corners
        #   - relatively narrow mouth
        # ====================================================

        narrow_mouth = 1.0 - min(
            mouth_width / face_width / max(self.smile_on, 1e-6),
            1.0,
        )

        sad_score = max(
            0.0,
            frown_score * 2.0,
        ) + (
            0.05 * narrow_mouth
        )

        if self.sad:

            self.sad = (
                sad_score >= self.sad_off
            )

        else:

            self.sad = (
                sad_score >= self.sad_on
            )

        # ====================================================
        # 9. GRIMACE
        # ====================================================
        #
        # Approximation based on:
        #   - wide mouth
        #   - relatively large lip opening
        #
        # This is intentionally a geometric demo signal.
        # ====================================================

        mouth_width_ratio = (
            mouth_width / face_width
        )

        mouth_height_ratio = (
            mouth_height / face_width
        )

        grimace_score = (
            mouth_width_ratio
            + mouth_height_ratio
        )

        if self.grimacing:

            self.grimacing = (
                grimace_score >= self.grimace_off
            )

        else:

            self.grimacing = (
                grimace_score >= self.grimace_on
            )

        # ====================================================
        # 10. NOSE POSITION
        # ====================================================

        nose_x = float(nose[0])
        nose_y = float(nose[1])

        # Frame center
        center_x = w / 2.0
        center_y = h / 2.0

        # Pixel offsets from frame center
        nose_offset_x = nose_x - center_x
        nose_offset_y = nose_y - center_y

        # Normalize by half-frame dimensions
        # approximately -1 to +1
        normalized_x = (
            nose_offset_x / max(w / 2.0, 1.0)
        )

        normalized_y = (
            nose_offset_y / max(h / 2.0, 1.0)
        )

        # Distance from frame center
        nose_distance = float(
            np.sqrt(
                normalized_x ** 2
                + normalized_y ** 2
            )
        )

        # ====================================================
        # 11. NOSE DIRECTION
        # ====================================================

        direction_threshold = 0.08

        horizontal = ""

        if normalized_x < -direction_threshold:
            horizontal = "LEFT"

        elif normalized_x > direction_threshold:
            horizontal = "RIGHT"

        else:
            horizontal = "CENTER"

        vertical = ""

        if normalized_y < -direction_threshold:
            vertical = "ABOVE"

        elif normalized_y > direction_threshold:
            vertical = "BELOW"

        else:
            vertical = "CENTER"

        if (
            horizontal == "CENTER"
            and vertical == "CENTER"
        ):
            nose_direction = "CENTER"

        elif horizontal == "CENTER":
            nose_direction = vertical

        elif vertical == "CENTER":
            nose_direction = horizontal

        else:
            nose_direction = (
                f"{vertical}-{horizontal}"
            )

        # ====================================================
        # Return all signals
        # ====================================================

        return FaceSignals(
            ear=ear,

            blink=blink,
            eyes_closed=eyes_closed,

            smile_score=smile_score,
            smiling=self.smiling,

            frown_score=frown_score,
            frowning=self.frowning,

            sad_score=sad_score,
            sad=self.sad,

            grimace_score=grimace_score,
            grimacing=self.grimacing,

            nose_x=nose_x,
            nose_y=nose_y,

            nose_offset_x=nose_offset_x,
            nose_offset_y=nose_offset_y,

            nose_distance=nose_distance,
            nose_direction=nose_direction,
        )




