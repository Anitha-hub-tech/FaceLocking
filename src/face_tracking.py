import argparse

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np

from src.align import align_face_5pt
from src.face_signals import FaceSignalExtractor
from src.recognize import (
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    HaarFaceMesh5pt,
    load_db_npz,
)


class LockState(Enum):
    SEARCHING = auto()
    LOCKED = auto()
    LOST = auto()


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))

    return inter / float(area_a + area_b - inter)


def center(box):
    x1, y1, x2, y2 = box

    return np.array(
        [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        ],
        dtype=np.float32,
    )


@dataclass
class TrackingSignal:
    error_x: float
    error_y: float
    horizontal: str
    vertical: str


class LockedFaceTracker:
    def __init__(
        self,
        target_name: str,
        detector,
        embedder,
        matcher,
        verify_every: int = 10,
        lost_timeout: int = 24,
        ema_alpha: float = 0.30,
        dead_zone: float = 0.07,
    ):
        self.target_name = target_name
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher

        self.verify_every = verify_every
        self.lost_timeout = lost_timeout
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone

        self.state = LockState.SEARCHING

        self.last_box = None
        self.smooth_center = None

        self.lost_frames = 0
        self.frame_index = 0

    @staticmethod
    def box(face):
        return (
            face.x1,
            face.y1,
            face.x2,
            face.y2,
        )

    def identity(self, frame, face):
        aligned, _ = align_face_5pt(
            frame,
            face.kps,
            out_size=(112, 112),
        )

        return self.matcher.match(
            self.embedder.embed(aligned)
        )

    def target_is_verified(self, frame, face) -> bool:
        match = self.identity(frame, face)

        return (
            match.accepted
            and match.name == self.target_name
        )

    def acquire(self, frame, faces):
        best = None
        best_similarity = -1.0

        for face in faces:
            match = self.identity(frame, face)

            if (
                match.accepted
                and match.name == self.target_name
                and match.similarity > best_similarity
            ):
                best = face
                best_similarity = match.similarity

        return best

    def associate(self, faces):
        if self.last_box is None or not faces:
            return None

        last_center = center(self.last_box)

        last_diag = max(
            np.linalg.norm(
                np.array(
                    [
                        self.last_box[2] - self.last_box[0],
                        self.last_box[3] - self.last_box[1],
                    ],
                    dtype=np.float32,
                )
            ),
            1.0,
        )

        ranked = []

        for face in faces:
            box = self.box(face)

            overlap = iou(
                self.last_box,
                box,
            )

            displacement = (
                np.linalg.norm(
                    center(box) - last_center
                )
                / last_diag
            )

            score = overlap - 0.35 * displacement

            ranked.append(
                (score, face)
            )

        score, candidate = max(
            ranked,
            key=lambda item: item[0],
        )

        return (
            candidate
            if score > -0.30
            else None
        )

    def update(self, frame):
        self.frame_index += 1

        faces = self.detector.detect(
            frame,
            max_faces=8,
        )

        if self.state == LockState.SEARCHING:

            candidate = self.acquire(
                frame,
                faces,
            )

        else:

            candidate = self.associate(faces)

            if (
                candidate is not None
                and (
                    self.state == LockState.LOST
                    or self.frame_index % self.verify_every == 0
                )
                and not self.target_is_verified(
                    frame,
                    candidate,
                )
            ):
                candidate = None

        if candidate is None:

            self.lost_frames += 1

            if self.last_box is not None:
                self.state = LockState.LOST

            if self.lost_frames > self.lost_timeout:

                self.state = LockState.SEARCHING

                self.last_box = None
                self.smooth_center = None

            return None, None

        self.state = LockState.LOCKED
        self.lost_frames = 0

        self.last_box = self.box(candidate)

        raw_center = center(
            self.last_box
        )

        if self.smooth_center is None:

            self.smooth_center = raw_center

        else:

            a = self.ema_alpha

            self.smooth_center = (
                a * raw_center
                + (1.0 - a) * self.smooth_center
            )

        return (
            candidate,
            self.position_signal(frame.shape),
        )

    def position_signal(self, shape) -> TrackingSignal:

        height, width = shape[:2]

        ex = float(
            (
                self.smooth_center[0]
                - width / 2.0
            )
            / (width / 2.0)
        )

        ey = float(
            (
                self.smooth_center[1]
                - height / 2.0
            )
            / (height / 2.0)
        )

        horizontal = "CENTER"
        vertical = "CENTER"

        if ex < -self.dead_zone:
            horizontal = "LEFT"

        elif ex > self.dead_zone:
            horizontal = "RIGHT"

        if ey < -self.dead_zone:
            vertical = "UP"

        elif ey > self.dead_zone:
            vertical = "DOWN"

        return TrackingSignal(
            ex,
            ey,
            horizontal,
            vertical,
        )


def draw_label(
    frame,
    text,
    xy,
    color,
    scale=0.62,
):
    cv2.putText(
        frame,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_nose_tracking(
    frame,
    nose_x,
    nose_y,
):
    """
    Draw a line from the center of the camera frame
    to the detected nose tip.
    """

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # Center of the camera frame
    # --------------------------------------------------------

    frame_center = (
        width // 2,
        height // 2,
    )

    # --------------------------------------------------------
    # Nose position
    # --------------------------------------------------------

    nose_point = (
        int(nose_x),
        int(nose_y),
    )

    # --------------------------------------------------------
    # Draw horizontal and vertical center lines
    # --------------------------------------------------------

    cv2.line(
        frame,
        (frame_center[0], 0),
        (frame_center[0], height),
        (255, 255, 0),
        1,
    )

    cv2.line(
        frame,
        (0, frame_center[1]),
        (width, frame_center[1]),
        (255, 255, 0),
        1,
    )

    # --------------------------------------------------------
    # Draw line from frame center to nose
    # --------------------------------------------------------

    cv2.line(
        frame,
        frame_center,
        nose_point,
        (0, 255, 255),
        3,
    )

    # --------------------------------------------------------
    # Draw frame center
    # --------------------------------------------------------

    cv2.circle(
        frame,
        frame_center,
        7,
        (255, 255, 0),
        -1,
    )

    # --------------------------------------------------------
    # Draw nose point
    # --------------------------------------------------------

    cv2.circle(
        frame,
        nose_point,
        8,
        (0, 255, 255),
        -1,
    )

    # --------------------------------------------------------
    # Label the nose
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "NOSE",
        (
            nose_point[0] + 10,
            nose_point[1] - 10,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        required=True,
        help="enrolled identity to lock",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.34,
    )

    args = parser.parse_args()

    # ========================================================
    # Face detector
    # ========================================================

    detector = HaarFaceMesh5pt(
        min_size=(70, 70),
        debug=False,
    )

    # ========================================================
    # ArcFace
    # ========================================================

    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )

    # ========================================================
    # Face database
    # ========================================================

    matcher = FaceDBMatcher(
        load_db_npz(
            Path("data/db/face_db.npz")
        ),
        dist_thresh=args.threshold,
    )

    # ========================================================
    # Tracker
    # ========================================================

    tracker = LockedFaceTracker(
        args.target,
        detector,
        embedder,
        matcher,
    )

    # ========================================================
    # Facial signals
    # ========================================================

    signals = FaceSignalExtractor()

    # ========================================================
    # Camera
    # ========================================================

    cap = cv2.VideoCapture(
        args.camera
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Camera {args.camera} not available"
        )

    # Try 1280x720
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280,
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720,
    )

    # ========================================================
    # Blink state
    # ========================================================

    blink_total = 0
    previous_blink = False

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            # =================================================
            # Track target
            # =================================================

            locked_face, position = tracker.update(
                frame
            )

            # =================================================
            # TARGET FOUND
            # =================================================

            if locked_face is not None:

                x1, y1, x2, y2 = tracker.box(
                    locked_face
                )

                # ------------------------------------------------
                # Draw face bounding box
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                # ------------------------------------------------
                # Analyze face
                # ------------------------------------------------

                signal = signals.analyze(
                    frame,
                    tracker.box(locked_face),
                )

                if signal is not None:

                    # ============================================
                    # Blink counter
                    # ============================================

                    if signal.blink and not previous_blink:
                        blink_total += 1

                    previous_blink = signal.blink

                    # ============================================
                    # Basic facial signals
                    # ============================================

                    draw_label(
                        frame,
                        f"EAR: {signal.ear:.2f}",
                        (20, 40),
                        (0, 255, 255),
                    )

                    draw_label(
                        frame,
                        f"Smile: {signal.smile_score:.2f}",
                        (20, 70),
                        (0, 255, 255),
                    )

                    draw_label(
                        frame,
                        f"Blinks: {blink_total}",
                        (20, 100),
                        (0, 255, 255),
                    )

                    # ============================================
                    # Frown
                    # ============================================

                    frown_status = (
                        "YES"
                        if signal.frowning
                        else "NO"
                    )

                    draw_label(
                        frame,
                        f"Frown: {frown_status}",
                        (20, 130),
                        (0, 200, 255),
                    )

                    # ============================================
                    # Sad expression estimate
                    # ============================================

                    sad_status = (
                        "YES"
                        if signal.sad
                        else "NO"
                    )

                    draw_label(
                        frame,
                        f"Sad estimate: {sad_status}",
                        (20, 160),
                        (0, 165, 255),
                    )

                    # ============================================
                    # Grimace
                    # ============================================

                    grimace_status = (
                        "YES"
                        if signal.grimacing
                        else "NO"
                    )

                    draw_label(
                        frame,
                        f"Grimace: {grimace_status}",
                        (20, 190),
                        (0, 165, 255),
                    )

                    # ============================================
                    # Nose tracking
                    # ============================================

                    draw_nose_tracking(
                        frame,
                        signal.nose_x,
                        signal.nose_y,
                    )

                    # ============================================
                    # Nose information
                    # ============================================

                    draw_label(
                        frame,
                        f"Nose: {signal.nose_direction}",
                        (20, 220),
                        (0, 255, 255),
                    )

                    draw_label(
                        frame,
                        f"Nose distance: {signal.nose_distance:.3f}",
                        (20, 250),
                        (0, 255, 255),
                    )

                    # ============================================
                    # Nose coordinates
                    # ============================================

                    draw_label(
                        frame,
                        f"Nose XY: "
                        f"{int(signal.nose_x)}, "
                        f"{int(signal.nose_y)}",
                        (20, 280),
                        (0, 255, 255),
                    )

                    # ============================================
                    # Eyes closed warning
                    # ============================================

                    if signal.eyes_closed:

                        draw_label(
                            frame,
                            "EYES CLOSED",
                            (20, 320),
                            (0, 0, 255),
                            scale=0.75,
                        )

                # ------------------------------------------------
                # Locked label
                # ------------------------------------------------

                draw_label(
                    frame,
                    f"LOCKED: {args.target}",
                    (
                        x1,
                        max(30, y1 - 10),
                    ),
                    (0, 255, 0),
                )

                # ------------------------------------------------
                # Overall face position
                # ------------------------------------------------

                if position is not None:

                    draw_label(
                        frame,
                        f"H: {position.horizontal}",
                        (20, 360),
                        (255, 255, 0),
                    )

                    draw_label(
                        frame,
                        f"V: {position.vertical}",
                        (20, 390),
                        (255, 255, 0),
                    )

            # =================================================
            # TARGET NOT FOUND
            # =================================================

            else:

                if tracker.state == LockState.LOST:

                    status = "FACE NOT DETECTED"

                    # --------------------------------------------
                    # Red warning
                    # --------------------------------------------

                    cv2.rectangle(
                        frame,
                        (10, 10),
                        (
                            frame.shape[1] - 10,
                            70,
                        ),
                        (0, 0, 255),
                        3,
                    )

                    draw_label(
                        frame,
                        status,
                        (30, 52),
                        (0, 0, 255),
                        scale=0.85,
                    )

                else:

                    status = "SEARCHING"

                    draw_label(
                        frame,
                        status,
                        (20, 40),
                        (0, 165, 255),
                    )

            # =================================================
            # Controls
            # =================================================

            draw_label(
                frame,
                "Q: Quit   R: Reset   +/-: Threshold   D: Debug",
                (
                    20,
                    frame.shape[0] - 20,
                ),
                (255, 255, 255),
                scale=0.50,
            )

            # =================================================
            # Display
            # =================================================

            cv2.imshow(
                "Locked Face Tracking",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            # -------------------------------------------------
            # Quit
            # -------------------------------------------------

            if key == ord("q"):
                break

            # -------------------------------------------------
            # Reset tracker
            # -------------------------------------------------

            elif key == ord("r"):

                tracker.state = LockState.SEARCHING
                tracker.last_box = None
                tracker.smooth_center = None
                tracker.lost_frames = 0

                signals.reset()

                blink_total = 0
                previous_blink = False

            # -------------------------------------------------
            # Increase threshold
            # -------------------------------------------------

            elif key in (
                ord("+"),
                ord("="),
            ):

                args.threshold = min(
                    0.90,
                    args.threshold + 0.01,
                )

                tracker.matcher.dist_thresh = (
                    args.threshold
                )

                print(
                    f"Threshold: {args.threshold:.2f}"
                )

            # -------------------------------------------------
            # Decrease threshold
            # -------------------------------------------------

            elif key in (
                ord("-"),
                ord("_"),
            ):

                args.threshold = max(
                    0.05,
                    args.threshold - 0.01,
                )

                tracker.matcher.dist_thresh = (
                    args.threshold
                )

                print(
                    f"Threshold: {args.threshold:.2f}"
                )

            # -------------------------------------------------
            # Debug information
            # -------------------------------------------------

            elif key == ord("d"):

                print(
                    f"State: {tracker.state.name}"
                )

                print(
                    f"Lost frames: "
                    f"{tracker.lost_frames}"
                )

                print(
                    f"Threshold: "
                    f"{args.threshold:.2f}"
                )

    finally:

        cap.release()

        signals.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
