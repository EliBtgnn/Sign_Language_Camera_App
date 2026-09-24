from __future__ import annotations
import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def normalize_landmarks(landmarks, handedness: str | None = None) -> np.ndarray:
    points = np.asarray([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    points -= points[0]  # wrist at origin

    scale = np.linalg.norm(points[:, :2], axis=1).max()
    if scale < 1e-6:
        return np.zeros(63, dtype=np.float32)

    points /= scale

    if handedness and handedness.lower() == "left":
        points[:, 0] *= -1.0

    return points.reshape(-1).astype(np.float32)


def draw_hand(frame, landmarks, handedness: str | None = None):
    import cv2

    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

    for x, y in pts:
        cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        cv2.circle(frame, (x, y), 2, (0, 0, 0), -1)

    if pts:
        cv2.putText(
            frame, handedness or "Hand",
            (pts[0][0] + 8, pts[0][1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
