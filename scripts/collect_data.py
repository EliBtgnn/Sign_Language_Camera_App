from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path

import cv2
import mediapipe as mp
from features import normalize_landmarks

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
DATA_PATH = ROOT / "data" / "landmarks.csv"


def create_landmarker():
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def ensure_csv():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        with DATA_PATH.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["label"] + [f"f{i}" for i in range(63)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run: python scripts/download_model.py")

    ensure_csv()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    collected = 0
    recording = False
    last_save = 0.0

    print(f"Collecting: {args.label}")
    print("SPACE=start/stop recording, Q=quit")

    with create_landmarker() as landmarker, DATA_PATH.open(
        "a", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)

        while collected < args.samples:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                handedness = (
                    result.handedness[0][0].category_name
                    if result.handedness and result.handedness[0] else None
                )
                features = normalize_landmarks(landmarks, handedness)

                if recording and time.time() - last_save >= 0.05:
                    writer.writerow([args.label] + features.tolist())
                    f.flush()
                    collected += 1
                    last_save = time.time()

                h, w = frame.shape[:2]
                for lm in landmarks:
                    cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0,255,0), -1)

            status = "RECORDING" if recording else "PAUSED"
            cv2.putText(frame, f"{args.label} {status} {collected}/{args.samples}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,255), 2)
            cv2.putText(frame, "SPACE=start/stop  Q=quit",
                        (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            cv2.imshow("Collect sign data", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:
                recording = not recording

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {collected} samples for {args.label}")


if __name__ == "__main__":
    main()
