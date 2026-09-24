from __future__ import annotations
import argparse
from collections import Counter, deque
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from features import draw_hand, normalize_landmarks

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
CLASSIFIER_PATH = ROOT / "models" / "gesture_classifier.joblib"


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


def stable_prediction(history, min_votes):
    if not history:
        return None, 0.0
    label, votes = Counter(history).most_common(1)[0]
    return (label, votes / len(history)) if votes >= min_votes else (None, 0.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--smoothing", type=int, default=9)
    parser.add_argument("--min-votes", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Run: python scripts/download_model.py")
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError("Run: python src/train.py")

    classifier = joblib.load(CLASSIFIER_PATH)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    history = deque(maxlen=max(1, args.smoothing))
    output_text = ""
    current_stable = None

    with create_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )

            prediction = None
            confidence = 0.0

            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                handedness = (
                    result.handedness[0][0].category_name
                    if result.handedness and result.handedness[0] else None
                )
                draw_hand(frame, landmarks, handedness)

                features = normalize_landmarks(
                    landmarks, handedness
                ).reshape(1, -1)

                probabilities = classifier.predict_proba(features)[0]
                i = int(np.argmax(probabilities))
                confidence = float(probabilities[i])

                if confidence >= args.threshold:
                    prediction = str(classifier.classes_[i])
                    history.append(prediction)
                else:
                    history.clear()
            else:
                history.clear()

            stable, stability = stable_prediction(history, args.min_votes)
            if stable is not None:
                current_stable = stable

            cv2.rectangle(frame, (10, 10), (700, 120), (0, 0, 0), -1)
            cv2.putText(frame, f"Prediction: {prediction or 'No confident sign'}",
                        (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}  Stability: {stability:.2f}",
                        (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            cv2.putText(frame, f"Text: {output_text}",
                        (25, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            cv2.putText(frame, "SPACE=append  C=clear  Q=quit",
                        (20, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255,255,255), 1)

            cv2.imshow("Real-Time Sign Language Recognizer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                output_text = ""
            elif key == 32 and current_stable:
                output_text += current_stable
                history.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
