from pathlib import Path
from urllib.request import urlopen

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"


def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        print(f"Model already exists: {MODEL_PATH}")
        return

    print("Downloading MediaPipe hand landmarker model...")
    with urlopen(MODEL_URL, timeout=60) as response:
        data = response.read()

    MODEL_PATH.write_bytes(data)
    if MODEL_PATH.stat().st_size < 100_000:
        MODEL_PATH.unlink(missing_ok=True)
        raise RuntimeError("Downloaded model looks invalid or incomplete.")

    print(f"Saved model to {MODEL_PATH}")
    print(f"Size: {MODEL_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
