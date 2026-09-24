# Real-Time Sign Language Gesture Recognizer

A real-time hand gesture recognition system built with **Python, OpenCV, MediaPipe, and scikit-learn**.

The project uses a webcam to detect hand landmarks, extracts a normalized representation of the hand, and classifies custom gestures with a **Random Forest** model. Predictions are smoothed over time to make the output more stable.

> **Note:** This is a gesture recognition prototype, not a complete sign-language translator. It currently focuses on single-hand, mostly static gestures.

## Demo

<!-- video here -->

<!-- ![Demo](assets/demo.gif) -->

## How It Works

```text
Webcam
   ↓
OpenCV
   ↓
MediaPipe Hand Landmarks
   ↓
Feature Normalization (63 features)
   ↓
Random Forest
   ↓
Temporal Smoothing
   ↓
Predicted Gesture
```

MediaPipe detects **21 hand landmarks**, each with x/y/z coordinates.

The landmarks are normalized to reduce the effect of hand position and scale, then flattened into **63 features** for the classifier.

A short prediction history is also used to reduce frame-to-frame flickering.

## Tech Stack

* **Python**
* **OpenCV** — webcam and real-time video processing
* **MediaPipe** — hand landmark detection
* **scikit-learn** — Random Forest classification
* **NumPy / Pandas** — data processing
* **Joblib** — model persistence

## Getting Started

### Requirements

* Python 3.10+
* Webcam
* Windows, macOS, or Linux

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Create a virtual environment

#### Windows

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download the MediaPipe model

```bash
python scripts/download_model.py
```

This downloads the hand landmark model to:

```text
models/hand_landmarker.task
```

## Collecting Data

The classifier is trained on data collected directly from the webcam.

For example:

```bash
python src/collect_data.py --label A --samples 300
python src/collect_data.py --label B --samples 300
python src/collect_data.py --label C --samples 300
```

Press **Space** to start/stop recording and **Q** to quit.

When collecting data, try to vary:

* hand position
* distance from the camera
* rotation
* lighting
* background

The collected landmarks are stored in:

```text
data/landmarks.csv
```

You can define your own vocabulary, for example:

```text
HELLO
YES
NO
THANK_YOU
```

## Training

Train the classifier with:

```bash
python src/train.py
```

The script evaluates the model on held-out data and saves the trained classifier to:

```text
models/gesture_classifier.joblib
```

## Running the Recognizer

Start the real-time application:

```bash
python src/app.py
```

### Controls

| Key     | Action                               |
| ------- | ------------------------------------ |
| `Q`     | Quit                                 |
| `C`     | Clear accumulated text               |
| `Space` | Append the current stable prediction |

### Useful Options

```bash
python src/app.py --camera 1
python src/app.py --smoothing 15
python src/app.py --threshold 0.70
```

## Limitations & Future Work

The current version focuses on **single-hand static gestures**.

Possible extensions include:

### More Gestures

Add more classes and retrain the model with a larger and more diverse dataset.

### Two-Handed Gestures

Extend the pipeline to detect and consistently represent two hands.

### Dynamic Signs

Some signs depend on movement rather than a single pose. A future version could process sequences of landmarks using an **LSTM, GRU, or Transformer** instead of classifying individual frames.

```text
Landmark Sequence
        ↓
Temporal Model
        ↓
Gesture / Sign
```

### Full Sign-Language Translation

A more complete system could combine hand, body, and facial landmarks with a temporal model and a language-decoding stage.

```text
Hands + Pose + Face
        ↓
Temporal Model
        ↓
Sign Sequence
        ↓
Language Model
        ↓
Sentence
```

## What I Learned

This project was an opportunity to explore the full pipeline of a small machine-learning application:

* Real-time computer vision
* Feature engineering from landmark data
* Collecting and preparing a custom dataset
* Training and evaluating a classifier
* Handling noisy real-time predictions
* Turning an ML model into an interactive application


## License
! This project is intended for educational and experimental use.

If you plan to distribute it, add an appropriate license and check the licenses of the dependencies, models, and datasets you use.
