# Development Guide

This document explains the main components of the project and how they fit together. It is intended for anyone who wants to understand the implementation or extend the recognizer.

## Architecture

The project is split into a few simple stages:

```text
Webcam
   ↓
Hand Landmark Detection
   ↓
Feature Extraction
   ↓
Random Forest Classifier
   ↓
Temporal Smoothing
   ↓
Predicted Gesture
```

Each stage has a separate responsibility, which makes it easier to experiment with different models or features without changing the entire pipeline.

## Components

### Hand Landmark Detection

`src/app.py` and `src/collect_data.py` use **MediaPipe Hand Landmarker** to detect the hand and its 21 landmarks.

The detector is only responsible for locating the hand. It does **not** determine which gesture is being performed.

### Feature Extraction

`src/features.py` converts the 21 landmarks into a 63-value feature vector.

The coordinates are normalized relative to the wrist and scaled to reduce sensitivity to:

* hand position
* hand size
* distance from the camera

This gives the classifier a more consistent representation of the same gesture.

### Classifier

`src/train.py` trains a **Random Forest** classifier using the collected landmark features.

Random Forest is a useful baseline for this project because the input is a relatively small tabular dataset and the model is simple to train and inspect.

The classifier can be replaced later with another model without changing the landmark detection stage.

### Temporal Smoothing

Real-time predictions can change from frame to frame, even when the hand is not moving.

The application therefore keeps a short history of recent predictions and selects the majority label.

```text
Frame predictions:

A A A B A A A
      ↓
   majority
      ↓
      A
```

This simple approach helps reduce visual flickering without requiring a more complex temporal model.

## Changing the Vocabulary

The labels stored in `data/landmarks.csv` become the classes used by the classifier.

To add a new gesture:

```bash
python src/collect_data.py --label NEW_SIGN --samples 500
python src/train.py
python src/app.py
```

When collecting data, try to include different hand positions, orientations, lighting conditions, and backgrounds.

For a larger vocabulary, keeping the number of samples reasonably balanced between classes is also important.

## From Gestures to Words

The current application lets the user press **Space** to append the current stable prediction to the output text.

For example:

```text
HELLO → YES → THANK_YOU
```

A more robust system would need to distinguish between:

* a new sign starting
* the same sign being held
* a sign ending
* the next sign beginning

This could be handled with **debouncing**, confidence thresholds, and sign-transition detection so that holding a gesture does not append it repeatedly.

## Extending to Dynamic Signs

The current classifier operates on individual frames, which works well for static gestures but not for signs where movement is part of the meaning.

A dynamic-sign pipeline could instead process a sequence of frames:

```text
Frame 1 ─┐
Frame 2  │
Frame 3  │
  ...    ├──→ Temporal Model ──→ Sign
Frame 30 │
Frame 31 ┘
```

Possible temporal models include:

* LSTM
* GRU
* Transformer

Useful features could include:

* normalized x/y/z landmarks
* frame-to-frame landmark velocity
* hand orientation
* handedness
* body pose landmarks
* facial landmarks

## Two-Handed Signs

The current pipeline focuses on one hand.

To support two-handed gestures, the system would need to:

1. detect up to two hands;
2. consistently identify the left and right hand;
3. extract features for both;
4. concatenate the representations before classification.

For example:

```text
Left Hand  → 63 features ─┐
                           ├──→ Model
Right Hand → 63 features ─┘
```

This would allow the same general architecture to be extended without changing the entire application.

## Production-Oriented Architecture

For a more robust real-time implementation, the pipeline could be separated into asynchronous stages:

```text
Camera Thread
      ↓
Frame Queue
      ↓
MediaPipe
      ↓
Landmark Sequence Buffer
      ↓
Temporal Model
      ↓
Confidence / Debounce Layer
      ↓
UI or API
```

Separating these components makes it easier to control latency and prevent slow model inference from blocking the camera loop.

MediaPipe's Python Tasks API provides image, video, and live-stream processing modes. A future version could use the live-stream API for asynchronous processing rather than performing every operation directly inside the main camera loop.

## Model Experiments

The Random Forest is only a baseline. Different models could be compared depending on the type of input.

### Static Gestures

Potential models include:

* Random Forest
* SVM
* Gradient Boosting
* Small MLP

### Dynamic Gestures

For sequences of landmarks, possible approaches include:

* LSTM / GRU
* Transformer-based models

When comparing models, it is useful to keep the dataset and evaluation protocol fixed so that the results are actually comparable.

## Evaluation

A random frame-level train/test split can give overly optimistic results because frames from the same recording session may be very similar.

A better experiment is a **signer-independent split**, where people appearing in the test set are not present during training.

For example:

```text
People 1–3 → Training
Person 4   → Validation
Person 5   → Test
```

Useful metrics include:

* Accuracy
* Macro F1
* Per-class recall
* Confusion matrix
* FPS
* End-to-end latency

For a real-time application, model accuracy is only part of the evaluation. Responsiveness and prediction stability also matter.

## Responsible Use

This project is an experimental and educational prototype, not an authoritative sign-language interpreter.

Recognition errors can have serious consequences in contexts such as:

* medical communication
* legal services
* education
* emergency situations

Applications in these settings would require substantially more data, testing, validation, and human oversight.

## Possible Next Steps

Some natural directions for extending the project are:

* [ ] Add more gesture classes
* [ ] Improve the data collection pipeline
* [ ] Support two-handed gestures
* [ ] Add dynamic gesture recognition
* [ ] Experiment with temporal models
* [ ] Add signer-independent evaluation
* [ ] Improve prediction debouncing
* [ ] Add a real-time performance benchmark
