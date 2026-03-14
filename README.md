# Push-up Judge

Real-time push-up counter and form classifier using a webcam or video file.

**Pipeline:** YOLO person detector → YOLO pose estimator → feature engineering → BiLSTM form classifier + rep detector → live HUD overlay.

---

## Project Structure

```
pushup_tracker/
├── run_webcam.py           ← main entry point
├── requirements.txt
├── realtime/
│   ├── config.py           ← all tuneable settings (webcam index, video path, thresholds)
│   ├── pipeline.py         ← end-to-end InferencePipeline
│   ├── rep_detector.py     ← adaptive state-machine rep counter
│   ├── features.py         ← per-frame feature extraction
│   ├── models.py           ← model loaders (YOLO + BiLSTM)
│   └── visualizer.py       ← HUD, skeleton, bounding-box drawing
├── person_detector/models/ ← trained YOLO person detector
├── keypoint_detector/models/ ← trained YOLO pose estimator
└── lstm_classifier/models/ ← trained BiLSTM form classifier
```

---

## Setup

### 1. Prerequisites

- Python 3.10 or 3.11
- A working webcam **or** a push-up video file

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Apple Silicon (M-series Mac):** PyTorch will automatically use the MPS backend for GPU acceleration — no extra steps needed.

---

## Running the App

### Webcam (default)

```bash
python run_webcam.py
```

Uses camera index `0` by default. Change `WEBCAM_INDEX` in [realtime/config.py](realtime/config.py) if you have multiple cameras.

### Specific webcam index

```bash
python run_webcam.py --source 1
```

### Pre-recorded video file

```bash
python run_webcam.py --source /path/to/your/video.mp4
```

Annotated output is saved automatically to `realtime/output/` when running a video file.

### Run on video without saving output

```bash
python run_webcam.py --source /path/to/your/video.mp4 --no-save
```

### Headless / SSH (no display window)

```bash
python run_webcam.py --no-display
```

### Set a default video in config

Edit [realtime/config.py](realtime/config.py):

```python
TEST_VIDEO = "/path/to/your/video.mp4"   # leave "" for webcam
```

Then just run `python run_webcam.py` with no arguments.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Q` or `Esc` | Quit and print session summary |
| `R` | Reset rep counter and pipeline state |
| `S` | Toggle output saving on/off |

---

## Configuration

All settings live in [realtime/config.py](realtime/config.py). Key options:

| Setting | Default | Description |
|---------|---------|-------------|
| `WEBCAM_INDEX` | `0` | Camera index when no video is set |
| `TEST_VIDEO` | `""` | Default video path (overrides webcam) |
| `DISPLAY_W / DISPLAY_H` | `960 / 540` | Output window size |
| `REP_SIGNAL` | `"shoulder_height"` | Signal used for rep counting (`"elbow_angle"`, `"shoulder_height"`, `"hip_height"`, `"shoulder_hip_dist"`, `"nose_hip_dist"`) |
| `REP_REQUIRE_POSE` | `True` | Only count reps when a valid push-up pose is detected |
| `VERDICT_HOLD_SEC` | `1.0` | Seconds to display the form verdict banner |

---

## Sanity-Check Tests

Run the unit tests to verify the pipeline without a camera:

```bash
python realtime/_test_pipeline.py
```

Expected output:

```
All imports OK
Test 1: sitting still at 160° …   PASS ✓
Test 2: random noise …             PASS ✓
...
9/9 tests passed – pipeline ready.
```

---

## Models Used

| Component | Model |
|-----------|-------|
| Person detector | Fine-tuned YOLO (see `person_detector/models/best_model/`) |
| Pose estimator | Fine-tuned YOLO-pose (see `keypoint_detector/models/best_model/`) |
| Form classifier | BiLSTM (15 features × 20 frames, val F1 = 0.846) |
