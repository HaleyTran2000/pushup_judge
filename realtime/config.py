"""
config.py – All paths and constants for the real-time push-up judge pipeline.
Edit WEBCAM_INDEX if necessary (0 = first built-in camera).
"""
import json
from pathlib import Path

# ── Root directory (one level up from realtime/) ─────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Person detector ───────────────────────────────────────────────────────────
_PD_INFO_PATH = ROOT / "person_detector" / "models" / "best_model" / "best_model_info.json"
PD_INFO       = json.loads(_PD_INFO_PATH.read_text())
PD_MODEL_PATH = Path(PD_INFO["pt_path"])
PD_CONF       = PD_INFO["conf_thresh"]

# ── Pose estimator ────────────────────────────────────────────────────────────
_POSE_JSON = json.loads((ROOT / "keypoint_detector" / "best_model_choice.json").read_text())
POSE_ID    = _POSE_JSON["model_id"]   # e.g. 'yolov8n-pose.pt'
POSE_PATH  = ROOT / "keypoint_detector" / POSE_ID

# ── LSTM classifier ───────────────────────────────────────────────────────────
_MODEL_DIR  = ROOT / "lstm_classifier" / "models"
LSTM_CKPT   = _MODEL_DIR / "best_model.pt"
LSTM_CFG    = json.loads((_MODEL_DIR / "model_config.json").read_text())

# Derived from LSTM_CFG
FEATURE_COLS = LSTM_CFG["feature_cols"]
T_FIXED      = LSTM_CFG["T_fixed"]
F_DIM        = LSTM_CFG["n_features"]
THRESHOLD    = LSTM_CFG["threshold"]

# ── Keypoint confidence threshold (must match training pipeline) ──────────────
CONF_THRESH = 0.3

# ── Person-tracking settings ──────────────────────────────────────────────────
LOCK_INTERVAL_SEC = 20.0    # re-lock to largest-bbox person every N seconds
IOU_MATCH_THRESH  = 0.30    # min IoU to consider "same person" between re-locks

# ── Display settings ──────────────────────────────────────────────────────────
WINDOW_TITLE     = "Push-up Judge  (press Q to quit)"
DISPLAY_W        = 960          # resize display window width (0 = no resize)
DISPLAY_H        = 540          # resize display window height

# How long (seconds) to show the big verdict banner after each rep
VERDICT_HOLD_SEC = 1.0

# ── Visualization helpers ────────────────────────────────────────────────────
KEYPOINT_TRAIL_LEN = 18
KEYPOINT_TRAIL_KPS = (0, 5, 6, 11, 12)  # nose, shoulders, hips
SIGNAL_PLOT_LEN    = 120

# ── Rep detector settings (state machine) ────────────────────────────────────
# Select the signal used for rep counting.
REP_SIGNAL = "shoulder_height"  # "elbow_angle" | "shoulder_height" | "hip_height" | "shoulder_hip_dist" | "nose_hip_dist"

# Require a plausible push-up pose to enable rep counting.
REP_REQUIRE_POSE = True
# Hip angles must be fairly straight (degrees). Uses whichever side is valid.
REP_MIN_HIP_ANGLE = 130.0
# Body straightness (perpendicular distance) must be below this.
REP_MAX_BODY_STRAIGHTNESS = 0.12

# Normalize elbow angle into [0, 1] using these bounds.
REP_ANGLE_MIN = 80.0
REP_ANGLE_MAX = 170.0

# Normalize vertical distance signals into [0, 1] using these bounds.
# Distance is computed in normalized image coordinates.
REP_DIST_MIN = 0.03
REP_DIST_MAX = 0.30

# Normalize vertical height signals into [0, 1] using these bounds.
# Height is a normalized y value in image coordinates.
REP_HEIGHT_MIN = 0.15
REP_HEIGHT_MAX = 0.85

# Hysteresis thresholds in normalized space (conservative defaults).
REP_DOWN_THRESH = 0.45
REP_UP_THRESH   = 0.60
# Minimum slope (per second) required to change state.
REP_MIN_SLOPE   = 0.015
# EMA smoothing for the normalized signal.
REP_EMA_ALPHA   = 0.55
# Time-based cooldown between reps.
REP_MIN_GAP_SEC = 0.20
# Pre-roll frames to include before descent (seconds).
REP_PRE_ROLL_SEC = 0.20

# ── Adaptive midline settings (curve-based, no fixed time window) ───────────
# Use recent peaks/troughs to compute a midline and hysteresis band.
REP_USE_ADAPTIVE = True
REP_EXTREMA_HISTORY = 2
REP_MIDLINE_BAND = 0.06
REP_MIN_RANGE = 0.04
REP_WARMUP_SAMPLES = 10
REP_PLATEAU_FRAMES = 4
REP_PLATEAU_SLOPE  = 0.02

# ── Webcam / video source ───────────────────────────────────────────────────
WEBCAM_INDEX  = 0       # default camera index when no video is set

# ┌─────────────────────────────────────────────────────────────
# │  TEST_VIDEO  –  set a path here to run on a video file instead of webcam  │
# │  e.g.  TEST_VIDEO = "/Users/you/Videos/mypushup.mp4"                       │
# │  Leave as empty string "" to use the webcam (WEBCAM_INDEX).               │
# └─────────────────────────────────────────────────────────────
TEST_VIDEO = "/Users/haleytran/Downloads/pushup_judge/3.mp4"  #/Users/haleytran/Downloads/pushup_judge/1.mp4 ← paste your video path here

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = ROOT / "realtime" / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_VIDEO = str(OUTPUT_DIR / "test_2.mp4")
