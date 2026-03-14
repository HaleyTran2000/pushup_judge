"""
pipeline.py – InferencePipeline: ties together detection → pose → features →
              rep detection → LSTM classification.

Usage
-----
    from realtime.pipeline import InferencePipeline

    pipeline = InferencePipeline()          # loads all models
    result   = pipeline.process_frame(bgr_frame)
    # result is a ProcessResult namedtuple
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from realtime.config import (
    PD_CONF, IOU_MATCH_THRESH, LOCK_INTERVAL_SEC, VERDICT_HOLD_SEC,
    CONF_THRESH, KEYPOINT_TRAIL_LEN, KEYPOINT_TRAIL_KPS, SIGNAL_PLOT_LEN,
    REP_DOWN_THRESH, REP_UP_THRESH, REP_SIGNAL,
)
from realtime.features import extract_frame_features, add_velocity_inplace
from realtime.models   import load_models, classify_rep
from realtime.rep_detector import RepDetector
from realtime.visualizer   import draw_skeleton, draw_bounding_box, draw_hud, draw_keypoint_trails


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ProcessResult:
    """Data returned by InferencePipeline.process_frame()."""
    rep_count:   int          = 0
    n_correct:   int          = 0
    n_wrong:     int          = 0
    last_label:  str          = ""
    last_conf:   float        = 0.0
    elbow_angle: float        = float("nan")
    rep_fired:   bool         = False           # True on the frame a rep was detected
    rep_ready:   bool         = True            # True when recovery seen (detector primed)
    rep_history: list         = field(default_factory=list)  # [(label, conf), ...]


# ── Pipeline ──────────────────────────────────────────────────────────────────

class InferencePipeline:
    """
    Full end-to-end push-up judge pipeline.

    Parameters
    ----------
    fps : expected frames-per-second of the source (can be updated via
          ``reset(fps=...)`` once the actual FPS is known from the capture).
    """

    def __init__(self, fps: float = 30.0):
        self.person_model, self.pose_model, self.lstm_model = load_models()
        self._fps = fps
        self._init_state(fps)

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _init_state(self, fps: float) -> None:
        self.rep_detector   = RepDetector(fps=fps)
        self.prev_feat: Optional[dict] = None
        self.last_label:  str   = ""
        self.last_conf:   float = 0.0
        self.rep_count:   int   = 0
        self.n_correct:   int   = 0
        self.n_wrong:     int   = 0
        self.elbow_angle: float = float("nan")
        self.rep_history: list  = []
        self.frame_idx:   int   = 0
        # Verdict hold: counts down from VERDICT_HOLD_SEC*fps to 0 after each rep
        self._verdict_hold_total: int     = max(1, int(VERDICT_HOLD_SEC * fps))
        self._verdict_hold_remaining: int = 0
        # Person-lock state
        self.locked_box: Optional[list] = None
        self.last_lock_frame: int      = -(int(LOCK_INTERVAL_SEC * fps) + 1)
        self._lock_interval_frames: int = int(LOCK_INTERVAL_SEC * fps)
        self._signal_hist: list = []
        self._signal_hist_max: int = max(20, SIGNAL_PLOT_LEN)
        self._kp_trails: dict = {k: [] for k in KEYPOINT_TRAIL_KPS}

    def reset(self, fps: Optional[float] = None) -> None:
        """Reset all state (call between sessions or when FPS changes)."""
        self._init_state(fps if fps is not None else self._fps)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _box_iou(a: list, b: list) -> float:
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter  = max(0, xB - xA) * max(0, yB - yA)
        aA = (a[2] - a[0]) * (a[3] - a[1])
        aB = (b[2] - b[0]) * (b[3] - b[1])
        union = aA + aB - inter
        return inter / union if union > 0 else 0.0

    def _update_lock(self, boxes: np.ndarray) -> None:
        """Re-lock or track the person bounding box."""
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        if (
            self.locked_box is None
            or (self.frame_idx - self.last_lock_frame) >= self._lock_interval_frames
        ):
            self.locked_box      = boxes[int(np.argmax(areas))].tolist()
            self.last_lock_frame = self.frame_idx
        else:
            ious     = [self._box_iou(self.locked_box, boxes[i].tolist()) for i in range(len(boxes))]
            best_iou = max(ious)
            if best_iou >= IOU_MATCH_THRESH:
                self.locked_box = boxes[int(np.argmax(ious))].tolist()

    # ── main entry ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> ProcessResult:
        """
        Run the full pipeline on a single BGR frame.

        The frame is annotated **in-place** (skeleton, bounding box, HUD).

        Returns
        -------
        ProcessResult with up-to-date counters and a flag indicating whether
        a rep was detected on this particular frame.
        """
        self.frame_idx += 1
        H, W = frame.shape[:2]
        kps_arr = None

        # ── 1. Person detection ───────────────────────────────────────────────
        person_results = self.person_model.predict(
            frame, conf=PD_CONF, imgsz=640, verbose=False
        )
        if len(person_results[0].boxes) > 0:
            boxes_raw = person_results[0].boxes.xyxy.cpu().numpy()
            self._update_lock(boxes_raw)

        # ── 2. Crop + pose estimation ─────────────────────────────────────────
        if self.locked_box is not None:
            x1, y1, x2, y2 = [int(v) for v in self.locked_box]
            pad  = 10
            x1c  = max(0, x1 - pad);  y1c = max(0,  y1 - pad)
            x2c  = min(W, x2 + pad);  y2c = min(H,  y2 + pad)
            crop = frame[y1c:y2c, x1c:x2c]

            draw_bounding_box(frame, [x1c, y1c, x2c, y2c])

            if crop.size > 0:
                pose_results = self.pose_model.predict(
                    crop, conf=0.3, imgsz=256, verbose=False
                )
                if (
                    len(pose_results[0].keypoints) > 0
                    and pose_results[0].keypoints.data is not None
                    and len(pose_results[0].keypoints.data) > 0
                ):
                    kps_raw = pose_results[0].keypoints.data[0].cpu().numpy()
                    kps_arr = kps_raw.copy()
                    kps_arr[:, 0] += x1c   # back to full-frame coords
                    kps_arr[:, 1] += y1c

        # ── 3. Feature extraction ─────────────────────────────────────────────
        rep_fired = False
        if kps_arr is not None:
            feat = extract_frame_features(kps_arr, H, W)
            if self.prev_feat is not None:
                add_velocity_inplace(self.prev_feat, feat)
            else:
                feat["shoulder_velocity"]    = 0.0
                feat["hip_velocity"]         = 0.0
                feat["elbow_angle_velocity"] = 0.0
            self.prev_feat   = feat
            self.elbow_angle = feat.get("mean_elbow_angle", float("nan"))

            # ── 4. Rep detection ──────────────────────────────────────────────
            if self.rep_detector.push(feat):
                label, conf = classify_rep(self.lstm_model, self.rep_detector.last_window)
                self.rep_count   += 1
                self.last_label   = label
                self.last_conf    = conf
                self.rep_history.append((label, conf))
                if label == "correct":
                    self.n_correct += 1
                else:
                    self.n_wrong   += 1
                rep_fired = True
                self._verdict_hold_remaining = self._verdict_hold_total  # start countdown
                print(
                    f"  Rep {self.rep_count:2d} → {label:7s}  "
                    f"({conf * 100:.1f}%)  [frame {self.frame_idx}]"
                )

            draw_skeleton(frame, kps_arr)
            for k in KEYPOINT_TRAIL_KPS:
                if kps_arr[k, 2] > CONF_THRESH:
                    pt = (int(kps_arr[k, 0]), int(kps_arr[k, 1]))
                    self._kp_trails[k].append(pt)
                    if len(self._kp_trails[k]) > KEYPOINT_TRAIL_LEN:
                        self._kp_trails[k].pop(0)
            draw_keypoint_trails(frame, self._kp_trails)

            if not np.isnan(self.rep_detector.last_norm):
                self._signal_hist.append(self.rep_detector.last_norm)
                if len(self._signal_hist) > self._signal_hist_max:
                    self._signal_hist = self._signal_hist[-self._signal_hist_max:]

        # ── 5. HUD overlay ────────────────────────────────────────────────────
        rep_ready = self.rep_detector.recovery_seen
        hud_elbow = self.elbow_angle if REP_SIGNAL == "elbow_angle" else float("nan")
        # Decrement verdict hold counter every frame
        if self._verdict_hold_remaining > 0:
            self._verdict_hold_remaining -= 1
        draw_hud(
            frame,
            self.rep_count, self.n_correct, self.n_wrong,
            self.last_label, self.last_conf, hud_elbow,
            rep_ready=rep_ready,
            verdict_hold=self._verdict_hold_remaining,
            verdict_hold_total=self._verdict_hold_total,
            signal_hist=self._signal_hist,
            signal_down=self.rep_detector.last_down_line,
            signal_up=self.rep_detector.last_up_line,
            rep_state=self.rep_detector.last_state,
        )

        return ProcessResult(
            rep_count   = self.rep_count,
            n_correct   = self.n_correct,
            n_wrong     = self.n_wrong,
            last_label  = self.last_label,
            last_conf   = self.last_conf,
            elbow_angle = self.elbow_angle,
            rep_fired   = rep_fired,
            rep_ready   = rep_ready,
            rep_history = list(self.rep_history),
        )

    # ── session summary ───────────────────────────────────────────────────────

    def print_summary(self) -> None:
        print("=" * 47)
        print("   PUSH-UP SESSION SUMMARY")
        print("=" * 47)
        print(f"  Total reps   : {self.rep_count}")
        print(f"  Correct form : {self.n_correct}  "
              f"({100 * self.n_correct / max(self.rep_count, 1):.0f}%)")
        print(f"  Wrong form   : {self.n_wrong}")
        if self.rep_history:
            print()
            print("  Rep-by-rep:")
            for i, (lbl, conf) in enumerate(self.rep_history, 1):
                mark = "[OK]" if lbl == "correct" else "[!!]"
                print(f"    Rep {i:2d}: {mark} {lbl:7s}  {conf * 100:.1f}%")
        print("=" * 47)
