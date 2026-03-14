"""
rep_detector.py – Adaptive push-up rep detector (state machine).

Design goals
------------
- Normalize the signal (angle) so it is scale/position invariant.
- Hysteresis thresholds to avoid bounce.
- Velocity check to ignore tiny jitter.
- Time-based cooldown so fast and slow reps both count.
"""
import numpy as np

from realtime.config import (
    REP_SIGNAL,
    REP_ANGLE_MIN, REP_ANGLE_MAX,
    REP_DIST_MIN, REP_DIST_MAX,
    REP_HEIGHT_MIN, REP_HEIGHT_MAX,
    REP_DOWN_THRESH, REP_UP_THRESH,
    REP_MIN_SLOPE, REP_EMA_ALPHA,
    REP_MIN_GAP_SEC, REP_PRE_ROLL_SEC,
    REP_REQUIRE_POSE, REP_MIN_HIP_ANGLE, REP_MAX_BODY_STRAIGHTNESS,
    REP_USE_ADAPTIVE, REP_EXTREMA_HISTORY, REP_MIDLINE_BAND, REP_MIN_RANGE,
    REP_WARMUP_SAMPLES, REP_PLATEAU_FRAMES, REP_PLATEAU_SLOPE,
)


class RepDetector:
    """Detects completed push-up reps from a stream of per-frame features."""

    def __init__(
        self,
        fps: float = 30.0,
        signal: str = REP_SIGNAL,
        angle_min: float = REP_ANGLE_MIN,
        angle_max: float = REP_ANGLE_MAX,
        dist_min: float = REP_DIST_MIN,
        dist_max: float = REP_DIST_MAX,
        down_thresh: float = REP_DOWN_THRESH,
        up_thresh: float = REP_UP_THRESH,
        min_slope: float = REP_MIN_SLOPE,
        ema_alpha: float = REP_EMA_ALPHA,
        min_rep_gap_sec: float = REP_MIN_GAP_SEC,
        pre_roll_sec: float = REP_PRE_ROLL_SEC,
    ):
        self.fps = fps
        self._dt = 1.0 / max(fps, 1e-6)

        self.SIGNAL = signal
        self.ANGLE_MIN = angle_min
        self.ANGLE_MAX = angle_max
        self.DIST_MIN = dist_min
        self.DIST_MAX = dist_max
        self.HEIGHT_MIN = REP_HEIGHT_MIN
        self.HEIGHT_MAX = REP_HEIGHT_MAX
        self.DOWN_THRESH = down_thresh
        self.UP_THRESH = up_thresh
        self.MIN_SLOPE = min_slope
        self.EMA_ALPHA = ema_alpha
        self.MIN_REP_GAP_SEC = min_rep_gap_sec
        self.PRE_ROLL_SEC = pre_roll_sec
        self.PRE_ROLL_FRAMES = max(1, int(pre_roll_sec * fps))

        self.REQUIRE_POSE = REP_REQUIRE_POSE
        self.MIN_HIP_ANGLE = REP_MIN_HIP_ANGLE
        self.MAX_BODY_STRAIGHTNESS = REP_MAX_BODY_STRAIGHTNESS

        self.USE_ADAPTIVE = REP_USE_ADAPTIVE
        self.EXTREMA_HISTORY = max(2, int(REP_EXTREMA_HISTORY))
        self.MIDLINE_BAND = REP_MIDLINE_BAND
        self.MIN_RANGE = REP_MIN_RANGE
        self.WARMUP_SAMPLES = max(6, int(REP_WARMUP_SAMPLES))
        self.PLATEAU_FRAMES = max(2, int(REP_PLATEAU_FRAMES))
        self.PLATEAU_SLOPE = REP_PLATEAU_SLOPE

        # For compatibility with previous debug output
        self.MIN_REP_GAP = max(1, int(min_rep_gap_sec * fps))

        self.frame_idx: int = 0
        self.last_rep_time: float = -1e9
        self.recovery_seen: bool = True
        self.last_window: list = []

        self._state: str = "up"
        self._ema: float | None = None
        self._prev_ema: float | None = None
        self._pre_buffer: list = []
        self._rep_buffer: list = []

        self.last_norm: float = float("nan")
        self.last_ema: float = float("nan")
        self.last_slope: float = float("nan")
        self.last_state: str = self._state

        self._prev_slope: float | None = None
        self._peaks: list = []
        self._troughs: list = []
        self.last_up_line: float = float("nan")
        self.last_down_line: float = float("nan")
        self._recent_ema: list = []
        self._plateau_count: int = 0
        self._plateau_mode: bool = False
        self._plateau_value: float = float("nan")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _normalize_angle(self, angle: float) -> float:
        denom = max(self.ANGLE_MAX - self.ANGLE_MIN, 1e-6)
        norm = (angle - self.ANGLE_MIN) / denom
        return float(np.clip(norm, 0.0, 1.0))

    def _normalize_dist(self, dist: float) -> float:
        denom = max(self.DIST_MAX - self.DIST_MIN, 1e-6)
        norm = (dist - self.DIST_MIN) / denom
        return float(np.clip(norm, 0.0, 1.0))

    def _normalize_height(self, height: float) -> float:
        denom = max(self.HEIGHT_MAX - self.HEIGHT_MIN, 1e-6)
        norm = (height - self.HEIGHT_MIN) / denom
        return float(np.clip(norm, 0.0, 1.0))

    def _signal_from_feat(self, feat: dict) -> float:
        if self.SIGNAL == "shoulder_height":
            sh = feat.get("shoulder_height_norm", np.nan)
            if np.isnan(sh):
                return np.nan
            return self._normalize_height(sh)

        if self.SIGNAL == "hip_height":
            hi = feat.get("hip_height_norm", np.nan)
            if np.isnan(hi):
                return np.nan
            return self._normalize_height(hi)

        if self.SIGNAL == "shoulder_hip_dist":
            sh = feat.get("shoulder_height_norm", np.nan)
            hi = feat.get("hip_height_norm", np.nan)
            if np.isnan(sh) or np.isnan(hi):
                return np.nan
            return self._normalize_dist(abs(sh - hi))

        if self.SIGNAL == "nose_hip_dist":
            nose = feat.get("nose_height_norm", np.nan)
            hi   = feat.get("hip_height_norm", np.nan)
            if np.isnan(nose) or np.isnan(hi):
                return np.nan
            return self._normalize_dist(abs(nose - hi))

        angle = feat.get("mean_elbow_angle", np.nan)
        if np.isnan(angle):
            return np.nan
        return self._normalize_angle(angle)

    def _pose_ok(self, feat: dict) -> bool:
        if not self.REQUIRE_POSE:
            return True

        straightness = feat.get("body_straightness", np.nan)
        l_hip = feat.get("left_hip_angle", np.nan)
        r_hip = feat.get("right_hip_angle", np.nan)
        hip_angles = [a for a in (l_hip, r_hip) if not np.isnan(a)]

        has_straightness = not np.isnan(straightness)
        has_hip = len(hip_angles) > 0

        if not has_straightness and not has_hip:
            return False

        if has_straightness and straightness > self.MAX_BODY_STRAIGHTNESS:
            return False

        if has_hip and max(hip_angles) < self.MIN_HIP_ANGLE:
            return False

        return True

    # ── public API ────────────────────────────────────────────────────────────

    def push(self, feat: dict) -> bool:
        """Add one frame of features. Returns True when a rep is detected."""
        self.frame_idx += 1

        if not self._pose_ok(feat):
            self._state = "up"
            self._ema = None
            self._prev_ema = None
            self._pre_buffer = []
            self._rep_buffer = []
            self.recovery_seen = True
            self.last_state = self._state
            return False

        norm = self._signal_from_feat(feat)
        if np.isnan(norm):
            return False

        self.last_norm = float(norm)

        self._pre_buffer.append(feat)
        if len(self._pre_buffer) > self.PRE_ROLL_FRAMES:
            self._pre_buffer.pop(0)

        if self._ema is None:
            self._ema = norm
            self._prev_ema = norm
            return False

        self._prev_ema = self._ema
        self._ema = (self.EMA_ALPHA * norm) + (1.0 - self.EMA_ALPHA) * self._ema
        slope = (self._ema - self._prev_ema) / self._dt

        self.last_ema = float(self._ema)
        self.last_slope = float(slope)

        self._recent_ema.append(self.last_ema)
        if len(self._recent_ema) > self.WARMUP_SAMPLES:
            self._recent_ema.pop(0)

        if abs(slope) <= self.PLATEAU_SLOPE:
            self._plateau_count += 1
        else:
            self._plateau_count = 0

        if self._plateau_count >= self.PLATEAU_FRAMES:
            self._plateau_mode = True
            self._plateau_value = float(self._ema)
            if self._state == "down":
                self._troughs.append(float(self._ema))
                if len(self._troughs) > self.EXTREMA_HISTORY:
                    self._troughs.pop(0)
            else:
                self._peaks.append(float(self._ema))
                if len(self._peaks) > self.EXTREMA_HISTORY:
                    self._peaks.pop(0)

        if self._prev_slope is not None:
            if self._prev_slope > 0 and slope <= 0 and abs(self._prev_slope) >= self.MIN_SLOPE:
                self._peaks.append(float(self._prev_ema))
                if len(self._peaks) > self.EXTREMA_HISTORY:
                    self._peaks.pop(0)
            if self._prev_slope < 0 and slope >= 0 and abs(self._prev_slope) >= self.MIN_SLOPE:
                self._troughs.append(float(self._prev_ema))
                if len(self._troughs) > self.EXTREMA_HISTORY:
                    self._troughs.pop(0)
        self._prev_slope = float(slope)

        t = self.frame_idx * self._dt
        fired = False

        use_adaptive = self.USE_ADAPTIVE and self._peaks and self._troughs
        if use_adaptive and not self._plateau_mode:
            peak_avg = float(np.mean(self._peaks))
            trough_avg = float(np.mean(self._troughs))
            rng = peak_avg - trough_avg
            if rng >= self.MIN_RANGE:
                mid = (peak_avg + trough_avg) / 2.0
                band = max(0.02, self.MIDLINE_BAND * rng)
                up_line = min(1.0, mid + band)
                down_line = max(0.0, mid - band)
            else:
                use_adaptive = False

        if self.USE_ADAPTIVE and self._plateau_mode:
            mid = self._plateau_value
            band = max(0.02, self.MIDLINE_BAND * self.MIN_RANGE)
            up_line = min(1.0, mid + band)
            down_line = max(0.0, mid - band)
            use_adaptive = True

        if not use_adaptive and self.USE_ADAPTIVE and len(self._recent_ema) >= self.WARMUP_SAMPLES:
            r_min = float(np.min(self._recent_ema))
            r_max = float(np.max(self._recent_ema))
            rng = r_max - r_min
            if rng >= self.MIN_RANGE:
                mid = (r_max + r_min) / 2.0
                band = max(0.02, self.MIDLINE_BAND * rng)
                up_line = min(1.0, mid + band)
                down_line = max(0.0, mid - band)
                use_adaptive = True

        if not use_adaptive:
            up_line = self.UP_THRESH
            down_line = self.DOWN_THRESH

        self.last_up_line = float(up_line)
        self.last_down_line = float(down_line)

        if self._state == "up":
            if self._ema <= down_line and slope <= -self.MIN_SLOPE:
                self._state = "down"
                self.recovery_seen = False
                self._rep_buffer = list(self._pre_buffer)
        else:
            self._rep_buffer.append(feat)
            if (
                self._ema >= up_line
                and slope >= self.MIN_SLOPE
                and (t - self.last_rep_time) >= self.MIN_REP_GAP_SEC
            ):
                fired = True
                self.last_rep_time = t
                self.last_window = list(self._rep_buffer) if self._rep_buffer else list(self._pre_buffer)
                self._rep_buffer = []
                self._state = "up"
                self.recovery_seen = True
                self._plateau_mode = False

            self.last_state = self._state

        if self._state == "up" and self._ema >= self.UP_THRESH:
            self.recovery_seen = True
            if self._plateau_mode and slope > self.MIN_SLOPE:
                self._plateau_mode = False

        return fired

    def get_buffer(self) -> list:
        """Return a copy of the current rep buffer (for debugging)."""
        return list(self._rep_buffer)
