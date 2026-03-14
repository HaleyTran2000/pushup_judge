"""
features.py – Per-frame feature extraction helpers.
Identical logic to feature_engineering/1.feature_engineering.ipynb.
"""
import numpy as np
from realtime.config import CONF_THRESH, FEATURE_COLS

# A NaN-filled template dict for one frame
NAN_FRAME: dict = {col: np.nan for col in FEATURE_COLS}


# ── Geometry primitives ───────────────────────────────────────────────────────

def angle_3pts(a, b, c) -> float:
    """Return the angle at vertex *b* formed by points a-b-c (degrees)."""
    ba = np.asarray(a, float) - np.asarray(b, float)
    bc = np.asarray(c, float) - np.asarray(b, float)
    n  = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / n, -1.0, 1.0))))


def perpendicular_dist(p, line_a, line_b) -> float:
    """Perpendicular distance from point *p* to the infinite line through a-b."""
    ab = line_b - line_a
    t  = np.dot(p - line_a, ab) / (np.dot(ab, ab) + 1e-9)
    return float(np.linalg.norm(p - (line_a + t * ab)))


def euclidean(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


# ── Frame-level feature extraction ───────────────────────────────────────────

def extract_frame_features(kps: np.ndarray, img_h: int, img_w: int) -> dict:
    """
    Convert raw YOLO keypoints to a dict of push-up features.

    Parameters
    ----------
    kps   : (17, 3) array  [x_px, y_px, conf]  (COCO keypoint order)
    img_h : frame height in pixels
    img_w : frame width  in pixels

    Returns
    -------
    dict with 15 keys matching FEATURE_COLS; missing/low-conf values are np.nan.
    """
    H, W = float(img_h), float(img_w)

    def pt(i):
        """Return normalised (x, y) for keypoint i if confidence is high enough."""
        return np.array([kps[i, 0] / W, kps[i, 1] / H]) if kps[i, 2] > CONF_THRESH else None

    nose  = pt(0);  l_sh = pt(5);  r_sh = pt(6)
    l_el  = pt(7);  r_el = pt(8)
    l_wr  = pt(9);  r_wr = pt(10)
    l_hi  = pt(11); r_hi = pt(12)
    l_kn  = pt(13); r_kn = pt(14)
    l_an  = pt(15); r_an = pt(16)

    d = dict(NAN_FRAME)

    # Elbow angles
    if l_sh is not None and l_el is not None and l_wr is not None:
        d["left_elbow_angle"]  = angle_3pts(l_sh, l_el, l_wr)
    if r_sh is not None and r_el is not None and r_wr is not None:
        d["right_elbow_angle"] = angle_3pts(r_sh, r_el, r_wr)
    vals = [v for v in [d["left_elbow_angle"], d["right_elbow_angle"]] if not np.isnan(v)]
    d["mean_elbow_angle"] = float(np.mean(vals)) if vals else np.nan
    if not np.isnan(d["left_elbow_angle"]) and not np.isnan(d["right_elbow_angle"]):
        d["elbow_symmetry"] = abs(d["left_elbow_angle"] - d["right_elbow_angle"])

    # Hip angles
    if l_sh is not None and l_hi is not None and l_kn is not None:
        d["left_hip_angle"]  = angle_3pts(l_sh, l_hi, l_kn)
    if r_sh is not None and r_hi is not None and r_kn is not None:
        d["right_hip_angle"] = angle_3pts(r_sh, r_hi, r_kn)

    # Body straightness
    for sh, hi, an in [(l_sh, l_hi, l_an), (r_sh, r_hi, r_an)]:
        if sh is not None and hi is not None and an is not None:
            d["body_straightness"] = perpendicular_dist(hi, sh, an)
            break

    # Height / width ratios
    sh_pts = [v for v in [l_sh, r_sh] if v is not None]
    hi_pts = [v for v in [l_hi, r_hi] if v is not None]
    if sh_pts:
        d["shoulder_height_norm"] = float(np.mean([p[1] for p in sh_pts]))
    if hi_pts:
        d["hip_height_norm"]      = float(np.mean([p[1] for p in hi_pts]))
    if l_wr is not None and r_wr is not None:
        d["wrist_width_norm"]     = euclidean(l_wr, r_wr)
    if l_sh is not None and r_sh is not None:
        d["shoulder_width_norm"]  = euclidean(l_sh, r_sh)
    if nose is not None:
        d["nose_height_norm"]     = float(nose[1])

    return d


def add_velocity_inplace(prev_feat: dict, cur_feat: dict) -> None:
    """
    Compute velocity features by differencing current and previous frames.
    Mutates *cur_feat* in-place.
    """
    def delta(key: str) -> float:
        v_cur  = cur_feat.get(key, np.nan)
        v_prev = prev_feat.get(key, np.nan)
        return float(v_cur - v_prev) if not (np.isnan(v_cur) or np.isnan(v_prev)) else np.nan

    cur_feat["shoulder_velocity"]    = delta("shoulder_height_norm")
    cur_feat["hip_velocity"]         = delta("hip_height_norm")
    cur_feat["elbow_angle_velocity"] = delta("mean_elbow_angle")
