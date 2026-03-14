"""
visualizer.py – OpenCV drawing helpers for the push-up HUD overlay and skeleton.
"""
import numpy as np
import cv2

# COCO 17-keypoint skeleton connections
SKELETON = [
    (5, 6),  (5, 7),  (6, 8),  (7, 9),  (8, 10),   # shoulders, upper arms
    (5, 11), (6, 12), (11, 12),                       # torso
    (11, 13),(12, 14),(13, 15),(14, 16),               # legs
    (0, 5),  (0, 6),                                  # nose → shoulders
]

# Colours (BGR)
_SKEL_LINE  = (0, 255, 255)  # cyan
_SKEL_DOT   = (0, 255, 0)    # green
_BOX_COLOUR = (255, 165, 0)  # orange


def draw_skeleton(frame: np.ndarray, kps: np.ndarray, conf_thresh: float = 0.3) -> None:
    """
    Draw COCO push-up skeleton on *frame* in-place.

    Parameters
    ----------
    frame       : BGR image (H, W, 3)
    kps         : (17, 3) array  [x_px, y_px, conf]  – full-frame coordinates
    conf_thresh : minimum keypoint confidence to draw
    """
    for i1, i2 in SKELETON:
        if kps[i1, 2] > conf_thresh and kps[i2, 2] > conf_thresh:
            cv2.line(
                frame,
                (int(kps[i1, 0]), int(kps[i1, 1])),
                (int(kps[i2, 0]), int(kps[i2, 1])),
                _SKEL_LINE, 2,
            )
    for i in range(17):
        if kps[i, 2] > conf_thresh:
            cv2.circle(frame, (int(kps[i, 0]), int(kps[i, 1])), 4, _SKEL_DOT, -1)


def draw_keypoint_trails(frame: np.ndarray, trails: dict) -> None:
    """Draw short motion trails for keypoints in *trails* (idx -> list of (x,y))."""
    for _, pts in trails.items():
        if len(pts) < 2:
            continue
        for i in range(1, len(pts)):
            p0 = pts[i - 1]
            p1 = pts[i]
            t = i / max(len(pts) - 1, 1)
            col = (int(50 + 205 * t), int(200 - 120 * t), int(255 - 150 * t))
            cv2.line(frame, p0, p1, col, 2)


def draw_bounding_box(frame: np.ndarray, box: list) -> None:
    """Draw the tracked-person bounding box (orange)."""
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), _BOX_COLOUR, 2)


def draw_hud(
    frame: np.ndarray,
    rep_count: int,
    n_correct: int,
    n_wrong: int,
    last_label: str,
    last_conf: float,
    elbow_angle: float,
    rep_ready: bool = True,
    verdict_hold: int = 0,
    verdict_hold_total: int = 1,
    signal_hist: list | None = None,
    signal_down: float | None = None,
    signal_up: float | None = None,
    rep_state: str | None = None,
) -> None:
    """
    Draw a semi-transparent heads-up display on *frame* in-place.

    Verdict display has two modes:
      • Active hold  (verdict_hold > 0): large full-width coloured banner
        that fills the bottom ~20 % of the frame, fading opacity as the
        countdown runs down.  Impossible to miss.
      • Persistent reminder (hold == 0, last_label set): small label in
        the top-left panel so the user remembers the last verdict.

    Parameters
    ----------
    verdict_hold       : frames remaining in the active hold period
    verdict_hold_total : total frames for the hold (used to compute alpha)
    rep_ready          : True when arm is extended enough for the next rep
    """
    H, W = frame.shape[:2]

    # ── determine verdict colour ──────────────────────────────────────────────
    has_verdict = bool(last_label and last_label not in ("", "unknown"))
    if has_verdict:
        is_correct  = last_label == "correct"
        vcolor_bgr  = (50, 205, 50) if is_correct else (50, 50, 220)   # green / red
        vtext       = f"{last_label.upper()}  {last_conf * 100:.0f}%"
    else:
        is_correct  = False
        vcolor_bgr  = (200, 200, 200)
        vtext       = ""

    # ── semi-transparent stats panel (top-left) ───────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (310, 165), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, f"Reps : {rep_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Good : {n_correct}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (100, 255, 100), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Bad  : {n_wrong}",   (10, 86),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (100, 100, 255), 2, cv2.LINE_AA)

    # ── last verdict as small persistent line in the panel ───────────────────
    if has_verdict and verdict_hold == 0:
        small_col = (100, 230, 100) if is_correct else (100, 100, 230)
        cv2.putText(frame, f"Last : {last_label} {last_conf*100:.0f}%", (10, 112),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, small_col, 1, cv2.LINE_AA)

    # ── elbow angle text + bar gauge ──────────────────────────────────────────
    if not np.isnan(elbow_angle):
        cv2.putText(frame, f"Elbow: {elbow_angle:.0f}\u00b0", (10, 134),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
        bar_x, bar_y, bar_w, bar_h = 10, 140, 200, 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                      (80, 80, 80), -1)
        filled = int(np.clip(elbow_angle / 180.0, 0, 1) * bar_w)
        if elbow_angle < 115:
            bar_colour = (60, 60, 220)
        elif elbow_angle > 145:
            bar_colour = (60, 200, 60)
        else:
            bar_colour = (60, 200, 220)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                      bar_colour, -1)
        for thresh_deg, tc in [(110, (0, 0, 255)), (145, (0, 255, 0))]:
            mx = bar_x + int(thresh_deg / 180.0 * bar_w)
            cv2.line(frame, (mx, bar_y - 2), (mx, bar_y + bar_h + 2), tc, 1)

    # ── READY / WAIT indicator ────────────────────────────────────────────────
    ready_text  = "READY" if rep_ready else "EXTEND ARMS"
    ready_color = (60, 220, 60) if rep_ready else (40, 160, 220)
    cv2.putText(frame, ready_text, (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ready_color, 2, cv2.LINE_AA)

    # ── Signal trace (top-right) ─────────────────────────────────────────────
    if signal_hist:
        plot_w, plot_h = 240, 110
        x0, y0 = W - plot_w - 10, 10
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (x0, y0), (x0 + plot_w, y0 + plot_h), (30, 30, 30), -1)
        cv2.addWeighted(overlay3, 0.55, frame, 0.45, 0, frame)

        vals = np.array(signal_hist[-plot_w:], dtype=float)
        vals = np.clip(vals, 0.0, 1.0)
        for i in range(1, len(vals)):
            x1 = x0 + i - 1
            x2 = x0 + i
            y1 = y0 + plot_h - int(vals[i - 1] * (plot_h - 10)) - 5
            y2 = y0 + plot_h - int(vals[i] * (plot_h - 10)) - 5
            cv2.line(frame, (x1, y1), (x2, y2), (80, 220, 220), 1)

        if signal_down is not None and not np.isnan(signal_down):
            yd = y0 + plot_h - int(signal_down * (plot_h - 10)) - 5
            cv2.line(frame, (x0, yd), (x0 + plot_w, yd), (80, 120, 255), 1)
        if signal_up is not None and not np.isnan(signal_up):
            yu = y0 + plot_h - int(signal_up * (plot_h - 10)) - 5
            cv2.line(frame, (x0, yu), (x0 + plot_w, yu), (80, 255, 120), 1)

        if rep_state:
            cv2.putText(frame, f"State: {rep_state}", (x0 + 8, y0 + plot_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Big verdict banner (active hold period) ───────────────────────────────
    if has_verdict and verdict_hold > 0:
        # Alpha fades from 0.75 → 0.25 over the hold period
        alpha = 0.25 + 0.50 * (verdict_hold / max(verdict_hold_total, 1))

        banner_h = int(H * 0.20)          # 20 % of frame height
        banner_y = H - banner_h

        # Draw coloured background strip
        overlay2 = frame.copy()
        r, g, b   = vcolor_bgr[2], vcolor_bgr[1], vcolor_bgr[0]   # BGR→RGB
        panel_col = (int(r * 0.25), int(g * 0.25), int(b * 0.25)) # dark tint
        cv2.rectangle(overlay2, (0, banner_y), (W, H), panel_col, -1)
        cv2.addWeighted(overlay2, alpha, frame, 1 - alpha, 0, frame)

        # Thick coloured top border
        cv2.rectangle(frame, (0, banner_y), (W, banner_y + 4), vcolor_bgr, -1)

        # Verdict text – scale font to fit width
        font       = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.8
        thickness  = 3
        (tw, th), _ = cv2.getTextSize(vtext, font, font_scale, thickness)
        while tw > W - 40 and font_scale > 0.6:
            font_scale -= 0.1
            (tw, th), _ = cv2.getTextSize(vtext, font, font_scale, thickness)

        ty = banner_y + (banner_h + th) // 2
        xv = (W - tw) // 2

        # shadow
        cv2.putText(frame, vtext, (xv + 2, ty + 2), font, font_scale,
                    (0, 0, 0), thickness + 2, cv2.LINE_AA)
        # main text
        cv2.putText(frame, vtext, (xv, ty), font, font_scale,
                    vcolor_bgr, thickness, cv2.LINE_AA)

        # Progress bar showing how long the banner will stay
        pb_h  = 5
        pb_w  = int(W * verdict_hold / max(verdict_hold_total, 1))
        cv2.rectangle(frame, (0, H - pb_h), (pb_w, H), vcolor_bgr, -1)
