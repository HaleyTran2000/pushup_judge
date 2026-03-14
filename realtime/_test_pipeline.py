"""
_test_pipeline.py – Quick sanity-check for the full realtime pipeline.
Run: python realtime/_test_pipeline.py  (from project root, with venv active)
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))

import numpy as np

# ── imports ────────────────────────────────────────────────────────────────────
from realtime.config import FEATURE_COLS, T_FIXED, THRESHOLD
from realtime.features import extract_frame_features, add_velocity_inplace, NAN_FRAME
from realtime.rep_detector import RepDetector
from realtime.visualizer import draw_hud, draw_skeleton, draw_bounding_box
from realtime.pipeline import ProcessResult
print("All imports OK\n")

# ── RepDetector settings ───────────────────────────────────────────────────────
rd0 = RepDetector(fps=30.0)
print(
    f"RepDetector thresholds:\n"
    f"  ANGLE_MIN/MAX   = {rd0.ANGLE_MIN}° / {rd0.ANGLE_MAX}°\n"
    f"  DOWN/UP_THRESH  = {rd0.DOWN_THRESH:.2f} / {rd0.UP_THRESH:.2f}\n"
    f"  MIN_SLOPE       = {rd0.MIN_SLOPE:.2f} per sec\n"
    f"  EMA_ALPHA       = {rd0.EMA_ALPHA:.2f}\n"
    f"  MIN_REP_GAP     = {rd0.MIN_REP_GAP} fr ({rd0.MIN_REP_GAP/30:.2f} s)\n"
)

results = {}

# ── Test 1: sitting still at ~160° — should NOT fire ──────────────────────────
print("Test 1: sitting still at 160° for 3 s … ", end="", flush=True)
rd = RepDetector(fps=30.0)
fired = sum(rd.push({"mean_elbow_angle": 160.0 + np.random.randn()*2}) for _ in range(90))
results["T1_still"] = (fired == 0, fired)
print(f"reps={fired}  {'PASS ✓' if fired == 0 else 'FAIL ✗'}  (expected 0)")

# ── Test 2: random jumpy angles 100-170° — no real pattern — should NOT fire ──
print("Test 2: random 100-170° arm noise    … ", end="", flush=True)
rd = RepDetector(fps=30.0)
fired = 0
np.random.seed(42)
for _ in range(120):
    if rd.push({"mean_elbow_angle": float(np.random.uniform(100, 170))}):
        fired += 1
results["T2_noise"] = (fired == 0, fired)
print(f"reps={fired}  {'PASS ✓' if fired == 0 else 'FAIL ✗'}  (expected 0)")

# ── Test 3: partial bend to 120° only — should NOT fire ───────────────────────
print("Test 3: partial bend only → 120°     … ", end="", flush=True)
rd = RepDetector(fps=30.0)
fired = 0
for _ in range(60):
    rd.push({"mean_elbow_angle": 160.0})                       # extended
for a in np.linspace(160, 120, 20):
    rd.push({"mean_elbow_angle": float(a)})                    # partial bend
for a in np.linspace(120, 160, 20):
    if rd.push({"mean_elbow_angle": float(a)}): fired += 1
for _ in range(20):
    if rd.push({"mean_elbow_angle": 160.0}): fired += 1
results["T3_partial"] = (fired == 0, fired)
print(f"reps={fired}  {'PASS ✓' if fired == 0 else 'FAIL ✗'}  (expected 0)")

# ── Test 4: real push-up arc 160→75→160 over ~1.5 s — MUST fire exactly once ─
print("Test 4: real push-up 160°→75°→160°  … ", end="", flush=True)
rd = RepDetector(fps=30.0)
fired = 0
for _ in range(65):                                            # 2 s extended
    rd.push({"mean_elbow_angle": 160.0})
for a in np.linspace(160, 75, 22):                            # ~0.7 s down
    rd.push({"mean_elbow_angle": float(a)})
for a in np.linspace(75, 160, 22):                            # ~0.7 s up
    if rd.push({"mean_elbow_angle": float(a)}): fired += 1
for _ in range(20):
    if rd.push({"mean_elbow_angle": 160.0}): fired += 1
results["T4_real"] = (fired == 1, fired)
print(f"reps={fired}  {'PASS ✓' if fired == 1 else 'FAIL ✗'}  (expected 1)")

# ── Test 5: 3 consecutive push-ups — MUST fire 3 times ────────────────────────
print("Test 5: 3 consecutive push-ups       … ", end="", flush=True)
rd = RepDetector(fps=30.0)
fired = 0
for _ in range(3):
    for _ in range(40):
        rd.push({"mean_elbow_angle": 160.0})
    for a in np.linspace(160, 75, 20):
        rd.push({"mean_elbow_angle": float(a)})
    for a in np.linspace(75, 160, 20):
        if rd.push({"mean_elbow_angle": float(a)}): fired += 1
    for _ in range(10):
        if rd.push({"mean_elbow_angle": 160.0}): fired += 1
results["T5_three"] = (fired == 3, fired)
print(f"reps={fired}  {'PASS ✓' if fired == 3 else 'FAIL ✗'}  (expected 3)")

# ── Test 6: bend to 130° (just above valley gate 115°) — should NOT fire ─────
print("Test 6: bend to 130° (barely bent)   … ", end="", flush=True)
rd = RepDetector(fps=30.0)
fired = 0
for _ in range(60):
    rd.push({"mean_elbow_angle": 160.0})
for a in np.linspace(160, 130, 20):
    rd.push({"mean_elbow_angle": float(a)})
for a in np.linspace(130, 160, 20):
    if rd.push({"mean_elbow_angle": float(a)}): fired += 1
for _ in range(20):
    if rd.push({"mean_elbow_angle": 160.0}): fired += 1
results["T6_shallow"] = (fired == 0, fired)
print(f"reps={fired}  {'PASS ✓' if fired == 0 else 'FAIL ✗'}  (expected 0)")

# ── Test 7: fast push-up (~0.5 s per rep) — MUST fire once ───────────────────
print("Test 7: fast push-up (~0.5 s / rep)  … ", end="", flush=True)
rd = RepDetector(fps=30.0)
fired = 0
for _ in range(40):                                            # 1.3 s extended
    rd.push({"mean_elbow_angle": 160.0})
for a in np.linspace(160, 70, 8):                             # ~0.27 s down
    rd.push({"mean_elbow_angle": float(a)})
for a in np.linspace(70, 160, 7):                             # ~0.23 s up
    if rd.push({"mean_elbow_angle": float(a)}): fired += 1
for _ in range(15):
    if rd.push({"mean_elbow_angle": 160.0}): fired += 1
results["T7_fast"] = (fired == 1, fired)
print(f"reps={fired}  {'PASS ✓' if fired == 1 else 'FAIL ✗'}  (expected 1)")

# ── Test 8: slow push-up (~3.5 s per rep) — MUST fire once ───────────────────
print("Test 8: slow push-up (~3.5 s / rep)  … ", end="", flush=True)
rd = RepDetector(fps=30.0)
fired = 0
for _ in range(40):                                            # 1.3 s extended
    rd.push({"mean_elbow_angle": 160.0})
for a in np.linspace(160, 72, 52):                            # ~1.7 s slow down
    rd.push({"mean_elbow_angle": float(a)})
for a in np.linspace(72, 160, 52):                            # ~1.7 s slow up
    if rd.push({"mean_elbow_angle": float(a)}): fired += 1
for _ in range(20):
    if rd.push({"mean_elbow_angle": 160.0}): fired += 1
results["T8_slow"] = (fired == 1, fired)
print(f"reps={fired}  {'PASS ✓' if fired == 1 else 'FAIL ✗'}  (expected 1)")

# ── Test 9: feature extraction sanity ─────────────────────────────────────────
print("\nTest 9: feature extraction from mock keypoints … ", end="", flush=True)
kps = np.zeros((17, 3), dtype=np.float32)
# Set a few confident keypoints in a plausible push-up position
kps[5]  = [300, 200, 0.9]  # left shoulder
kps[6]  = [340, 200, 0.9]  # right shoulder
kps[7]  = [260, 240, 0.9]  # left elbow
kps[8]  = [380, 240, 0.9]  # right elbow
kps[9]  = [240, 210, 0.9]  # left wrist
kps[10] = [400, 210, 0.9]  # right wrist
kps[11] = [305, 270, 0.9]  # left hip
kps[12] = [335, 270, 0.9]  # right hip
kps[0]  = [320, 170, 0.9]  # nose
feat = extract_frame_features(kps, img_h=480, img_w=640)
missing = [k for k in FEATURE_COLS if k not in feat]
all_present = len(missing) == 0
results["T9_features"] = (all_present, feat)
print(f"{'PASS ✓' if all_present else 'FAIL ✗ missing: ' + str(missing)}")
if all_present:
    for k in ["mean_elbow_angle", "shoulder_height_norm", "hip_height_norm"]:
        print(f"    {k} = {feat[k]:.3f}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
passed = sum(1 for v in results.values() if v[0])
total  = len(results)
print(f"  {passed}/{total} tests passed")
if passed == total:
    print("  All tests PASSED – pipeline ready.")
else:
    print("  SOME TESTS FAILED:")
    for name, (ok, val) in results.items():
        if not ok:
            print(f"    {name}: got {val}")
print("=" * 50)
