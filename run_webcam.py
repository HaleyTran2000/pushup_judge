#!/usr/bin/env python3
"""
run_webcam.py – Real-time push-up judge on webcam (or a video file).

Usage
-----
    # Default: uses config.TEST_VIDEO if set, otherwise webcam 0
    python run_webcam.py

    # Specify webcam index
    python run_webcam.py --source 1

    # Run on a pre-recorded video file  (auto-saves annotated output)
    python run_webcam.py --source /path/to/video.mp4

    # Run on video WITHOUT saving output
    python run_webcam.py --source /path/to/video.mp4 --no-save

    # Webcam + force save
    python run_webcam.py --save

    # Disable the live OpenCV window (headless, e.g. for SSH)
    python run_webcam.py --no-display

Keyboard shortcuts (when display window is open)
-------------------------------------------------
    Q / Esc  – quit and print session summary
    R        – reset state (rep counter, locking, etc.)
    S        – toggle saving on/off mid-session
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# ── make sure the project root is on sys.path ─────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from realtime.config   import WINDOW_TITLE, DISPLAY_W, DISPLAY_H, OUTPUT_VIDEO, \
                               TEST_VIDEO, WEBCAM_INDEX
from realtime.pipeline import InferencePipeline


# ── argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    # Default source: TEST_VIDEO from config if set, otherwise WEBCAM_INDEX
    default_source = TEST_VIDEO if TEST_VIDEO else str(WEBCAM_INDEX)

    p = argparse.ArgumentParser(
        description="Real-time push-up judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="TIP: edit realtime/config.py → TEST_VIDEO to set a default video path.",
    )
    p.add_argument(
        "--source", default=default_source,
        help=(
            "Webcam index (int) or path to a video file.  "
            f"Default: config.TEST_VIDEO if set, else webcam {WEBCAM_INDEX}"
        ),
    )
    p.add_argument(
        "--save", action="store_true",
        help="Force save annotated output (automatic for video-file sources).",
    )
    p.add_argument(
        "--no-save", dest="no_save", action="store_true",
        help="Suppress automatic saving even when the source is a video file.",
    )
    p.add_argument(
        "--no-display", dest="no_display", action="store_true",
        help="Do not open an OpenCV window (useful for headless/SSH runs).",
    )
    p.add_argument(
        "--output", default=None,
        help="Override output video path.",
    )
    return p.parse_args()


# ── writer helper ─────────────────────────────────────────────────────────────

def open_writer(path: str, fps: float, W: int, H: int) -> cv2.VideoWriter:
    """Try avc1 (H.264) first; fall back to MJPG AVI."""
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(path, fourcc, fps, (W, H))
    if writer.isOpened():
        print(f"[save] writing to {path}  (H.264 / mp4)")
        return writer
    fb = path.replace(".mp4", ".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(fb, fourcc, fps, (W, H))
    print(f"[save] avc1 unavailable – writing to {fb}  (MJPG / avi)")
    return writer


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Resolve source (int = webcam index, str = video file path)
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Auto-generate output filename from source name
    if args.output:
        out_path = args.output
    elif isinstance(source, int):
        out_path = str(Path(OUTPUT_VIDEO).parent / f"webcam{source}_annotated.mp4")
    else:
        stem     = Path(source).stem
        out_path = str(Path(OUTPUT_VIDEO).parent / f"{stem}_annotated.mp4")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frm  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   # 0 for live webcam
    source_label = f"webcam {source}" if isinstance(source, int) else Path(source).name

    # Auto-save when source is a video file (suppress with --no-save)
    is_video_file = isinstance(source, str)
    saving = args.save or (is_video_file and not args.no_save)

    print(f"\nSource  : {source_label}  ({W}x{H} @ {fps:.1f} fps"
          + (f", {n_frm} frames" if n_frm > 0 else " [live]") + ")")
    if is_video_file:
        print(f"Video   : {Path(source).resolve()}")
    if saving:
        print(f"Output  : {out_path}")
    else:
        print(f"Output  : (not saving – pass --save or use a video-file source)")
    print("Keys    : Q/Esc quit  |  R reset  |  S toggle save\n")

    # Load pipeline
    pipeline = InferencePipeline(fps=fps)

    # VideoWriter
    writer = open_writer(out_path, fps, W, H) if saving else None

    # For video files, track how long each frame takes so we can pace display
    _target_frame_ms = int(1000.0 / fps) if fps > 0 else 33

    # Display window
    show = not args.no_display
    if show:
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        if DISPLAY_W and DISPLAY_H:
            cv2.resizeWindow(WINDOW_TITLE, DISPLAY_W, DISPLAY_H)

    # ── main loop ─────────────────────────────────────────────────────────────
    frame_idx = 0
    t0        = time.time()

    while cap.isOpened():
        t_frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            if isinstance(source, int):
                # Webcam stall – retry once
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break   # video EOF

        frame_idx += 1
        pipeline.process_frame(frame)   # annotates frame in-place

        # Optionally write to file
        if saving and writer is not None:
            writer.write(frame)

        # Show window
        if show:
            display = frame
            if DISPLAY_W and DISPLAY_H and (W != DISPLAY_W or H != DISPLAY_H):
                display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            cv2.imshow(WINDOW_TITLE, display)

            # Pace video display to source FPS; webcam just polls at 1 ms
            if is_video_file:
                proc_ms  = int((time.time() - t_frame_start) * 1000)
                wait_ms  = max(1, _target_frame_ms - proc_ms)
            else:
                wait_ms = 1

            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), ord("Q"), 27):   # Q or Esc → quit
                break
            elif key in (ord("r"), ord("R")):     # R → reset
                pipeline.reset(fps=fps)
                print("[reset]  pipeline state cleared.")
            elif key in (ord("s"), ord("S")):     # S → toggle saving
                if saving:
                    if writer is not None:
                        writer.release()
                        writer = None
                    saving = False
                    print("[save] stopped.")
                else:
                    writer = open_writer(out_path, fps, W, H)
                    saving = True

        # Progress print for video files
        if n_frm > 0 and frame_idx % 150 == 0:
            elapsed  = time.time() - t0
            proc_fps = frame_idx / elapsed
            pct      = frame_idx / n_frm * 100
            print(f"  [{pct:5.1f}%]  frame {frame_idx}/{n_frm}  "
                  f"{proc_fps:.1f} fps  reps={pipeline.rep_count}")

    # ── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s  "
          f"({frame_idx / max(elapsed, 1e-6):.1f} fps)")
    if out_path and (saving or writer is not None):
        print(f"Saved   : {out_path}")

    pipeline.print_summary()


if __name__ == "__main__":
    main()
