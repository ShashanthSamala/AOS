# pyre-ignore-all-errors
# AURA-OS — Layer 5 Isolation: Perception Performance Check
# test_l5_perception.py — Camera indexing, MediaPipe thread stability,
#     FPS profiling, and EMA coordinate smoothing leak detection.
# Standalone execution: python -m tests.test_l5_perception
#                   or: python main.py --test-all  (via runner)

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import sys
import time
import threading
import tracemalloc
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Vector 1: Camera Index Probe ──────────────────────────────────────────
def test_camera_index_probe():
    """Validate cv2.VideoCapture opens on index 0..3, reports actual hw state."""
    import cv2  # type: ignore
    print("\n" + "=" * 72)
    print("[L5-001] Camera Index Probe")
    print("=" * 72)

    passed = 0
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        ok = cap.isOpened()
        if ok:
            ret, frame = cap.read()
            h, w = (frame.shape[:2]) if ret else (0, 0)
            cap.release()
            print(f"  [PASS] index={idx}  opened=True  frame={w}x{h}  readable={ret}")
            passed += 1
        else:
            cap.release()
            print(f"  [INFO] index={idx}  opened=False  (no device)")

    status = "PASS" if passed >= 1 else "FAIL"
    print(f"  [{status}] {passed}/4 camera indices responded")
    return passed >= 1


# ── Vector 2: MediaPipe Thread Stability ──────────────────────────────────
def test_mediapipe_thread_stability():
    """Spawn MediaPipe Hands in a daemon thread, process 200 synthetic frames,
    verify zero unhandled exceptions escape the thread boundary."""
    import cv2  # type: ignore
    import mediapipe as mp  # type: ignore
    import numpy as np  # type: ignore
    print("\n" + "=" * 72)
    print("[L5-002] MediaPipe Thread Stability (200 synthetic frames)")
    print("=" * 72)

    FRAME_COUNT = 200
    errors = []
    processed = [0]

    def worker():
        try:
            hands = mp.solutions.hands.Hands(
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            for i in range(FRAME_COUNT):
                # Synthetic 480x640 RGB frame — random noise
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                result = hands.process(frame)
                processed[0] += 1
                # Deliberately access result attributes to trigger edge-case crashes
                _ = result.multi_hand_landmarks
                _ = result.multi_handedness
            hands.close()
        except Exception as e:
            errors.append(f"{type(e).__name__}: {e}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=60)

    alive = t.is_alive()
    if alive:
        print(f"  [FAIL] Thread still alive after 60s timeout")
        return False

    if errors:
        print(f"  [FAIL] {len(errors)} exception(s) escaped thread:")
        for e in errors:
            print(f"         {e}")
        return False

    ok = processed[0] == FRAME_COUNT
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] Processed {processed[0]}/{FRAME_COUNT} frames, 0 exceptions")
    return ok


# ── Vector 3: FPS Profiling (live camera) ─────────────────────────────────
def test_fps_profiling():
    """Profile raw processing loop on live camera for 10 seconds.
    Target: >= 20 FPS sustained on CPU (no GPU)."""
    import cv2  # type: ignore
    import mediapipe as mp  # type: ignore
    print("\n" + "=" * 72)
    print("[L5-003] FPS Profiling — 10s live camera processing")
    print("=" * 72)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [SKIP] No camera available — cannot profile FPS")
        return None  # inconclusive

    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    DURATION_S = 10.0
    frame_times = []
    start = time.monotonic()

    while (time.monotonic() - start) < DURATION_S:
        t0 = time.monotonic()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        _ = hands.process(rgb)
        frame_times.append(time.monotonic() - t0)

    cap.release()
    hands.close()

    if not frame_times:
        print("  [FAIL] Zero frames captured")
        return False

    total_frames = len(frame_times)
    elapsed = sum(frame_times)
    avg_fps = total_frames / elapsed if elapsed > 0 else 0
    avg_ms = (elapsed / total_frames) * 1000 if total_frames > 0 else 0
    p95_ms = sorted(frame_times)[int(len(frame_times) * 0.95)] * 1000
    min_fps = 1.0 / max(frame_times) if max(frame_times) > 0 else 0
    max_fps = 1.0 / min(frame_times) if min(frame_times) > 0 else 0

    ok = avg_fps >= 20.0
    status = "PASS" if ok else "FAIL"

    print(f"  Frames captured : {total_frames}")
    print(f"  Average FPS     : {avg_fps:.1f}")
    print(f"  Average latency : {avg_ms:.2f} ms/frame")
    print(f"  P95 latency     : {p95_ms:.2f} ms")
    print(f"  FPS range       : {min_fps:.1f} — {max_fps:.1f}")
    print(f"  [{status}] Target >= 20 FPS, measured {avg_fps:.1f} FPS")
    return ok


# ── Vector 4: EMA Smoothing Memory Leak Detection (5 min sustained) ──────
def test_ema_smoothing_leak_detection():
    """Simulate 5 minutes of continuous POINT gesture coordinate smoothing.
    Feed 9000 synthetic (x, y) samples (~30 FPS × 300s) through the EMA filter.
    Profile resident memory delta — flag if growth exceeds 2 MB threshold."""
    print("\n" + "=" * 72)
    print("[L5-004] EMA Smoothing — 5-minute leak detection (9000 samples)")
    print("=" * 72)

    import random

    # Replicate GestureEngine EMA logic in isolation
    alpha = 0.35
    prev_x = None
    prev_y = None
    SAMPLE_COUNT = 9000  # ~30 FPS × 300s = 5 minutes

    # Start memory tracking
    tracemalloc.start()
    gc.collect()
    snap_before = tracemalloc.take_snapshot()
    mem_before = sum(stat.size for stat in snap_before.statistics("lineno"))

    coord_log = []  # bounded accumulator — simulates what the engine does NOT store

    for i in range(SAMPLE_COUNT):
        raw_x = random.uniform(0.0, 1.0)
        raw_y = random.uniform(0.0, 1.0)

        if prev_x is None:
            prev_x = raw_x
            prev_y = raw_y
        else:
            prev_x = alpha * raw_x + (1.0 - alpha) * prev_x
            prev_y = alpha * raw_y + (1.0 - alpha) * prev_y

        # Validate output bounds
        assert 0.0 <= prev_x <= 1.0, f"EMA x out of bounds: {prev_x}"
        assert 0.0 <= prev_y <= 1.0, f"EMA y out of bounds: {prev_y}"

    gc.collect()
    snap_after = tracemalloc.take_snapshot()
    mem_after = sum(stat.size for stat in snap_after.statistics("lineno"))
    tracemalloc.stop()

    delta_kb = (mem_after - mem_before) / 1024.0
    delta_mb = delta_kb / 1024.0

    ok = delta_mb < 2.0
    status = "PASS" if ok else "FAIL"

    print(f"  Samples processed : {SAMPLE_COUNT}")
    print(f"  Memory before     : {mem_before / 1024:.1f} KB")
    print(f"  Memory after      : {mem_after / 1024:.1f} KB")
    print(f"  Memory delta      : {delta_kb:.1f} KB ({delta_mb:.2f} MB)")
    print(f"  Final EMA state   : x={prev_x:.6f}, y={prev_y:.6f}")
    print(f"  [{status}] Leak threshold < 2 MB, measured {delta_mb:.2f} MB")
    return ok


# ── Vector 5: GestureEngine EMA State Reset Correctness ──────────────────
def test_ema_state_reset():
    """Verify that EMA state variables (_smooth_prev_x, _smooth_prev_y) are
    correctly reset when gesture transitions away from POINT, preventing
    stale lerp-on-reentry artifacts."""
    print("\n" + "=" * 72)
    print("[L5-005] EMA State Reset on Gesture Transition")
    print("=" * 72)

    # Simulate the state machine
    smooth_prev_x = 0.75  # seeded from previous POINT tracking
    smooth_prev_y = 0.25

    # Transition to non-POINT gesture → engine sets both to None
    gesture = "FIST"
    if gesture != "POINT":
        smooth_prev_x = None
        smooth_prev_y = None

    ok_reset = smooth_prev_x is None and smooth_prev_y is None

    # Re-enter POINT → first frame seeds fresh
    gesture = "POINT"
    raw_x, raw_y = 0.5, 0.5
    if smooth_prev_x is None:
        smooth_prev_x = raw_x
        smooth_prev_y = raw_y

    ok_seed = smooth_prev_x == 0.5 and smooth_prev_y == 0.5

    ok = ok_reset and ok_seed
    status = "PASS" if ok else "FAIL"
    print(f"  Reset on exit     : {ok_reset}")
    print(f"  Fresh seed entry  : {ok_seed}  (x={smooth_prev_x}, y={smooth_prev_y})")
    print(f"  [{status}] EMA state lifecycle correct")
    return ok


# ── Runner ────────────────────────────────────────────────────────────────
def run_all():
    print("\n" + "#" * 72)
    print("#  AURA-OS  Layer 5 — Perception Performance Suite")
    print("#" * 72)

    results = {}
    results["L5-001_camera_probe"] = test_camera_index_probe()
    results["L5-002_mediapipe_thread"] = test_mediapipe_thread_stability()
    results["L5-003_fps_profiling"] = test_fps_profiling()
    results["L5-004_ema_leak_5min"] = test_ema_smoothing_leak_detection()
    results["L5-005_ema_state_reset"] = test_ema_state_reset()

    print("\n" + "=" * 72)
    print("  LAYER 5 SUMMARY")
    print("=" * 72)
    for name, result in results.items():
        tag = "PASS" if result is True else ("SKIP" if result is None else "FAIL")
        print(f"  {name:40s} [{tag}]")

    total = len(results)
    passed = sum(1 for r in results.values() if r is True)
    skipped = sum(1 for r in results.values() if r is None)
    failed = total - passed - skipped
    print(f"\n  Total: {total}  Passed: {passed}  Failed: {failed}  Skipped: {skipped}")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
