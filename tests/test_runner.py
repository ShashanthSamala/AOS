# pyre-ignore-all-errors
# AURA-OS — Unified Test Runner
# test_runner.py — CLI entry point for all layer isolation tests.
# Usage:
#   python tests/test_runner.py --test-all         Run all layers
#   python tests/test_runner.py --layer 5          Run single layer
#   python tests/test_runner.py --layer 4 3        Run multiple layers
#   python tests/test_runner.py --test-all --skip-camera   Skip camera-dependent tests

import os
import sys
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Suppress noisy imports
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


BANNER = r"""
    ╔═══════════════════════════════════════════════════╗
    ║     AURA-OS  VERIFICATION SUITE  v1.0             ║
    ║     4-Layer Architecture Isolation Matrix          ║
    ║     Target: Xubuntu CPU-only (no GPU)              ║
    ╚═══════════════════════════════════════════════════╝
"""


def run_layer(layer_num):
    """Import and execute a single layer's test suite. Returns (passed: bool, name: str)."""
    if layer_num == 5:
        from tests.test_l5_perception import run_all
        return run_all(), "Layer 5 — Perception Performance"
    elif layer_num == 4:
        from tests.test_l4_foundation import run_all
        return run_all(), "Layer 4 — Foundation Determinism"
    elif layer_num == 3:
        from tests.test_l3_kernel import run_all
        return run_all(), "Layer 3 — Kernel Synchronization"
    elif layer_num == 2:
        from tests.test_l2_mcp import run_all
        return run_all(), "Layer 2 — MCP Security Audit"
    else:
        print(f"[ERROR] Unknown layer: {layer_num}")
        return False, f"Layer {layer_num} — UNKNOWN"


def main():
    parser = argparse.ArgumentParser(
        description="AURA-OS Verification Suite — 4-Layer Architecture Isolation Matrix",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--test-all", action="store_true",
        help="Run all layer isolation tests (L5 → L2)",
    )
    parser.add_argument(
        "--layer", nargs="+", type=int, choices=[2, 3, 4, 5],
        help="Run specific layer(s). Example: --layer 4 3",
    )
    parser.add_argument(
        "--skip-camera", action="store_true",
        help="Skip tests that require a live camera (L5-001, L5-003)",
    )
    args = parser.parse_args()

    if not args.test_all and not args.layer:
        parser.print_help()
        sys.exit(1)

    print(BANNER)

    layers = [5, 4, 3, 2] if args.test_all else sorted(args.layer, reverse=True)

    if args.skip_camera:
        os.environ["AOS_SKIP_CAMERA"] = "1"

    suite_start = time.monotonic()
    suite_results = {}

    for layer_num in layers:
        try:
            ok, name = run_layer(layer_num)
            suite_results[name] = ok
        except Exception as e:
            name = f"Layer {layer_num}"
            print(f"\n[FATAL] {name} crashed: {type(e).__name__}: {e}")
            suite_results[name] = False

    suite_duration = time.monotonic() - suite_start

    # ── Final Summary ─────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  AURA-OS  FULL VERIFICATION REPORT")
    print("═" * 72)
    for name, ok in suite_results.items():
        tag = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {name:45s} [{tag}]")

    total = len(suite_results)
    passed = sum(1 for v in suite_results.values() if v)
    failed = total - passed

    print(f"\n  Layers tested : {total}")
    print(f"  Passed        : {passed}")
    print(f"  Failed        : {failed}")
    print(f"  Duration      : {suite_duration:.1f}s")

    if failed == 0:
        print("\n  ✓  ALL LAYERS PASSED — Architecture verification COMPLETE")
    else:
        print(f"\n  ✗  {failed} LAYER(S) FAILED — Review output above")

    print("═" * 72)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
