# pyre-ignore-all-errors
# AURA-OS — Perception Layer
# gesture_engine.py — Camera + MediaPipe hand gesture detection engine
# Author: Samala Shashanth | Project: AURA-OS

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["QT_QPA_PLATFORM"] = "xcb"  # suppress Qt threading warnings in OpenCV
import cv2  # type: ignore
import math
import mediapipe as mp  # type: ignore
import time
import threading
import queue
from typing import Any, Dict, Optional, Tuple
import pyautogui  # type: ignore
from perception.gesture_stabilizer import GestureStabilizer  # type: ignore

class GestureEngine:
    """Real-time hand gesture detection using MediaPipe."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg: Dict[str, Any] = config or {}
        perception_cfg: Dict[str, Any] = cfg.get("perception", {})

        self.camera_index = perception_cfg.get("camera_index", 0)
        self.max_hands = perception_cfg.get("max_hands", 1)
        self.detection_confidence = perception_cfg.get("detection_confidence", 0.8)
        self.tracking_confidence = perception_cfg.get("tracking_confidence", 0.5)
        self.cooldown_ms = perception_cfg.get("gesture_cooldown_ms", 500)

        # MediaPipe setup
        self.mp_hands: Any = mp.solutions.hands
        self.mp_draw: Any = mp.solutions.drawing_utils
        self.hands: Any = self.mp_hands.Hands(
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
        )

        # Screen size for coordinate mapping
        self._screen_w, self._screen_h = pyautogui.size()

        # Gesture stabilizer for debouncing
        gesture_stability = perception_cfg.get("gesture_stability_threshold", 3)
        gesture_cooldown = perception_cfg.get("gesture_cooldown_ms", 800)
        self.stabilizer = GestureStabilizer(
            stability_threshold=gesture_stability,
            cooldown_ms=gesture_cooldown
        )

        # State
        self._gesture_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._running: Any = False
        self._last_gesture_time: float = 0
        self._last_gesture: Optional[str] = None
        self._cap: Any = None  # cv2.VideoCapture
        self._thread: Any = None
        self._last_screen_pos: Optional[Tuple[int, int]] = None  # Last known cursor position for stable click
        self._pinch_frame_count: int = 0
        self._holding_gesture: Optional[str] = None
        self._gesture_hold_start: float = 0.0
        self._gesture_executed: bool = False

        # Spatial coordinate smoothing — 1D exponential moving average (EMA)
        # α ∈ (0, 1]: lower = smoother/laggier, higher = more responsive/noisier
        self._smooth_alpha: float = perception_cfg.get("cursor_smoothing_alpha", 0.35)
        self._smooth_prev_x: Optional[float] = None  # EMA state: previous smoothed X (normalised 0..1)
        self._smooth_prev_y: Optional[float] = None  # EMA state: previous smoothed Y (normalised 0..1)

        # Virtual margin mapping — maps central (1-2*margin) of camera to full screen
        self._margin: float = perception_cfg.get("edge_margin", 0.10)

        # False-positive rejection gates
        self._min_visibility: float = perception_cfg.get("min_landmark_visibility", 0.6)
        self._min_palm_span: float = perception_cfg.get("min_palm_span", 0.04)

        # Angle-based classification thresholds (degrees)
        self._finger_angle: float = perception_cfg.get("finger_extend_angle", 160)
        self._thumb_angle: float = perception_cfg.get("thumb_extend_angle", 150)
        self._pinch_ratio: float = perception_cfg.get("pinch_ratio_threshold", 0.30)

    def _distance(self, p1, p2):
        """Euclidean distance between two landmarks."""
        return ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5

    def _remap(self, val: float) -> float:
        """Remap camera coordinate from [margin, 1-margin] → [0.0, 1.0] with clamp.

        This allows the hand to reach screen corners without going to the very
        edge of the camera frame where MediaPipe loses tracking.
        """
        m = self._margin
        return max(0.0, min(1.0, (val - m) / (1.0 - 2.0 * m)))

    def _is_valid_hand(self, landmarks, handedness_score: float = 1.0) -> bool:
        """Reject ghost / phantom hand detections.

        Two checks:
        1. Handedness classification confidence (from MediaPipe's
           ``multi_handedness``) must exceed the configured minimum.
           NOTE: We use this instead of per-landmark ``visibility`` because
           MediaPipe hands do NOT reliably populate the visibility field
           (it is always ~0.0 for hand landmarks).
        2. Palm span (wrist → middle-finger MCP) must exceed the configured
           minimum to filter out tiny noise blobs.
        """
        # Handedness confidence gate (replaces broken visibility check)
        if handedness_score < self._min_visibility:
            return False

        # Minimum palm span gate
        palm_span = self._distance(landmarks[0], landmarks[9])
        if palm_span < self._min_palm_span:
            return False

        return True

    def _joint_angle(self, a, b, c) -> float:
        """Compute the angle (degrees) at point *b* formed by segments a→b and b→c.

        Used for angle-based finger extension detection.  An extended finger
        has an angle close to 180°; a bent finger is typically < 140°.
        """
        ba = (a.x - b.x, a.y - b.y)
        bc = (c.x - b.x, c.y - b.y)
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
        if mag_ba * mag_bc < 1e-8:
            return 0.0
        cos_angle = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
        return math.degrees(math.acos(cos_angle))

    def _count_fingers(self, landmarks, hand_label="Right"):
        """Count number of raised fingers using angle-based joint analysis.

        Instead of simple y-coordinate comparison (which breaks when the hand
        is tilted), we measure the joint angle at each finger's PIP joint.
        An angle > threshold means the finger is extended.

        Thumb uses IP joint (landmarks 2→3→4) with a separate threshold.

        Args:
            landmarks: MediaPipe hand landmark list
            hand_label: "Right" or "Left" — currently unused (angle math is
                        handedness-invariant) but kept for API compatibility.
        """
        count: int = 0

        # Thumb — angle at IP joint (landmarks 2 → 3 → 4)
        thumb_angle = self._joint_angle(landmarks[2], landmarks[3], landmarks[4])
        if thumb_angle > self._thumb_angle:
            count += 1

        # Index, Middle, Ring, Pinky — angle at PIP joint
        # Each finger: (MCP, PIP, TIP) landmark IDs
        finger_joints = [
            (5, 6, 8),    # Index:  MCP=5, PIP=6, TIP=8
            (9, 10, 12),  # Middle: MCP=9, PIP=10, TIP=12
            (13, 14, 16), # Ring:   MCP=13, PIP=14, TIP=16
            (17, 18, 20), # Pinky:  MCP=17, PIP=18, TIP=20
        ]
        for mcp, pip, tip in finger_joints:
            angle = self._joint_angle(landmarks[mcp], landmarks[pip], landmarks[tip])
            if angle > self._finger_angle:
                count += 1

        return count

    def classify(self, landmarks, hand_label="Right"):
        """
        Classify hand gesture from landmarks.
        6 Core Gestures:
        - POINT (1 finger): index finger pointing
        - PINCH (thumb+index close relative to palm span): thumb touching index
        - THREE (3 fingers): three fingers up
        - FOUR (4 fingers): four fingers up
        - FIVE (5 fingers): all five fingers up (open palm)
        - FIST (0 fingers): closed fist

        Plus: PEACE (2 fingers) for app cycling gesture

        Priority: PINCH > POINT > finger count gestures

        Uses scale-invariant pinch detection (normalised by palm span) and
        angle-based finger counting for tilt-robust classification.

        Args:
            landmarks: MediaPipe hand landmark list
            hand_label: "Right" or "Left" for handedness-aware classification
        """
        # Scale-invariant PINCH: normalise thumb-index distance by palm span
        palm_span = self._distance(landmarks[0], landmarks[9])  # wrist → middle MCP
        pinch_dist = self._distance(landmarks[4], landmarks[8])  # thumb tip → index tip
        pinch_ratio = pinch_dist / max(palm_span, 1e-6)
        if pinch_ratio < self._pinch_ratio:
            return "PINCH"

        fingers = self._count_fingers(landmarks, hand_label)

        if fingers == 0:
            return "FIST"
        elif fingers == 1:
            return "POINT"
        elif fingers == 2:
            return "PEACE"      # App cycle gesture
        elif fingers == 3:
            return "THREE"
        elif fingers == 4:
            return "FOUR"
        elif fingers == 5:
            return "FIVE"
        else:
            return f"{fingers}_FINGERS"

    def _detection_loop(self, show_video=True):
        """Main camera loop — runs in a thread with exception protection."""
        try:
            self._cap = cv2.VideoCapture(self.camera_index)

            if not self._cap.isOpened():
                print("[PERCEPTION] ERROR: Camera not found.")
                self._running = False
                return

            print("[PERCEPTION] Gesture engine started. Press Q on video window to stop.")
            print("[PERCEPTION] TIP: Click on another window to control it with gestures.")

            if show_video:
                # Create a small always-on-top window that doesn't steal focus
                cv2.namedWindow("AURA-OS | Gesture Detection", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("AURA-OS | Gesture Detection", 320, 240)
                cv2.setWindowProperty(
                    "AURA-OS | Gesture Detection",
                    cv2.WND_PROP_TOPMOST, 1
                )

            last_action_text = ""

            while self._running:  # type: ignore
                try:
                    ret, frame = self._cap.read()  # type: ignore
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = self.hands.process(rgb)  # type: ignore

                    gesture = None
                    fingertip = None

                    if result.multi_hand_landmarks:  # type: ignore
                        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):  # type: ignore
                            # Extract handedness for correct thumb detection
                            hand_label = "Right"
                            handedness_score = 1.0
                            if result.multi_handedness:  # type: ignore
                                hand_label = result.multi_handedness[idx].classification[0].label  # type: ignore
                                handedness_score = result.multi_handedness[idx].classification[0].score  # type: ignore

                            if show_video:
                                self.mp_draw.draw_landmarks(
                                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                                )

                            # Reject phantom / ghost hand detections
                            if not self._is_valid_hand(hand_landmarks.landmark, handedness_score):
                                continue

                            gesture = self.classify(hand_landmarks.landmark, hand_label)

                            # Track index fingertip (landmark 8) for POINT gesture
                            tip = hand_landmarks.landmark[8]
                            fingertip = {
                                "x": tip.x,  # 0.0 to 1.0 (left to right)
                                "y": tip.y,  # 0.0 to 1.0 (top to bottom)
                            }

                            if show_video:
                                # Show detected gesture
                                cv2.putText(
                                    frame, gesture, (10, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2,
                                )
                                # Draw fingertip circle
                                h, w, _ = frame.shape
                                cx, cy = int(tip.x * w), int(tip.y * h)
                                cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

                    # Show last action on screen
                    if show_video and last_action_text:
                        cv2.putText(
                            frame, last_action_text, (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1,
                        )

                    REQUIRE_RELEASE_GESTURES = {"PEACE", "FIVE", "THREE", "FIST", "FOUR", "FOUR_FINGERS"}
                    if not gesture or gesture not in REQUIRE_RELEASE_GESTURES:
                        self._holding_gesture = None
                        self._gesture_executed = False

                    if gesture != "PINCH":
                        self._pinch_frame_count = 0

                    # Reset EMA state when hand leaves POINT — prevents stale lerp on re-entry
                    if gesture != "POINT":
                        self._smooth_prev_x = None
                        self._smooth_prev_y = None

                    # POINT gesture: move cursor continuously (no cooldown)
                    if gesture == "POINT" and fingertip:
                        # Virtual margin remap — hand at 10% camera → screen 0,
                        #                        hand at 90% camera → screen max
                        raw_x: float = self._remap(fingertip["x"])
                        raw_y: float = self._remap(fingertip["y"])

                        # EMA smoothing: y_t = α·x_t + (1 − α)·y_{t−1}
                        if self._smooth_prev_x is None:
                            # First frame — seed filter state, zero transient delay
                            self._smooth_prev_x = raw_x
                            self._smooth_prev_y = raw_y
                        else:
                            a = self._smooth_alpha
                            self._smooth_prev_x = a * raw_x + (1.0 - a) * self._smooth_prev_x
                            self._smooth_prev_y = a * raw_y + (1.0 - a) * self._smooth_prev_y  # type: ignore[operator]

                        # Map smoothed normalised coords → screen pixel space with boundary clamp
                        screen_x = max(0, min(int(self._smooth_prev_x * self._screen_w), self._screen_w - 1))
                        screen_y = max(0, min(int(self._smooth_prev_y * self._screen_h), self._screen_h - 1))  # type: ignore[arg-type]
                        pyautogui.moveTo(screen_x, screen_y, _pause=False)
                        self._last_screen_pos = (screen_x, screen_y)
                        last_action_text = f">> POINT ({screen_x}, {screen_y})"
                        # Still emit event but less frequently for logging
                        now = time.time() * 1000
                        if now - self._last_gesture_time > self.cooldown_ms:
                            self._gesture_queue.put({
                                "gesture": gesture,
                                "timestamp": time.time(),
                                "fingertip": fingertip,
                                "screen_pos": (screen_x, screen_y),
                            })
                            self._last_gesture = gesture
                            self._last_gesture_time = now

                    # PINCH gesture: click at last known cursor position
                    elif gesture == "PINCH":
                        self._pinch_frame_count += 1
                        if self._pinch_frame_count >= 2:
                            now = time.time() * 1000
                            if (self._last_gesture != "PINCH" or
                                    now - self._last_gesture_time > self.cooldown_ms):
                                # Click at last known position (don't move cursor)
                                if self._last_screen_pos:
                                    pyautogui.click(self._last_screen_pos[0], self._last_screen_pos[1], _pause=False)  # type: ignore
                                else:
                                    pyautogui.click(_pause=False)  # type: ignore
                                self._gesture_queue.put({
                                    "gesture": "PINCH",
                                    "timestamp": time.time(),
                                    "screen_pos": self._last_screen_pos,
                                })
                                self._last_gesture = "PINCH"
                                self._last_gesture_time = now
                                last_action_text = f">> CLICK at {self._last_screen_pos}"

                    elif gesture:
                        if gesture in REQUIRE_RELEASE_GESTURES:
                            if gesture != self._holding_gesture:
                                self._holding_gesture = gesture
                                self._gesture_hold_start = time.time()
                                self._gesture_executed = False
                                last_action_text = f">> HOLDING {gesture}..."
                            else:
                                elapsed = time.time() - self._gesture_hold_start
                                if elapsed >= 3.0:
                                    if not self._gesture_executed:
                                        self._gesture_queue.put({
                                            "gesture": gesture,
                                            "timestamp": time.time(),
                                            "fingertip": fingertip,
                                        })
                                        last_action_text = f">> {gesture} (EXECUTED)"
                                        self._last_gesture = gesture
                                        self._last_gesture_time = time.time() * 1000
                                        self._gesture_executed = True
                                    else:
                                        last_action_text = f">> {gesture} (EXECUTED)"
                                else:
                                    last_action_text = f">> HOLDING {gesture}... ({3.0 - elapsed:.1f}s)"
                        else:
                            # Other gestures: use gesture stabilizer for debouncing
                            if self.stabilizer.should_accept(gesture):
                                self._gesture_queue.put({
                                    "gesture": gesture,
                                    "timestamp": time.time(),
                                    "fingertip": fingertip,
                                })
                                last_action_text = f">> {gesture} (STABLE)"
                                self._last_gesture = gesture
                                self._last_gesture_time = time.time() * 1000

                    if show_video:
                        cv2.imshow("AURA-OS | Gesture Detection", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            self._running = False
                            break

                except Exception as e:
                    print(f"[PERCEPTION] Frame processing error: {e}")
                    # Skip bad frame and continue
                    continue

        except Exception as e:
            print(f"[PERCEPTION] Detection loop error: {e}")
        finally:
            # GUARANTEED cleanup - always runs even if error occurs
            try:
                if self._cap:
                    self._cap.release()
            except Exception as e:
                print(f"[PERCEPTION] Error releasing camera: {e}")

            if show_video:
                try:
                    cv2.destroyAllWindows()
                except Exception as e:
                    print(f"[PERCEPTION] Error destroying windows: {e}")

            print("[PERCEPTION] Gesture engine stopped.")

    def start(self, show_video=True, threaded=True):
        """
        Start the gesture detection engine.

        Args:
            show_video: Whether to show the camera feed window
            threaded: Run in a background thread (non-blocking)
        """
        self._running = True
        if threaded:
            self._thread = threading.Thread(
                target=self._detection_loop,
                args=(show_video,),
                daemon=True,
            )
            self._thread.start()
        else:
            self._detection_loop(show_video)

    def stop(self):
        """Stop the gesture detection engine."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def get_gesture(self, timeout=None):
        """
        Get the next detected gesture (blocking).

        Args:
            timeout: Max seconds to wait (None = wait forever)

        Returns:
            dict with {gesture, timestamp} or None on timeout
        """
        try:
            return self._gesture_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self):
        return self._running
