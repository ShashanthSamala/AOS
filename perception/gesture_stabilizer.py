# AURA-OS — Perception Layer
# gesture_stabilizer.py — Gesture stability checking and debouncing
# Author: Samala Shashanth | Project: AURA-OS

"""
Prevents gesture flicking and repeated actions by requiring gestures
to be consistent across multiple frames before accepting them.

Instead of accepting a gesture on first detection, wait for it to stabilize
over N consecutive frames.
"""

from collections import deque
from typing import Optional, Dict
import time


class GestureStabilizer:
    """Stabilizes gesture recognition by requiring consistency."""

    def __init__(self, stability_threshold: int = 3, cooldown_ms: int = 800):
        """
        Initialize gesture stabilizer.

        Args:
            stability_threshold: Require N consistent frames before accepting gesture
            cooldown_ms: Minimum milliseconds between accepting same gesture
        """
        self.stability_threshold = stability_threshold  # 3 frames = ~100ms at 30fps
        self.cooldown_ms = cooldown_ms

        # Track consecutive identical gestures
        self._gesture_buffer = deque(maxlen=stability_threshold)
        self._last_accepted_gesture = None
        self._last_accepted_time = 0

    def should_accept(self, gesture: str) -> bool:
        """
        Check if gesture should be accepted (stable + not in cooldown).

        Args:
            gesture: Current detected gesture

        Returns:
            True if gesture is stable and not in cooldown
        """
        now = time.time() * 1000

        # Add to buffer
        self._gesture_buffer.append(gesture)

        # Not enough samples yet
        if len(self._gesture_buffer) < self.stability_threshold:
            return False

        # Check if all samples are identical
        if not all(g == gesture for g in self._gesture_buffer):
            # Inconsistent - keep buffering
            return False

        # Gesture is stable, now check cooldown
        if gesture == self._last_accepted_gesture:
            elapsed = now - self._last_accepted_time
            if elapsed < self.cooldown_ms:
                # Still in cooldown - reject
                return False

        # Gesture is stable and not in cooldown - ACCEPT
        self._last_accepted_gesture = gesture
        self._last_accepted_time = now
        self._gesture_buffer.clear()  # Reset buffer after acceptance
        return True

    def reset(self):
        """Reset stabilizer state."""
        self._gesture_buffer.clear()
        self._last_accepted_gesture = None
        self._last_accepted_time = 0

    def get_buffer_status(self) -> Dict:
        """Get current buffer status for debugging."""
        return {
            "buffered_gestures": list(self._gesture_buffer),
            "buffer_size": len(self._gesture_buffer),
            "threshold": self.stability_threshold,
            "last_accepted": self._last_accepted_gesture,
            "is_stable": len(self._gesture_buffer) == self.stability_threshold and
                        all(g == list(self._gesture_buffer)[0] for g in self._gesture_buffer),
        }
