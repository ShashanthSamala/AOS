# AURA-OS — Perception Layer
# gesture_map.py — Gesture-to-action mapping (configurable)
# Author: Samala Shashanth | Project: AURA-OS

"""
Gesture-to-action mapping for direct mode (bypasses LLM).
Set a gesture value to None to route it through LLM reasoning instead.

These defaults can be overridden by config/settings.yaml under 'gesture_map'.
"""

# Default gesture → action mapping for the 6 core gestures
# POINT (index finger) → move_cursor (cursor follows hand)
# PINCH (thumb + index touching) → click
# THREE (three fingers) → open_app (customizable, default: code)
# FOUR (four fingers) → open_browser
# FIVE (all five fingers) → minimize_all
# FIST (closed fist) → screenshot
# PEACE (two fingers) → None (special handling for app cycling)
DEFAULT_GESTURE_MAP = {
    "POINT": "move_cursor",      # Point finger for cursor tracking
    "PINCH": "click",            # Thumb + index pinch = click
    "THREE": "open_app",         # Three fingers = open app
    "FOUR": "open_browser",      # Four fingers = browser
    "FIVE": "minimize_all",      # Five fingers = minimize all
    "FIST": "screenshot",        # Fist = screenshot
    "PEACE": None,               # Two fingers = special handling (app cycling)
}


class GestureMap:
    """Manages gesture-to-action mappings with config override support."""

    def __init__(self, config=None):
        # Start with defaults
        self._map = dict(DEFAULT_GESTURE_MAP)

        # Override with config values if provided
        if config and "gesture_map" in config:
            for gesture, action in config["gesture_map"].items():
                self._map[gesture.upper()] = action

    def lookup(self, gesture):
        """
        Look up action for a gesture.

        Args:
            gesture: Gesture name (e.g. 'FIST')

        Returns:
            Action string if direct-mapped, None if should go to LLM
        """
        return self._map.get(gesture.upper())

    def is_direct(self, gesture):
        """Check if gesture has a direct mapping (not None)."""
        return self.lookup(gesture) is not None

    def set_mapping(self, gesture, action):
        """Update a gesture mapping at runtime."""
        self._map[gesture.upper()] = action

    def get_all(self):
        """Return all current mappings."""
        return dict(self._map)

    def __repr__(self):
        lines = ["Gesture Mappings:"]
        for gesture, action in self._map.items():
            mode = "direct" if action else "→ LLM"
            lines.append(f"  {gesture:15s} → {action or mode}")
        return "\n".join(lines)
