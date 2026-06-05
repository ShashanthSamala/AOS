# AURA-OS — Kernel Layer
# context_manager.py — Conversation/interaction history store
# Author: Samala Shashanth | Project: AURA-OS

import time


class ContextManager:
    """Stores recent gesture→action interactions for LLM context."""

    def __init__(self, max_history=10):
        self._history = []
        self.max_history = max_history

    def add(self, gesture, action, reasoning="", result=None):
        """
        Record a gesture→action interaction.

        Args:
            gesture: Detected gesture name
            action: Action that was executed
            reasoning: LLM's reasoning (if used)
            result: Execution result
        """
        entry = {
            "gesture": gesture,
            "action": action,
            "reasoning": reasoning,
            "result": str(result) if result else None,
            "timestamp": time.time(),
        }
        self._history.append(entry)

        # Trim to max size
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

    def get_history(self, n=None):
        """
        Get recent interaction history.

        Args:
            n: Number of entries to return (default: all)

        Returns:
            List of interaction dicts
        """
        if n is None:
            return list(self._history)
        return list(self._history[-n:])

    def get_context_string(self, n=5):
        """
        Format recent history as a string for LLM context.

        Returns:
            Formatted string of recent interactions
        """
        recent = self.get_history(n)
        if not recent:
            return "No previous interactions."

        lines = []
        for entry in recent:
            lines.append(
                f"Gesture: {entry['gesture']} → Action: {entry['action']}"
            )
        return "\n".join(lines)

    def clear(self):
        """Clear all history."""
        self._history.clear()

    @property
    def size(self):
        return len(self._history)

    def __repr__(self):
        return f"ContextManager(entries={self.size}, max={self.max_history})"
