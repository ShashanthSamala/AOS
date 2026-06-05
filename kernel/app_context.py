# AURA-OS — Kernel Layer
# app_context.py — Non-blocking X11 active-window context watcher
# Author: Samala Shashanth | Project: AURA-OS

import subprocess
import time
from typing import Optional, Dict, Tuple

# Window-class substrings → canonical context tokens.
# Checked in order; first match wins.  Keep the table sorted
# by expected frequency to minimize linear-scan cost.
_WINDOW_CLASS_MAP: Dict[str, str] = {
    "firefox":          "BROWSER",
    "chromium":         "BROWSER",
    "google-chrome":    "BROWSER",
    "brave":            "BROWSER",
    "navigator":        "BROWSER",       # Firefox WM_CLASS fallback
    "xfce4-terminal":   "XFCE4-TERMINAL",
    "gnome-terminal":   "TERMINAL",
    "alacritty":        "TERMINAL",
    "kitty":            "TERMINAL",
    "konsole":          "TERMINAL",
    "xterm":            "TERMINAL",
    "code":             "VSCODE",
    "vscodium":         "VSCODE",
    "vlc":              "VLC",
    "mpv":              "MPV",
    "totem":            "VIDEO",
    "celluloid":        "VIDEO",
    "nautilus":         "FILE-MANAGER",
    "thunar":           "FILE-MANAGER",
    "nemo":             "FILE-MANAGER",
    "libreoffice":      "OFFICE",
    "soffice":          "OFFICE",
    "evince":           "PDF-VIEWER",
    "gimp":             "GIMP",
    "inkscape":         "INKSCAPE",
    "slack":            "SLACK",
    "discord":          "DISCORD",
    "telegram":         "TELEGRAM",
    "spotify":          "SPOTIFY",
    "thunderbird":      "EMAIL",
}

# Minimum subprocess throttle window in seconds.
_DEFAULT_THROTTLE_S: float = 0.4


class AppContextManager:
    """Non-blocking X11 active-window context watcher.

    Uses xdotool (preferred) or wmctrl to read the focused window
    title and WM_CLASS.  Subprocess invocations are temporally
    gated by a 400 ms monotonic-clock cache so the main loop is
    never stalled by redundant shell forks.
    """

    __slots__ = (
        "_throttle_s",
        "_cached_token",
        "_cached_raw",
        "_last_poll_mono",
    )

    def __init__(self, throttle_s: float = _DEFAULT_THROTTLE_S):
        self._throttle_s: float = max(throttle_s, 0.0)
        self._cached_token: str = "DESKTOP"
        self._cached_raw: str = ""
        # Seed to -inf so the very first call always polls
        self._last_poll_mono: float = -1e9

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def get_context(self) -> str:
        """Return the current uppercase context token.

        If the monotonic delta since the last subprocess probe is
        under the throttle window, the cached token is returned
        without spawning any child process.
        """
        now = time.monotonic()
        if (now - self._last_poll_mono) < self._throttle_s:
            return self._cached_token

        raw = self._poll_active_window()
        self._last_poll_mono = time.monotonic()

        if raw is None:
            # X server unreachable or no focused window
            self._cached_token = "DESKTOP"
            self._cached_raw = ""
            return self._cached_token

        self._cached_raw = raw
        self._cached_token = self._normalize(raw)
        return self._cached_token

    # Backwards-compatible alias used by kernel_loop._resolve_action
    def get_app_name(self) -> str:
        """Alias returning lowercase form for prompt injection."""
        return self.get_context().lower()

    def get_active_window(self) -> Optional[str]:
        """Return the raw (untokenized) window title string."""
        self.get_context()  # ensure cache is fresh
        return self._cached_raw or None

    def get_context_report(self) -> Dict:
        """Structured report for diagnostics."""
        token = self.get_context()
        return {
            "window_title": self._cached_raw or "unknown",
            "context_token": token,
            "app_name": token.lower(),
            "cached": (time.monotonic() - self._last_poll_mono) < self._throttle_s,
        }

    def clear_cache(self) -> None:
        """Force next call to re-probe the window manager."""
        self._last_poll_mono = -1e9
        self._cached_token = "DESKTOP"
        self._cached_raw = ""

    # ------------------------------------------------------------------
    #  Internals — subprocess probes
    # ------------------------------------------------------------------

    def _poll_active_window(self) -> Optional[str]:
        """Single-shot subprocess probe.  xdotool first, wmctrl fallback."""
        title = self._xdotool_probe()
        if title is not None:
            return title
        return self._wmctrl_probe()

    @staticmethod
    def _xdotool_probe() -> Optional[str]:
        """xdotool getactivewindow getwindowname — single fork, ~2 ms."""
        try:
            proc = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True,
                text=True,
                timeout=0.5,        # hard kill after 500 ms
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return None

    @staticmethod
    def _wmctrl_probe() -> Optional[str]:
        """wmctrl -lp fallback — parses active-window line.

        wmctrl marks the active desktop as the second column; we
        match the first entry whose desktop field is not '-1'
        (sticky windows) by looking at the currently focused
        desktop reported by wmctrl -d.  Simpler heuristic: take
        the last listed window (topmost in stack).
        """
        try:
            proc = subprocess.run(
                ["wmctrl", "-l"],
                capture_output=True,
                text=True,
                timeout=0.5,
            )
            if proc.returncode != 0:
                return None
            # Best-effort: last line is typically the focused window
            lines = [l for l in proc.stdout.splitlines() if l.strip()]
            if lines:
                parts = lines[-1].split(maxsplit=3)
                if len(parts) >= 4:
                    return parts[3]
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return None

    # ------------------------------------------------------------------
    #  String normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(raw_title: str) -> str:
        """Map a raw X11 window title to an uppercase context token.

        Strategy:
          1. Lower-case the title.
          2. Scan _WINDOW_CLASS_MAP for a substring match.
          3. If no match, extract the first dash-delimited or
             space-delimited word and upper-case it.
          4. Strip non-alphanumeric noise.

        Returns an uppercase token such as "BROWSER", "XFCE4-TERMINAL",
        "VLC", or "DESKTOP".
        """
        if not raw_title:
            return "DESKTOP"

        lower = raw_title.lower()

        # Pass 1 — class-map lookup (substring match)
        for needle, token in _WINDOW_CLASS_MAP.items():
            if needle in lower:
                return token

        # Pass 2 — heuristic extraction
        # Titles like "~ — bash — xfce4-terminal" or "file.py - VSCode"
        # Use common separators to grab the trailing app identifier.
        for sep in (" — ", " - ", " – ", " | "):
            if sep in raw_title:
                candidate = raw_title.rsplit(sep, 1)[-1].strip()
                # Re-check the class map against the candidate
                cand_lower = candidate.lower()
                for needle, token in _WINDOW_CLASS_MAP.items():
                    if needle in cand_lower:
                        return token
                # Return cleaned candidate as-is
                clean = "".join(
                    c if (c.isalnum() or c in "-_") else ""
                    for c in candidate
                ).upper()
                if clean:
                    return clean

        # Pass 3 — first word fallback
        first = raw_title.split()[0] if raw_title.split() else "DESKTOP"
        clean = "".join(
            c if (c.isalnum() or c in "-_") else "" for c in first
        ).upper()
        return clean or "DESKTOP"
