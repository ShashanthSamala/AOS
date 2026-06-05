# AURA-OS — MCP Tools Layer
# system_tools.py — System-level actions (terminal, browser, info)
# Author: Samala Shashanth | Project: AURA-OS

import subprocess
import webbrowser
import psutil
import platform
import time
from pathlib import Path


def open_terminal():
    """Open a new terminal window with timeout protection."""
    terminals = [
        "gnome-terminal", "xfce4-terminal", "konsole",
        "mate-terminal", "xterm", "lxterminal",
    ]

    for term in terminals:
        try:
            subprocess.Popen(
                [term],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return f"Opened {term}"
        except FileNotFoundError:
            continue

    return "Error: No supported terminal found"


def open_browser(url="https://www.google.com"):
    """Open URL in default browser."""
    webbrowser.open(url)
    return f"Opened browser: {url}"


def show_desktop():
    """Minimize all windows to show desktop (uses wmctrl or xdotool)."""
    try:
        subprocess.run(["wmctrl", "-k", "on"], check=True, timeout=5)
        return "Desktop shown"
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        try:
            subprocess.run(
                ["xdotool", "key", "super+d"],
                check=True,
                timeout=5
            )
            return "Desktop shown (xdotool)"
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return "Error: wmctrl/xdotool not available"


def get_system_info():
    """Get CPU, RAM, and disk usage (non-blocking)."""
    try:
        # Use interval=0 for non-blocking CPU check (returns last known value)
        cpu = psutil.cpu_percent(interval=0)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "platform": platform.system(),
            "cpu_percent": cpu,
            "ram_total_gb": round(ram.total / (1024**3), 1),
            "ram_used_percent": ram.percent,
            "disk_total_gb": round(disk.total / (1024**3), 1),
            "disk_used_percent": disk.percent,
        }
    except Exception as e:
        return {"error": str(e)}


def get_battery():
    """Get battery status if available."""
    battery = psutil.sensors_battery()
    if battery is None:
        return {"status": "No battery detected (desktop)"}
    return {
        "percent": battery.percent,
        "plugged_in": battery.power_plugged,
        "time_left_min": (
            round(battery.secsleft / 60) if battery.secsleft > 0 else "N/A"
        ),
    }


def volume_up():
    """Increase system volume with timeout."""
    try:
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+5%"],
            check=True,
            timeout=3
        )
        return "Volume increased"
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "Error: pactl not available"


def volume_down():
    """Decrease system volume with timeout."""
    try:
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-5%"],
            check=True,
            timeout=3
        )
        return "Volume decreased"
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "Error: pactl not available"


def play_pause():
    """Toggle media play/pause with timeout."""
    try:
        subprocess.run(
            ["playerctl", "play-pause"],
            check=True,
            timeout=3
        )
        return "Play/pause toggled"
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "Error: playerctl not available"


def lock_screen():
    """Lock the screen with timeout."""
    lockers = [
        ["loginctl", "lock-session"],
        ["xdg-screensaver", "lock"],
        ["gnome-screensaver-command", "--lock"],
    ]
    for cmd in lockers:
        try:
            subprocess.run(cmd, check=True, timeout=5)
            return "Screen locked"
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
    return "Error: No screen locker found"


_ACTIVE_APP_PROC = None

def open_app(app_name):
    """Open an application by name under strict single-app execution policy.

    Args:
        app_name: Name of the application to open (e.g., 'code', 'firefox', 'thunar')
    """
    global _ACTIVE_APP_PROC
    try:
        # Check if the app is already running — focus it instead of re-launching
        try:
            result = subprocess.run(
                ["wmctrl", "-a", app_name],
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            if result.returncode == 0:
                return f"Focused: {app_name}"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Terminate only the single process we previously launched (if any)
        if _ACTIVE_APP_PROC is not None and _ACTIVE_APP_PROC.poll() is None:
            try:
                _ACTIVE_APP_PROC.terminate()
                _ACTIVE_APP_PROC.wait(timeout=2)
            except Exception:
                try:
                    _ACTIVE_APP_PROC.kill()
                except Exception:
                    pass

        _ACTIVE_APP_PROC = subprocess.Popen(
            [app_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return f"Opened: {app_name}"
    except FileNotFoundError:
        return f"Error: Application '{app_name}' not found"
    except Exception as e:
        return f"Error: Failed to open '{app_name}': {e}"


def minimize_all():
    """Minimize all windows to show desktop."""
    try:
        subprocess.run(["wmctrl", "-k", "on"], check=True, timeout=5)
        return "All windows minimized"
    except (FileNotFoundError, subprocess.CalledProcessError):
        try:
            subprocess.run(
                ["xdotool", "key", "super+d"],
                check=True,
                timeout=5
            )
            return "All windows minimized (xdotool)"
        except (FileNotFoundError, subprocess.CalledProcessError):
            return "Error: wmctrl/xdotool not available"


def screenshot():
    """Take a screenshot and save to Pictures folder.

    Uses scrot if available, otherwise falls back to gnome-screenshot or import.
    """
    try:
        timestamp = int(time.time())
        screenshot_dir = Path.home() / "Pictures"
        screenshot_dir.mkdir(exist_ok=True, parents=True)
        screenshot_path = screenshot_dir / f"screenshot_{timestamp}.png"

        # Try scrot first
        try:
            subprocess.run(
                ["scrot", str(screenshot_path)],
                check=True,
                timeout=5
            )
            return f"Screenshot saved: {screenshot_path}"
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Try gnome-screenshot
        try:
            subprocess.run(
                ["gnome-screenshot", "-f", str(screenshot_path)],
                check=True,
                timeout=5
            )
            return f"Screenshot saved: {screenshot_path}"
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        # Try import (ImageMagick)
        try:
            subprocess.run(
                ["import", "-window", "root", str(screenshot_path)],
                check=True,
                timeout=5
            )
            return f"Screenshot saved: {screenshot_path}"
        except (FileNotFoundError, subprocess.CalledProcessError):
            return "Error: No screenshot tool available (install scrot, gnome-screenshot, or imagemagick)"

    except Exception as e:
        return f"Error: Screenshot failed: {e}"


def do_nothing():
    """No action — intentional pass."""
    return "No action taken"
