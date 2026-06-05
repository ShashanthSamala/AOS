# AURA-OS — MCP Tools Layer
# mouse_tools.py — Mouse control actions via pyautogui
# Author: Samala Shashanth | Project: AURA-OS

import pyautogui

# Safety: prevent pyautogui from throwing errors at screen edge
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1


def click():
    """Perform a left mouse click at current cursor position."""
    pyautogui.click()
    return "Left click performed"


def right_click():
    """Perform a right mouse click at current cursor position."""
    pyautogui.rightClick()
    return "Right click performed"


def double_click():
    """Perform a double left click at current cursor position."""
    pyautogui.doubleClick()
    return "Double click performed"


def scroll_up(amount=5):
    """Scroll up by given amount."""
    pyautogui.scroll(amount)
    return f"Scrolled up by {amount}"


def scroll_down(amount=5):
    """Scroll down by given amount."""
    pyautogui.scroll(-amount)
    return f"Scrolled down by {amount}"


def move_cursor(x=0, y=0, relative=True):
    """
    Move cursor to position.

    Args:
        x: X coordinate (or offset if relative)
        y: Y coordinate (or offset if relative)
        relative: If True, move relative to current position
    """
    if relative:
        pyautogui.moveRel(x, y, duration=0.1)
        return f"Cursor moved by ({x}, {y})"
    else:
        pyautogui.moveTo(x, y, duration=0.1)
        return f"Cursor moved to ({x}, {y})"
