# AURA-OS — MCP Tools Layer
# keyboard_tools.py — Keyboard control actions via pyautogui
# Author: Samala Shashanth | Project: AURA-OS

import pyautogui


def press_key(key="enter"):
    """
    Press a single key.

    Args:
        key: Key name (e.g. 'enter', 'space', 'tab', 'escape')
    """
    pyautogui.press(key)
    return f"Pressed key: {key}"


def hotkey(*keys):
    """
    Press a keyboard shortcut.

    Args:
        keys: Keys to press together (e.g. 'ctrl', 'c')
    """
    if not keys:
        keys = ("ctrl", "c")
    pyautogui.hotkey(*keys)
    return f"Hotkey pressed: {'+'.join(keys)}"


def type_text(text=""):
    """
    Type text string.

    Args:
        text: String to type
    """
    if text:
        pyautogui.typewrite(text, interval=0.02)
    return f"Typed: {text}"
