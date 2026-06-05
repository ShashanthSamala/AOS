# AURA-OS — Foundation Layer
# prompt_template.py — System prompts and prompt builders for gesture interpretation
# Author: Samala Shashanth | Project: AURA-OS

SYSTEM_PROMPT = """You are AURA-OS, an intelligent agentic operating system.
You receive hand gesture inputs detected by a camera and decide what mouse/keyboard/OS action to perform.
ALWAYS reply with ONLY valid JSON in exactly this format, nothing else:
{"action": "<action_name>", "parameters": {}, "reasoning": "<one line explanation>"}

Available actions:
click, right_click, double_click,
scroll_up, scroll_down,
move_cursor,
open_terminal, open_browser, show_desktop,
volume_up, volume_down, play_pause,
lock_screen, do_nothing

Rules:
- Pick the single most appropriate action for the gesture.
- If unsure, use "do_nothing".
- Keep reasoning under 15 words.
- Never output anything except the JSON object."""


def build_prompt(gesture, context=None, extra_info=None):
    """
    Build a prompt for the LLM from a detected gesture.

    Args:
        gesture: Name of detected gesture (e.g. 'FIST', 'OPEN_PALM')
        context: Uppercase context token from AppContextManager (e.g. 'BROWSER', 'XFCE4-TERMINAL')
        extra_info: Optional dict with additional context (e.g. battery, finger count)

    Returns:
        Formatted prompt string
    """
    if context is None:
        context = "DESKTOP"
    prompt = f"Gesture detected: {gesture}\nActive context: {context}"

    if extra_info:
        for key, value in extra_info.items():
            prompt += f"\n{key}: {value}"

    prompt += "\nWhat action should I perform?"
    return prompt


def get_available_actions():
    """Return list of all supported action names."""
    return [
        "click", "right_click", "double_click",
        "scroll_up", "scroll_down",
        "move_cursor",
        "open_terminal", "open_browser", "show_desktop",
        "volume_up", "volume_down", "play_pause",
        "lock_screen", "do_nothing"
    ]
