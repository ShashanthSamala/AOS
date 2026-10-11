# AURA-OS — Foundation Layer
# prompt_template.py — System prompts and prompt builders for gesture interpretation
# Author: Samala Shashanth | Project: AURA-OS

SYSTEM_PROMPT = """You are AURA-OS, an intelligent agentic operating system.
You receive human intent signals from AURA-OS perception systems and decide what mouse/keyboard/OS action to perform.
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
- Pick the single most appropriate action for the detected intent.
- If unsure, use "do_nothing".
- Keep reasoning under 15 words.
- Never output anything except the JSON object."""


def build_prompt(
    gesture=None,
    context=None,
    extra_info=None,
    intent=None,
):
    """
    Build a prompt for the LLM from a legacy gesture or semantic intent.

    The legacy ``gesture`` argument remains supported for KernelLoop.
    The active semantic path should provide ``intent`` instead.

    Args:
        gesture: Legacy gesture token, e.g. 'FIST'.
        context: Uppercase context token from AppContextManager.
        extra_info: Optional dict with additional context.
        intent: Semantic HumanIntent value, e.g. 'open_app'.

    Returns:
        Formatted prompt string.
    """
    if context is None:
        context = "DESKTOP"

    if intent is not None:
        prompt = f"Intent detected: {intent}\nActive context: {context}"
    else:
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
