# AURA-OS — Foundation Layer
# action_validator.py — LLM response validation and schema checking
# Author: Samala Shashanth | Project: AURA-OS

"""
Validates LLM responses before execution to prevent crashes and invalid actions.

Safety-first approach: Only execute actions that have been explicitly validated.
"""

from typing import Dict, Any, Optional


# Valid actions that can be executed
# These must match the tools registered in mcp_server
VALID_ACTIONS = {
    # Mouse tools
    "click",
    "right_click",
    "double_click",
    "scroll_up",
    "scroll_down",
    "move_cursor",
    # Keyboard tools
    "press_key",
    "hotkey",
    "type_text",
    # System tools
    "open_terminal",
    "open_browser",
    "open_app",
    "show_desktop",
    "minimize_all",
    "screenshot",
    "get_system_info",
    "get_battery",
    "volume_up",
    "volume_down",
    "play_pause",
    "lock_screen",
    "do_nothing",
}

# Parameter types expected for each action
# None = no parameters expected
# Dict[param_name] = {type: <type>, required: bool, description: str}
ACTION_SCHEMAS = {
    # Actions with no parameters
    "click": None,
    "right_click": None,
    "double_click": None,
    "scroll_up": None,
    "scroll_down": None,
    "open_terminal": None,
    "show_desktop": None,
    "minimize_all": None,
    "screenshot": None,
    "get_system_info": None,
    "get_battery": None,
    "volume_up": None,
    "volume_down": None,
    "play_pause": None,
    "lock_screen": None,
    "do_nothing": None,
    # Actions with optional parameters
    "move_cursor": {
        "x": {"type": int, "required": False, "description": "X coordinate"},
        "y": {"type": int, "required": False, "description": "Y coordinate"},
    },
    "open_browser": {
        "url": {"type": str, "required": False, "description": "URL to open"},
    },
    "open_app": {
        "app_name": {"type": str, "required": True, "description": "Application name to open"},
    },
    "press_key": {
        "key": {"type": str, "required": True, "description": "Key name to press"},
    },
    "hotkey": {
        "keys": {"type": list, "required": True, "description": "List of keys for hotkey"},
    },
    "type_text": {
        "text": {"type": str, "required": True, "description": "Text to type"},
    },
}


class ValidationError(Exception):
    """Raised when action validation fails."""
    pass


class ActionValidator:
    """Validates LLM responses and action parameters."""

    @staticmethod
    def validate_action(response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an LLM response to ensure it can be safely executed.

        Args:
            response: Dict with expected keys 'action' and optional 'parameters'

        Returns:
            Validated response dict (may be modified to add safe defaults)

        Raises:
            ValidationError: If response is invalid
        """
        if not isinstance(response, dict):
            raise ValidationError(f"Response must be dict, got {type(response)}")

        action = response.get("action")
        if not action:
            raise ValidationError("Response missing 'action' field")

        if not isinstance(action, str):
            raise ValidationError(f"Action must be string, got {type(action)}")

        # Check action is valid
        if action not in VALID_ACTIONS:
            raise ValidationError(
                f"Unknown action '{action}'. "
                f"Valid actions: {sorted(VALID_ACTIONS)}"
            )

        # Validate parameters
        parameters = response.get("parameters", {})
        if parameters is None:
            parameters = {}

        if not isinstance(parameters, dict):
            raise ValidationError(
                f"Parameters must be dict, got {type(parameters)}"
            )

        ActionValidator._validate_parameters(action, parameters)

        # Return validated response with cleaned parameters
        return {
            "action": action,
            "parameters": parameters,
            "reasoning": response.get("reasoning", ""),
            "latency_ms": response.get("latency_ms", 0),
        }

    @staticmethod
    def _validate_parameters(action: str, parameters: Dict[str, Any]) -> None:
        """
        Validate parameters for a specific action.

        Args:
            action: Action name
            parameters: Parameter dict

        Raises:
            ValidationError: If parameters are invalid for this action
        """
        schema = ACTION_SCHEMAS.get(action)

        # No schema = no parameters expected
        if schema is None:
            if parameters:
                # Warn but don't fail - just ignore extra parameters
                pass
            return

        # Validate each parameter in schema
        for param_name, param_spec in schema.items():
            if param_spec["required"] and param_name not in parameters:
                raise ValidationError(
                    f"Action '{action}' requires parameter '{param_name}'"
                )

            if param_name in parameters:
                param_value = parameters[param_name]
                expected_type = param_spec["type"]

                # Type checking
                if expected_type == list:
                    if not isinstance(param_value, list):
                        raise ValidationError(
                            f"Parameter '{param_name}' must be list, "
                            f"got {type(param_value)}"
                        )
                elif expected_type == int:
                    if not isinstance(param_value, int):
                        raise ValidationError(
                            f"Parameter '{param_name}' must be int, "
                            f"got {type(param_value)}"
                        )
                elif expected_type == str:
                    if not isinstance(param_value, str):
                        raise ValidationError(
                            f"Parameter '{param_name}' must be str, "
                            f"got {type(param_value)}"
                        )

    @staticmethod
    def is_valid_action(action_name: str) -> bool:
        """Check if an action name is valid without raising exceptions."""
        return action_name in VALID_ACTIONS

    @staticmethod
    def get_valid_actions() -> set:
        """Return set of all valid actions."""
        return set(VALID_ACTIONS)
