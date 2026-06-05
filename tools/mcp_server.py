# AURA-OS — MCP Tools Layer
# mcp_server.py — Tool registry and dispatcher with timeout protection
# Author: Samala Shashanth | Project: AURA-OS

import threading
import time


class MCPServer:
    """
    Model Context Protocol tool server with timeout protection.
    Registers tools and dispatches action requests with execution timeouts.
    Prevents system freeze by enforcing timeouts on all tool execution.
    """

    def __init__(self, default_timeout: float = 15.0):
        """
        Initialize MCP server.

        Args:
            default_timeout: Default timeout in seconds for tool execution
        """
        self._tools = {}
        self.default_timeout = default_timeout

    def register_tool(self, name, handler, description=""):
        """
        Register a tool with the MCP server.

        Args:
            name: Action name (e.g. 'click', 'open_browser')
            handler: Callable that executes the action
            description: Human-readable description of what the tool does
        """
        self._tools[name] = {
            "handler": handler,
            "description": description,
        }

    def execute(self, action, parameters=None):
        """
        Execute a registered tool by action name WITH TIMEOUT PROTECTION.

        Args:
            action: Action name to execute
            parameters: Dict of parameters to pass to the handler

        Returns:
            dict with {status, result} or {status, error, timed_out}
        """
        parameters = parameters or {}

        if action not in self._tools:
            return {
                "status": "error",
                "message": f"Unknown action: {action}",
                "available": list(self._tools.keys()),
            }

        # Execute with timeout protection
        return self._execute_with_timeout(
            action=action,
            handler=self._tools[action]["handler"],
            parameters=parameters
        )

    def _execute_with_timeout(self, action: str, handler, parameters: dict):
        """Execute handler with timeout protection."""
        result_container = {"result": None, "error": None, "completed": False}
        start = time.time()

        def target():
            try:
                result_container["result"] = handler(**parameters)
                result_container["completed"] = True
            except Exception as e:
                result_container["error"] = str(e)
                result_container["completed"] = True

        # Run in daemon thread so it doesn't block forever
        thread = threading.Thread(target=target, daemon=True)
        thread.start()

        # Wait for completion with timeout
        thread.join(timeout=self.default_timeout)
        duration = round((time.time() - start) * 1000)

        if thread.is_alive():
            # Timeout - thread still running
            return {
                "status": "timeout",
                "action": action,
                "message": f"Action '{action}' exceeded {self.default_timeout}s timeout",
                "duration_ms": duration,
                "timed_out": True,
            }

        if result_container["error"]:
            return {
                "status": "error",
                "action": action,
                "message": result_container["error"],
                "duration_ms": duration,
                "timed_out": False,
            }

        return {
            "status": "ok",
            "action": action,
            "result": result_container["result"],
            "duration_ms": duration,
            "timed_out": False,
        }

    def list_tools(self):
        """Return list of registered tools with descriptions."""
        return [
            {"name": name, "description": info["description"]}
            for name, info in self._tools.items()
        ]

    def has_tool(self, action):
        """Check if a tool is registered."""
        return action in self._tools

    def __repr__(self):
        return f"MCPServer({len(self._tools)} tools registered, timeout={self.default_timeout}s)"
