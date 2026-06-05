# AURA-OS — Tools Layer
# timeout_executor.py — Safely execute tools with timeout protection
# Author: Samala Shashanth | Project: AURA-OS

"""
Prevents system freeze by enforcing timeouts on all tool execution.
Uses threading with daemon threads to ensure timeout works even for
blocking subprocess calls.
"""

import threading
import time
from typing import Any, Callable, Optional, Dict


class TimeoutExecutor:
    """Executes functions with timeout protection."""

    def __init__(self, default_timeout: float = 15.0):
        """
        Initialize timeout executor.

        Args:
            default_timeout: Default timeout in seconds (default: 15s)
        """
        self.default_timeout = default_timeout
        self._results = {}
        self._errors = {}

    def execute_with_timeout(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        timeout: float = None,
        name: str = "task"
    ) -> Dict[str, Any]:
        """
        Execute a function with timeout protection.

        Args:
            func: Function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            timeout: Timeout in seconds (uses default if None)
            name: Name for debugging

        Returns:
            Dict with {status, result, error, duration_ms, timed_out}
        """
        kwargs = kwargs or {}
        timeout = timeout or self.default_timeout
        start = time.time()

        result_container = {"result": None, "error": None, "completed": False}

        def target():
            try:
                result_container["result"] = func(*args, **kwargs)
                result_container["completed"] = True
            except Exception as e:
                result_container["error"] = str(e)
                result_container["completed"] = True

        # Run in daemon thread
        thread = threading.Thread(target=target, daemon=True)
        thread.start()

        # Wait for completion with timeout
        thread.join(timeout=timeout)

        duration = round((time.time() - start) * 1000)

        if thread.is_alive():
            # Thread still running after timeout
            return {
                "status": "timeout",
                "result": None,
                "error": f"Execution exceeded {timeout}s timeout",
                "duration_ms": duration,
                "timed_out": True,
            }

        if result_container["error"]:
            return {
                "status": "error",
                "result": None,
                "error": result_container["error"],
                "duration_ms": duration,
                "timed_out": False,
            }

        return {
            "status": "ok",
            "result": result_container["result"],
            "error": None,
            "duration_ms": duration,
            "timed_out": False,
        }


# Singleton executor
_executor = TimeoutExecutor(default_timeout=15.0)


def execute_with_timeout(
    func: Callable,
    args: tuple = (),
    kwargs: dict = None,
    timeout: float = None,
    name: str = "task"
) -> Dict[str, Any]:
    """
    Execute function with timeout (singleton convenience function).

    Args:
        func: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        timeout: Timeout in seconds
        name: Name for debugging

    Returns:
        Dict with execution result
    """
    return _executor.execute_with_timeout(func, args, kwargs, timeout, name)
