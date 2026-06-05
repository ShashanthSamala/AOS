# AURA-OS — Kernel Layer
# kernel_loop.py — Main OS event loop connecting all layers
# Author: Samala Shashanth | Project: AURA-OS
# Bug fixes applied: task parameter usage, handedness, cooldown

import time
from typing import Any, Dict, Optional
from kernel.scheduler import Scheduler
from kernel.context_manager import ContextManager
from kernel.logger import get_logger
from kernel.resource_manager import ResourceMonitor, ResourceState
from kernel.app_context import AppContextManager
from kernel.task_retry import RetryPolicy
from kernel.metrics import MetricsCollector
from foundation.llm_interface import LLMInterface
from foundation.prompt_template import SYSTEM_PROMPT, build_prompt
from foundation.action_validator import ActionValidator, ValidationError
from perception.gesture_map import GestureMap


class KernelLoop:
    """
    The AURA-OS kernel loop.

    Receives gesture events, determines actions (direct or via LLM),
    schedules tasks, and dispatches them to the MCP tool server.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, mcp_server: Any = None):
        self.config: Dict[str, Any] = config or {}
        kernel_cfg: Dict[str, Any] = self.config.get("kernel", {})

        # Mode: "direct" = use gesture_map, "llm" = always use LLM
        self.mode = kernel_cfg.get("mode", "direct")
        max_queue = kernel_cfg.get("scheduler_max_queue", 100)
        history_size = kernel_cfg.get("context_history_size", 10)

        # Initialize components
        self.llm = LLMInterface(config_path=None)
        self.llm.model = self.config.get("llm", {}).get("model", "tinyllama")
        self.scheduler = Scheduler(max_queue=max_queue)
        self.context = ContextManager(max_history=history_size)
        self.gesture_map = GestureMap(self.config)
        self.mcp = mcp_server
        self.log = get_logger("kernel", self.config)

        # Resource monitoring
        resource_cfg = self.config.get("resources", {})
        self.resource_monitor = ResourceMonitor(resource_cfg)

        # Application context awareness (400 ms throttle window)
        self.app_context = AppContextManager(throttle_s=0.4)

        # Task retry policy for transient failures
        self.retry_policy = RetryPolicy(max_retries=3)

        # Metrics collection
        self.metrics = MetricsCollector(max_entries=1000)
        self._gesture_counter = 0

        self._running = False
        self._resource_state = ResourceState.HEALTHY

    def process_gesture(self, gesture_event):
        """
        Process a single gesture event through the full pipeline.

        Pipeline: gesture → resolve action → schedule → execute → log

        Args:
            gesture_event: dict with {gesture, timestamp, fingertip?, screen_pos?}

        Returns:
            dict with {action, reasoning, result}
        """
        start_time = time.time()
        gesture = gesture_event["gesture"]
        self.log.info(f"Gesture received: {gesture}")

        # Fetch live context token (internally throttled to ≤1 subprocess / 400 ms)
        context_token = self.app_context.get_context()
        self.log.debug(f"Context: {context_token}")

        # POINT/move_cursor is handled directly in gesture engine for smooth tracking
        if gesture == "POINT" and "screen_pos" in gesture_event:
            pos = gesture_event["screen_pos"]
            reasoning = f"Cursor tracking at ({pos[0]}, {pos[1]})"
            self.context.add(gesture, "move_cursor", reasoning)
            return {
                "gesture": gesture,
                "action": "move_cursor",
                "reasoning": reasoning,
                "result": {"status": "ok"},
                "latency_ms": 0,
            }

        # PINCH/click is handled directly in gesture engine for stable cursor
        if gesture == "PINCH":
            pos = gesture_event.get("screen_pos", ("?", "?"))
            reasoning = f"Pinch click at {pos}"
            self.context.add(gesture, "click", reasoning)
            return {
                "gesture": gesture,
                "action": "click",
                "reasoning": reasoning,
                "result": {"status": "ok"},
                "latency_ms": 0,
            }

        # Step 1: Resolve action
        action_data = self._resolve_action(gesture)
        action = action_data.get("action", "do_nothing")
        parameters = action_data.get("parameters", {})
        reasoning = action_data.get("reasoning", "")

        self.log.info(f"Action resolved: {action} ({reasoning})")

        # Step 2: Create task (not enqueued — executed inline)
        from kernel.scheduler import Task
        task = Task(
            command=gesture,
            action=action,
            parameters=parameters,
            reasoning=reasoning,
        )

        # Step 3: Execute via MCP
        result = self._execute_task(task)

        # Step 4: Record in context history
        self.context.add(gesture, action, reasoning, result)

        # Step 5: Record metrics
        duration_ms = int((time.time() - start_time) * 1000)
        self.metrics.record_gesture(
            gesture=gesture,
            latency_ms=duration_ms,
            success=(result.get("status") == "ok")
        )

        # Log metrics periodically
        self._gesture_counter += 1
        if self._gesture_counter % 100 == 0:
            stats = self.metrics.get_statistics()
            self.log.info(f"Metrics Summary: {stats}")

        return {
            "gesture": gesture,
            "action": action,
            "reasoning": reasoning,
            "result": result,
            "latency_ms": action_data.get("latency_ms", 0),
        }

    def _resolve_action(self, gesture):
        """
        Determine what action to perform for a gesture.

        In 'direct' mode: lookup gesture_map first, fall back to LLM if not degraded.
        In 'llm' mode: always send to LLM (unless resources critical).

        Special handling:
        - PEACE gesture: Cycle to next favorite app for THREE gesture
        - THREE gesture: Pass current app as parameter to open_app
        - Resource-aware: Skip LLM if DEGRADED/CRITICAL
        """
        # Check resource state
        self._resource_state = self.resource_monitor.check(
            queue_size=self.scheduler.size,
            max_queue=self.config.get("kernel", {}).get("scheduler_max_queue", 100)
        )

        if self._resource_state == ResourceState.CRITICAL:
            # In CRITICAL state, only accept direct-mapped gestures
            if not self.gesture_map.is_direct(gesture):
                return {
                    "action": "do_nothing",
                    "parameters": {},
                    "reasoning": f"CRITICAL resource state: rejecting unmapped gesture {gesture}",
                    "latency_ms": 0,
                }

        # Direct mode — try gesture map first
        if self.mode == "direct":
            direct_action = self.gesture_map.lookup(gesture)

            # Handle PEACE gesture specially (app cycling)
            if gesture == "PEACE" and direct_action is None:
                # Cycle to next favorite app
                current_idx = self.config.get("app_cycle_index", 0)
                favorite_apps = self.config.get("favorite_apps", ["code", "firefox"])
                next_idx = (current_idx + 1) % len(favorite_apps)
                next_app = favorite_apps[next_idx]

                # Update config (in-memory only)
                self.config["app_cycle_index"] = next_idx

                return {
                    "action": "open_app",
                    "parameters": {"app_name": next_app},
                    "reasoning": f"App cycle: {next_app} (index {next_idx}/{len(favorite_apps)})",
                    "latency_ms": 0,
                }

            # Handle THREE gesture (pass app name as parameter)
            if gesture == "THREE" and direct_action == "open_app":
                app_name = self.config.get("three_finger_app", "code")
                return {
                    "action": "open_app",
                    "parameters": {"app_name": app_name},
                    "reasoning": f"Direct mapping: Open app '{app_name}'",
                    "latency_ms": 0,
                }

            # Other direct mappings
            if direct_action is not None:
                return {
                    "action": direct_action,
                    "parameters": {},
                    "reasoning": f"Direct mapping: {gesture} → {direct_action}",
                    "latency_ms": 0,
                }

            # If resources are degraded/critical, don't use LLM for unmapped gestures
            if self.resource_monitor.should_skip_llm(self._resource_state):
                return {
                    "action": "do_nothing",
                    "parameters": {},
                    "reasoning": f"Resource {self._resource_state.value.upper()}: skipping LLM for unmapped {gesture}",
                    "latency_ms": 0,
                }

        # LLM reasoning (for unmapped gestures or LLM mode)
        # Check resources one more time before expensive LLM call
        if self.resource_monitor.should_skip_llm(self._resource_state):
            return {
                "action": "do_nothing",
                "parameters": {},
                "reasoning": f"Resource {self._resource_state.value.upper()}: skipping LLM",
                "latency_ms": 0,
            }

        # Get live app context for smarter LLM reasoning
        context_token = self.app_context.get_context()
        context_str = self.context.get_context_string(n=3)
        extra_info = {
            "Recent history": context_str,
            "Current app": context_token,
        }

        prompt = build_prompt(
            gesture=gesture,
            context=context_token,
            extra_info=extra_info,
        )

        self.log.info(f"Sending to LLM: {gesture} (ctx: {context_token}, resource state: {self._resource_state.value})")
        result = self.llm.ask(prompt, system_prompt=SYSTEM_PROMPT)
        return result

    def _execute_task(self, task):
        """Execute a task via the MCP tool server with validation and retry logic."""
        # Use the passed task directly (no queue pop)
        task.mark_running()

        if self.mcp is None:
            self.log.warning("No MCP server — task not executed")
            self.scheduler.complete_task(task, {"status": "no_mcp"})
            return {"status": "no_mcp", "message": "MCP server not initialized"}

        # Validate action before execution
        try:
            ActionValidator.validate_action({
                "action": task.action,
                "parameters": task.parameters,
            })
        except ValidationError as e:
            error_msg = f"Action validation failed: {e}"
            self.log.error(error_msg)
            self.scheduler.fail_task(task, error_msg)
            return {"status": "error", "message": error_msg}

        # Initialize retry counter if needed
        if not hasattr(task, 'retry_count'):
            task.retry_count = 0

        try:
            result = self.mcp.execute(task.action, task.parameters)

            # Check for timeout and determine if should retry
            if result.get("timed_out"):
                if self.retry_policy.should_retry(result.get("message", ""), task.retry_count):
                    task.retry_count += 1
                    delay = self.retry_policy.get_backoff_delay(task.retry_count)
                    self.log.warning(
                        f"Retrying {task.action} (attempt {task.retry_count}) after {delay}s delay"
                    )
                    time.sleep(delay)
                    self.scheduler.requeue_task(task)
                    return {"status": "retry", "message": f"Rescheduled for retry (attempt {task.retry_count})"}

            self.scheduler.complete_task(task, result)
            self.log.info(f"Task {task.id} completed: {result}")
            return result

        except Exception as e:
            error_msg = str(e)
            if self.retry_policy.should_retry(error_msg, getattr(task, 'retry_count', 0)):
                task.retry_count = getattr(task, 'retry_count', 0) + 1
                delay = self.retry_policy.get_backoff_delay(task.retry_count)
                self.log.warning(
                    f"Retrying {task.action} after error (attempt {task.retry_count}): {error_msg}"
                )
                time.sleep(delay)
                self.scheduler.requeue_task(task)
                return {"status": "retry", "message": f"Rescheduled for retry (attempt {task.retry_count})"}
            else:
                self.scheduler.fail_task(task, error_msg)
                self.log.error(f"Task {task.id} failed: {error_msg}")
                return {"status": "error", "message": error_msg}

    def run(self, gesture_engine):
        """
        Main kernel loop — continuously processes gestures.

        Args:
            gesture_engine: GestureEngine instance providing gesture events
        """
        self._running = True
        self.log.info("AURA-OS Kernel started.")
        self.log.info(f"Mode: {self.mode}")
        self.log.info(f"Gesture mappings:\n{self.gesture_map}")

        while self._running and gesture_engine.is_running:
            # Block waiting for next gesture (1s timeout for clean shutdown)
            event = gesture_engine.get_gesture(timeout=1.0)
            if event is None:
                continue

            try:
                result = self.process_gesture(event)
                self.log.info(
                    f"[{result['gesture']}] → {result['action']} "
                    f"| {result['reasoning']}"
                )
            except Exception as e:
                self.log.error(f"Kernel error processing gesture: {e}")

        self.log.info("AURA-OS Kernel stopped.")

    def stop(self):
        """Stop the kernel loop."""
        self._running = False

    def get_status(self):
        """Return current kernel status."""
        return {
            "running": self._running,
            "mode": self.mode,
            "scheduler": self.scheduler.get_queue_status(),
            "context_size": self.context.size,
        }
