# pyre-ignore-all-errors
# AURA-OS — Kernel Layer
# orchestrator.py — Central Orchestrator Agent: async event loop pipeline
# Author: Samala Shashanth | Project: AURA-OS
#
# Governs five concurrent subsystems:
#   1. Async FIFO queue (asyncio.Queue, maxsize=5) — ingests raw tracking frames
#   2. 300 ms temporal state filter — deduplicates identical prototype_token_id
#   3. AosInputFrame packing — merges sensor data with desktop metadata
#   4. Blocking thread-delegated Ollama inference via ThreadPoolExecutor
#   5. Validation safety gate — drops malformed payloads, forwards good ones

import asyncio
import concurrent.futures
import dataclasses
import json
import queue
import threading
import time
from typing import Any, Dict, Optional, Tuple

from kernel.scheduler import Scheduler, Task
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


# ---------------------------------------------------------------------------
#  §1  AosInputFrame — standardised payload between perception → kernel
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, slots=True)
class AosInputFrame:
    """Immutable snapshot of a single perception event enriched with desktop
    metadata.  Flows through the orchestrator pipeline as the canonical
    inter-layer message format.

    Attributes:
        prototype_token_id: Gesture classification token, e.g. "FIST", "THREE".
        timestamp_mono:     Monotonic clock sample at capture (seconds).
        fingertip:          Normalised {x, y} in [0, 1] or None.
        screen_pos:         Mapped pixel coords or None.
        desktop_context:    Uppercase context token from AppContextManager.
        desktop_window_title: Raw X11 window title string.
    """
    prototype_token_id: str
    timestamp_mono: float
    fingertip: Optional[Dict[str, float]]
    screen_pos: Optional[Tuple[int, int]]
    desktop_context: str
    desktop_window_title: str


# ---------------------------------------------------------------------------
#  §2  TemporalStateFilter — 300 ms deduplication guard
# ---------------------------------------------------------------------------

class TemporalStateFilter:
    """Discards consecutive frames carrying the same ``prototype_token_id``
    when they arrive within the configured time envelope.

    Tracking gestures (``POINT``) always pass because each frame carries
    unique spatial data.  ``PINCH`` is also exempted as its frame-count
    gating already lives in the gesture engine.
    """

    _BYPASS_TOKENS = frozenset({"POINT", "PINCH"})

    __slots__ = ("_window_s", "_last_token", "_last_mono")

    def __init__(self, window_s: float = 0.3):
        self._window_s: float = window_s
        self._last_token: Optional[str] = None
        self._last_mono: float = -1e9  # seed so first call always passes

    def accept(self, frame: AosInputFrame) -> bool:
        """Return ``True`` if *frame* should be processed, ``False`` to
        discard as a redundant duplicate."""
        token = frame.prototype_token_id

        # Fast-path: tracking/click gestures are never deduplicated.
        if token in self._BYPASS_TOKENS:
            self._last_token = token
            self._last_mono = frame.timestamp_mono
            return True

        # Temporal dedup: same token within the window → discard.
        if (token == self._last_token
                and (frame.timestamp_mono - self._last_mono) < self._window_s):
            return False

        self._last_token = token
        self._last_mono = frame.timestamp_mono
        return True

    def reset(self) -> None:
        """Clear filter state (e.g. on stop/restart)."""
        self._last_token = None
        self._last_mono = -1e9


# ---------------------------------------------------------------------------
#  §3–5  CentralOrchestrator
# ---------------------------------------------------------------------------

class CentralOrchestrator:
    """Async event loop pipeline governing the AURA-OS kernel.

    Lifecycle::

        orchestrator = CentralOrchestrator(config, mcp_server)
        orchestrator.run(gesture_engine)   # blocks until stop()
        orchestrator.stop()
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        mcp_server: Any = None,
    ):
        self.config: Dict[str, Any] = config or {}
        kernel_cfg: Dict[str, Any] = self.config.get("kernel", {})

        # Operating mode: "direct" = gesture_map first, "llm" = always LLM
        self.mode: str = kernel_cfg.get("mode", "direct")

        # Async ingest queue — bounded buffer between vision thread and loop
        queue_size: int = kernel_cfg.get("ingest_queue_size", 5)
        self._ingest_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)

        # Temporal dedup filter (configurable, default 300 ms)
        filter_ms: int = kernel_cfg.get("temporal_filter_ms", 300)
        self._temporal_filter = TemporalStateFilter(window_s=filter_ms / 1000.0)

        # Desktop context provider (400 ms subprocess throttle)
        self._app_context = AppContextManager(throttle_s=0.4)

        # LLM interface
        self._llm = LLMInterface(config_path=None)
        self._llm.model = self.config.get("llm", {}).get("model", "tinyllama")

        # Single-thread executor for blocking Ollama calls
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ollama",
        )

        # Kernel sub-components (reused from existing codebase)
        max_queue = kernel_cfg.get("scheduler_max_queue", 100)
        history_size = kernel_cfg.get("context_history_size", 10)
        self.scheduler = Scheduler(max_queue=max_queue)
        self.context = ContextManager(max_history=history_size)
        self.gesture_map = GestureMap(self.config)
        self.mcp = mcp_server
        self.log = get_logger("orchestrator", self.config)

        # Resource monitoring
        resource_cfg = self.config.get("resources", {})
        self.resource_monitor = ResourceMonitor(resource_cfg)

        # Retry policy for transient MCP failures
        self.retry_policy = RetryPolicy(max_retries=3)

        # Metrics
        self.metrics = MetricsCollector(max_entries=1000)
        self._gesture_counter: int = 0

        # Lifecycle flags
        self._running: bool = False
        self._resource_state = ResourceState.HEALTHY
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ingest_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    #  §1  Ingest loop — bridges sync gesture queue → async pipeline
    # ------------------------------------------------------------------

    def _ingest_loop(self, gesture_engine) -> None:
        """Runs in a dedicated daemon thread.  Blocks on the synchronous
        ``gesture_engine.get_gesture()`` and packs accepted events into
        :class:`AosInputFrame` payloads, posting them into the async
        ingest queue.

        POINT and PINCH events are **not** routed through the async
        pipeline.  Their pyautogui actions are already executed in the
        gesture engine thread; we only record lightweight telemetry here.
        """
        self.log.info("[INGEST] Bridge thread started")
        _FAST_PATH_TOKENS = frozenset({"POINT", "PINCH"})

        while self._running and gesture_engine.is_running:
            try:
                raw_event = gesture_engine.get_gesture(timeout=0.5)
                if raw_event is None:
                    continue

                # Capture monotonic timestamp immediately
                mono_now = time.monotonic()

                # Fetch desktop context (internally throttled, <1 ms when cached)
                ctx_token = self._app_context.get_context()
                ctx_raw = self._app_context.get_active_window() or ""

                # Pack into canonical frame
                frame = AosInputFrame(
                    prototype_token_id=raw_event["gesture"],
                    timestamp_mono=mono_now,
                    fingertip=raw_event.get("fingertip"),
                    screen_pos=raw_event.get("screen_pos"),
                    desktop_context=ctx_token,
                    desktop_window_title=ctx_raw,
                )

                # POINT / PINCH fast-path: telemetry only, no async overhead
                if frame.prototype_token_id in _FAST_PATH_TOKENS:
                    self._record_fast_path(frame)
                    continue

                # Post to async queue (non-blocking, drop on overflow)
                if self._loop is not None:
                    try:
                        self._loop.call_soon_threadsafe(
                            self._try_enqueue, frame,
                        )
                    except RuntimeError:
                        # Loop already closed during shutdown
                        break

            except Exception as e:
                self.log.error(f"[INGEST] Error: {e}")
                continue

        self.log.info("[INGEST] Bridge thread stopped")

    def _try_enqueue(self, frame: AosInputFrame) -> None:
        """Thread-safe enqueue helper called via ``call_soon_threadsafe``.
        Drops the frame silently if the queue is at capacity (back-pressure)."""
        try:
            self._ingest_queue.put_nowait(frame)
        except asyncio.QueueFull:
            self.log.debug(
                f"[INGEST] Queue full — dropped {frame.prototype_token_id}"
            )

    def _record_fast_path(self, frame: AosInputFrame) -> None:
        """Record POINT/PINCH telemetry without entering the async pipeline.
        Kept intentionally minimal to stay within the 20–30 ms budget."""
        gesture = frame.prototype_token_id

        if gesture == "POINT" and frame.screen_pos:
            reasoning = f"Cursor tracking at ({frame.screen_pos[0]}, {frame.screen_pos[1]})"
            self.context.add(gesture, "move_cursor", reasoning)
        elif gesture == "PINCH":
            pos = frame.screen_pos or ("?", "?")
            reasoning = f"Pinch click at {pos}"
            self.context.add(gesture, "click", reasoning)

    # ------------------------------------------------------------------
    #  §2–5  Async processing pipeline
    # ------------------------------------------------------------------

    async def _process_loop(self) -> None:
        """Core async coroutine.  Continuously drains the ingest queue,
        applies the temporal filter, resolves actions (direct or LLM),
        validates results, and dispatches to MCP."""
        self.log.info("[ORCHESTRATOR] Async process loop started")

        while self._running:
            try:
                # Block (async) until a frame is available
                try:
                    frame: AosInputFrame = await asyncio.wait_for(
                        self._ingest_queue.get(), timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    continue

                # §2 — Temporal state filter (300 ms dedup)
                if not self._temporal_filter.accept(frame):
                    self.log.debug(
                        f"[FILTER] Discarded duplicate: {frame.prototype_token_id}"
                    )
                    continue

                self.log.info(
                    f"[ORCHESTRATOR] Processing: {frame.prototype_token_id} "
                    f"(ctx={frame.desktop_context})"
                )

                # §3 — Resolve action
                start_time = time.time()
                action_data = await self._resolve_action(frame)
                action = action_data.get("action", "do_nothing")
                parameters = action_data.get("parameters", {})
                reasoning = action_data.get("reasoning", "")

                self.log.info(f"[ORCHESTRATOR] Resolved: {action} ({reasoning})")

                # §5 — Create task and execute with validation safety gate
                task = Task(
                    command=frame.prototype_token_id,
                    action=action,
                    parameters=parameters,
                    reasoning=reasoning,
                )

                result = self._execute_task(task)

                # Record context history
                self.context.add(
                    frame.prototype_token_id, action, reasoning, result,
                )

                # Record metrics
                duration_ms = int((time.time() - start_time) * 1000)
                self.metrics.record_gesture(
                    gesture=frame.prototype_token_id,
                    latency_ms=duration_ms,
                    success=(result.get("status") == "ok"),
                )

                self._gesture_counter += 1
                if self._gesture_counter % 100 == 0:
                    stats = self.metrics.get_statistics()
                    self.log.info(f"[ORCHESTRATOR] Metrics: {stats}")

                self.log.info(
                    f"[{frame.prototype_token_id}] → {action} | {reasoning}"
                )

            except Exception as e:
                self.log.error(f"[ORCHESTRATOR] Pipeline error: {e}")

        self.log.info("[ORCHESTRATOR] Async process loop stopped")

    # ------------------------------------------------------------------
    #  Action resolution (direct map or LLM via executor)
    # ------------------------------------------------------------------

    async def _resolve_action(self, frame: AosInputFrame) -> Dict[str, Any]:
        """Determine the action for a frame.  Direct-mapped gestures are
        resolved synchronously; unmapped gestures are sent to the local
        Ollama instance via a blocking executor call (§4)."""
        gesture = frame.prototype_token_id

        # Resource check
        self._resource_state = self.resource_monitor.check(
            queue_size=self.scheduler.size,
            max_queue=self.config.get("kernel", {}).get("scheduler_max_queue", 100),
        )

        if self._resource_state == ResourceState.CRITICAL:
            if not self.gesture_map.is_direct(gesture):
                return {
                    "action": "do_nothing",
                    "parameters": {},
                    "reasoning": f"CRITICAL resource state: rejecting {gesture}",
                    "latency_ms": 0,
                }

        # Direct mode — gesture map lookup
        if self.mode == "direct":
            direct = self._try_direct_resolve(frame)
            if direct is not None:
                return direct

            # If resources degraded, skip LLM
            if self.resource_monitor.should_skip_llm(self._resource_state):
                return {
                    "action": "do_nothing",
                    "parameters": {},
                    "reasoning": f"Resource {self._resource_state.value}: skipping LLM for {gesture}",
                    "latency_ms": 0,
                }

        # LLM mode or unmapped gesture — check resources first
        if self.resource_monitor.should_skip_llm(self._resource_state):
            return {
                "action": "do_nothing",
                "parameters": {},
                "reasoning": f"Resource {self._resource_state.value}: skipping LLM",
                "latency_ms": 0,
            }

        # §4 — Blocking Ollama call delegated to ThreadPoolExecutor
        self.log.info(
            f"[ORCHESTRATOR] LLM inference: {gesture} "
            f"(ctx={frame.desktop_context}, resource={self._resource_state.value})"
        )

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._call_ollama, frame,
            )
            return result
        except Exception as e:
            self.log.error(f"[ORCHESTRATOR] LLM executor error: {e}")
            return {
                "action": "do_nothing",
                "parameters": {},
                "reasoning": f"LLM error: {e}",
                "latency_ms": 0,
            }

    def _try_direct_resolve(self, frame: AosInputFrame) -> Optional[Dict[str, Any]]:
        """Attempt direct gesture-map resolution.  Returns ``None`` if the
        gesture should be routed to the LLM."""
        gesture = frame.prototype_token_id
        direct_action = self.gesture_map.lookup(gesture)

        # PEACE — app cycling
        if gesture == "PEACE" and direct_action is None:
            current_idx = self.config.get("app_cycle_index", 0)
            favorite_apps = self.config.get("favorite_apps", ["code", "firefox"])
            next_idx = (current_idx + 1) % len(favorite_apps)
            next_app = favorite_apps[next_idx]
            self.config["app_cycle_index"] = next_idx
            return {
                "action": "open_app",
                "parameters": {"app_name": next_app},
                "reasoning": f"App cycle: {next_app} ({next_idx}/{len(favorite_apps)})",
                "latency_ms": 0,
            }

        # THREE — open_app with configured app name
        if gesture == "THREE" and direct_action == "open_app":
            app_name = self.config.get("three_finger_app", "code")
            return {
                "action": "open_app",
                "parameters": {"app_name": app_name},
                "reasoning": f"Direct: Open '{app_name}'",
                "latency_ms": 0,
            }

        # Other direct mappings
        if direct_action is not None:
            return {
                "action": direct_action,
                "parameters": {},
                "reasoning": f"Direct: {gesture} → {direct_action}",
                "latency_ms": 0,
            }

        return None  # route to LLM

    def _call_ollama(self, frame: AosInputFrame) -> Dict[str, Any]:
        """Blocking call to the local Ollama instance.  Executed inside the
        single-worker ``ThreadPoolExecutor`` to avoid stalling the async
        event loop.  This is §4 of the pipeline."""
        context_str = self.context.get_context_string(n=3)
        extra_info = {
            "Recent history": context_str,
            "Current app": frame.desktop_context,
            "Window": frame.desktop_window_title,
        }

        prompt = build_prompt(
            gesture=frame.prototype_token_id,
            context=frame.desktop_context,
            extra_info=extra_info,
        )

        return self._llm.ask(prompt, system_prompt=SYSTEM_PROMPT)

    # ------------------------------------------------------------------
    #  §5  Task execution with validation safety gate
    # ------------------------------------------------------------------

    def _execute_task(self, task: Task) -> Dict[str, Any]:
        """Execute a resolved task via MCP with full validation.  Malformed
        or invalid payloads are instantly dropped (systemic safety)."""
        task.mark_running()

        if self.mcp is None:
            self.log.warning("[ORCHESTRATOR] No MCP server — task skipped")
            self.scheduler.complete_task(task, {"status": "no_mcp"})
            return {"status": "no_mcp", "message": "MCP server not initialised"}

        # Validation gate — catch invalid actions before they reach MCP
        try:
            ActionValidator.validate_action({
                "action": task.action,
                "parameters": task.parameters,
            })
        except ValidationError as e:
            error_msg = f"Validation failed: {e}"
            self.log.error(f"[SAFETY] {error_msg} — payload dropped")
            self.scheduler.fail_task(task, error_msg)
            return {"status": "error", "message": error_msg}
        except Exception as e:
            error_msg = f"Unexpected validation error: {e}"
            self.log.error(f"[SAFETY] {error_msg} — payload dropped")
            self.scheduler.fail_task(task, error_msg)
            return {"status": "error", "message": error_msg}

        # Initialise retry counter
        if not hasattr(task, "retry_count"):
            task.retry_count = 0

        try:
            result = self.mcp.execute(task.action, task.parameters)

            # Retry on timeout (transient failure)
            if result.get("timed_out"):
                if self.retry_policy.should_retry(
                    result.get("message", ""), task.retry_count
                ):
                    task.retry_count += 1
                    delay = self.retry_policy.get_backoff_delay(task.retry_count)
                    self.log.warning(
                        f"[ORCHESTRATOR] Retry {task.action} "
                        f"(attempt {task.retry_count}) after {delay}s"
                    )
                    time.sleep(delay)
                    self.scheduler.requeue_task(task)
                    return {
                        "status": "retry",
                        "message": f"Rescheduled (attempt {task.retry_count})",
                    }

            self.scheduler.complete_task(task, result)
            self.log.info(f"[ORCHESTRATOR] Task {task.id} done: {result}")
            return result

        except Exception as e:
            error_msg = str(e)
            # Catch any malformed string or unexpected exception → drop
            self.log.error(
                f"[SAFETY] Execution error for {task.action}: {error_msg} "
                f"— payload dropped"
            )
            self.scheduler.fail_task(task, error_msg)
            return {"status": "error", "message": error_msg}

    # ------------------------------------------------------------------
    #  Lifecycle
    # ------------------------------------------------------------------

    def run(self, gesture_engine) -> None:
        """Boot the orchestrator.  Starts the ingest bridge thread and
        enters the async event loop on the calling (main) thread.

        Args:
            gesture_engine: ``GestureEngine`` instance providing raw events.
        """
        self._running = True
        self.log.info("=" * 55)
        self.log.info("AURA-OS Central Orchestrator — booting")
        self.log.info(f"  Mode            : {self.mode}")
        self.log.info(f"  Ingest queue    : maxsize={self._ingest_queue.maxsize}")
        self.log.info(f"  Temporal filter : {self._temporal_filter._window_s * 1000:.0f} ms")
        self.log.info(f"  Gesture map     :\n{self.gesture_map}")
        self.log.info("=" * 55)

        # Start ingest bridge thread
        self._ingest_thread = threading.Thread(
            target=self._ingest_loop,
            args=(gesture_engine,),
            daemon=True,
            name="orchestrator-ingest",
        )
        self._ingest_thread.start()

        # Run async event loop on main thread
        try:
            asyncio.run(self._async_main())
        except KeyboardInterrupt:
            self.log.info("[ORCHESTRATOR] Interrupted — shutting down")
        finally:
            self._running = False
            self._executor.shutdown(wait=False)
            if self._ingest_thread and self._ingest_thread.is_alive():
                self._ingest_thread.join(timeout=2)
            self.log.info("[ORCHESTRATOR] Shutdown complete")

    async def _async_main(self) -> None:
        """Async entry point — stores the loop reference for the ingest
        thread and runs the processing pipeline."""
        self._loop = asyncio.get_event_loop()
        await self._process_loop()

    def stop(self) -> None:
        """Signal the orchestrator to stop."""
        self._running = False
        self._temporal_filter.reset()

    def get_status(self) -> Dict[str, Any]:
        """Return current orchestrator status."""
        return {
            "running": self._running,
            "mode": self.mode,
            "engine": "orchestrator",
            "scheduler": self.scheduler.get_queue_status(),
            "context_size": self.context.size,
            "ingest_queue_size": (
                self._ingest_queue.qsize()
                if self._ingest_queue else 0
            ),
            "resource_state": self._resource_state.value,
        }
