from kernel.orchestrator import AosInputFrame, CentralOrchestrator
from perception.intent_signal import HumanIntentSignal


def make_frame(
    intent,
    prototype_token_id,
    *,
    target=None,
    timestamp=1.0,
    raw_gesture=None,
):
    signal = HumanIntentSignal(
        modality="gesture",
        intent=intent,
        target=target,
        confidence=1.0,
        timestamp=timestamp,
        raw={"gesture": raw_gesture or prototype_token_id},
    )

    return AosInputFrame(
        prototype_token_id=prototype_token_id,
        timestamp_mono=timestamp,
        fingertip=None,
        screen_pos=target,
        desktop_context="TEST",
        desktop_window_title="",
        intent_signal=signal,
    )


def make_orchestrator(config=None):
    orchestrator = CentralOrchestrator.__new__(CentralOrchestrator)

    config = config or {
        "kernel": {
            "mode": "direct",
        },
    }

    orchestrator.config = config
    orchestrator.mode = "direct"

    from perception.gesture_map import GestureMap

    orchestrator.gesture_map = GestureMap(config)

    return orchestrator


def test_direct_resolution_preserves_open_browser():
    orchestrator = make_orchestrator()

    frame = make_frame(
        intent="open_browser",
        prototype_token_id="FOUR",
    )

    result = orchestrator._try_direct_resolve(frame)

    assert result is not None
    assert result["action"] == "open_browser"


def test_direct_resolution_preserves_minimize_all():
    orchestrator = make_orchestrator()

    frame = make_frame(
        intent="minimize_all",
        prototype_token_id="FIVE",
    )

    result = orchestrator._try_direct_resolve(frame)

    assert result is not None
    assert result["action"] == "minimize_all"


def test_direct_resolution_preserves_screenshot():
    orchestrator = make_orchestrator()

    frame = make_frame(
        intent="screenshot",
        prototype_token_id="FIST",
    )

    result = orchestrator._try_direct_resolve(frame)

    assert result is not None
    assert result["action"] == "screenshot"


def test_semantic_intent_can_override_legacy_token_for_resolution():
    orchestrator = make_orchestrator()

    # Migration-proof case:
    # legacy token says FOUR, but the semantic intent says open_app.
    frame = make_frame(
        intent="open_app",
        prototype_token_id="FOUR",
    )

    result = orchestrator._try_direct_resolve(frame)

    assert result is not None
    assert result["action"] == "open_app"


def test_three_intent_uses_configured_three_finger_app():
    orchestrator = make_orchestrator({
        "kernel": {
            "mode": "direct",
        },
        "three_finger_app": "code",
    })

    frame = make_frame(
        intent="open_app",
        prototype_token_id="THREE",
    )

    result = orchestrator._try_direct_resolve(frame)

    assert result is not None
    assert result["action"] == "open_app"
    assert result["parameters"] == {"app_name": "code"}


def test_peace_intent_cycles_favorite_apps():
    orchestrator = make_orchestrator({
        "kernel": {
            "mode": "direct",
        },
        "favorite_apps": ["code", "firefox"],
        "app_cycle_index": 0,
    })

    frame = make_frame(
        intent="open_app",
        prototype_token_id="PEACE",
    )

    result = orchestrator._try_direct_resolve(frame)

    assert result is not None
    assert result["action"] == "open_app"
    assert result["parameters"] == {"app_name": "firefox"}
    assert orchestrator.config["app_cycle_index"] == 1


def test_resolve_action_uses_semantic_intent_under_critical_resources():
    orchestrator = make_orchestrator()

    from kernel.orchestrator import ResourceState

    class ResourceSpy:
        def __init__(self):
            self.calls = []

        def check(self, queue_size, max_queue):
            self.calls.append(("check", queue_size, max_queue))
            return ResourceState.CRITICAL

        def should_skip_llm(self, state):
            self.calls.append(("should_skip_llm", state))
            return True

    class SchedulerSpy:
        size = 0

    orchestrator.resource_monitor = ResourceSpy()
    orchestrator.scheduler = SchedulerSpy()

    # Deliberately use an unknown legacy token.
    # The semantic intent must be authoritative.
    frame = make_frame(
        intent="open_app",
        prototype_token_id="UNKNOWN",
    )

    import asyncio

    result = asyncio.run(
        orchestrator._resolve_action(frame)
    )

    assert result["action"] == "open_app"


def test_call_ollama_uses_semantic_intent_not_legacy_token():
    orchestrator = make_orchestrator()

    from kernel.context_manager import ContextManager
    orchestrator.context = ContextManager(max_history=3)

    from perception.intent_signal import HumanIntentSignal

    signal = HumanIntentSignal(
        modality="gesture",
        intent="open_app",
        target=None,
        confidence=1.0,
        timestamp=1.0,
        raw={"gesture": "UNKNOWN"},
    )

    frame = AosInputFrame(
        prototype_token_id="UNKNOWN",
        timestamp_mono=1.0,
        fingertip=None,
        screen_pos=None,
        desktop_context="TEST",
        desktop_window_title="",
        intent_signal=signal,
    )

    class LLMSPy:
        def __init__(self):
            self.prompt = None

        def ask(self, prompt, system_prompt=None):
            self.prompt = prompt
            return {
                "action": "do_nothing",
                "parameters": {},
                "reasoning": "test",
                "latency_ms": 0,
            }

    llm = LLMSPy()
    orchestrator._llm = llm

    result = orchestrator._call_ollama(frame)

    assert result["action"] == "do_nothing"
    assert "open_app" in llm.prompt
    assert "UNKNOWN" not in llm.prompt

def test_task_command_uses_semantic_intent_not_legacy_token():
    import asyncio

    orchestrator = make_orchestrator()

    frame = make_frame(
        intent="open_app",
        prototype_token_id="UNKNOWN",
    )

    # make_orchestrator() bypasses __init__, so provide the dependencies
    # required by the real _process_loop() without starting external services.
    from kernel.logger import get_logger
    from kernel.context_manager import ContextManager
    from kernel.metrics import MetricsCollector
    from kernel.resource_manager import ResourceMonitor

    orchestrator.log = get_logger("test_orchestrator", {})
    orchestrator._ingest_queue = asyncio.Queue()
    orchestrator._running = True
    orchestrator._resource_state = None
    orchestrator.context = ContextManager(max_history=3)
    orchestrator.metrics = MetricsCollector(max_entries=10)
    orchestrator.resource_monitor = ResourceMonitor({})
    orchestrator.scheduler = type(
        "SchedulerSpy",
        (),
        {"size": 0},
    )()
    orchestrator._temporal_filter = type(
        "FilterSpy",
        (),
        {"accept": lambda self, frame: True},
    )()

    captured = {}

    def capture_task(task):
        captured["task"] = task
        orchestrator._running = False
        return {"status": "ok"}

    orchestrator._execute_task = capture_task

    async def run_once():
        orchestrator._ingest_queue.put_nowait(frame)
        await orchestrator._process_loop()

    asyncio.run(run_once())

    assert "task" in captured
    assert captured["task"].command == "open_app"
    assert captured["task"].command != "UNKNOWN"

def test_metrics_use_semantic_intent_not_legacy_token():
    import asyncio

    orchestrator = make_orchestrator()

    frame = make_frame(
        intent="open_app",
        prototype_token_id="UNKNOWN",
    )

    from kernel.logger import get_logger
    from kernel.context_manager import ContextManager
    from kernel.metrics import MetricsCollector
    from kernel.resource_manager import ResourceMonitor

    orchestrator.log = get_logger("test_metrics", {})
    orchestrator._ingest_queue = asyncio.Queue()
    orchestrator._running = True
    orchestrator._resource_state = None
    orchestrator.context = ContextManager(max_history=3)
    orchestrator.metrics = MetricsCollector(max_entries=10)
    orchestrator.resource_monitor = ResourceMonitor({})
    orchestrator.scheduler = type(
        "SchedulerSpy",
        (),
        {"size": 0},
    )()
    orchestrator._temporal_filter = type(
        "FilterSpy",
        (),
        {"accept": lambda self, frame: True},
    )()

    captured = {}

    def capture_task(task):
        captured["task"] = task
        orchestrator._running = False
        return {"status": "ok"}

    orchestrator._execute_task = capture_task

    async def run_once():
        orchestrator._ingest_queue.put_nowait(frame)
        await orchestrator._process_loop()

    asyncio.run(run_once())

    assert "task" in captured

    metrics = orchestrator.metrics.get_recent(1)
    assert len(metrics) == 1
    assert metrics[0]["gesture"] == "open_app"
    assert metrics[0]["gesture"] != "UNKNOWN"
