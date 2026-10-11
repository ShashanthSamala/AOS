from kernel.orchestrator import CentralOrchestrator, AosInputFrame
from perception.intent_signal import HumanIntentSignal


def test_point_fast_path_uses_real_human_intent_signal():
    orchestrator = CentralOrchestrator.__new__(CentralOrchestrator)

    signal = HumanIntentSignal(
        modality="gesture",
        intent="move_cursor",
        target=(500, 300),
        confidence=0.95,
        timestamp=123.456,
        raw={
            "gesture": "POINT",
            "fingertip": {"x": 0.5, "y": 0.5},
        },
    )

    frame = AosInputFrame(
        prototype_token_id="POINT",
        timestamp_mono=123.456,
        fingertip={"x": 0.5, "y": 0.5},
        screen_pos=(500, 300),
        desktop_context="test",
        desktop_window_title="test",
        intent_signal=signal,
    )

    class FakeContext:
        def __init__(self):
            self.entries = []

        def add_intent(self, signal, action, reasoning="", result=None):
            self.entries.append(
                (signal, action, reasoning, result)
            )

    class FakeLogger:
        def __init__(self):
            self.errors = []

        def error(self, message):
            self.errors.append(message)

    orchestrator.context = FakeContext()
    orchestrator.log = FakeLogger()

    orchestrator._record_fast_path(frame)

    assert len(orchestrator.context.entries) == 1

    stored_signal, action, reasoning, result = orchestrator.context.entries[0]

    assert stored_signal is signal
    assert stored_signal.intent == "move_cursor"
    assert stored_signal.modality == "gesture"
    assert stored_signal.confidence == 0.95
    assert action == "move_cursor"
    assert reasoning == "Cursor tracking at (500, 300)"
    assert result is None
    assert orchestrator.log.errors == []


def test_point_fast_path_uses_semantic_intent_not_legacy_token():
    from perception.intent_signal import HumanIntentSignal

    orchestrator = CentralOrchestrator.__new__(CentralOrchestrator)

    signal = HumanIntentSignal(
        modality="gesture",
        intent="move_cursor",
        target=(100, 200),
        confidence=1.0,
        timestamp=1.0,
        raw={"gesture": "THREE"},
    )

    frame = AosInputFrame(
        prototype_token_id="THREE",
        timestamp_mono=1.0,
        fingertip={"x": 0.5, "y": 0.5},
        screen_pos=(100, 200),
        desktop_context="TEST",
        desktop_window_title="",
        intent_signal=signal,
    )

    class ContextSpy:
        def __init__(self):
            self.calls = []

        def add_intent(self, *args):
            self.calls.append(args)

    orchestrator.context = ContextSpy()

    orchestrator._record_fast_path(frame)

    assert len(orchestrator.context.calls) == 1
    assert orchestrator.context.calls[0][0] is signal
    assert orchestrator.context.calls[0][1] == "move_cursor"
