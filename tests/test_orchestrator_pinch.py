from kernel.orchestrator import CentralOrchestrator, AosInputFrame
from perception.intent_signal import HumanIntentSignal


class FakeContext:
    def __init__(self):
        self.entries = []

    def add_intent(self, signal, action, reasoning="", result=None):
        self.entries.append((signal, action, reasoning, result))


class FakeLogger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)


def make_orchestrator():
    orchestrator = CentralOrchestrator.__new__(CentralOrchestrator)
    orchestrator.context = FakeContext()
    orchestrator.log = FakeLogger()
    return orchestrator


def make_pinch_frame(signal=None):
    return AosInputFrame(
        prototype_token_id="PINCH",
        timestamp_mono=123.456,
        fingertip={"x": 0.5, "y": 0.5},
        screen_pos=(500, 300),
        desktop_context="test",
        desktop_window_title="test",
        intent_signal=signal,
    )


def test_pinch_fast_path_uses_real_human_intent_signal():
    orchestrator = make_orchestrator()

    signal = HumanIntentSignal(
        modality="gesture",
        intent="click",
        target=(500, 300),
        confidence=0.95,
        timestamp=123.456,
        raw={
            "gesture": "PINCH",
            "fingertip": {"x": 0.5, "y": 0.5},
        },
    )

    frame = make_pinch_frame(signal)

    orchestrator._record_fast_path(frame)

    assert len(orchestrator.context.entries) == 1

    stored_signal, action, reasoning, result = (
        orchestrator.context.entries[0]
    )

    assert stored_signal is signal
    assert stored_signal.intent == "click"
    assert stored_signal.modality == "gesture"
    assert stored_signal.confidence == 0.95
    assert action == "click"
    assert reasoning == "Pinch click at (500, 300)"
    assert result is None
    assert orchestrator.log.errors == []


def test_pinch_fast_path_rejects_missing_human_intent_signal():
    orchestrator = make_orchestrator()

    frame = make_pinch_frame()

    orchestrator._record_fast_path(frame)

    assert orchestrator.context.entries == []
    assert orchestrator.log.errors == [
        "[CONTEXT] PINCH fast path missing HumanIntentSignal"
    ]
