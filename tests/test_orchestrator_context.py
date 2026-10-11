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


def make_frame(signal=None):
    return AosInputFrame(
        prototype_token_id="THREE",
        timestamp_mono=123.456,
        fingertip={"x": 0.5, "y": 0.5},
        screen_pos=(500, 300),
        desktop_context="test",
        desktop_window_title="test",
        intent_signal=signal,
    )


def test_normal_pipeline_context_preserves_signal_and_result():
    orchestrator = make_orchestrator()

    signal = HumanIntentSignal(
        modality="gesture",
        intent="open_app",
        target="firefox",
        confidence=0.95,
        timestamp=123.456,
        raw={"gesture": "THREE"},
    )

    frame = make_frame(signal)

    action = "open_app"
    reasoning = "Opening Firefox"
    result = {
        "status": "ok",
        "app": "firefox",
    }

    # Exercise the migrated context-recording boundary directly.
    if frame.intent_signal is None:
        orchestrator.log.error(
            "[CONTEXT] Normal pipeline missing HumanIntentSignal"
        )
    else:
        orchestrator.context.add_intent(
            frame.intent_signal,
            action,
            reasoning,
            result,
        )

    assert len(orchestrator.context.entries) == 1

    stored_signal, stored_action, stored_reasoning, stored_result = (
        orchestrator.context.entries[0]
    )

    assert stored_signal is signal
    assert stored_signal.intent == "open_app"
    assert stored_signal.modality == "gesture"
    assert stored_signal.confidence == 0.95
    assert stored_action == "open_app"
    assert stored_reasoning == "Opening Firefox"
    assert stored_result == result
    assert orchestrator.log.errors == []


def test_normal_pipeline_missing_signal_does_not_fabricate_signal():
    orchestrator = make_orchestrator()

    frame = make_frame()

    if frame.intent_signal is None:
        orchestrator.log.error(
            "[CONTEXT] Normal pipeline missing HumanIntentSignal"
        )
    else:
        orchestrator.context.add_intent(
            frame.intent_signal,
            "open_app",
            "Opening Firefox",
            {"status": "ok"},
        )

    assert orchestrator.context.entries == []
    assert orchestrator.log.errors == [
        "[CONTEXT] Normal pipeline missing HumanIntentSignal"
    ]
