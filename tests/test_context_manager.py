import pytest

from kernel.context_manager import ContextManager
from perception.intent_signal import HumanIntentSignal


def make_signal():
    return HumanIntentSignal(
        modality="gesture",
        intent="open_app",
        target="firefox",
        confidence=0.95,
        timestamp=1234.567,
        raw={
            "gesture": "THREE",
            "fingertip": {"x": 0.5, "y": 0.5},
        },
    )


def test_add_intent_preserves_real_signal_object_and_metadata():
    context = ContextManager()
    signal = make_signal()

    context.add_intent(
        signal,
        action="open_firefox",
        reasoning="THREE gesture mapped to open_app",
        result="success",
    )

    assert context.size == 1

    entry = context.get_history()[0]

    # Exact signal object is preserved.
    assert entry["intent_signal"] is signal

    # Semantic metadata comes from the real signal.
    assert entry["intent"] == signal.intent
    assert entry["modality"] == signal.modality
    assert entry["target"] == signal.target
    assert entry["confidence"] == signal.confidence
    assert entry["timestamp"] == signal.timestamp
    assert entry["intent_signal"].raw == signal.raw

    # Action-side metadata remains intact.
    assert entry["action"] == "open_firefox"
    assert entry["reasoning"] == "THREE gesture mapped to open_app"
    assert entry["result"] == "success"


def test_add_intent_rejects_non_signal_input():
    context = ContextManager()

    with pytest.raises(TypeError):
        context.add_intent(
            "THREE",
            action="open_firefox",
        )

    assert context.size == 0


def test_add_intent_renders_semantic_context():
    context = ContextManager()
    signal = make_signal()

    context.add_intent(
        signal,
        action="open_firefox",
    )

    assert context.get_context_string() == (
        "Intent: open_app → Action: open_firefox"
    )


def test_legacy_add_remains_unchanged():
    context = ContextManager()

    context.add(
        "THREE",
        "open_firefox",
        reasoning="legacy path",
        result="success",
    )

    entry = context.get_history()[0]

    assert entry["gesture"] == "THREE"
    assert entry["action"] == "open_firefox"
    assert entry["reasoning"] == "legacy path"
    assert entry["result"] == "success"

    # Legacy add() must not fabricate semantic data.
    assert "intent_signal" not in entry
    assert "intent" not in entry
    assert "modality" not in entry


def test_context_string_supports_mixed_legacy_and_semantic_entries():
    context = ContextManager()

    context.add(
        "THREE",
        "open_firefox",
    )

    signal = make_signal()

    context.add_intent(
        signal,
        "focus_firefox",
    )

    assert context.get_context_string() == (
        "Gesture: THREE → Action: open_firefox\n"
        "Intent: open_app → Action: focus_firefox"
    )


def test_add_intent_respects_history_limit():
    context = ContextManager(max_history=2)

    signal1 = HumanIntentSignal(
        modality="gesture",
        intent="click",
        target=(100, 200),
        confidence=1.0,
        timestamp=1.0,
        raw={"gesture": "PINCH"},
    )

    signal2 = HumanIntentSignal(
        modality="gesture",
        intent="open_app",
        target="firefox",
        confidence=0.9,
        timestamp=2.0,
        raw={"gesture": "THREE"},
    )

    signal3 = HumanIntentSignal(
        modality="voice",
        intent="screenshot",
        target=None,
        confidence=0.99,
        timestamp=3.0,
        raw={"text": "take a screenshot"},
    )

    context.add_intent(signal1, "click")
    context.add_intent(signal2, "open_firefox")
    context.add_intent(signal3, "screenshot")

    history = context.get_history()

    assert len(history) == 2
    assert history[0]["intent_signal"] is signal2
    assert history[1]["intent_signal"] is signal3
