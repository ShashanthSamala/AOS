from kernel.orchestrator import AosInputFrame, TemporalStateFilter
from perception.intent_signal import HumanIntentSignal


def make_frame(
    intent,
    timestamp,
    *,
    modality="gesture",
    target=None,
):
    signal = HumanIntentSignal(
        modality=modality,
        intent=intent,
        target=target,
        confidence=1.0,
        timestamp=timestamp,
        raw={"intent": intent},
    )

    return AosInputFrame(
        prototype_token_id={
            "move_cursor": "POINT",
            "click": "PINCH",
        }.get(intent, intent.upper()),
        timestamp_mono=timestamp,
        fingertip=None,
        screen_pos=None,
        desktop_context="TEST",
        desktop_window_title="",
        intent_signal=signal,
    )


def test_first_intent_is_accepted():
    filt = TemporalStateFilter(window_s=0.3)

    frame = make_frame("open_app", 1.0)

    assert filt.accept(frame) is True


def test_same_intent_within_window_is_rejected():
    filt = TemporalStateFilter(window_s=0.3)

    assert filt.accept(make_frame("open_app", 1.0)) is True
    assert filt.accept(make_frame("open_app", 1.1)) is False


def test_same_intent_after_window_is_accepted():
    filt = TemporalStateFilter(window_s=0.3)

    assert filt.accept(make_frame("open_app", 1.0)) is True
    assert filt.accept(make_frame("open_app", 1.31)) is True


def test_different_intent_within_window_is_accepted():
    filt = TemporalStateFilter(window_s=0.3)

    assert filt.accept(make_frame("open_app", 1.0)) is True
    assert filt.accept(make_frame("open_browser", 1.1)) is True


def test_point_bypasses_deduplication():
    filt = TemporalStateFilter(window_s=0.3)

    assert filt.accept(make_frame("move_cursor", 1.0)) is True
    assert filt.accept(make_frame("move_cursor", 1.1)) is True


def test_pinch_bypasses_deduplication():
    filt = TemporalStateFilter(window_s=0.3)

    assert filt.accept(make_frame("click", 1.0)) is True
    assert filt.accept(make_frame("click", 1.1)) is True


def test_reset_clears_filter_state():
    filt = TemporalStateFilter(window_s=0.3)

    assert filt.accept(make_frame("open_app", 1.0)) is True
    assert filt.accept(make_frame("open_app", 1.1)) is False

    filt.reset()

    assert filt.accept(make_frame("open_app", 1.2)) is True


def test_filter_semantics_are_based_on_intent_not_legacy_token():
    filt = TemporalStateFilter(window_s=0.3)

    first = make_frame("open_app", 1.0)
    second = make_frame("open_app", 1.1)

    # Deliberately give both frames different legacy tokens.
    first = AosInputFrame(
        prototype_token_id="THREE",
        timestamp_mono=first.timestamp_mono,
        fingertip=first.fingertip,
        screen_pos=first.screen_pos,
        desktop_context=first.desktop_context,
        desktop_window_title=first.desktop_window_title,
        intent_signal=first.intent_signal,
    )

    second = AosInputFrame(
        prototype_token_id="FOUR",
        timestamp_mono=second.timestamp_mono,
        fingertip=second.fingertip,
        screen_pos=second.screen_pos,
        desktop_context=second.desktop_context,
        desktop_window_title=second.desktop_window_title,
        intent_signal=second.intent_signal,
    )

    assert filt.accept(first) is True
    assert filt.accept(second) is False
