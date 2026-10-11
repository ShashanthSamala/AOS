from unittest.mock import Mock

import pytest

from kernel.orchestrator import CentralOrchestrator


class FakeTTS:
    def __init__(self, enabled=True, error=None):
        self.enabled = enabled
        self.error = error
        self.messages = []
        self.closed = False

    def speak(self, text):
        if self.error:
            raise self.error
        self.messages.append(text)

    def close(self):
        self.closed = True


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


def make_orchestrator(tts=None):
    orchestrator = CentralOrchestrator.__new__(CentralOrchestrator)
    orchestrator.tts = tts or FakeTTS()
    orchestrator.log = FakeLogger()
    return orchestrator


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("open_app", "Opening application."),
        ("open_browser", "Opening browser."),
        ("minimize_all", "Minimizing windows."),
        ("screenshot", "Screenshot taken."),
    ],
)
def test_successful_allowlisted_action_speaks_confirmation(
    action, expected, monkeypatch
):
    orchestrator = make_orchestrator()
    queued = []

    class FakeThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            queued.append(self.kwargs)

    monkeypatch.setattr("kernel.orchestrator.threading.Thread", FakeThread)

    orchestrator._queue_speech_confirmation(action, {"status": "ok"})

    assert len(queued) == 1
    assert queued[0]["args"] == (expected,)
    assert queued[0]["daemon"] is True
    assert queued[0]["name"] == "aos-tts"


@pytest.mark.parametrize("status", ["error", "timeout", "retry", "no_mcp"])
def test_unsuccessful_action_does_not_queue_speech(status, monkeypatch):
    orchestrator = make_orchestrator()
    monkeypatch.setattr(
        "kernel.orchestrator.threading.Thread",
        Mock(side_effect=AssertionError("Speech must not be queued")),
    )

    orchestrator._queue_speech_confirmation("open_browser", {"status": status})


def test_unknown_action_does_not_queue_speech(monkeypatch):
    orchestrator = make_orchestrator()
    monkeypatch.setattr(
        "kernel.orchestrator.threading.Thread",
        Mock(side_effect=AssertionError("Speech must not be queued")),
    )

    orchestrator._queue_speech_confirmation("type_text", {"status": "ok"})


def test_disabled_tts_does_not_queue_speech(monkeypatch):
    orchestrator = make_orchestrator(FakeTTS(enabled=False))
    monkeypatch.setattr(
        "kernel.orchestrator.threading.Thread",
        Mock(side_effect=AssertionError("Disabled TTS must not be queued")),
    )

    orchestrator._queue_speech_confirmation("open_browser", {"status": "ok"})


def test_speech_exception_is_logged_without_propagating():
    orchestrator = make_orchestrator(
        FakeTTS(error=RuntimeError("synthetic test failure"))
    )

    orchestrator._speak_safely("Opening browser.")

    assert len(orchestrator.log.warnings) == 1
    assert "[TTS] Speech output failed:" in orchestrator.log.warnings[0]


def test_disabled_tts_engine_reports_unavailable():
    from perception.tts_engine import TTSEngine

    engine = TTSEngine({"tts": {"enabled": False}})
    try:
        assert engine.enabled is False
        assert engine.is_available() is False
        engine.speak("This must not play.")
    finally:
        engine.close()
