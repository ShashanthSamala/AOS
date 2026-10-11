import threading
import time

from kernel.orchestrator import CentralOrchestrator
from perception.intent_signal import HumanIntentSignal


class FakeGestureEngine:
    is_running = True

    def __init__(self, signal):
        self.signal = signal
        self._returned = False

    def get_gesture(self, timeout=None):
        if not self._returned:
            self._returned = True
            return self.signal

        time.sleep(0.05)
        return None


def test_ingest_accepts_voice_signal_without_gesture():
    signal = HumanIntentSignal(
        modality="voice",
        intent="open_app",
        target="firefox",
        confidence=0.94,
        timestamp=time.time(),
        raw={},
    )

    orchestrator = CentralOrchestrator.__new__(CentralOrchestrator)

    class FakeLog:
        def info(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    class FakeAppContext:
        def get_context(self):
            return "TEST"

        def get_active_window(self):
            return ""

    class FakeLoop:
        def call_soon_threadsafe(self, callback, frame):
            orchestrator.captured_frame = frame
            orchestrator._running = False

    orchestrator.log = FakeLog()
    orchestrator._app_context = FakeAppContext()
    orchestrator._loop = FakeLoop()
    orchestrator._running = True

    engine = FakeGestureEngine(signal)

    thread = threading.Thread(
        target=orchestrator._ingest_loop,
        args=(engine,),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=1)

    assert hasattr(orchestrator, "captured_frame")
    assert orchestrator.captured_frame.prototype_token_id == "UNKNOWN"
    assert orchestrator.captured_frame.intent_signal is signal
import threading
import time

from kernel.orchestrator import CentralOrchestrator
from perception.intent_signal import HumanIntentSignal


class FakeGestureEngine:
    is_running = True

    def __init__(self, signal):
        self.signal = signal
        self._returned = False

    def get_gesture(self, timeout=None):
        if not self._returned:
            self._returned = True
            return self.signal

        time.sleep(0.05)
        return None


def test_ingest_accepts_voice_signal_without_gesture():
    signal = HumanIntentSignal(
        modality="voice",
        intent="open_app",
        target="firefox",
        confidence=0.94,
        timestamp=time.time(),
        raw={},
    )

    orchestrator = CentralOrchestrator.__new__(CentralOrchestrator)

    class FakeLog:
        def info(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    class FakeAppContext:
        def get_context(self):
            return "TEST"

        def get_active_window(self):
            return ""

    class FakeLoop:
        def call_soon_threadsafe(self, callback, frame):
            orchestrator.captured_frame = frame
            orchestrator._running = False

    orchestrator.log = FakeLog()
    orchestrator._app_context = FakeAppContext()
    orchestrator._loop = FakeLoop()
    orchestrator._running = True

    engine = FakeGestureEngine(signal)

    thread = threading.Thread(
        target=orchestrator._ingest_loop,
        args=(engine,),
        daemon=True,
    )
    thread.start()
    thread.join(timeout=1)

    assert hasattr(orchestrator, "captured_frame")
    assert orchestrator.captured_frame.prototype_token_id == "UNKNOWN"
    assert orchestrator.captured_frame.intent_signal is signal
