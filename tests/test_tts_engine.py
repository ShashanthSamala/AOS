import os
import threading
import time
from pathlib import Path

import pytest

from perception.tts_engine import TTSEngine


def make_engine(timeout=1):
    engine = TTSEngine({
        "tts": {
            "enabled": True,
            "python": "/nonexistent/aos-tts-python",
            "timeout": timeout,
        }
    })
    # These tests exercise locking and protocol handling, not real audio.
    engine._available = True
    return engine


def test_speak_skips_when_lock_is_busy(caplog):
    engine = make_engine()
    engine._lock.acquire()

    try:
        started = time.monotonic()
        engine.speak("This should be skipped.")
        elapsed = time.monotonic() - started

        assert elapsed < 0.5
        assert "TTS is busy; skipping this confirmation." in caplog.text
        assert engine._process is None
    finally:
        engine._lock.release()
        engine.close()


def test_reader_times_out_when_worker_is_silent():
    engine = make_engine(timeout=1)
    read_fd, write_fd = os.pipe()

    try:
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            started = time.monotonic()

            with pytest.raises(TimeoutError, match="did not respond"):
                engine._readline_with_timeout(stream)

            elapsed = time.monotonic() - started
            assert 0.8 <= elapsed < 2.0
    finally:
        os.close(write_fd)
        engine.close()


class FakeStream:
    def __init__(self, fd):
        self._fd = fd

    def fileno(self):
        return self._fd


class FakeWorker:
    def __init__(self):
        self.stdout = FakeStream(987654)
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return None if not self.terminated and not self.killed else -15

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        if not self.terminated and not self.killed:
            raise TimeoutError("simulated worker wait timeout")
        return -15


def test_worker_termination_clears_stdout_buffer():
    engine = make_engine()
    worker = FakeWorker()
    engine._stdout_buffers[987654] = bytearray(b"partial response")

    engine._terminate_worker(worker)

    assert worker.terminated
    assert 987654 not in engine._stdout_buffers
    engine.close()


def test_close_resets_state_when_lock_is_busy():
    engine = make_engine()
    worker = FakeWorker()
    engine._process = worker
    engine._ready = True
    engine._stdout_buffers[987654] = bytearray(b"partial response")
    engine._lock.acquire()

    try:
        engine.close()

        assert worker.terminated
        assert engine._process is None
        assert engine._ready is False
        assert 987654 not in engine._stdout_buffers
    finally:
        engine._lock.release()
        engine.close()
