"""Local KittenTTS speech output for AURA-OS."""

from __future__ import annotations

import selectors
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any


_WORKER_SCRIPT = r'''
import contextlib
import json
import sys
import traceback

import sounddevice as sd
from kittentts import KittenTTS

model_id, voice, device, sample_rate, speed = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    int(sys.argv[4]),
    float(sys.argv[5]),
)

try:
    model = KittenTTS(model_id)
    print(json.dumps({"ready": True}), flush=True)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "stop":
                break

            text = request.get("text", "").strip()
            if not text:
                print(json.dumps({"ok": True}), flush=True)
                continue

            with contextlib.redirect_stdout(sys.stderr):
                audio = model.generate(text, voice=voice, speed=speed)
            sd.play(audio, samplerate=sample_rate, device=device)
            sd.wait()
            print(json.dumps({"ok": True}), flush=True)

        except Exception as exc:
            print(json.dumps({
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }), flush=True)
except Exception as exc:
    print(json.dumps({
        "ready": False,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }), flush=True)
'''


class TTSEngine:
    """Persistent local KittenTTS worker with serialized speech playback."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        tts_config = config.get("tts", {})
        self.enabled = bool(tts_config.get("enabled", True))

        self.python = Path(
            tts_config.get(
                "python",
                str(Path.home() / ".venvs/aos-tts/bin/python"),
            )
        ).expanduser()
        self.model_id = tts_config.get(
            "model_id", "KittenML/kitten-tts-nano-0.8"
        )
        self.voice = tts_config.get("voice", "Bella")
        self.sample_rate = int(tts_config.get("sample_rate", 24000))
        self.output_device = int(tts_config.get("output_device", 5))
        self.speed = float(tts_config.get("speed", 1.5))
        self.timeout = int(tts_config.get("timeout", 120))

        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stdout_buffers: dict[int, bytearray] = {}
        self._available = self.enabled and self.python.is_file()
        self._ready = False

    def _readline_with_timeout(self, stream) -> str:
        """Read one newline-delimited response with a bounded wait."""
        if self.timeout <= 0:
            raise ValueError("TTS timeout must be greater than zero")

        buffers = getattr(self, "_stdout_buffers", None)
        if buffers is None:
            buffers = {}
            self._stdout_buffers = buffers

        fd = stream.fileno()
        buffer = buffers.setdefault(fd, bytearray())

        import time
        import os

        deadline = time.monotonic() + self.timeout

        while True:
            newline = buffer.find(b"\n")
            if newline >= 0:
                line = bytes(buffer[:newline + 1])
                del buffer[:newline + 1]
                return line.decode("utf-8")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"KittenTTS worker did not respond within {self.timeout}s"
                )

            with selectors.DefaultSelector() as selector:
                selector.register(stream, selectors.EVENT_READ)
                if not selector.select(timeout=remaining):
                    raise TimeoutError(
                        f"KittenTTS worker did not respond within {self.timeout}s"
                    )

            chunk = os.read(fd, 4096)
            if not chunk:
                if buffer:
                    line = bytes(buffer)
                    buffer.clear()
                    return line.decode("utf-8")
                return ""

            buffer.extend(chunk)

    def is_available(self) -> bool:
        """Return whether the worker is initialized and ready."""
        with self._lock:
            return (
                self._available
                and self._ready
                and self._process is not None
                and self._process.poll() is None
            )

    def _terminate_worker(self, process: subprocess.Popen[str]) -> None:
        """Terminate a worker with bounded waits and clear its buffer."""
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                import logging
                logging.getLogger(__name__).error(
                    "KittenTTS worker did not exit after kill."
                )
        except OSError as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Could not terminate KittenTTS worker: %s", exc
            )
        finally:
            if process.stdout is not None:
                try:
                    self._stdout_buffers.pop(process.stdout.fileno(), None)
                except (OSError, ValueError):
                    pass

    def _start_worker(self) -> None:
        if not self._available:
            raise RuntimeError(
                f"KittenTTS Python environment not found: {self.python}"
            )

        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "2")

        process = subprocess.Popen(
            [
                str(self.python), "-u", "-c", _WORKER_SCRIPT,
                self.model_id, self.voice, str(self.output_device),
                str(self.sample_rate), str(self.speed),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )
        self._process = process
        self._ready = False

        try:
            assert process.stdout is not None
            line = self._readline_with_timeout(process.stdout)
            if not line:
                raise RuntimeError(
                    "KittenTTS worker exited before becoming ready"
                )

            response = json.loads(line)
            if not response.get("ready"):
                raise RuntimeError(
                    "KittenTTS worker failed to initialize: "
                    f"{response.get('error', 'unknown error')}"
                )
            self._ready = True
        except Exception:
            self._terminate_worker(process)
            if self._process is process:
                self._process = None
            self._ready = False
            raise

    def speak(self, text: str) -> None:
        """Speak without waiting indefinitely for a busy engine lock."""
        if not self.enabled:
            return
        if not isinstance(text, str) or not text.strip():
            return

        # Never let speech confirmation stall the gesture pipeline.
        if not self._lock.acquire(timeout=0.1):
            import logging
            logging.getLogger(__name__).warning(
                "TTS is busy; skipping this confirmation."
            )
            return

        try:
            if not self._ready:
                self._start_worker()

            process = self._process
            if process is None or process.poll() is not None:
                self._ready = False
                raise RuntimeError("KittenTTS worker is not running")

            assert process.stdin is not None
            assert process.stdout is not None

            try:
                process.stdin.write(
                    json.dumps({"command": "speak", "text": text.strip()}) + "\n"
                )
                process.stdin.flush()

                line = self._readline_with_timeout(process.stdout)
                if not line:
                    raise RuntimeError("KittenTTS worker stopped unexpectedly")

                response = json.loads(line)
                if not response.get("ok"):
                    raise RuntimeError(
                        f"KittenTTS speech failed: {response.get('error')}"
                    )
            except (TimeoutError, BrokenPipeError, OSError, ValueError) as exc:
                self._ready = False
                self._terminate_worker(process)
                if self._process is process:
                    self._process = None
                if isinstance(exc, TimeoutError):
                    raise
                raise RuntimeError(
                    f"KittenTTS worker communication failed: {exc}"
                ) from exc
        finally:
            self._lock.release()

    def close(self) -> None:
        """Stop the worker without waiting indefinitely for the engine lock."""
        if not self._lock.acquire(timeout=0.1):
            # Interrupt a stuck request without waiting for the engine lock.
            process = self._process
            if process is not None:
                self._terminate_worker(process)
            if self._process is process:
                self._process = None
                self._ready = False
            return

        try:
            process = self._process
            self._process = None
            self._ready = False

            if process is None or process.poll() is not None:
                return

            try:
                if process.stdin is not None:
                    process.stdin.write('{"command":"stop"}\n')
                    process.stdin.flush()
                process.wait(timeout=2)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self._terminate_worker(process)
        finally:
            self._lock.release()
