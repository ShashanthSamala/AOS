# AOS — Agentic Operating System

## What is AOS?

Everything changes with time — that is the nature of time itself.

Computers evolved from room-sized machines to devices that fit in a pocket,
not by adding more, but by removing what was unnecessary and replacing it
with something better. This has always been true of both hardware and
software.

When I enrolled in my B.Tech (AI & Data Science) program, the standard
curriculum followed a familiar path: programming languages, data structures,
operating systems, databases. At some point I stopped and asked a question
that bothered me:

**Why has the operating system stayed fundamentally the same since it was
invented?**

The purpose of an operating system is to be the interface between hardware
and user. That was true in the 1970s and it is still the definition today.
But for most of that time, the user had to adapt to the machine — learning
its conventions, its file systems, its input devices. That was acceptable in
the early years. Then AI arrived and introduced new possibilities across
every domain.

I decided to do something about it. That decision became AOS.

---

AOS is not AI controlling your entire system. It is about making an existing
operating system smarter in ways that are actually useful to the person using
it — starting with me, as a student who wanted a system that understood my
context and responded to my intent.

The interaction layer was the other half of the idea. Keyboards and mice are
not bad — they are simply not the only option. They are devices we invented
to communicate with machines, not the only way communication can work.
Touchscreens exist but are expensive. A camera, however, is already in most
computers. Gesture control through a camera requires no additional hardware.
Voice is the most natural human output. Eyes and neural interfaces are
further out — but the architecture should not assume they are impossible.

These two ideas combined into one project: an OS layer that understands
intent expressed through hands, voice, and eventually other human signals,
and translates that intent into action — without requiring the user to adapt
to the machine.

**Current implementation:** Camera-based gesture input and local KittenTTS speech output are implemented. Voice input and speech-to-text (STT) are not yet integrated; eyes and neural interfaces remain future directions.

---

## Architecture

The rule that governs every design decision in AOS:

> **The AI proposes. The kernel decides. ActionValidator sits between
> every AI output and every tool call.**

The AI is not the operating system kernel. It reasons and suggests. The
kernel controls what is allowed, what tools exist, execution, timeouts,
retries, resource constraints, and logging. This separation is intentional
and non-negotiable.





### Full Architecture

```
                         ┌──────────────────────────┐
                         │          HUMAN           │
                         │                          │
                         │  Hands   Voice   Eyes    │
                         │         Neural*          │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │       PERCEPTION LAYER          │
                    │                                 │
                    │  MediaPipe Tasks API            │
                    │  Gesture detection              │
                    │  Landmark tracking              │
                    │        │                        │
                    │        ▼                        │
                    │  HumanIntentSignal              │
                    │  ┌─────────────────────────┐   │
                    │  │ modality  (gesture/voice)│   │
                    │  │ intent    (click/open..) │   │
                    │  │ target    (pos/app name) │   │
                    │  │ confidence               │   │
                    │  │ timestamp                │   │
                    │  │ raw       (metadata)     │   │
                    │  └─────────────────────────┘   │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │          AOS KERNEL             │
                    │                                 │
                    │  CentralOrchestrator            │
                    │    TemporalStateFilter          │
                    │    ResourceMonitor              │
                    │    ContextManager               │
                    │    Scheduler                    │
                    │    TaskRetry                    │
                    │    MetricsCollector             │
                    │    Logger                       │
                    │                                 │
                    │         Intent Router           │
                    │              │                  │
                    │    ┌─────────┴──────────┐       │
                    │    ▼                    ▼       │
                    │  Direct path          LLM path  │
                    │  (low latency)        (semantic) │
                    │  move_cursor          qwen:4b  │
                    │  click                via Ollama│
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │        FOUNDATION LAYER         │
                    │                                 │
                    │  ┌───────────────────────────┐  │
                    │  │     ActionValidator        │  │
                    │  │                           │  │
                    │  │  Validates every AI output│  │
                    │  │  before any tool executes │  │
                    │  │                           │  │
                    │  │  Cannot be bypassed       │  │
                    │  │  Cannot be removed        │  │
                    │  │  Cannot be routed around  │  │
                    │  └───────────────────────────┘  │
                    │                                 │
                    │  LLMInterface  PromptTemplate   │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │       CAPABILITY LAYER          │
                    │                                 │
                    │  Tool Registry (MCP server)     │
                    │                                 │
                    │  mouse_tools                    │
                    │    click · move · scroll        │
                    │                                 │
                    │  keyboard_tools                 │
                    │    type · hotkey · press        │
                    │                                 │
                    │  system_tools                   │
                    │    open_app · screenshot        │
                    │    volume · minimize · lock     │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │      PLATFORM ADAPTER           │
                    │                                 │
                    │  X11 (current — legacy)         │
                    │  KDE Wayland ← Phase 4 target   │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │            LINUX                │
                    │  Kernel · Desktop · Devices     │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │         OUTPUT LAYER            │
                    │                                 │
                    │  KittenTTS                      │
                    │  Local CPU-only TTS             │
                    │  Daemon thread — non-blocking   │
                    │  Always on the machine          │
                    │  with the speakers              │
                    └─────────────────────────────────┘

* future research direction
```

---

### Data Flow — One Complete Cycle

```
Camera frame
     │
     ▼
MediaPipe Tasks API
     │
     ▼
HumanIntentSignal(modality="gesture", intent="click", ...)
     │
     ▼
CentralOrchestrator._ingest_loop()
     │
     ▼
TemporalStateFilter   ← drops duplicates
     │
     ▼
Intent Router
     │
     ├── Direct intent?  → execute immediately (no LLM)
     │
     └── Semantic intent? → LLMInterface.ask()
                                │
                                ▼
                           qwen:4b (Ollama)
                                │
                                ▼
                         ActionValidator    ← SAFETY GATE
                                │
                                ▼
                         Tool execution
                                │
                                ▼
                         ContextManager.add_intent()
                                │
                                ▼
                         KittenTTS.speak()  ← daemon thread
```

---

## Hardware

**Development machine:** ThinkPad, Intel i5-4210M, Intel HD 4600 (no CUDA),
7.6 GB RAM, Arch Linux, KDE Plasma, Wayland

**Target deployment:** Raspberry Pi 5 (Phase 7)

---

## Setup

> Arch Linux only. These instructions do not apply to Ubuntu or Windows.

```bash
# Clone
git clone -b prototype https://github.com/ShashanthSamala/AOS.git
cd AOS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull qwen:4b via Ollama
ollama pull qwen:4b
```

KittenTTS requires a separate Python 3.11 environment — see
`perception/tts_engine.py` for setup details.

---

## Running AOS

```bash
source .venv/bin/activate
python main.py
```

AOS will initialize the gesture engine, orchestrator, and LLM interface.
Point a hand at the camera. POINT moves the cursor. PINCH clicks.
Other gestures trigger semantic resolution through the kernel.

---

```
