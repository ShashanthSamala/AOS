# AURA-OS — Gesture-Driven Agentic Operating System

<p align="center">
  <strong>Control your entire desktop with hand gestures — no mouse, no keyboard, no touch.</strong>
</p>

<p align="center">
  <em>AURA-OS is a minimal, platform-independent agentic operating system layer that sits on top of your Linux desktop. It uses real-time hand tracking via a webcam, interprets gestures through a modular perception pipeline, and translates them into OS-level actions — cursor movement, clicks, app launching, media control, and more. When a gesture doesn't have a direct mapping, a local LLM reasons about the user's intent and decides what to do.</em>
</p>

---

## 🧠 What is AURA-OS?

AURA-OS is not a traditional operating system — it's an **agentic OS layer**. It augments your existing Linux desktop with a gesture-driven, AI-powered interface. Instead of replacing your OS, it sits on top and provides a new way to interact with it.

The core idea is simple: **your hand becomes the controller**. Point to move the cursor. Pinch to click. Show three fingers to open an app. Make a fist to take a screenshot. Every gesture is captured by your webcam, classified in real time, and executed as an OS action — all locally, with zero cloud dependency.

What makes AURA-OS *agentic* is the LLM fallback. When the system encounters an unrecognized gesture or ambiguous context, it doesn't fail — it sends the gesture data to a local LLM (TinyLlama via Ollama) which reasons about what you probably intended and picks the right tool to call. The system learns from context: it knows what app you're currently using, your recent gesture history, and the system resource state.

---

## ✅ What We've Built So Far (Phase 1 — Prototype)

Phase 1 is the fully working foundation. Everything listed below is implemented, integrated, and functional.

### 🖐️ Perception Layer — Real-Time Gesture Recognition

- **MediaPipe Hands + OpenCV** pipeline for low-latency hand tracking at camera frame rate
- **7 core gestures** detected using angle-based finger classification (not ML-based gesture classification — pure geometric logic for reliability):

| Gesture | Fingers | Action | Description |
|---------|---------|--------|-------------|
| **POINT** | ☝️ Index only | `move_cursor` | Cursor follows fingertip position in real time |
| **PINCH** | 👌 Thumb + index touch | `click` | Left click at current cursor position |
| **FIST** | ✊ None | `screenshot` | Take a screenshot and save to disk |
| **THREE** | 🤟 Three fingers | `open_app` | Open configurable app (default: VS Code) |
| **FOUR** | 🖐️ Four fingers | `open_browser` | Launch Firefox |
| **FIVE** | 🖐️ All five | `minimize_all` | Show desktop / minimize all windows |
| **PEACE** | ✌️ Two fingers | `app_cycle` | Cycle through a list of favorite apps |

- **Gesture stabilizer** with cooldown debouncing (configurable, default 500ms) to prevent accidental repeated triggers
- **Cursor smoothing** via Exponential Moving Average (EMA) — configurable alpha for responsiveness vs. stability tradeoff
- **Scale-invariant pinch detection** — pinch distance measured as a ratio of palm span, so it works at any distance from the camera
- **False-positive rejection** — minimum landmark visibility, minimum palm span, and angle thresholds to filter noise and phantom detections
- **Left/right hand support** — handedness-aware thumb detection via MediaPipe
- **Edge margin mapping** — maps the central zone of the camera frame to the full screen, so you can reach screen corners without stretching your arm

### 🔧 MCP Tool Server — Modular Action Execution

A lightweight, local Model-Context-Protocol-inspired tool server with **19 registered tools**:

| Category | Tools |
|----------|-------|
| **Mouse** | `click`, `right_click`, `double_click`, `scroll_up`, `scroll_down`, `move_cursor` |
| **Keyboard** | `press_key`, `hotkey`, `type_text` |
| **System** | `open_terminal`, `open_browser`, `open_app`, `show_desktop`, `minimize_all`, `screenshot`, `get_system_info`, `get_battery`, `volume_up`, `volume_down`, `play_pause`, `lock_screen`, `do_nothing` |

Every gesture action maps to a tool call. The MCP server validates parameters and executes tools safely.

### ⚙️ Kernel — The Brain

- **Dual engine support**:
  - **CentralOrchestrator** (default) — async pipeline with `asyncio` for high-throughput gesture processing
  - **KernelLoop** (legacy) — synchronous fallback for simpler environments
- **Dual mode**:
  - `direct` mode — pure gesture-map lookup, zero LLM latency
  - `llm` mode — every gesture is interpreted by the LLM for maximum flexibility
- **FIFO task scheduler** with configurable queue depth
- **Context manager** — maintains a rolling history of recent gestures for LLM context
- **App context awareness** — detects the currently focused window so the LLM can make context-aware decisions
- **Resource manager** — monitors CPU, RAM, and battery in real time; auto-switches to degraded mode under pressure
- **Task retry with exponential backoff** — failed tool calls are retried gracefully
- **Performance metrics** — tracks gesture-to-action latency, throughput, and error rates
- **Structured logging** — file + console logging with configurable verbosity

### 🤖 Foundation Layer — LLM Integration

- **Ollama API client** — connects to a locally running TinyLlama model
- **Prompt template system** — structured system prompts that describe available tools, current context, and constraints
- **Action validator** — validates LLM-suggested actions against the tool registry before execution
- **Graceful fallback** — if the LLM is unavailable, the system seamlessly operates in direct mode

### 🧪 Test Suite

- **4 layered test modules** covering MCP tools, kernel logic, foundation/LLM, and perception
- **Custom test runner** with formatted output and per-layer reporting
- Tests run without requiring a webcam or LLM (mocked dependencies)

### 📁 Configuration

All settings are centralized in `config/settings.yaml` — gesture mappings, LLM parameters, perception thresholds, kernel mode, resource limits, and logging. Every tunable parameter is documented inline.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│                    main.py                       │
│              (Boot & orchestration)              │
├──────────────┬───────────────┬───────────────────┤
│  Perception  │    Kernel     │    Foundation     │
│              │               │                   │
│ gesture_     │ orchestrator  │ llm_interface     │
│  engine      │ kernel_loop   │ prompt_template   │
│ gesture_map  │ scheduler     │ action_validator  │
│ gesture_     │ context_mgr   │                   │
│  stabilizer  │ resource_mgr  │                   │
│              │ app_context   │                   │
│              │ task_retry    │                   │
│              │ metrics       │                   │
├──────────────┴───────────────┴───────────────────┤
│                 Tools (MCP)                      │
│   mouse_tools · keyboard_tools · system_tools    │
└──────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Webcam
- Linux desktop (X11 or Wayland with XDG support)
- [Ollama](https://ollama.ai) (optional — for LLM reasoning mode)

### Installation

```bash
# Clone the repository
git clone https://github.com/ShashanthSamala/AOS.git
cd AOS

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install opencv-python mediapipe pyautogui pyyaml requests psutil

# (Optional) Pull TinyLlama for LLM reasoning mode
ollama pull tinyllama
```

### Run

```bash
source venv/bin/activate
python main.py
```

A small camera preview window will appear. Show hand gestures to control your desktop. Press **Q** on the video window or **Ctrl+C** to stop.

### Run Tests

```bash
python -m tests.test_runner
```

---

## 📁 Project Structure

```
AOS/
├── main.py                    # Entry point — boots all layers
├── config/
│   └── settings.yaml          # All configuration
├── perception/                # Camera + gesture detection
│   ├── gesture_engine.py      # MediaPipe hand tracking engine
│   ├── gesture_map.py         # Gesture → action mappings
│   └── gesture_stabilizer.py  # Debouncing / stabilization
├── kernel/                    # Core OS logic
│   ├── orchestrator.py        # Async central pipeline (default engine)
│   ├── kernel_loop.py         # Synchronous fallback engine
│   ├── scheduler.py           # FIFO task queue
│   ├── context_manager.py     # Gesture history context
│   ├── resource_manager.py    # CPU/RAM/battery monitoring
│   ├── app_context.py         # Active window detection
│   ├── task_retry.py          # Retry with backoff
│   ├── metrics.py             # Performance metrics
│   └── logger.py              # Logging setup
├── foundation/                # LLM + validation
│   ├── llm_interface.py       # Ollama API client
│   ├── prompt_template.py     # System prompts
│   └── action_validator.py    # Action validation
├── tools/                     # MCP tool server
│   ├── mcp_server.py          # Tool registry + executor
│   ├── mouse_tools.py         # Click, scroll, move
│   ├── keyboard_tools.py      # Keypress, hotkey, typing
│   └── system_tools.py        # Apps, volume, screenshot
├── tests/                     # Layered test suite
│   ├── test_runner.py         # Formatted test runner
│   ├── test_l2_mcp.py         # MCP tool tests
│   ├── test_l3_kernel.py      # Kernel logic tests
│   ├── test_l4_foundation.py  # Foundation/LLM tests
│   └── test_l5_perception.py  # Perception pipeline tests
└── gesture.py                 # Standalone gesture test script
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Hand tracking | MediaPipe Hands |
| Computer vision | OpenCV |
| Desktop control | PyAutoGUI |
| LLM | TinyLlama via Ollama |
| Async runtime | asyncio |
| Config | PyYAML |
| Resource monitoring | psutil |
| Testing | unittest (stdlib) |

---

## 🗺️ Roadmap — What's Coming Next

### Phase 2 — Voice + Multimodal Input
- [ ] **Voice commands** — Wake-word activated speech-to-text for hybrid gesture + voice control
- [ ] **Dynamic gesture learning** — Users can record and assign custom gestures to any tool
- [ ] **Two-hand gestures** — Track both hands simultaneously for complex multi-gesture combos (e.g., left hand selects, right hand acts)
- [ ] **Gesture chaining** — Sequential gesture combos (e.g., FIST → POINT = drag-and-drop)

### Phase 3 — Smarter Agent
- [ ] **Upgraded LLM** — Move from TinyLlama to a more capable local model (Phi-3, Mistral, etc.)
- [ ] **Proactive actions** — The agent suggests or auto-performs actions based on patterns (e.g., "you always open VS Code after Firefox — should I do that?")
- [ ] **Contextual tool selection** — Different gesture mappings per-app (e.g., PINCH = "play/pause" in a media player, "click" elsewhere)
- [ ] **Natural language tool creation** — Describe a new tool in plain English and the LLM generates it

### Phase 4 — Cross-Platform & Ecosystem
- [ ] **Windows & macOS support** — Replace Linux-specific subprocess calls with cross-platform abstractions
- [ ] **Plugin system** — Third-party tool plugins (Spotify control, smart home, notifications, etc.)
- [ ] **Web dashboard** — Real-time monitoring UI showing gesture feed, metrics, system state, and tool history
- [ ] **Mobile companion app** — Use phone camera as an alternative to webcam

### Phase 5 — Full Agentic OS
- [ ] **Multi-agent collaboration** — Specialized agents for different domains (file management, web browsing, coding)
- [ ] **Persistent memory** — The OS remembers your preferences, habits, and frequently used workflows
- [ ] **Task automation** — "When I show FIST, take a screenshot AND open it in the editor"
- [ ] **Accessibility mode** — Optimized for users with limited mobility

---

## 📄 License

This project is part of a personal portfolio. All rights reserved.

---

**Author:** Samala Shashanth  
**Project:** AURA-OS — Agentic Operating System  
**Branch:** `prototype`
