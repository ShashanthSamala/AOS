# AURA-OS — Gesture-Driven Agentic Operating System

> **Control your desktop with hand gestures.** AURA-OS uses a webcam, MediaPipe hand tracking, and a local LLM to translate real-time hand gestures into OS-level actions — clicks, scrolling, app launching, and more.

---

## ✨ Features

- **6 core gestures** — POINT (cursor), PINCH (click), FIST, THREE, FOUR, FIVE — each mapped to a desktop action
- **Real-time hand tracking** — MediaPipe Hands + OpenCV for low-latency finger detection
- **LLM reasoning fallback** — unmapped gestures are interpreted by a local TinyLlama model via Ollama
- **MCP tool server** — modular tool registry for mouse, keyboard, and system control
- **Resource-aware kernel** — CPU/RAM/battery monitoring with automatic degraded mode
- **Configurable** — YAML-based settings for gestures, LLM, kernel mode, and resource thresholds
- **Left/right hand support** — handedness-aware thumb detection via MediaPipe

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│                    main.py                       │
│              (Boot & orchestration)              │
├──────────────┬───────────────┬───────────────────┤
│  Perception  │    Kernel     │    Foundation     │
│              │               │                   │
│ gesture_     │ kernel_loop   │ llm_interface     │
│  engine      │ scheduler     │ prompt_template   │
│ gesture_map  │ context_mgr   │ action_validator  │
│ gesture_     │ resource_mgr  │                   │
│  stabilizer  │ app_context   │                   │
│              │ task_retry    │                   │
│              │ metrics       │                   │
├──────────────┴───────────────┴───────────────────┤
│                 Tools (MCP)                      │
│   mouse_tools · keyboard_tools · system_tools    │
└──────────────────────────────────────────────────┘
```

## 🖐️ Gesture Mappings

| Gesture | Fingers | Action | Description |
|---------|---------|--------|-------------|
| **POINT** | ☝️ Index only | `move_cursor` | Cursor follows fingertip position |
| **PINCH** | 👌 Thumb + index touch | `click` | Click at last cursor position |
| **FIST** | ✊ None | `screenshot` | Take a screenshot |
| **THREE** | 🤟 Three fingers | `open_app` | Open configurable app (default: VS Code) |
| **FOUR** | 🖐️ Four fingers | `open_browser` | Open Firefox |
| **FIVE** | 🖐️ All five | `minimize_all` | Show desktop |
| **PEACE** | ✌️ Two fingers | App cycle | Cycle through favorite apps list |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Webcam
- Linux desktop (X11 or Wayland with XDG support)
- [Ollama](https://ollama.ai) (optional — for LLM reasoning mode)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/AOS.git
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

## ⚙️ Configuration

All settings are in [`config/settings.yaml`](config/settings.yaml):

- **`perception.gesture_cooldown_ms`** — Minimum time between gesture actions (default: 500ms)
- **`kernel.mode`** — `"direct"` (gesture map lookup) or `"llm"` (always use LLM reasoning)
- **`gesture_map`** — Customize which gesture triggers which action
- **`three_finger_app`** — App opened by the THREE gesture
- **`favorite_apps`** — List of apps to cycle through with PEACE gesture
- **`resources`** — CPU/RAM/battery thresholds for degraded mode

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
│   ├── kernel_loop.py         # Main event loop
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
└── gesture.py                 # Standalone gesture test script
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Hand tracking | MediaPipe Hands |
| Computer vision | OpenCV |
| Desktop control | PyAutoGUI |
| LLM | TinyLlama via Ollama |
| Config | PyYAML |
| Resource monitoring | psutil |

## 📄 License

This project is part of a personal portfolio. All rights reserved.

---

**Author:** Samala Shashanth  
**Project:** AURA-OS — Agentic Operating System
