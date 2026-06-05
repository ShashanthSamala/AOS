# pyre-ignore-all-errors
# AURA-OS — Main Entry Point
# main.py — Boots and connects all layers
# Author: Samala Shashanth | Project: AURA-OS

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"
import sys
import signal
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
import yaml  # type: ignore
from typing import Dict, Any

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from foundation.llm_interface import LLMInterface
from perception.gesture_engine import GestureEngine
from perception.gesture_map import GestureMap

from kernel.kernel_loop import KernelLoop
from kernel.orchestrator import CentralOrchestrator
from kernel.logger import get_logger
from tools.mcp_server import MCPServer
from tools import mouse_tools, keyboard_tools, system_tools

BANNER = r"""
    ╔═══════════════════════════════════════════════════╗
    ║                                                   ║
    ║     █████╗ ██╗   ██╗██████╗  █████╗               ║
    ║    ██╔══██╗██║   ██║██╔══██╗██╔══██╗              ║
    ║    ███████║██║   ██║██████╔╝███████║              ║
    ║    ██╔══██║██║   ██║██╔══██╗██╔══██║              ║
    ║    ██║  ██║╚██████╔╝██║  ██║██║  ██║              ║
    ║    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝              ║
    ║                                                   ║
    ║    Agentic Operating System — Phase 1             ║
    ║    Gesture-Driven · LLM-Powered · Local-First     ║
    ║                                                   ║
    ╚═══════════════════════════════════════════════════╝
"""


def load_config() -> Dict[str, Any]:
    """Load configuration from settings.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "config", "settings.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def build_mcp_server():
    """Create and configure the MCP tool server with all available tools."""
    mcp = MCPServer()

    # Mouse tools
    mcp.register_tool("click", mouse_tools.click, "Left mouse click")
    mcp.register_tool("right_click", mouse_tools.right_click, "Right mouse click")
    mcp.register_tool("double_click", mouse_tools.double_click, "Double left click")
    mcp.register_tool("scroll_up", mouse_tools.scroll_up, "Scroll up")
    mcp.register_tool("scroll_down", mouse_tools.scroll_down, "Scroll down")
    mcp.register_tool("move_cursor", mouse_tools.move_cursor, "Move cursor")

    # Keyboard tools
    mcp.register_tool("press_key", keyboard_tools.press_key, "Press a key")
    mcp.register_tool("hotkey", keyboard_tools.hotkey, "Keyboard shortcut")
    mcp.register_tool("type_text", keyboard_tools.type_text, "Type text")

    # System tools
    mcp.register_tool("open_terminal", system_tools.open_terminal, "Open terminal")
    mcp.register_tool("open_browser", system_tools.open_browser, "Open browser to URL")
    mcp.register_tool("open_app", system_tools.open_app, "Open application by name (e.g. 'code', 'firefox')")
    mcp.register_tool("show_desktop", system_tools.show_desktop, "Show desktop / minimize all")
    mcp.register_tool("minimize_all", system_tools.minimize_all, "Minimize all windows")
    mcp.register_tool("screenshot", system_tools.screenshot, "Take a screenshot")
    mcp.register_tool("get_system_info", system_tools.get_system_info, "Get system info")
    mcp.register_tool("get_battery", system_tools.get_battery, "Battery status")
    mcp.register_tool("volume_up", system_tools.volume_up, "Volume up")
    mcp.register_tool("volume_down", system_tools.volume_down, "Volume down")
    mcp.register_tool("play_pause", system_tools.play_pause, "Play/pause media")
    mcp.register_tool("lock_screen", system_tools.lock_screen, "Lock screen")
    mcp.register_tool("do_nothing", system_tools.do_nothing, "No action")

    return mcp


def main():
    """AURA-OS boot sequence."""
    print(BANNER)

    # Step 1: Load config
    config = load_config()
    log = get_logger("main", config)
    log.info("Loading configuration...")

    # Step 2: Check LLM availability
    llm = LLMInterface()
    if llm.is_available():
        log.info(f"LLM ready: {llm.model}")
    else:
        log.warning(f"LLM model '{llm.model}' not available — direct mode only")
        config.setdefault("kernel", {})["mode"] = "direct"

    # Step 3: Build MCP server
    mcp = build_mcp_server()
    log.info(f"MCP server initialized with {len(mcp.list_tools())} tools")
    for tool in mcp.list_tools():
        log.info(f"  Tool: {tool['name']:20s} — {tool['description']}")

    # Step 4: Select and initialize kernel engine
    engine_type = config.get("kernel", {}).get("engine", "orchestrator")
    log.info(f"Engine selection: {engine_type}")

    if engine_type == "orchestrator":
        engine = CentralOrchestrator(config=config, mcp_server=mcp)
        log.info(f"Central Orchestrator initialized (mode: {engine.mode})")
    else:
        engine = KernelLoop(config=config, mcp_server=mcp)
        log.info(f"Legacy KernelLoop initialized (mode: {engine.mode})")

    # Step 5: Initialize gesture engine
    gesture_engine = GestureEngine(config=config)
    log.info("Gesture engine initialized")

    # Step 6: Handle graceful shutdown
    def on_shutdown(sig, frame):
        log.info("Shutdown signal received...")
        gesture_engine.stop()
        engine.stop()

    signal.signal(signal.SIGINT, on_shutdown)
    signal.signal(signal.SIGTERM, on_shutdown)

    # Step 7: Start!
    log.info("=" * 50)
    log.info("AURA-OS is booting up...")
    log.info(f"Engine: {engine_type.upper()}")
    log.info("Show hand gestures to the camera to control your system.")
    log.info("Press Q on the video window or Ctrl+C to shutdown.")
    log.info("=" * 50)

    gesture_engine.start(show_video=True, threaded=True)
    engine.run(gesture_engine)

    # Cleanup
    gesture_engine.stop()
    log.info("AURA-OS shutdown complete.")


if __name__ == "__main__":
    main()
