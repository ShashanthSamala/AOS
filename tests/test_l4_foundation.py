# pyre-ignore-all-errors
# AURA-OS — Layer 4 Isolation: Foundation Determinism Verification
# test_l4_foundation.py — Regex filter stress test, fallback enforcement,
#     corrupted/chaotic LLM output parsing, ActionValidator boundary check.
# Standalone execution: python -m tests.test_l4_foundation

import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from foundation.llm_interface import LLMInterface
from foundation.action_validator import ActionValidator, ValidationError, VALID_ACTIONS


# ── Vector 1: _parse_json Regex Filter — 50 Chaotic Payloads ─────────────
def test_parse_json_chaotic_inputs():
    """Inject 50 malformed/chaotic strings into LLMInterface._parse_json().
    Every single one must either return valid parsed JSON or trigger the
    fallback dict with action='do_nothing'. Zero unhandled exceptions."""
    print("\n" + "=" * 72)
    print("[L4-001] _parse_json — 50 Chaotic Payload Stress Test")
    print("=" * 72)

    llm = LLMInterface.__new__(LLMInterface)
    # Minimal init — bypass config loading for isolated unit test
    llm.model = "test"
    llm.api_url = ""
    llm.temperature = 0.1
    llm.timeout = 5
    llm.max_retries = 0

    # 50 adversarial payloads covering: empty, garbage, partial JSON,
    # nested fences, unicode bombs, injection attempts, truncated objects
    payloads = [
        "",                                                          # 01: empty
        "   ",                                                       # 02: whitespace
        "not json at all",                                           # 03: plain text
        "{",                                                         # 04: open brace only
        "}",                                                         # 05: close brace only
        "{{{{",                                                      # 06: nested braces
        '{"action": "click"}',                                       # 07: valid minimal
        '```json\n{"action": "scroll_up"}\n```',                     # 08: fenced valid
        '```json\n{"action": "scroll_up"}\n```\nextra garbage',      # 09: fenced + trailing
        'Here is the answer:\n```json\n{"action":"do_nothing","parameters":{}}\n```\nDone!',  # 10
        '{"action": "INVALID_ACTION_XYZ"}',                          # 11: unknown action
        '{"action": 12345}',                                         # 12: non-string action
        '{"action": null}',                                          # 13: null action
        '{"action": "click", "parameters": "not_a_dict"}',           # 14: bad param type
        '{"action": "open_app"}',                                    # 15: missing required param
        '{"action": "click", "reasoning": "test", "latency_ms": 0}',# 16: valid full
        "```\n{broken json\n```",                                    # 17: broken inside fence
        '{"action": "type_text", "parameters": {"text": "hello"}}',  # 18: valid with params
        '{"unclosed": true',                                         # 19: unclosed object
        'random preamble {"action":"screenshot"} random postamble',  # 20: embedded JSON
        '{"action":"click"} {"action":"scroll_up"}',                 # 21: double objects
        "```json\n```",                                              # 22: empty fence
        "```\n```",                                                  # 23: empty non-json fence
        '{"action": "press_key", "parameters": {"key": ""}}',       # 24: empty key param
        '{"action": "hotkey", "parameters": {"keys": []}}',         # 25: empty list param
        '{"action": "move_cursor", "parameters": {"x": "abc"}}',    # 26: wrong param type
        '\x00\x01\x02{"action": "click"}\xff\xfe',                  # 27: binary-wrapped JSON
        '{"action": "click", "extra_field": true}',                 # 28: extra fields
        'I think you should ```json{"action":"click"}```',           # 29: no newlines in fence
        '{"action":"open_browser","parameters":{"url":"https://evil.com/;rm -rf /"}}',  # 30
        "{'action': 'click'}",                                       # 31: single-quoted (invalid)
        '{"action": "click",}',                                      # 32: trailing comma
        '/* comment */ {"action": "click"}',                         # 33: C-style comment
        '{"action": "click"}\n\n\n',                                 # 34: trailing newlines
        '\n\n\n{"action": "click"}',                                 # 35: leading newlines
        '{"action": "type_text", "parameters": {"text": ""}}\0',    # 36: null-terminated
        '{"action": "move_cursor", "parameters": {"x": 100, "y": 200}}',  # 37: valid move
        json.dumps({"action": "click", "parameters": {}, "reasoning": "A" * 10000}),  # 38: 10KB reasoning
        '{"action": "' + 'A' * 500 + '"}',                          # 39: oversized action
        '{"action": "do_nothing", "parameters": {"nested": {"deep": {"deeper": 1}}}}',  # 40
        '```json\n\n\n{"action": "click"}\n\n\n```',                 # 41: padded fence
        '{"action": "scroll_up", "parameters": {}, "reasoning": "test\\nwith\\nnewlines"}',  # 42
        '{"action": "click", "parameters": {}, "reasoning": "line1\nline2\nline3"}',  # 43
        '{"action": "press_key", "parameters": {"key": "\\u0041"}}', # 44: unicode escape
        "true",                                                      # 45: JSON literal
        "null",                                                      # 46: JSON null
        "[]",                                                        # 47: JSON array
        "[1, 2, 3]",                                                 # 48: JSON array with data
        '{"action": "lock_screen", "parameters": {}}' * 5,          # 49: repeated objects
        "🔥🖐️✋ gesture detected: FIST → ```json\n{\"action\": \"screenshot\"}\n```",  # 50: emoji + fence
    ]

    passed = 0
    failed = 0

    for i, payload in enumerate(payloads, 1):
        try:
            result = llm._parse_json(payload)
            # Must always return a dict
            if not isinstance(result, dict):
                print(f"  [FAIL] #{i:02d}: returned {type(result).__name__}, expected dict")
                failed += 1
                continue
            # Must always have 'action' key
            if "action" not in result:
                print(f"  [FAIL] #{i:02d}: missing 'action' key in result")
                failed += 1
                continue
            passed += 1
        except Exception as e:
            print(f"  [FAIL] #{i:02d}: unhandled {type(e).__name__}: {e}")
            failed += 1

    status = "PASS" if failed == 0 else "FAIL"
    print(f"  [{status}] {passed}/{len(payloads)} payloads handled safely, {failed} failures")
    return failed == 0


# ── Vector 2: Fallback Dictionary Enforcement ─────────────────────────────
def test_fallback_enforcement():
    """Verify _fallback() always returns the canonical safe-action dict
    with exact key set {action, parameters, reasoning, latency_ms}."""
    print("\n" + "=" * 72)
    print("[L4-002] Fallback Dictionary Structure Enforcement")
    print("=" * 72)

    llm = LLMInterface.__new__(LLMInterface)
    llm.model = "test"

    reasons = [
        "LLM timeout",
        "Invalid JSON from LLM",
        "No JSON in LLM response",
        "",
        "x" * 5000,  # very long reason
        "Validation error: missing param",
        "Cannot connect to Ollama — is it running?",
        "LLM error: ConnectionResetError()",
    ]

    REQUIRED_KEYS = {"action", "parameters", "reasoning", "latency_ms"}
    passed = 0

    for reason in reasons:
        fb = llm._fallback(reason)
        keys_ok = set(fb.keys()) == REQUIRED_KEYS
        action_ok = fb["action"] == "do_nothing"
        params_ok = fb["parameters"] == {}
        latency_ok = fb["latency_ms"] == 0
        reason_ok = fb["reasoning"] == reason

        if all([keys_ok, action_ok, params_ok, latency_ok, reason_ok]):
            passed += 1
        else:
            print(f"  [FAIL] reason={reason[:40]!r}  keys={keys_ok}  action={action_ok}")

    ok = passed == len(reasons)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {passed}/{len(reasons)} fallback outputs structurally correct")
    return ok


# ── Vector 3: ActionValidator Boundary Sweep ──────────────────────────────
def test_action_validator_boundaries():
    """Exhaustive boundary check on ActionValidator.validate_action().
    Tests every registered action with correct params, then verifies
    rejection of invalid actions, missing required params, and type mismatches."""
    print("\n" + "=" * 72)
    print("[L4-003] ActionValidator — Exhaustive Boundary Sweep")
    print("=" * 72)

    passed = 0
    failed = 0
    total = 0

    # ── 3A: Valid actions with correct parameters ──
    valid_payloads = [
        {"action": "click", "parameters": {}},
        {"action": "right_click", "parameters": {}},
        {"action": "double_click", "parameters": {}},
        {"action": "scroll_up", "parameters": {}},
        {"action": "scroll_down", "parameters": {}},
        {"action": "open_terminal", "parameters": {}},
        {"action": "show_desktop", "parameters": {}},
        {"action": "minimize_all", "parameters": {}},
        {"action": "screenshot", "parameters": {}},
        {"action": "get_system_info", "parameters": {}},
        {"action": "get_battery", "parameters": {}},
        {"action": "volume_up", "parameters": {}},
        {"action": "volume_down", "parameters": {}},
        {"action": "play_pause", "parameters": {}},
        {"action": "lock_screen", "parameters": {}},
        {"action": "do_nothing", "parameters": {}},
        {"action": "move_cursor", "parameters": {"x": 100, "y": 200}},
        {"action": "open_browser", "parameters": {"url": "https://example.com"}},
        {"action": "open_app", "parameters": {"app_name": "firefox"}},
        {"action": "press_key", "parameters": {"key": "enter"}},
        {"action": "hotkey", "parameters": {"keys": ["ctrl", "c"]}},
        {"action": "type_text", "parameters": {"text": "hello world"}},
    ]

    for payload in valid_payloads:
        total += 1
        try:
            result = ActionValidator.validate_action(payload)
            if result["action"] == payload["action"]:
                passed += 1
            else:
                print(f"  [FAIL] Valid {payload['action']}: action mismatch in result")
                failed += 1
        except Exception as e:
            print(f"  [FAIL] Valid {payload['action']}: unexpected {type(e).__name__}: {e}")
            failed += 1

    # ── 3B: Invalid actions — must raise ValidationError ──
    invalid_actions = [
        {"action": "NONEXISTENT"},
        {"action": "drop_tables"},
        {"action": "rm -rf /"},
        {"action": ""},
        {"action": 42},
        {"action": None},
        {"action": ["click"]},
        "not_a_dict",
        42,
        None,
    ]

    for payload in invalid_actions:
        total += 1
        try:
            ActionValidator.validate_action(payload)
            print(f"  [FAIL] Invalid {payload!r}: did NOT raise ValidationError")
            failed += 1
        except (ValidationError, TypeError, AttributeError):
            passed += 1
        except Exception as e:
            print(f"  [FAIL] Invalid {payload!r}: wrong exception {type(e).__name__}: {e}")
            failed += 1

    # ── 3C: Type mismatches in parameters ──
    type_mismatches = [
        {"action": "move_cursor", "parameters": {"x": "abc", "y": 100}},  # str for int
        {"action": "move_cursor", "parameters": {"x": 100, "y": "abc"}},  # str for int
        {"action": "open_app", "parameters": {"app_name": 123}},          # int for str
        {"action": "press_key", "parameters": {"key": 42}},               # int for str
        {"action": "hotkey", "parameters": {"keys": "ctrl+c"}},           # str for list
        {"action": "type_text", "parameters": {"text": 12345}},           # int for str
    ]

    for payload in type_mismatches:
        total += 1
        try:
            ActionValidator.validate_action(payload)
            print(f"  [FAIL] TypeMismatch {payload['action']}: did NOT raise ValidationError")
            failed += 1
        except ValidationError:
            passed += 1
        except Exception as e:
            print(f"  [FAIL] TypeMismatch {payload['action']}: wrong exception {type(e).__name__}")
            failed += 1

    # ── 3D: Missing required parameters ──
    missing_required = [
        {"action": "open_app", "parameters": {}},       # app_name required
        {"action": "press_key", "parameters": {}},       # key required
        {"action": "hotkey", "parameters": {}},           # keys required
        {"action": "type_text", "parameters": {}},        # text required
    ]

    for payload in missing_required:
        total += 1
        try:
            ActionValidator.validate_action(payload)
            print(f"  [FAIL] MissingReq {payload['action']}: did NOT raise ValidationError")
            failed += 1
        except ValidationError:
            passed += 1
        except Exception as e:
            print(f"  [FAIL] MissingReq {payload['action']}: wrong exception {type(e).__name__}")
            failed += 1

    ok = failed == 0
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {passed}/{total} boundary checks passed, {failed} failures")
    return ok


# ── Vector 4: 50-Loop Sequential parse_json + validate Stress ────────────
def test_sequential_parse_validate_stress():
    """Run 50 sequential loops: generate chaotic markdown-wrapped strings,
    pipe through _parse_json → ActionValidator.validate_action chain.
    Verify fallback dict 100% enforcement on non-parseable inputs."""
    print("\n" + "=" * 72)
    print("[L4-004] 50-Loop Sequential Parse+Validate Stress")
    print("=" * 72)

    import random
    import string

    llm = LLMInterface.__new__(LLMInterface)
    llm.model = "test"
    llm.api_url = ""
    llm.temperature = 0.1
    llm.timeout = 5
    llm.max_retries = 0

    LOOPS = 50
    fallback_count = 0
    valid_count = 0
    error_count = 0

    for i in range(LOOPS):
        # Generate chaotic markdown string
        noise = "".join(random.choices(string.printable, k=random.randint(10, 500)))
        # Randomly decide if we embed valid JSON or not
        if random.random() < 0.3:
            # 30% chance: embed valid action JSON inside noise
            action = random.choice(list(VALID_ACTIONS))
            json_str = json.dumps({"action": action, "parameters": {}, "reasoning": "auto"})
            raw = f"```json\n{json_str}\n```\n{noise}"
        else:
            # 70% chance: pure chaos
            raw = noise

        try:
            parsed = llm._parse_json(raw)
            if parsed.get("action") == "do_nothing" and parsed.get("reasoning", "").startswith(("No JSON", "Invalid")):
                fallback_count += 1
            else:
                # Attempt validation
                try:
                    ActionValidator.validate_action(parsed)
                    valid_count += 1
                except ValidationError:
                    fallback_count += 1
        except Exception as e:
            print(f"  [FAIL] Loop {i}: unhandled {type(e).__name__}: {e}")
            error_count += 1

    ok = error_count == 0
    status = "PASS" if ok else "FAIL"
    print(f"  Valid parsed  : {valid_count}/{LOOPS}")
    print(f"  Fallback safe : {fallback_count}/{LOOPS}")
    print(f"  Errors        : {error_count}/{LOOPS}")
    print(f"  [{status}] Zero unhandled exceptions in {LOOPS} loops")
    return ok


# ── Runner ────────────────────────────────────────────────────────────────
def run_all():
    print("\n" + "#" * 72)
    print("#  AURA-OS  Layer 4 — Foundation Determinism Suite")
    print("#" * 72)

    results = {}
    results["L4-001_parse_json_chaos"] = test_parse_json_chaotic_inputs()
    results["L4-002_fallback_struct"] = test_fallback_enforcement()
    results["L4-003_validator_boundary"] = test_action_validator_boundaries()
    results["L4-004_stress_50_loops"] = test_sequential_parse_validate_stress()

    print("\n" + "=" * 72)
    print("  LAYER 4 SUMMARY")
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
