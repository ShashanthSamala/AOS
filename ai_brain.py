# AURA-OS — Layer 4: Reasoning Layer
# ai_brain.py — LLM interprets gestures and decides OS actions
# Author: Samala Shashanth | Project: AURA-OS

import requests
import json
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
CURRENT_MODEL = "tinyllama"

SYSTEM_PROMPT = """You are AURA-OS, an intelligent operating system assistant.
You receive hand gesture inputs and decide what OS action to perform.
ALWAYS reply with ONLY valid JSON in exactly this format, nothing else:
{"action": "<action_name>", "reasoning": "<one line explanation>"}

Available actions:
click, new_tab, close_tab, scroll_up, scroll_down,
volume_up, volume_down, play_pause, open_browser,
show_desktop, lock_screen, open_terminal, do_nothing"""

def ask_brain(gesture: str, context: str = "desktop", battery: int = 100) -> dict:
    prompt = f"""Gesture detected: {gesture}
Active application: {context}
Battery level: {battery}%
What OS action should I perform?"""

    try:
        start = time.time()
        resp = requests.post(OLLAMA_URL, json={
            "model": CURRENT_MODEL,
            "system": SYSTEM_PROMPT,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1}
        }, timeout=60)

        latency = round((time.time() - start) * 1000)
        print(f"[DEBUG RAW] {resp.text}")
        raw = resp.json()["response"].strip()

        # Clean up backticks if model adds them
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Extract just the JSON part
        start_idx = raw.find("{")
        end_idx = raw.rfind("}") + 1
        raw = raw[start_idx:end_idx]

        result = json.loads(raw)
        result["latency_ms"] = latency
        return result

    except Exception as e:
        print(f"[BRAIN ERROR] {e}")
        return {"action": "do_nothing", "reasoning": "LLM error fallback", "latency_ms": 0}


# --- Test it directly ---
if __name__ == "__main__":
    test_gestures = ["FIST", "OPEN_PALM", "THREE", "PEACE", "POINT"]

    print("AURA-OS Layer 4 — Brain Test")
    print("=" * 40)

    for gesture in test_gestures:
        result = ask_brain(gesture, context="browser", battery=75)
        print(f"Gesture: {gesture}")
        print(f"Action:  {result.get('action')}")
        print(f"Reason:  {result.get('reasoning')}")
        print(f"Latency: {result.get('latency_ms')}ms")
        print("-" * 40)
