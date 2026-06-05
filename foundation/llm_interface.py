# AURA-OS — Foundation Layer
# llm_interface.py — Ollama REST API wrapper
# Author: Samala Shashanth | Project: AURA-OS

import requests
import json
import time
import yaml
import os

from .action_validator import ActionValidator, ValidationError


class LLMInterface:
    """Wraps the Ollama REST API for local LLM inference."""

    def __init__(self, config_path=None):
        config = self._load_config(config_path)
        llm_cfg = config.get("llm", {})

        self.model = llm_cfg.get("model", "tinyllama")
        self.api_url = llm_cfg.get("api_url", "http://localhost:11434/api/generate")
        self.temperature = llm_cfg.get("temperature", 0.1)
        self.timeout = llm_cfg.get("timeout", 60)
        self.max_retries = llm_cfg.get("max_retries", 2)

    def _load_config(self, config_path):
        """Load config from YAML file."""
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "config", "settings.yaml"
            )
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def ask(self, prompt, system_prompt=""):
        """
        Send a prompt to the LLM and return parsed + validated JSON action.

        Returns:
            dict with keys: action, parameters, reasoning, latency_ms
        """
        for attempt in range(self.max_retries + 1):
            try:
                start = time.time()
                resp = requests.post(self.api_url, json={
                    "model": self.model,
                    "system": system_prompt,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature}
                }, timeout=self.timeout)

                latency = round((time.time() - start) * 1000)
                resp.raise_for_status()

                raw = resp.json().get("response", "").strip()
                result = self._parse_json(raw)
                result["latency_ms"] = latency

                # Validate action before returning
                try:
                    validated = ActionValidator.validate_action(result)
                    return validated
                except ValidationError as e:
                    return self._fallback(f"Validation error: {e}")

            except requests.exceptions.Timeout:
                if attempt < self.max_retries:
                    continue
                return self._fallback("LLM timeout")
            except requests.exceptions.ConnectionError:
                return self._fallback("Cannot connect to Ollama — is it running?")
            except Exception as e:
                if attempt < self.max_retries:
                    continue
                return self._fallback(f"LLM error: {e}")

    def _parse_json(self, raw):
        """Extract and parse JSON from LLM response text."""
        # Strip markdown code fences
        raw = raw.replace("```json", "").replace("```", "").strip()

        # Find JSON object boundaries
        start_idx = raw.find("{")
        end_idx = raw.rfind("}") + 1

        if start_idx == -1 or end_idx <= start_idx:
            return self._fallback("No JSON in LLM response")

        json_str = raw[start_idx:end_idx]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return self._fallback("Invalid JSON from LLM")

    def _fallback(self, reason):
        """Return a safe fallback action."""
        return {
            "action": "do_nothing",
            "parameters": {},
            "reasoning": reason,
            "latency_ms": 0
        }

    def is_available(self):
        """Check if Ollama is running and the model is loaded."""
        try:
            resp = requests.get(
                self.api_url.replace("/api/generate", "/api/tags"),
                timeout=5
            )
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in m for m in models)
        except Exception:
            return False
