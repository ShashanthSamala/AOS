# AURA-OS — Kernel Layer
# metrics.py — Performance metrics collection and reporting
# Author: Samala Shashanth | Project: AURA-OS

"""
Collects performance metrics and provides insights into system behavior.
Tracks latencies, success rates, and patterns for debugging and optimization.
"""

import time
from collections import deque
from typing import Dict, List
from statistics import mean, median, stdev


class MetricsCollector:
    """Collects and aggregates performance metrics."""

    def __init__(self, max_entries: int = 1000):
        """
        Initialize metrics collector.

        Args:
            max_entries: Maximum entries to keep (default: 1000)
        """
        self.max_entries = max_entries
        self.metrics = deque(maxlen=max_entries)
        self.start_time = time.time()

    def record_gesture(self, gesture: str, latency_ms: int, success: bool):
        """
        Record a gesture event.

        Args:
            gesture: Gesture name
            latency_ms: Execution latency in milliseconds
            success: Whether gesture executed successfully
        """
        self.metrics.append({
            "timestamp": time.time(),
            "type": "gesture",
            "gesture": gesture,
            "latency_ms": latency_ms,
            "success": success,
        })

    def record_llm_call(self, latency_ms: int, success: bool, reason: str = ""):
        """
        Record an LLM call metric.

        Args:
            latency_ms: LLM latency in milliseconds
            success: Whether call succeeded
            reason: Optional reason for failure
        """
        self.metrics.append({
            "timestamp": time.time(),
            "type": "llm",
            "latency_ms": latency_ms,
            "success": success,
            "reason": reason,
        })

    def record_action(self, action: str, latency_ms: int, success: bool):
        """
        Record an action execution.

        Args:
            action: Action name
            latency_ms: Execution latency in milliseconds
            success: Whether action succeeded
        """
        self.metrics.append({
            "timestamp": time.time(),
            "type": "action",
            "action": action,
            "latency_ms": latency_ms,
            "success": success,
        })

    def get_statistics(self) -> Dict:
        """Get aggregate statistics over all recorded metrics."""
        if not self.metrics:
            return {
                "total_events": 0,
                "uptime_seconds": time.time() - self.start_time,
            }

        metrics_list = list(self.metrics)
        latencies = [m["latency_ms"] for m in metrics_list]
        successes = [m for m in metrics_list if m["success"]]

        stats = {
            "total_events": len(metrics_list),
            "uptime_seconds": int(time.time() - self.start_time),
            "success_rate": f"{(len(successes) / len(metrics_list) * 100):.1f}%",
            "latency_ms": {
                "min": min(latencies),
                "max": max(latencies),
                "mean": f"{mean(latencies):.1f}",
                "median": median(latencies),
            }
        }

        if len(latencies) > 1:
            stats["latency_ms"]["stdev"] = f"{stdev(latencies):.1f}"

        # Per-gesture stats
        gesture_metrics = {}
        for m in metrics_list:
            if m["type"] == "gesture":
                g = m["gesture"]
                if g not in gesture_metrics:
                    gesture_metrics[g] = {"count": 0, "latencies": []}
                gesture_metrics[g]["count"] += 1
                gesture_metrics[g]["latencies"].append(m["latency_ms"])

        if gesture_metrics:
            stats["gestures"] = {
                g: {
                    "count": v["count"],
                    "avg_latency_ms": f"{mean(v['latencies']):.1f}",
                }
                for g, v in gesture_metrics.items()
            }

        return stats

    def clear(self):
        """Clear all metrics."""
        self.metrics.clear()

    def get_recent(self, n: int = 10) -> List[Dict]:
        """Get recent n metrics."""
        return list(self.metrics)[-n:] if self.metrics else []
