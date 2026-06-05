# AURA-OS — Kernel Layer
# resource_manager.py — System resource monitoring and graceful degradation
# Author: Samala Shashanth | Project: AURA-OS

"""
Monitors system resources (CPU, RAM, battery, queue depth) and degrades
operating mode gracefully to maintain stability under load.

Resource states:
- HEALTHY: Normal operation with LLM enabled
- DEGRADED: High resource usage; LLM disabled for unmapped gestures
- CRITICAL: Severe resource constraints; direct mode only
"""

import psutil
import time
from enum import Enum
from typing import Dict, Optional


class ResourceState(Enum):
    """System resource state."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class ResourceMonitor:
    """Monitors system resources and provides degradation guidance."""

    def __init__(self, config=None):
        """
        Initialize resource monitor with config thresholds.

        Config keys:
        - cpu_threshold: CPU usage % above which to degrade (default: 80)
        - ram_threshold: RAM usage % above which to degrade (default: 85)
        - battery_threshold: Battery % below which to go critical (default: 10)
        - queue_threshold: Queue fullness % to degrade (default: 80)
        - check_interval: Min seconds between checks (default: 2)
        """
        cfg = config or {}
        self.cpu_threshold = cfg.get("cpu_threshold", 80)
        self.ram_threshold = cfg.get("ram_threshold", 85)
        self.battery_threshold = cfg.get("battery_threshold", 10)
        self.queue_threshold = cfg.get("queue_threshold", 80)
        self.check_interval = cfg.get("check_interval", 2.0)

        self._last_check_time = 0
        self._last_state = ResourceState.HEALTHY
        self._metrics = {}

    def check(self, queue_size: int = 0, max_queue: int = 100) -> ResourceState:
        """
        Check current resource state.

        Args:
            queue_size: Current number of tasks in queue
            max_queue: Maximum queue capacity

        Returns:
            ResourceState (HEALTHY, DEGRADED, or CRITICAL)
        """
        now = time.time()
        # Only check periodically to avoid overhead
        if now - self._last_check_time < self.check_interval:
            return self._last_state

        self._last_check_time = now

        # Collect metrics
        self._metrics = self._collect_metrics(queue_size, max_queue)

        # Determine state
        state = self._determine_state(self._metrics)
        self._last_state = state
        return state

    def _collect_metrics(self, queue_size: int, max_queue: int) -> Dict:
        """Collect current resource metrics."""
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        battery = psutil.sensors_battery()

        queue_percent = int((queue_size / max_queue) * 100) if max_queue > 0 else 0

        metrics = {
            "cpu_percent": cpu,
            "ram_percent": ram.percent,
            "queue_percent": queue_percent,
            "timestamp": time.time(),
        }

        if battery:
            metrics["battery_percent"] = battery.percent
            metrics["battery_plugged"] = battery.power_plugged
        else:
            metrics["battery_percent"] = 100  # Assume OK if no battery
            metrics["battery_plugged"] = True

        return metrics

    def _determine_state(self, metrics: Dict) -> ResourceState:
        """Determine resource state based on metrics."""
        # Critical thresholds (highest priority)
        if (metrics["battery_percent"] < self.battery_threshold
                and not metrics["battery_plugged"]):
            return ResourceState.CRITICAL

        # Degraded thresholds
        if (metrics["cpu_percent"] > self.cpu_threshold
                or metrics["ram_percent"] > self.ram_threshold
                or metrics["queue_percent"] > self.queue_threshold):
            return ResourceState.DEGRADED

        return ResourceState.HEALTHY

    def get_metrics(self) -> Dict:
        """Return last collected metrics."""
        return dict(self._metrics)

    def should_skip_llm(self, state: Optional[ResourceState] = None) -> bool:
        """
        Check if LLM should be skipped based on resource state.

        In DEGRADED/CRITICAL: skip LLM for unmapped gestures
        (use direct mode only)

        Args:
            state: ResourceState (uses last check if None)

        Returns:
            True if LLM should be skipped
        """
        if state is None:
            state = self._last_state
        return state in (ResourceState.DEGRADED, ResourceState.CRITICAL)

    def should_accept_gesture(self, state: Optional[ResourceState] = None) -> bool:
        """
        Check if system should accept new gestures.

        In CRITICAL: only accept critical gestures

        Args:
            state: ResourceState (uses last check if None)

        Returns:
            True if gesture should be accepted
        """
        if state is None:
            state = self._last_state
        # Accept all gestures unless critical
        return state != ResourceState.CRITICAL

    def get_health_report(self, state: Optional[ResourceState] = None) -> Dict:
        """
        Get human-readable health report.

        Args:
            state: ResourceState (uses last check if None)

        Returns:
            Dict with status, metrics, recommendations
        """
        if state is None:
            state = self._last_state

        metrics = self.get_metrics()
        recommendations = []

        if metrics["cpu_percent"] > self.cpu_threshold:
            recommendations.append("High CPU usage - close unnecessary apps")
        if metrics["ram_percent"] > self.ram_threshold:
            recommendations.append("High memory usage - free up RAM")
        if metrics["queue_percent"] > self.queue_threshold:
            recommendations.append("Task queue backlog - slowdown detected")
        if (metrics["battery_percent"] < self.battery_threshold
                and not metrics.get("battery_plugged", True)):
            recommendations.append("Low battery - connect power or reduce gestures")

        return {
            "state": state.value,
            "metrics": metrics,
            "skip_llm": self.should_skip_llm(state),
            "accept_gestures": self.should_accept_gesture(state),
            "recommendations": recommendations,
        }
