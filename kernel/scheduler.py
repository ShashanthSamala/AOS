# AURA-OS — Kernel Layer
# scheduler.py — FIFO task queue for agent request scheduling
# Author: Samala Shashanth | Project: AURA-OS

import time
import uuid
from collections import deque
from typing import Any, Optional


class Task:
    """Represents a single scheduled task in the kernel."""

    def __init__(self, command: str, action: str, parameters: Optional[dict] = None, reasoning: str = ""):
        self.id: str = str(uuid.uuid4()).split("-")[0]
        self.command = command
        self.action = action
        self.parameters = parameters or {}
        self.reasoning = reasoning
        self.status = "queued"        # queued → running → done / failed
        self.created_at: float = time.time()
        self.completed_at: Optional[float] = None
        self.result: Optional[Any] = None

    def mark_running(self):
        self.status = "running"

    def mark_done(self, result=None):
        self.status = "done"
        self.completed_at = time.time()
        self.result = result

    def mark_failed(self, error=None):
        self.status = "failed"
        self.completed_at = time.time()
        self.result = {"error": str(error)}

    def to_dict(self):
        return {
            "id": self.id,
            "command": self.command,
            "action": self.action,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
            "status": self.status,
            "created_at": self.created_at,
        }

    def __repr__(self):
        return f"Task({self.id}: {self.action} [{self.status}])"


class Scheduler:
    """FIFO task scheduler for the AURA-OS agent kernel."""

    def __init__(self, max_queue=100):
        self._queue = deque()
        self._history = []
        self.max_queue = max_queue

    def add_task(self, command, action, parameters=None, reasoning=""):
        """
        Create and enqueue a new task.

        Returns:
            The created Task object
        """
        if len(self._queue) >= self.max_queue:
            # Drop oldest task if queue is full
            dropped = self._queue.popleft()
            dropped.mark_failed("Dropped: queue overflow")
            self._history.append(dropped)

        task = Task(command, action, parameters, reasoning)
        self._queue.append(task)
        return task

    def get_next(self):
        """
        Dequeue and return the next task (FIFO).

        Returns:
            Task object or None if queue is empty
        """
        if self._queue:
            task = self._queue.popleft()
            task.mark_running()
            return task
        return None

    def peek(self):
        """View the next task without removing it."""
        return self._queue[0] if self._queue else None

    def complete_task(self, task, result=None):
        """Mark a task as completed and move to history."""
        task.mark_done(result)
        self._history.append(task)

    def fail_task(self, task, error=None):
        """Mark a task as failed and move to history."""
        task.mark_failed(error)
        self._history.append(task)

    def requeue_task(self, task):
        """Re-insert a task at the front of the queue for retry."""
        task.status = "queued"
        self._queue.appendleft(task)

    @property
    def size(self):
        return len(self._queue)

    @property
    def history(self):
        return list(self._history)

    def get_queue_status(self):
        """Return summary of current queue state."""
        return {
            "queued": len(self._queue),
            "completed": sum(1 for t in self._history if t.status == "done"),
            "failed": sum(1 for t in self._history if t.status == "failed"),
        }

    def __repr__(self):
        status = self.get_queue_status()
        return (
            f"Scheduler(queued={status['queued']}, "
            f"done={status['completed']}, failed={status['failed']})"
        )
