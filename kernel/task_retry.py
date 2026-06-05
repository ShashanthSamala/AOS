# AURA-OS — Kernel Layer
# task_retry.py — Retry policy for transient failures
# Author: Samala Shashanth | Project: AURA-OS

"""
Handles retry logic for failed tasks with exponential backoff.
Distinguishes between transient errors (network, temporary) and
permanent errors (invalid action, validation failure).
"""

import time
from enum import Enum
from typing import Optional


class ErrorType(Enum):
    """Classification of errors."""
    TRANSIENT = "transient"      # Retry-able (network, busy)
    PERMANENT = "permanent"      # Don't retry (validation, config)
    UNKNOWN = "unknown"          # Unknown - don't retry to be safe


class RetryPolicy:
    """Retry policy with exponential backoff."""

    def __init__(self, max_retries: int = 3):
        """
        Initialize retry policy.

        Args:
            max_retries: Maximum number of retries (default: 3)
        """
        self.max_retries = max_retries
        self.backoff_delays = [0.0, 0.5, 1.0]  # Delays in seconds

    def should_retry(self, error_msg: str, retry_count: int) -> bool:
        """
        Determine if a failed task should be retried.

        Args:
            error_msg: Error message
            retry_count: Number of retries already attempted

        Returns:
            True if should retry, False otherwise
        """
        if retry_count >= self.max_retries:
            return False

        error_type = self._classify_error(error_msg)
        return error_type == ErrorType.TRANSIENT

    def get_backoff_delay(self, retry_count: int) -> float:
        """
        Get backoff delay for retry attempt.

        Args:
            retry_count: Number of retries already attempted

        Returns:
            Delay in seconds
        """
        if retry_count < len(self.backoff_delays):
            return self.backoff_delays[retry_count]
        # Exponential backoff for attempts beyond initial list
        return 1.0 * (2 ** (retry_count - len(self.backoff_delays)))

    def _classify_error(self, error_msg: str) -> ErrorType:
        """Classify error as transient or permanent."""
        error_lower = error_msg.lower()

        # Transient errors - retry-able
        transient_keywords = [
            "timeout",
            "connection",
            "network",
            "refused",
            "unavailable",
            "busy",
            "try",  # "try again"
        ]
        for keyword in transient_keywords:
            if keyword in error_lower:
                return ErrorType.TRANSIENT

        # Permanent errors - don't retry
        permanent_keywords = [
            "validation",
            "invalid",
            "unknown action",
            "not found",
            "parameter",
            "type error",
        ]
        for keyword in permanent_keywords:
            if keyword in error_lower:
                return ErrorType.PERMANENT

        # Default: don't retry unknown errors to avoid infinite loops
        return ErrorType.UNKNOWN

    def get_error_type(self, error_msg: str) -> str:
        """Get error type as string."""
        return self._classify_error(error_msg).value
