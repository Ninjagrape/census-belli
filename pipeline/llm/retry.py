"""
Retry policy for transient LLM failures.

The project's error-handling rule is three retries with exponential backoff
from a 2 second base. Both vendor SDKs retry some failures internally, so
this wrapper is configured to cover what they do not: it counts attempts
across the whole call so that a stage's log shows the true number of round
trips, and it treats only transient conditions as retryable.

**A rate limiter that says when to come back is obeyed.** Blind exponential
backoff answers "retry in 29s" with 2.5 seconds, fails again, and burns every
remaining attempt on a limit that was never going to lift -- which is what a
free-tier Gemini key did on 2026-09-19, turning a quota error into what looked
like a hang. When a provider supplies a delay, :class:`TransientLLMError`
carries it and it wins over the computed one.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

import structlog

__all__ = ["DEFAULT_BASE_DELAY", "DEFAULT_MAX_RETRIES", "TransientLLMError", "with_backoff"]

logger = structlog.get_logger()

DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 2.0
# Cap so a long backoff chain cannot stall a batch for minutes on one record.
_MAX_DELAY = 30.0
# A provider that names its own delay may exceed that cap, within reason: the
# point of waiting at all is that the wait is long enough to work. Beyond this
# the limit is better reported than slept through.
_MAX_SERVER_DELAY = 120.0

T = TypeVar("T")


class TransientLLMError(Exception):
    """
    A failure worth retrying: rate limit, timeout, connection drop, 5xx.

    Provider clients translate their SDK's exceptions into this so the retry
    policy stays vendor-neutral.

    Attributes:
        retry_after: Seconds the provider asked us to wait, when it said so.
            None means it did not, and the computed backoff applies.
    """

    def __init__(self, *args: object, retry_after: float | None = None) -> None:
        """Create the error.

        Args:
            *args: Passed to ``Exception``.
            retry_after: The provider's own suggested delay, in seconds.
        """
        super().__init__(*args)
        self.retry_after = retry_after


def with_backoff(
    operation: Callable[[], T],
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_BASE_DELAY,
    context: dict[str, object] | None = None,
) -> tuple[T, int]:
    """
    Run an operation, retrying transient failures with exponential backoff.

    Args:
        operation: Zero-argument callable performing one attempt. It must
            raise :class:`TransientLLMError` for retryable conditions and any
            other exception for permanent ones.
        max_retries: Maximum retries after the initial attempt.
        base_delay: Seconds for the first backoff; doubles each retry.
        context: Extra fields to include in retry log lines, e.g. the stage
            name and battle id.

    Returns:
        A ``(result, attempts)`` pair, where ``attempts`` counts every round
        trip made including the successful one.

    Raises:
        TransientLLMError: If every attempt failed transiently.
        Exception: Whatever the operation raised for a permanent failure.
    """
    log_context = context or {}
    last_error: TransientLLMError | None = None

    for attempt in range(max_retries + 1):
        try:
            return operation(), attempt + 1
        except TransientLLMError as e:
            last_error = e
            if attempt == max_retries:
                break
            # Jitter so that a batch hitting a rate limit does not resynchronise
            # every worker onto the same retry instant.
            delay = min(base_delay * (2**attempt) + random.uniform(0, 1), _MAX_DELAY)

            # A provider that named a delay knows better than the formula.
            suggested = getattr(e, "retry_after", None)
            honoured = False
            if suggested is not None and suggested > delay:
                delay = min(float(suggested), _MAX_SERVER_DELAY)
                honoured = True

            logger.warning(
                "llm_call_retrying",
                attempt=attempt + 1,
                max_attempts=max_retries + 1,
                delay_seconds=round(delay, 1),
                server_suggested=honoured,
                error=str(e),
                **log_context,
            )
            time.sleep(delay)

    assert last_error is not None  # unreachable: the loop only breaks after a failure
    raise last_error
