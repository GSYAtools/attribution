import time
from collections.abc import Callable
from typing import TypeVar

from src.providers.errors import ProviderError


T = TypeVar("T")


def execute_with_retry(
    operation: Callable[[], T],
    max_retries: int,
    initial_delay_seconds: float = 2.0,
) -> T:
    """
    Execute a provider operation with controlled retries.

    max_retries is the number of retries after the initial attempt.
    Only errors explicitly marked as retryable are retried.
    """

    if max_retries < 0:
        raise ValueError(
            "max_retries cannot be negative."
        )

    attempt = 0

    while True:
        try:
            return operation()

        except ProviderError as exc:
            if not exc.retryable:
                raise

            if attempt >= max_retries:
                raise

            delay = (
                initial_delay_seconds
                * (2 ** attempt)
            )

            attempt += 1

            print(
                f"Retryable provider error "
                f"({exc.category}). "
                f"Retry {attempt}/{max_retries} "
                f"in {delay:.1f}s."
            )

            if delay > 0:
                time.sleep(delay)
