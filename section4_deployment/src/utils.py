"""Small standalone utilities shared across the project."""
import functools
import logging
import time

logger = logging.getLogger(__name__)


def retry(times: int = 3, delay_s: float = 2.0, exceptions: tuple = (Exception,)):
    """
    Retry decorator with linear backoff. Used for anything that talks to the
    network (model downloads) where transient failures are common and
    shouldn't kill the whole run on the first hiccup.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: BLE001 - intentionally broad, this is a generic retry helper
                    last_exc = exc
                    logger.warning("Attempt %d/%d failed for %s: %s", attempt, times, func.__name__, exc)
                    if attempt < times:
                        time.sleep(delay_s * attempt)
            raise last_exc
        return wrapper
    return decorator
