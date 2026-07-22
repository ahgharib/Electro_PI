"""
Custom exception hierarchy.

Keeping these distinct (instead of raising bare Exception/RuntimeError
everywhere) means calling code -- the CLI, the benchmark runner, and later
Section 4's API layer -- can catch exactly what it expects and give the user
an actionable message instead of a raw stack trace.
"""


class QuantizationBenchmarkError(Exception):
    """Base class for every custom exception in this project."""


class DeviceUnavailableError(QuantizationBenchmarkError):
    """Raised when a specifically requested device (--device gpu) is not available.
    Deliberately NOT auto-recovered -- if you explicitly asked for GPU numbers,
    silently benchmarking on CPU instead would produce misleading results."""


class UnsupportedPrecisionError(QuantizationBenchmarkError):
    """Raised when a precision is requested that the resolved device can't support
    (e.g. bitsandbytes 4-bit/8-bit on a CPU-only run)."""


class ModelLoadError(QuantizationBenchmarkError):
    """Raised when a model fails to load after every fallback attempt is exhausted."""


class GenerationError(QuantizationBenchmarkError):
    """Raised for non-recoverable failures during text generation."""


class ModelNotReadyError(QuantizationBenchmarkError):
    """Raised by the API when a request arrives before the model has finished loading
    (or after it failed to load) -- mapped to HTTP 503 in the FastAPI layer."""
