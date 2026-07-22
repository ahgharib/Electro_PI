"""
Abstract inference engine interface.

Section 3 needs to compare fp16 vs quantized versions of the SAME model.
Section 4 will deploy behind an API. Both need to swap backends without
rewriting the calling code -- every engine implements this same contract.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import perf_counter
import traceback


@dataclass
class GenerationResult:
    prompt: str
    output: str
    generation_time_s: float
    output_tokens: int
    precision_label: str  # e.g. "fp16", "4bit", "4bit (auto-fallback from fp16 OOM)"
    error: str | None = None
    error_traceback: str | None = None  # full traceback -- some exceptions (e.g. bare AttributeError) carry no message

    @property
    def tokens_per_second(self) -> float:
        if self.generation_time_s <= 0 or self.output_tokens == 0:
            return 0.0
        return self.output_tokens / self.generation_time_s


class InferenceEngine(ABC):
    """
    Common contract for any backend. Concrete engines implement
    load/_generate_raw/unload; generate() wraps timing + error handling so
    a single failed prompt never crashes an entire benchmark run.
    """

    precision_label: str = "unknown"

    @abstractmethod
    def load(self) -> None:
        """Load weights into memory/VRAM. Must set self._loaded = True."""
        raise NotImplementedError

    @abstractmethod
    def _generate_raw(self, prompt: str, max_new_tokens: int) -> tuple[str, int]:
        """Return (output_text, output_token_count). Timing is handled by generate()."""
        raise NotImplementedError

    @abstractmethod
    def unload(self) -> None:
        """Release model/VRAM. Must be safe to call even if load() never succeeded."""
        raise NotImplementedError

    def generate(self, prompt: str, max_new_tokens: int = 256) -> GenerationResult:
        start = perf_counter()
        try:
            output_text, output_tokens = self._generate_raw(prompt, max_new_tokens)
            elapsed = perf_counter() - start
            return GenerationResult(
                prompt=prompt,
                output=output_text,
                generation_time_s=elapsed,
                output_tokens=output_tokens,
                precision_label=self.precision_label,
            )
        except Exception as exc:  # noqa: BLE001 - record ANY per-prompt failure without killing the whole run
            elapsed = perf_counter() - start
            return GenerationResult(
                prompt=prompt,
                output="",
                generation_time_s=elapsed,
                output_tokens=0,
                precision_label=self.precision_label,
                error=f"{type(exc).__name__}: {exc}",
                error_traceback=traceback.format_exc(),
            )

    def generate_stream(self, prompt: str, max_new_tokens: int = 256):
        """
        Optional: engines that support token-by-token streaming override this
        as a generator yielding text chunks. Default raises NotImplementedError
        so a caller gets a clear error instead of silently falling back to
        something unexpected.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support streaming generation.")

    def __enter__(self) -> "InferenceEngine":
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.unload()
