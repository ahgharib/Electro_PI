"""
Transformers-backed engine: fp16/bf16 baseline + bitsandbytes 4-bit/8-bit
quantization (NF4, double quantization).

Two deliberate fallback mechanisms (not incidental -- each is logged and
surfaced in the benchmark report so results stay honest about what actually
ran):

  1. Load-time CUDA OOM: if fp16/bf16 loading runs out of VRAM, automatically
     retry once at 4-bit instead of just crashing. This is realistic --
     you're on an 8GB card, headroom is genuinely tight.
  2. Model fallback chain: if the primary model_id fails to load for any
     other reason (bad repo id, gated repo, network issue), retry with a
     smaller, known-good fallback model so a single bad model id doesn't
     block the whole benchmark run.

Both are OFF by switches (allow_oom_fallback, fallback_model_id=None) so
they can be disabled if you want a strict "fail loudly" run instead.
"""
import logging

from ..device import resolve_device
from ..exceptions import ModelLoadError, UnsupportedPrecisionError
from .base import InferenceEngine

logger = logging.getLogger(__name__)

VALID_PRECISIONS = {"fp16", "bf16", "4bit", "8bit"}
DEFAULT_FALLBACK_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class HFTransformersEngine(InferenceEngine):
    def __init__(
        self,
        model_id: str,
        precision: str = "fp16",
        device_mode: str = "auto",
        fallback_model_id: str | None = DEFAULT_FALLBACK_MODEL,
        allow_oom_fallback: bool = True,
    ):
        if precision not in VALID_PRECISIONS:
            raise ValueError(f"precision must be one of {VALID_PRECISIONS}, got {precision!r}")

        self.model_id = model_id
        self.precision = precision
        self.precision_label = precision
        self.device_mode = device_mode
        self.fallback_model_id = fallback_model_id
        self.allow_oom_fallback = allow_oom_fallback

        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._device: str | None = None

        # Populated during load() -- surfaced in the benchmark report so a
        # reader can see at a glance whether a fallback path was taken.
        self.resolved_model_id = model_id
        self.auto_fallback_triggered = False
        self.fallback_reason: str | None = None

    def load(self) -> None:
        self._device = resolve_device(self.device_mode)

        if self.precision in ("4bit", "8bit") and self._device != "cuda":
            raise UnsupportedPrecisionError(
                f"precision={self.precision!r} requires a CUDA GPU (bitsandbytes has no CPU kernel). "
                f"Resolved device was {self._device!r}. Use --device gpu on a CUDA machine, or switch to "
                f"--precision fp16/bf16 to run on CPU."
            )

        candidates = [self.model_id]
        if self.fallback_model_id and self.fallback_model_id != self.model_id:
            candidates.append(self.fallback_model_id)

        last_error: Exception | None = None
        for i, candidate in enumerate(candidates):
            try:
                self._try_load(candidate, self.precision)
                self.resolved_model_id = candidate
                if i > 0:
                    self.auto_fallback_triggered = True
                    self.fallback_reason = f"Primary model {self.model_id!r} failed to load ({last_error}); used fallback model {candidate!r}"
                    logger.warning(self.fallback_reason)
                return
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and self.allow_oom_fallback and self.precision in ("fp16", "bf16"):
                    logger.warning("CUDA OOM loading %s at %s -- retrying at 4-bit.", candidate, self.precision)
                    try:
                        self._try_load(candidate, "4bit")
                        self.resolved_model_id = candidate
                        self.precision_label = f"4bit (auto-fallback from {self.precision} OOM)"
                        self.auto_fallback_triggered = True
                        self.fallback_reason = f"CUDA OOM at {self.precision} on {candidate!r}, auto-downgraded to 4-bit"
                        return
                    except Exception as inner_exc:  # noqa: BLE001
                        last_error = inner_exc
                        continue
                last_error = exc
                continue
            except ModelLoadError as exc:
                last_error = exc
                continue

        raise ModelLoadError(
            f"Exhausted all candidate models {candidates} at precision={self.precision!r}. Last error: {last_error}"
        )

    def _try_load(self, model_id: str, precision: str) -> None:
        """One concrete load attempt. Raises RuntimeError on CUDA OOM (so load() can
        decide whether to retry at 4-bit) or ModelLoadError on any other failure."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        quant_config = None
        torch_dtype = torch.float32
        if self._device == "cuda":
            if precision == "fp16":
                torch_dtype = torch.float16
            elif precision == "bf16":
                torch_dtype = torch.bfloat16
            elif precision in ("4bit", "8bit"):
                from transformers import BitsAndBytesConfig

                quant_config = BitsAndBytesConfig(
                    load_in_4bit=(precision == "4bit"),
                    load_in_8bit=(precision == "8bit"),
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch_dtype if quant_config is None else None,
                quantization_config=quant_config,
                device_map="auto" if self._device == "cuda" else None,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise  # let load() decide whether to retry at 4-bit
            raise ModelLoadError(f"Failed to load {model_id!r}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - e.g. repo not found, gated repo, network error
            raise ModelLoadError(f"Failed to load {model_id!r}: {exc}") from exc

        if self._device == "cpu":
            model.to("cpu")
        model.eval()

        self._model = model
        self._tokenizer = tokenizer
        self._loaded = True

    def _generate_raw(self, prompt: str, max_new_tokens: int) -> tuple[str, int]:
        import torch

        if not self._loaded:
            raise RuntimeError("Engine not loaded — call load() or use as a context manager.")

        messages = [{"role": "user", "content": prompt}]
        encoded = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )

        # transformers >=5.x returns a BatchEncoding (dict-like: input_ids, attention_mask).
        # Older versions could return a bare tensor. Handle both so this survives version upgrades.
        if hasattr(encoded, "input_ids"):
            input_ids = encoded["input_ids"].to(self._model.device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._model.device)
        else:
            input_ids = encoded.to(self._model.device)
            attention_mask = None

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,  # deterministic -> fair fp16 vs quantized comparison
            pad_token_id=self._tokenizer.eos_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        with torch.no_grad():
            output_ids = self._model.generate(input_ids, **gen_kwargs)

        new_tokens = output_ids[0][input_ids.shape[-1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, len(new_tokens)

    def unload(self) -> None:
        import gc

        self._model = None
        self._tokenizer = None
        self._loaded = False
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
