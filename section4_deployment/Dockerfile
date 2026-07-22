# CPU-only by design: this must build and run on ANY machine, including an
# examiner's with no GPU and no NVIDIA container toolkit. GPU acceleration is
# available by running natively (outside Docker) with a CUDA-enabled torch --
# see README "Running with a GPU". This keeps `docker build && docker run`
# a true zero-assumptions path per the task requirement.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU-only torch wheel -- small, fast to install, no CUDA toolkit required.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY src/ src/
COPY prompts/ prompts/

# Small model + CPU by default -- guarantees a smooth first run for anyone
# without a GPU. Override at `docker run` time (see README) to use the
# larger Section 3 model and/or a GPU if running natively instead.
ENV MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
ENV FALLBACK_MODEL_ID=""
ENV PRECISION="fp16"
ENV DEVICE_MODE="cpu"
ENV MAX_CONCURRENT_GENERATIONS="1"
ENV HF_HOME="/app/hf_cache"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
