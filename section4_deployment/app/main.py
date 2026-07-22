"""
FastAPI application.

Built as a factory (`create_app(service)`) rather than a bare module-level
app so tests can inject a fake ModelService without touching torch/GPU/network
-- see tests/test_api.py. `uvicorn app.main:app` still works normally via the
module-level `app` built at the bottom with the real Settings-driven service.
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.config import Settings
from app.model_service import ModelService
from app.schemas import GenerateRequest, GenerateResponse, HealthResponse, StatsResponse
from src.exceptions import ModelNotReadyError

logger = logging.getLogger(__name__)


def create_app(service: ModelService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await service.startup()
        yield
        await service.shutdown()

    app = FastAPI(title="Section 4 — Quantized LLM Serving API", lifespan=lifespan)

    @app.get("/health", response_model=HealthResponse)
    async def health():
        engine = service.engine
        return HealthResponse(
            status="ok" if service.ready else "not_ready",
            ready=service.ready,
            model_id=getattr(engine, "resolved_model_id", None) if service.ready else None,
            precision=engine.precision_label if service.ready else None,
            device=getattr(engine, "_device", None) if service.ready else None,
            fallback_triggered=getattr(engine, "auto_fallback_triggered", False),
        )

    @app.get("/stats", response_model=StatsResponse)
    async def stats():
        return StatsResponse(**service.get_stats())

    @app.post("/generate", response_model=GenerateResponse)
    async def generate(req: GenerateRequest):
        try:
            result = await service.generate(req.prompt, req.max_new_tokens)
        except ModelNotReadyError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        if result.error:
            raise HTTPException(status_code=500, detail=result.error)

        return GenerateResponse(
            output=result.output,
            output_tokens=result.output_tokens,
            generation_time_s=round(result.generation_time_s, 4),
            tokens_per_second=round(result.tokens_per_second, 3),
            precision=result.precision_label,
            device=getattr(service.engine, "_device", None) or "unknown",
        )

    @app.post("/generate/stream")
    async def generate_stream(req: GenerateRequest):
        if not service.ready:
            raise HTTPException(status_code=503, detail=service.load_error or "Model is not loaded yet.")

        def event_source():
            try:
                for chunk in service.stream_generate_sync(req.prompt, req.max_new_tokens):
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:  # noqa: BLE001 - surface generation errors as an SSE event instead of a broken stream
                logger.exception("Streaming generation failed")
                yield f"data: [ERROR] {type(exc).__name__}: {exc}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    return app


def build_default_service() -> ModelService:
    """Builds the real, Settings-driven service used by `uvicorn app.main:app`."""
    from src.engines.transformers_engine import HFTransformersEngine
    from src.monitoring import MetricsLogger

    settings = Settings()
    if settings.hf_cache_dir:
        os.environ["HF_HOME"] = settings.hf_cache_dir
        logger.info("HF cache redirected to: %s", settings.hf_cache_dir)

    engine = HFTransformersEngine(
        model_id=settings.model_id,
        precision=settings.precision,
        device_mode=settings.device_mode,
        fallback_model_id=(settings.fallback_model_id or None),
    )
    metrics = MetricsLogger(settings.metrics_log_path)
    return ModelService(engine=engine, metrics=metrics, max_concurrent_generations=settings.max_concurrent_generations)


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
app = create_app(build_default_service())
