from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    max_new_tokens: int = Field(200, ge=1, le=1024)


class GenerateResponse(BaseModel):
    output: str
    output_tokens: int
    generation_time_s: float
    tokens_per_second: float
    precision: str
    device: str


class HealthResponse(BaseModel):
    status: str
    ready: bool
    model_id: str | None = None
    precision: str | None = None
    device: str | None = None
    fallback_triggered: bool = False


class StatsResponse(BaseModel):
    active_requests: int
    queue_depth: int
    total_requests_served: int
    total_requests_failed: int
    avg_generation_time_s: float
    max_concurrent_generations: int
