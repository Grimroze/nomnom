import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from circuit_breaker import ModelRouter
from rate_limiter import RateLimiter
from redis_client import redis_client
from semantic_cache import SemanticCache

app = FastAPI(
    title="NomNom LLM Gateway",
    description="Resilient Distributed LLM Gateway with Semantic Caching and Circuit Breaking",
    version="1.0.0",
)

# Initialize Core Services
limiter = RateLimiter(limit=5, period=60)
cache = SemanticCache(threshold=0.85, ttl=3600)
router = ModelRouter(
    primary_model="llama3.1:latest",
    backup_model="gemma4:31b-cloud",
    failure_threshold=3,
    cooldown_seconds=30,
)


class ChatRequest(BaseModel):
    prompt: str
    client_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    source: str
    similarity_score: Optional[float] = None
    latency_ms: float
    rate_limit_remaining: int

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "NomNom LLM Gateway",
        "docs_url": "/docs",
        "stats_url": "/v1/stats",
    }

@app.post("/v1/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest, request: Request):
    start_time = time.time()
    client_id = payload.client_id or request.client.host or "127.0.0.1"

    # Metric: increment total requests
    redis_client.incr("metrics:total_requests")

    # Step 1: Rate Limiter Check
    is_blocked, remaining = limiter.is_rate_limited(client_id)
    if is_blocked:
        redis_client.incr("metrics:rate_limited")
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Maximum 5 requests per 60 seconds allowed.",
            headers={"Retry-After": "60", "X-RateLimit-Remaining": "0"},
        )

    # Step 2: Semantic Cache Lookup
    cached_response, similarity, matched_prompt = cache.get(payload.prompt)
    if cached_response is not None:
        redis_client.incr("metrics:cache_hits")
        latency = (time.time() - start_time) * 1000.0

        return ChatResponse(
            response=cached_response,
            source="SEMANTIC_CACHE",
            similarity_score=round(similarity, 4),
            latency_ms=round(latency, 2),
            rate_limit_remaining=remaining,
        )

    # Step 3: Cache Miss -> Route to LLM
    redis_client.incr("metrics:cache_misses")
    response_text, model_source = router.route(payload.prompt)

    # Step 4: Populate Cache with new LLM response
    cache.set(
        prompt=payload.prompt, response=response_text, model_used=model_source
    )

    latency = (time.time() - start_time) * 1000.0

    return ChatResponse(
        response=response_text,
        source=model_source,
        similarity_score=None,
        latency_ms=round(latency, 2),
        rate_limit_remaining=remaining,
    )


@app.get("/v1/stats")
def stats_endpoint():
    """Live analytics from Redis."""
    total = int(redis_client.get("metrics:total_requests") or 0)
    hits = int(redis_client.get("metrics:cache_hits") or 0)
    misses = int(redis_client.get("metrics:cache_misses") or 0)
    rate_limited = int(redis_client.get("metrics:rate_limited") or 0)
    circuit_state = redis_client.get("cb:state") or "CLOSED"

    hit_ratio = round((hits / total) * 100, 2) if total > 0 else 0.0

    return {
        "total_requests": total,
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_ratio_percent": hit_ratio,
        "rate_limited_requests": rate_limited,
        "circuit_breaker_state": circuit_state,
    }


@app.get("/health")
def health_endpoint():
    redis_ok = redis_client.ping()
    return {"status": "healthy", "redis_connected": redis_ok}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)