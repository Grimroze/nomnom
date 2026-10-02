import os
import redis

# Reads REDIS_HOST and REDIS_PORT from environment (defaults to localhost:6379 for local development)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_client = redis.Redis(
    host=REDIS_HOST, port=REDIS_PORT, decode_responses=True
)


def get_redis():
    """FastAPI Dependency Injection helper."""
    return redis_client


if __name__ == "__main__":
    redis_client.set("nomnom_status", "Real Redis is connected.")
    msg = redis_client.get("nomnom_status")
    print(f"[Connection Test]: {msg}")