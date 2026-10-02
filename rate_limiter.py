import time
import uuid
from redis_client import redis_client


class RateLimiter:

    def __init__(self, limit: int = 5, period: int = 60):
        self.r = redis_client
        self.limit = limit  # e.g., 5 requests
        self.period = period  # e.g., in 60 seconds

    def is_rate_limited(self, client_id: str) -> tuple[bool, int]:
        """Checks if the client (IP or User) has exceeded the rate limit.

        Returns: (is_blocked: bool, remaining_requests: int)
        """
        now = time.time()
        key = f"rate_limit:{client_id}"
        window_start = now - self.period

        # Salted member to prevent collision at exact same millisecond
        member = f"{now}:{uuid.uuid4().hex[:6]}"

        # Redis Pipeline for single network round-trip & atomicity
        pipe = self.r.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)  # 1. Purani requests delete
        pipe.zcard(key)  # 2. Window ke andar kitni hain
        pipe.expire(key, self.period + 5)  # 3. Key TTL for memory cleanup
        _, current_count, _ = pipe.execute()

        # Check limit
        if current_count >= self.limit:
            return True, 0  # Blocked!

        # Allowed! Log new request
        self.r.zadd(key, {member: now})
        remaining = self.limit - (current_count + 1)
        return False, remaining


# Quick Test
if __name__ == "__main__":
    limiter = RateLimiter(limit=3, period=10)
    user_ip = "127.0.0.1"

    print("--- Testing Real Redis Rate Limiter (Limit: 3 req / 10s) ---")
    for i in range(1, 6):
        blocked, remaining = limiter.is_rate_limited(user_ip)
        if blocked:
            print(f"Request {i}:  429 BLOCKED (Quota exceeded!)")
        else:
            print(
                f"Request {i}:  200 ALLOWED (Remaining quota: {remaining}/3)"
            )
        time.sleep(0.3)