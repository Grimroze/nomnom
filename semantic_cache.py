import json
import uuid
import numpy as np
from redis_client import redis_client
from sentence_transformers import SentenceTransformer

NEGATION_WORDS = {
    "not",
    "never",
    "no",
    "without",
    "dont",
    "don't",
    "cannot",
    "can't",
    "avoid",
    "disable",
    "uninstall",
}


class SemanticCache:

    def __init__(self, threshold: float = 0.85, ttl: int = 3600):
        self.r = redis_client
        self.threshold = threshold
        self.ttl = ttl
        print("[SemanticCache] Loading bge-small-en-v1.5 embedding model...")
        self.model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        print("[SemanticCache] Model loaded and ready.")

    def _get_embedding(self, text: str) -> list[float]:
        """Encodes text to a normalized 384-dimensional vector."""
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def _has_negation_mismatch(self, q1: str, q2: str) -> bool:
        """Layer 2 Guard: Returns True if polarity mismatches exist between queries."""
        w1 = set(q1.lower().replace("?", "").split()) & NEGATION_WORDS
        w2 = set(q2.lower().replace("?", "").split()) & NEGATION_WORDS
        return w1 != w2

    def set(self, prompt: str, response: str, model_used: str = "primary"):
        """Saves prompt, response, and vector into Redis Hash."""
        cache_id = f"cache:{uuid.uuid4().hex[:8]}"
        vector = self._get_embedding(prompt)

        self.r.hset(
            cache_id,
            mapping={
                "prompt": prompt,
                "response": response,
                "embedding": json.dumps(vector),
                "model_used": model_used,
            },
        )
        self.r.expire(cache_id, self.ttl)

    def get(self, user_prompt: str) -> tuple[str | None, float, str | None]:
        """Searches Redis for the highest similarity match above threshold.

        Returns: (cached_response, similarity_score, matched_prompt)
        """
        query_vec = np.array(self._get_embedding(user_prompt))

        best_score = -1.0
        best_response = None
        matched_prompt = None

        for key in self.r.scan_iter("cache:*"):
            data = self.r.hgetall(key)
            if not data or "embedding" not in data:
                continue

            cached_prompt = data["prompt"]
            cached_vec = np.array(json.loads(data["embedding"]))

            # Cosine similarity via dot product of normalized vectors
            score = float(np.dot(query_vec, cached_vec))

            if score >= self.threshold and score > best_score:
                # Layer 2: Negation Guard
                if self._has_negation_mismatch(user_prompt, cached_prompt):
                    print(
                        f"[Negation Guard Blocked] Score was {score:.3f}, but polarity mismatch detected with '{cached_prompt}'"
                    )
                    continue

                best_score = score
                best_response = data["response"]
                matched_prompt = cached_prompt

        if best_response:
            return best_response, best_score, matched_prompt

        return None, best_score, None


# Standalone Verification
if __name__ == "__main__":
    cache = SemanticCache(threshold=0.82)

    print("\n1. Seeding Redis with an entry...")
    cache.set(
        prompt="How do I install Docker on Ubuntu Linux?",
        response="Run: sudo apt update && sudo apt install docker.io",
        model_used="llama3.1:latest",
    )

    test_queries = [
        "What is the command to setup Docker in Ubuntu?",
        "How do I NOT install Docker on Ubuntu Linux?",
        "Best recipe for butter chicken",
    ]

    print("\n2. Testing Queries against Redis:")
    for q in test_queries:
        print(f"\nQuery: '{q}'")
        ans, score, matched = cache.get(q)
        if ans:
            print(f"CACHE HIT (Score: {score:.3f} | Matched: '{matched}')")
            print(f"Answer: {ans}")
        else:
            print(f"CACHE MISS (Best score: {score:.3f})")