import time
from langchain_ollama import ChatOllama
from redis_client import redis_client


class ModelRouter:

    def __init__(
        self,
        primary_model: str = "llama3.1:latest",
        backup_model: str = "gemma4:31b-cloud",
        failure_threshold: int = 3,
        cooldown_seconds: int = 30,
    ):
        self.r = redis_client
        self.primary_name = primary_model
        self.backup_name = backup_model
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

        self.primary_llm = ChatOllama(model=primary_model)
        self.backup_llm = ChatOllama(model=backup_model)

    def route(self, prompt: str) -> tuple[str, str]:
        """Routes prompt to Primary or Backup model depending on Circuit Breaker state.

        Returns: (response_text: str, source_model: str)
        """
        # 1. Check if Circuit Breaker is OPEN in Redis
        if self.r.get("cb:state") == "OPEN":
            print(
                f"[Circuit Breaker: OPEN] Primary '{self.primary_name}' locked. Routing directly to backup '{self.backup_name}'."
            )
            res = self.backup_llm.invoke(prompt)
            return res.content.strip(), f"BACKUP_{self.backup_name}"

        # 2. Circuit is CLOSED -> Try Primary
        try:
            print(f"[Circuit Breaker: CLOSED] Calling primary '{self.primary_name}'...")
            res = self.primary_llm.invoke(prompt)

            # On success, clear any recorded past failures
            self.r.delete("cb:failures")
            return res.content.strip(), f"PRIMARY_{self.primary_name}"

        except Exception as e:
            # 3. Primary failure handling
            fails = self.r.incr("cb:failures")
            print(
                f"[Primary Error] {e.__class__.__name__}: {e}. Failure count: {fails}/{self.failure_threshold}"
            )

            if fails >= self.failure_threshold:
                print(
                    f"[Circuit Breaker Tripped] Tripping circuit to OPEN for {self.cooldown_seconds}s."
                )
                self.r.setex("cb:state", self.cooldown_seconds, "OPEN")
                self.r.delete("cb:failures")

            # 4. Fallback execution
            print(f"[Fallback] Executing backup '{self.backup_name}'...")
            res = self.backup_llm.invoke(prompt)
            return res.content.strip(), f"BACKUP_{self.backup_name}"


# Standalone Test
if __name__ == "__main__":
    router = ModelRouter(
        primary_model="llama3.1:latest",
        backup_model="gemma4:31b-cloud",
        failure_threshold=3,
        cooldown_seconds=10,
    )

    test_prompt = "Reply in exactly 5 words: What is Redis?"
    print(f"Testing ModelRouter with prompt: '{test_prompt}'\n")

    response, source = router.route(test_prompt)
    print(f"\nResponse: {response}")
    print(f"Source:   {source}")