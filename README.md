# NomNom: Resilient Distributed LLM Gateway

A high-performance, fault-tolerant LLM Gateway engineered with FastAPI, Redis, and LangChain. Built to reduce inference costs, enforce tenant quota protection, and guarantee high availability across local and cloud LLM endpoints.

---

## System Architecture

```mermaid
flowchart TD
    Client["Client Application"] -->|POST /v1/chat| Gateway["NomNom FastAPI Gateway"]
    
    subgraph RateLimiting ["Layer 1: Quota Protection"]
        Gateway --> RL["Sliding Window Limiter - Redis ZSET"]
        RL -->|Quota Exceeded| Block["429 Too Many Requests"]
    end

    subgraph SemanticCaching ["Layer 2: Two-Stage Semantic Cache"]
        RL -->|Allowed| SC["Semantic Cache Lookup"]
        SC --> BGE["BGE-small-en-v1.5 Embedding"]
        BGE --> Guard{"Lexical Polarity Guard"}
        Guard -->|Polarity Mismatch Detected| CacheMiss["Cache Bypass - Miss"]
        Guard -->|Polarity Match| CacheHit["Return Cached Response - ~14ms"]
    end

    subgraph ResilienceRouting ["Layer 3: Distributed Circuit Breaker and Router"]
        CacheMiss --> Router{"Circuit Breaker State"}
        Router -->|State: CLOSED| Primary["Primary: Local Ollama llama3.1"]
        Router -->|State: OPEN| Backup["Backup: Cloud Ollama gemma4"]
        Primary -->|Connection Timeout / Error| Trip["Record Failure in Redis"]
        Trip -->|Failures >= 3| OpenCircuit["Trip Circuit to OPEN - TTL 30s"]
        OpenCircuit --> Backup
        Primary -->|Success| ResetFailures["Reset Failures - cb:failures = 0"]
    end

    subgraph Persistence ["Persistence and Analytics"]
        ResetFailures --> SaveCache["Persist in Redis Hash - TTL 3600s"]
        Backup --> SaveCache
        SaveCache --> Metrics["Increment Redis Telemetry Metrics"]
        Metrics --> Response["200 OK Response with Telemetry"]
    end
```

---

## Circuit Breaker State Machine

The distributed Circuit Breaker prevents cascading gateway latency during upstream inference outages. The state transitions are coordinated atomically across distributed nodes using Redis key-expiration semantics.

```mermaid
flowchart LR
    CLOSED["State: CLOSED<br/>(Normal Operation)<br/>Traffic -> Primary llama3.1"]
    OPEN["State: OPEN<br/>(Circuit Tripped)<br/>Traffic -> Backup gemma4"]

    CLOSED -->|"3 Failures Detected (cb:failures >= 3)"| OPEN
    OPEN -->|"Cooldown Expired (Redis TTL: 30s)"| CLOSED

    style CLOSED fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    style OPEN fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#b71c1c
```

---

## Execution Lifecycle: Cache Hit vs. Cache Miss

```mermaid
sequenceDiagram
    autonumber
    actor Client
    participant Gateway as FastAPI Gateway
    participant Redis as Redis (Port 6379)
    participant Model as Ollama LLM Service

    Note over Client,Redis: Scenario A: Cache Hit - Latency ~14 ms
    Client->>Gateway: POST /v1/chat with prompt
    Gateway->>Redis: ZREMRANGEBYSCORE and ZCARD (Sliding Window Log)
    Redis-->>Gateway: Quota OK
    Gateway->>Redis: Scan cache hashes and compute dot product
    Redis-->>Gateway: Match found with Score 0.942
    Gateway-->>Client: 200 OK from SEMANTIC_CACHE in 14.2 ms

    Note over Client,Model: Scenario B: Cache Miss - Latency ~2100 ms
    Client->>Gateway: POST /v1/chat with prompt
    Gateway->>Redis: Scan cache hashes and compute dot product
    Redis-->>Gateway: No match above threshold
    Gateway->>Redis: GET cb:state
    Redis-->>Gateway: CLOSED
    Gateway->>Model: invoke prompt on llama3.1
    Model-->>Gateway: Inference response generated
    Gateway->>Redis: HSET cache entry with TTL 3600s
    Gateway->>Redis: INCR metrics total and misses
    Gateway-->>Client: 200 OK from PRIMARY_llama3.1 in 2140 ms
```

---

## Key Technical Features

### 1. Sliding Window Log Rate Limiting
- Built on top of **Redis Sorted Sets (ZSET)** to eliminate the boundary spike vulnerability inherent to Fixed Window counters.
- Prunes expired timestamps (`ZREMRANGEBYSCORE`) and reads window cardinality (`ZCARD`) within a single Redis pipeline transaction to minimize network round-trip time (RTT).

### 2. Two-Stage Semantic Caching
- **Stage 1 (Vector Proximity):** Transforms prompts into 384-dimensional unit embeddings using `BAAI/bge-small-en-v1.5`. Because embeddings are pre-normalized, Cosine Similarity simplifies to a hardware-accelerated NumPy dot product.
- **Stage 2 (Lexical Polarity Guard):** Bi-encoder embedding models exhibit known blind spots for negation terms (`not`, `never`, `without`), frequently assigning similarity scores $> 0.88$ to opposing prompts (e.g., *"How to install"* vs *"How NOT to install"*). A sub-millisecond lexical XOR filter enforces polarity verification before any cached payload is returned.

### 3. Distributed Circuit Breaker & Fallback Router
- Prevents cascading gateway delays when upstream endpoints hang or fail (e.g., HTTP 410 model retirement, connection timeouts).
- Automatically trips to `OPEN` state upon 3 consecutive upstream failures, locking the dead endpoint for 30 seconds via Redis TTL keys and routing requests directly to the backup cloud model without latency penalty.

### 4. Real-Time Observability
- Exposes runtime telemetry via `/v1/stats`, including cache hit ratios, rate-limited request counts, and upstream circuit breaker statuses.

---

## Latency & Benchmark Metrics

| Request Path | Upstream Engine | Mean Latency | Latency Reduction |
| :--- | :--- | :--- | :--- |
| **Semantic Cache Hit** | Redis (Vector Dot Product) | **12 ms - 22 ms** | **99.3%** |
| **Primary Inference (Local)** | Ollama `llama3.1:latest` | **1800 ms - 3200 ms** | Baseline |
| **Fallback Inference (Cloud)** | Ollama `gemma4:31b-cloud` | **900 ms - 1600 ms** | Dynamic Failover |
| **Circuit Breaker Failover** | Redis State Bypass | **0 ms Overhead** | Immediate Routing |

---

## Project Structure

```text
nomnom/
├── redis_client.py       # Distributed Redis client with Docker/Local env config
├── rate_limiter.py       # Sliding Window Log limiter using Redis ZSET
├── semantic_cache.py     # 2-Stage Semantic Cache (BGE + Negation Guard)
├── circuit_breaker.py    # Multi-Model Router with Redis-backed Circuit Breaker
├── main.py               # FastAPI application (/v1/chat, /v1/stats, /health)
├── requirements.txt      # Locked production dependencies
├── Dockerfile            # Container build specification with preloaded model weights
├── docker-compose.yml    # Single-command orchestrator for Gateway and Redis
└── .gitignore            # Git exclusion rules
```

---

## Quickstart

### Option A: Running with Docker Compose (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nomnom.git
   cd nomnom
   ```

2. Launch the Gateway and Redis services:
   ```bash
   docker compose up --build
   ```

3. Open Swagger documentation at: `http://localhost:8000/docs`

---

### Option B: Local Python Development

1. Launch Redis container:
   ```bash
   docker run -d --name nomnom-redis -p 6379:6379 redis:latest
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Gateway:
   ```bash
   python main.py
   ```

---

## API Specification

### `POST /v1/chat`
Execute prompt with automated rate limiting, semantic caching, and model routing.

**Request Payload:**
```json
{
  "prompt": "How do I install Docker on Ubuntu Linux?",
  "client_id": "client_alpha"
}
```

**Response Payload (Cache Miss):**
```json
{
  "response": "Run: sudo apt update && sudo apt install docker.io",
  "source": "PRIMARY_llama3.1:latest",
  "similarity_score": null,
  "latency_ms": 2145.8,
  "rate_limit_remaining": 4
}
```

**Response Payload (Semantic Cache Hit):**
```json
{
  "response": "Run: sudo apt update && sudo apt install docker.io",
  "source": "SEMANTIC_CACHE",
  "similarity_score": 0.9339,
  "latency_ms": 14.1,
  "rate_limit_remaining": 3
}
```

### `GET /v1/stats`
Retrieve live telemetry recorded across the Redis cluster.

**Response Payload:**
```json
{
  "total_requests": 24,
  "cache_hits": 9,
  "cache_misses": 15,
  "cache_hit_ratio_percent": 37.5,
  "rate_limited_requests": 7,
  "circuit_breaker_state": "CLOSED"
}
```
