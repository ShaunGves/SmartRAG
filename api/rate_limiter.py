"""
api/rate_limiter.py

Per-IP rate limiting middleware for the FastAPI application.

Why rate limiting matters for an LLM API:
  - Each request runs inference on a GPU → real compute cost
  - Without limits, a single bad actor can exhaust the GPU budget
  - LLM APIs at every major company (OpenAI, Anthropic) use rate limits
  - Shows production system design thinking

Implementation: Token bucket algorithm
  - Each IP gets a "bucket" of tokens
  - Each request costs 1 token
  - Tokens refill at a fixed rate (requests_per_minute)
  - Burst allowance: bucket can hold up to burst_multiplier × per-minute limit

Limits enforced:
  - Per-minute  : prevents sudden bursts
  - Per-hour    : prevents sustained abuse
  - Per-day     : prevents daily quota exhaustion

Headers returned (matching OpenAI/Anthropic convention):
  X-RateLimit-Limit-Requests
  X-RateLimit-Remaining-Requests
  X-RateLimit-Reset-Requests
  Retry-After  (on 429)
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ─── Token Bucket ────────────────────────────────────────────────
@dataclass
class TokenBucket:
    """
    Token bucket for one client IP at one time window.

    Attributes:
        capacity     : Max tokens (= burst limit)
        tokens       : Current available tokens
        refill_rate  : Tokens added per second
        last_refill  : Unix timestamp of last refill
    """
    capacity: float
    tokens: float
    refill_rate: float       # tokens per second
    last_refill: float = field(default_factory=time.time)

    def consume(self, n: float = 1.0) -> Tuple[bool, float]:
        """
        Try to consume n tokens.

        Returns:
            (allowed, wait_seconds)
              allowed=True  if tokens available
              allowed=False if rate limited; wait_seconds = how long to wait
        """
        now = time.time()
        elapsed = now - self.last_refill

        # Refill bucket based on elapsed time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= n:
            self.tokens -= n
            return True, 0.0
        else:
            # How long until 1 token is available
            wait = (n - self.tokens) / self.refill_rate
            return False, wait

    @property
    def remaining(self) -> int:
        return max(0, int(self.tokens))

    @property
    def reset_in_seconds(self) -> float:
        """Seconds until bucket is full again."""
        missing = self.capacity - self.tokens
        return missing / self.refill_rate if self.refill_rate > 0 else 0


# ─── Rate Limit Store ────────────────────────────────────────────
class RateLimitStore:
    """
    In-memory store for per-IP token buckets across time windows.

    In production: replace with Redis for multi-worker deployments.
    Redis INCR + EXPIRE pattern is the standard for distributed rate limiting.
    """

    def __init__(self):
        # Three buckets per IP: minute, hour, day
        self._minute_buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=cfg.ratelimit.requests_per_minute * cfg.ratelimit.burst_multiplier,
                tokens=cfg.ratelimit.requests_per_minute * cfg.ratelimit.burst_multiplier,
                refill_rate=cfg.ratelimit.requests_per_minute / 60.0,
            )
        )
        self._hour_buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=cfg.ratelimit.requests_per_hour,
                tokens=cfg.ratelimit.requests_per_hour,
                refill_rate=cfg.ratelimit.requests_per_hour / 3600.0,
            )
        )
        self._day_buckets: Dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=cfg.ratelimit.requests_per_day,
                tokens=cfg.ratelimit.requests_per_day,
                refill_rate=cfg.ratelimit.requests_per_day / 86400.0,
            )
        )
        self._blocked_ips: Dict[str, float] = {}   # IP → unblock timestamp

    def check(self, ip: str) -> Tuple[bool, dict]:
        """
        Check if the IP is within rate limits.

        Returns:
            (allowed, rate_limit_info)
        """
        # Check if IP is temporarily blocked
        if ip in self._blocked_ips:
            if time.time() < self._blocked_ips[ip]:
                wait = self._blocked_ips[ip] - time.time()
                return False, {"window": "blocked", "retry_after": round(wait)}
            else:
                del self._blocked_ips[ip]

        # Check minute bucket
        allowed, wait = self._minute_buckets[ip].consume()
        if not allowed:
            return False, {
                "window": "minute",
                "limit": cfg.ratelimit.requests_per_minute,
                "remaining": 0,
                "retry_after": round(wait, 1),
            }

        # Check hour bucket
        allowed, wait = self._hour_buckets[ip].consume()
        if not allowed:
            return False, {
                "window": "hour",
                "limit": cfg.ratelimit.requests_per_hour,
                "remaining": 0,
                "retry_after": round(wait, 1),
            }

        # Check day bucket
        allowed, wait = self._day_buckets[ip].consume()
        if not allowed:
            return False, {
                "window": "day",
                "limit": cfg.ratelimit.requests_per_day,
                "remaining": 0,
                "retry_after": round(wait, 1),
            }

        return True, {
            "remaining_minute": self._minute_buckets[ip].remaining,
            "remaining_hour":   self._hour_buckets[ip].remaining,
            "remaining_day":    self._day_buckets[ip].remaining,
        }

    def stats(self) -> dict:
        return {
            "tracked_ips": len(self._minute_buckets),
            "blocked_ips": len(self._blocked_ips),
        }


# ─── FastAPI Middleware ───────────────────────────────────────────
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that enforces per-IP rate limits.

    Exempt paths: /health, /docs, /openapi.json, /
    Limited paths: /query, /ingest (anything that touches GPU)
    """

    EXEMPT_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
    LIMITED_PATHS = {"/query", "/ingest"}

    def __init__(self, app, store: Optional[RateLimitStore] = None):
        super().__init__(app)
        self.store = store or RateLimitStore()
        log.info("Rate limiter initialized")

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        # Only limit specified paths
        if request.url.path not in self.LIMITED_PATHS:
            return await call_next(request)

        # Skip if rate limiting disabled in config
        if not cfg.ratelimit.enabled:
            return await call_next(request)

        # Get client IP
        ip = self._get_client_ip(request)

        # Check rate limit
        allowed, info = self.store.check(ip)

        if not allowed:
            retry_after = info.get("retry_after", 60)
            log.warning(f"Rate limited: {ip} | window={info.get('window')} | retry_after={retry_after}s")

            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Too many requests. Rate limit window: {info.get('window')}.",
                    "retry_after_seconds": retry_after,
                },
                headers={
                    "Retry-After": str(int(retry_after)),
                    "X-RateLimit-Limit-Requests": str(cfg.ratelimit.requests_per_minute),
                    "X-RateLimit-Remaining-Requests": "0",
                },
            )

        # Request allowed — add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit-Requests"]     = str(cfg.ratelimit.requests_per_minute)
        response.headers["X-RateLimit-Remaining-Requests"] = str(info.get("remaining_minute", 0))
        response.headers["X-RateLimit-Limit-Hour"]         = str(cfg.ratelimit.requests_per_hour)
        response.headers["X-RateLimit-Remaining-Hour"]     = str(info.get("remaining_hour", 0))
        return response

    @staticmethod
    def _get_client_ip(request: Request) -> str:
        """Extract real client IP, respecting proxy headers."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        return request.client.host if request.client else "unknown"
