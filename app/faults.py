"""Deterministic fault injection + synthetic OpenAI-style rate-limit headers.

Off by default. A fault is chosen per request, in priority order:

1. Request header ``X-Stub-Fault: <spec>``  (``none`` disables env faults for that request)
2. Model suffix ``"model": "stub-model::fault=<spec>"`` (for clients that can't set headers;
   the suffix is stripped before the model id is echoed back)
3. Env ``STUB_FAULT=<spec>``, optionally scheduled with a deterministic request counter:
   ``STUB_FAULT_FIRST=N`` (only the first N chat requests) and/or
   ``STUB_FAULT_EVERY=N`` (every Nth chat request: N, 2N, ...). Neither set → every request.

``<spec>`` grammar::

    429[:retry_after_s]   OpenAI-shaped rate_limit_exceeded + Retry-After (default 1s)
    503[:retry_after_s]   server overloaded (Retry-After only if given)
    timeout[:ms]          sleep ms (default STUB_FAULT_TIMEOUT_MS or 30000), then answer
    drop[:k]              stream=true only: send k content chunks, then end the stream
                          with no finish_reason and no ``data: [DONE]`` (default k=2)

No randomness: the same config + request order always gives the same faults.
Synthetic values only — not OpenAI's real limits, and not a real rate limiter.
"""

from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass

MODEL_FAULT_SEP = "::fault="

_SPEC_RE = re.compile(r"^(429|503|timeout|drop)(?::(\d+(?:\.\d+)?))?$")


@dataclass(frozen=True)
class Fault:
    kind: str  # "429" | "503" | "timeout" | "drop"
    retry_after_s: float | None = None
    timeout_ms: float | None = None
    drop_after: int | None = None
    source: str = "env"  # "header" | "model" | "env"


class FaultSpecError(ValueError):
    pass


def parse_fault_spec(spec: str | None, *, source: str = "env") -> Fault | None:
    """Parse ``429:2`` / ``503`` / ``timeout:250`` / ``drop:3``; ``none``/empty → None."""
    if spec is None:
        return None
    s = spec.strip().lower()
    if s in ("", "none", "off", "0"):
        return None
    m = _SPEC_RE.match(s)
    if not m:
        raise FaultSpecError(
            f"invalid fault spec {spec!r}; expected 429[:s] | 503[:s] | timeout[:ms] | drop[:k] | none"
        )
    kind, arg = m.group(1), m.group(2)
    if kind == "429":
        return Fault("429", retry_after_s=float(arg) if arg else 1.0, source=source)
    if kind == "503":
        return Fault("503", retry_after_s=float(arg) if arg else None, source=source)
    if kind == "timeout":
        default_ms = float(os.getenv("STUB_FAULT_TIMEOUT_MS", "30000") or 30000)
        return Fault("timeout", timeout_ms=float(arg) if arg else default_ms, source=source)
    return Fault("drop", drop_after=int(float(arg)) if arg else 2, source=source)


def split_model_fault(model: str) -> tuple[str, str | None]:
    """``"stub-model::fault=429"`` → ``("stub-model", "429")``."""
    if MODEL_FAULT_SEP in model:
        base, spec = model.split(MODEL_FAULT_SEP, 1)
        return base, spec
    return model, None


def _int_env(name: str) -> int | None:
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        return None
    return n if n > 0 else None


class FaultInjector:
    """Holds the deterministic request counter + per-kind injected counts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.request_seq = 0
        self.injected: dict[str, int] = {}

    def reset(self) -> None:
        with self._lock:
            self.request_seq = 0
            self.injected = {}

    def _env_fault_applies(self, seq: int) -> bool:
        first = _int_env("STUB_FAULT_FIRST")
        every = _int_env("STUB_FAULT_EVERY")
        if first is None and every is None:
            return True
        return (first is not None and seq <= first) or (every is not None and seq % every == 0)

    def resolve(self, *, header: str | None, model_spec: str | None) -> Fault | None:
        """Pick the fault for this request (advances the env schedule counter)."""
        with self._lock:
            self.request_seq += 1
            seq = self.request_seq
        if header is not None:
            fault = parse_fault_spec(header, source="header")
        elif model_spec is not None:
            fault = parse_fault_spec(model_spec, source="model")
        else:
            fault = parse_fault_spec(os.getenv("STUB_FAULT"), source="env")
            if fault is not None and not self._env_fault_applies(seq):
                fault = None
        return fault

    def record(self, kind: str) -> None:
        with self._lock:
            self.injected[kind] = self.injected.get(kind, 0) + 1

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.injected)


faults = FaultInjector()


# ---------------------------------------------------------------------------
# Synthetic x-ratelimit-* headers (OpenAI header names; values are made up)
# ---------------------------------------------------------------------------

def _fmt_reset(seconds: float) -> str:
    """OpenAI-style duration string: ``1s``, ``6m0s``, ``250ms``."""
    if seconds < 1:
        return f"{int(round(seconds * 1000))}ms"
    total = int(round(seconds))
    m, s = divmod(total, 60)
    return f"{m}m{s}s" if m else f"{s}s"


class RateLimitWindow:
    """Fixed 60s window counting requests/tokens — reported in headers, never enforced."""

    WINDOW_S = 60.0

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._start = clock()
        self.requests = 0
        self.tokens = 0

    def reset(self) -> None:
        with self._lock:
            self._start = self._clock()
            self.requests = 0
            self.tokens = 0

    def headers(self, *, tokens: int = 0, exhausted: bool = False, retry_after_s: float | None = None) -> dict[str, str]:
        limit_req = _int_env("STUB_RATELIMIT_REQUESTS") or 60
        limit_tok = _int_env("STUB_RATELIMIT_TOKENS") or 100_000
        with self._lock:
            now = self._clock()
            if now - self._start >= self.WINDOW_S:
                self._start, self.requests, self.tokens = now, 0, 0
            self.requests += 1
            self.tokens += max(0, tokens)
            reset_s = max(0.0, self.WINDOW_S - (now - self._start))
            rem_req = max(0, limit_req - self.requests)
            rem_tok = max(0, limit_tok - self.tokens)
        if exhausted:
            rem_req = 0
            if retry_after_s is not None:
                reset_s = retry_after_s
        return {
            "x-ratelimit-limit-requests": str(limit_req),
            "x-ratelimit-remaining-requests": str(rem_req),
            "x-ratelimit-reset-requests": _fmt_reset(reset_s),
            "x-ratelimit-limit-tokens": str(limit_tok),
            "x-ratelimit-remaining-tokens": str(rem_tok),
            "x-ratelimit-reset-tokens": _fmt_reset(reset_s),
        }


rate_limits = RateLimitWindow()


def error_body(fault: Fault) -> tuple[int, dict]:
    """OpenAI-shaped error JSON for 429 / 503 faults."""
    if fault.kind == "429":
        return 429, {
            "error": {
                "message": "Rate limit reached for requests (stub fault injection).",
                "type": "requests",
                "param": None,
                "code": "rate_limit_exceeded",
            }
        }
    return 503, {
        "error": {
            "message": "The server is overloaded or not ready yet (stub fault injection).",
            "type": "server_error",
            "param": None,
            "code": "service_unavailable",
        }
    }


def fmt_retry_after(seconds: float) -> str:
    """Retry-After is delta-seconds (integer per RFC 9110); round up so clients never retry early."""
    whole = int(seconds)
    return str(whole if whole == seconds else whole + 1)
