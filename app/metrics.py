"""In-process latency / request metrics for the stub.

Exposes:
- JSON snapshot (humans / dashboards that prefer JSON)
- Prometheus text exposition (scrapers) — hand-rolled, no prometheus_client dep
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field


@dataclass
class MetricsStore:
    request_count: int = 0
    error_count: int = 0
    stream_request_count: int = 0
    total_latency_ms: float = 0.0
    last_latency_ms: float = 0.0
    total_ttft_ms: float = 0.0
    last_ttft_ms: float = 0.0
    ttft_sample_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        latency_ms: float,
        *,
        error: bool = False,
        stream: bool = False,
        ttft_ms: float | None = None,
    ) -> None:
        with self._lock:
            self.request_count += 1
            if stream:
                self.stream_request_count += 1
            if error:
                self.error_count += 1
            self.total_latency_ms += latency_ms
            self.last_latency_ms = latency_ms
            if ttft_ms is not None:
                self.total_ttft_ms += ttft_ms
                self.last_ttft_ms = ttft_ms
                self.ttft_sample_count += 1

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            avg = (
                self.total_latency_ms / self.request_count
                if self.request_count
                else 0.0
            )
            avg_ttft = (
                self.total_ttft_ms / self.ttft_sample_count
                if self.ttft_sample_count
                else 0.0
            )
            return {
                "request_count": self.request_count,
                "stream_request_count": self.stream_request_count,
                "error_count": self.error_count,
                "total_latency_ms": round(self.total_latency_ms, 3),
                "avg_latency_ms": round(avg, 3),
                "last_latency_ms": round(self.last_latency_ms, 3),
                "ttft_sample_count": self.ttft_sample_count,
                "avg_ttft_ms": round(avg_ttft, 3),
                "last_ttft_ms": round(self.last_ttft_ms, 3),
            }

    def prometheus_text(self, *, model: str | None = None) -> str:
        """Prometheus exposition format (text/plain; version=0.0.4).

        Hand-rolled teaching formatter — not the official client library.
        """
        snap = self.snapshot()
        model = model or os.getenv("STUB_MODEL_ID", "stub-model")
        # Escape label values lightly
        model_label = model.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        lines = [
            "# HELP stub_requests_total Total chat completion requests handled by the stub.",
            "# TYPE stub_requests_total counter",
            f'stub_requests_total{{model="{model_label}"}} {snap["request_count"]}',
            "# HELP stub_stream_requests_total Streaming (SSE) chat completion requests.",
            "# TYPE stub_stream_requests_total counter",
            f'stub_stream_requests_total{{model="{model_label}"}} {snap["stream_request_count"]}',
            "# HELP stub_errors_total Error responses from the stub.",
            "# TYPE stub_errors_total counter",
            f'stub_errors_total{{model="{model_label}"}} {snap["error_count"]}',
            "# HELP stub_latency_ms_sum Cumulative end-to-end latency in milliseconds.",
            "# TYPE stub_latency_ms_sum counter",
            f'stub_latency_ms_sum{{model="{model_label}"}} {snap["total_latency_ms"]}',
            "# HELP stub_latency_ms_avg Average end-to-end latency in milliseconds.",
            "# TYPE stub_latency_ms_avg gauge",
            f'stub_latency_ms_avg{{model="{model_label}"}} {snap["avg_latency_ms"]}',
            "# HELP stub_latency_ms_last Most recent end-to-end latency in milliseconds.",
            "# TYPE stub_latency_ms_last gauge",
            f'stub_latency_ms_last{{model="{model_label}"}} {snap["last_latency_ms"]}',
            "# HELP stub_ttft_samples_total Number of TTFT samples recorded.",
            "# TYPE stub_ttft_samples_total counter",
            f'stub_ttft_samples_total{{model="{model_label}"}} {snap["ttft_sample_count"]}',
            "# HELP stub_ttft_ms_avg Average time-to-first-token in milliseconds.",
            "# TYPE stub_ttft_ms_avg gauge",
            f'stub_ttft_ms_avg{{model="{model_label}"}} {snap["avg_ttft_ms"]}',
            "# HELP stub_ttft_ms_last Most recent time-to-first-token in milliseconds.",
            "# TYPE stub_ttft_ms_last gauge",
            f'stub_ttft_ms_last{{model="{model_label}"}} {snap["last_ttft_ms"]}',
            "",
        ]
        return "\n".join(lines)


metrics = MetricsStore()


class Timer:
    """Simple wall-clock timer in milliseconds."""

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000.0
