"""Structured JSON logging + optional Langfuse tracing.

Every agent step, MCP tool call, guardrail decision and LLM call funnels through
here so the eval harness can compute latency, cost and per-server success rates
from a single JSONL stream.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Iterator

from core.config import get_settings

_LOCK = Lock()
_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "astral", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    global _CONFIGURED
    with _LOCK:
        if _CONFIGURED:
            return
        settings = get_settings()
        root = logging.getLogger("astralgraph")
        root.setLevel(settings.log_level.upper())
        root.propagate = False

        console = logging.StreamHandler()
        console.setFormatter(_JsonFormatter())
        root.addHandler(console)

        log_file = Path(settings.log_dir) / "astralgraph.jsonl"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(_JsonFormatter())
        root.addHandler(file_handler)
        _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"astralgraph.{name}")


def log_event(logger: logging.Logger, message: str, /, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, message, extra={"astral": fields})


# --------------------------------------------------------------------------- #
# Session-scoped trace record (consumed by the eval harness and the API)
# --------------------------------------------------------------------------- #


@dataclass
class SpanRecord:
    name: str
    kind: str
    started_at: float
    duration_ms: float
    ok: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trace:
    """Collects everything that happened while answering one question."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    question: str = ""
    spans: list[SpanRecord] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    usd_cost: float = 0.0
    started_at: float = field(default_factory=time.perf_counter)

    def add(self, span: SpanRecord) -> None:
        self.spans.append(span)

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0

    def tool_calls(self) -> list[SpanRecord]:
        return [s for s in self.spans if s.kind == "mcp_tool"]

    def server_stats(self) -> dict[str, dict[str, int]]:
        stats: dict[str, dict[str, int]] = {}
        for span in self.tool_calls():
            server = str(span.detail.get("server", "unknown"))
            bucket = stats.setdefault(server, {"calls": 0, "ok": 0, "failed": 0})
            bucket["calls"] += 1
            bucket["ok" if span.ok else "failed"] += 1
        return stats

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "latency_ms": round(self.elapsed_ms, 1),
            "tool_calls": len(self.tool_calls()),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd_cost": round(self.usd_cost, 6),
            "server_stats": self.server_stats(),
            "spans": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "ms": round(s.duration_ms, 1),
                    "ok": s.ok,
                    **s.detail,
                }
                for s in self.spans
            ],
        }


@contextmanager
def span(trace: Trace | None, name: str, kind: str, **detail: Any) -> Iterator[dict[str, Any]]:
    """Time a block and attach it to the trace. Mutate the yielded dict to add fields."""
    started = time.perf_counter()
    mutable: dict[str, Any] = dict(detail)
    ok = True
    try:
        yield mutable
    except Exception as exc:  # noqa: BLE001 - recorded then re-raised
        ok = False
        mutable["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration = (time.perf_counter() - started) * 1000.0
        ok = ok and bool(mutable.pop("_ok", True))
        if trace is not None:
            trace.add(
                SpanRecord(
                    name=name,
                    kind=kind,
                    started_at=started,
                    duration_ms=duration,
                    ok=ok,
                    detail=mutable,
                )
            )
        log_event(
            get_logger("trace"),
            f"span:{kind}:{name}",
            session_id=getattr(trace, "session_id", None),
            duration_ms=round(duration, 1),
            ok=ok,
            **mutable,
        )


# --------------------------------------------------------------------------- #
# Langfuse (optional)
# --------------------------------------------------------------------------- #


def get_langfuse():  # pragma: no cover - network client
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception as exc:  # noqa: BLE001
        log_event(get_logger("telemetry"), "langfuse_init_failed", error=str(exc))
        return None


def push_trace_to_langfuse(trace: Trace, answer: str) -> None:  # pragma: no cover
    client = get_langfuse()
    if client is None:
        return
    try:
        root = client.trace(
            id=trace.session_id,
            name="astralgraph.ask",
            input=trace.question,
            output=answer,
            metadata=trace.summary(),
        )
        for s in trace.spans:
            root.span(name=f"{s.kind}:{s.name}", metadata={"ok": s.ok, **s.detail})
        client.flush()
    except Exception as exc:  # noqa: BLE001
        log_event(get_logger("telemetry"), "langfuse_push_failed", error=str(exc))
