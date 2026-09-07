"""Shared plumbing for the custom MCP servers.

Deliberately depends only on stdlib + ``httpx`` + ``python-dotenv`` so each server
process stays lightweight and independent of the agent runtime.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except Exception:  # noqa: BLE001
    pass

DEFAULT_TIMEOUT = 25.0
MAX_ATTEMPTS = 3
_CACHE_TTL = 120.0
_cache: dict[str, tuple[float, Any]] = {}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def nasa_key() -> str:
    return os.environ.get("NASA_API_KEY", "DEMO_KEY") or "DEMO_KEY"


def ok(data: Any, *, source: str, url: str, **extra: Any) -> dict[str, Any]:
    """Standard success envelope. ``source``/``url`` become provenance in the KG."""
    return {
        "ok": True,
        "source": {"name": source, "url": url, "retrieved_at": utc_now_iso()},
        "data": data,
        **extra,
    }


def err(message: str, *, source: str, url: str = "", hint: str = "") -> dict[str, Any]:
    return {
        "ok": False,
        "error": message,
        "hint": hint,
        "source": {"name": source, "url": url, "retrieved_at": utc_now_iso()},
        "data": None,
    }


async def fetch_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    cache_ttl: float = _CACHE_TTL,
) -> Any:
    """GET JSON with retry + exponential backoff and a short in-process cache."""
    cache_key = f"{url}|{sorted((params or {}).items())}"
    now = time.time()
    hit = _cache.get(cache_key)
    if hit and now - hit[0] < cache_ttl:
        return hit[1]

    last_exc: Exception | None = None
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = await client.get(url, params=params, headers=headers)
                if response.status_code == 429:
                    raise httpx.HTTPStatusError(
                        "rate limited (429) — set a real NASA_API_KEY instead of DEMO_KEY",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                _cache[cache_key] = (now, payload)
                return payload
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < MAX_ATTEMPTS - 1:
                    await asyncio.sleep(0.6 * (2**attempt))
    raise RuntimeError(f"request failed after {MAX_ATTEMPTS} attempts: {last_exc}")


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))
