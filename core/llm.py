"""Claude (Anthropic) wrapper with cost accounting and Pydantic-structured output.

Design notes
------------
* Structured output is obtained by forcing a *tool call* whose input schema is the
  Pydantic model's JSON schema. That is far more reliable than prompting for JSON.
* Every call updates the :class:`~core.telemetry.Trace` with token counts and an
  estimated USD cost, which the eval harness reports as cost/query.
* If no API key is configured the wrapper raises :class:`LLMUnavailable`; callers
  fall back to deterministic (non-LLM) behaviour so the pipeline still runs.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence, TypeVar

from pydantic import BaseModel, ValidationError

from core.config import get_settings
from core.telemetry import Trace, get_logger, log_event, span

log = get_logger("llm")

T = TypeVar("T", bound=BaseModel)

# USD per 1M tokens (public list prices; used for the cost column in the report).
PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-3-7-sonnet-20250219": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
}
_DEFAULT_PRICE = (3.00, 15.00)


class LLMUnavailable(RuntimeError):
    """Raised when no Anthropic API key is configured."""


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = PRICING.get(model, _DEFAULT_PRICE)
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


class ClaudeClient:
    """Thin async wrapper around the Anthropic Messages API."""

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self.settings = settings
        self.model = model or settings.model
        self._client: Any = None

    # -- lifecycle ---------------------------------------------------------
    def _ensure(self) -> Any:
        if self._client is None:
            if not self.settings.llm_enabled:
                raise LLMUnavailable(
                    "ANTHROPIC_API_KEY is not set — running in deterministic fallback mode."
                )
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    @property
    def available(self) -> bool:
        return self.settings.llm_enabled

    # -- accounting --------------------------------------------------------
    def _account(self, trace: Trace | None, usage: Any) -> dict[str, Any]:
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        cost = estimate_cost(self.model, in_tok, out_tok)
        if trace is not None:
            trace.input_tokens += in_tok
            trace.output_tokens += out_tok
            trace.usd_cost += cost
        return {"input_tokens": in_tok, "output_tokens": out_tok, "usd_cost": round(cost, 6)}

    # -- plain completion --------------------------------------------------
    async def complete(
        self,
        system: str,
        user: str,
        *,
        trace: Trace | None = None,
        name: str = "complete",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        client = self._ensure()
        with span(trace, name, "llm", model=self.model) as sp:
            resp = await client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.settings.max_tokens,
                temperature=self.settings.temperature if temperature is None else temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            sp.update(self._account(trace, resp.usage))
            return "".join(
                block.text for block in resp.content if getattr(block, "type", "") == "text"
            ).strip()

    # -- structured completion --------------------------------------------
    async def structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        trace: Trace | None = None,
        name: str = "structured",
        max_retries: int | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Return a validated instance of ``schema``.

        Implements the *validate-and-retry* guardrail: on a Pydantic
        ``ValidationError`` the raw output plus the error text are fed back to the
        model, up to ``max_retries`` extra attempts.
        """
        client = self._ensure()
        retries = self.settings.max_retries_per_agent if max_retries is None else max_retries
        tool_name = f"emit_{schema.__name__.lower()}"
        tool = {
            "name": tool_name,
            "description": f"Emit a valid {schema.__name__} object. All fields are required.",
            "input_schema": _json_schema(schema),
        }

        messages: list[dict[str, Any]] = [{"role": "user", "content": user}]
        last_error = ""

        for attempt in range(retries + 1):
            with span(
                trace, f"{name}#{attempt}", "llm", model=self.model, schema=schema.__name__
            ) as sp:
                resp = await client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens or self.settings.max_tokens,
                    temperature=self.settings.temperature,
                    system=system,
                    tools=[tool],
                    tool_choice={"type": "tool", "name": tool_name},
                    messages=messages,
                )
                sp.update(self._account(trace, resp.usage))
                payload: dict[str, Any] | None = None
                for block in resp.content:
                    if getattr(block, "type", "") == "tool_use":
                        payload = dict(block.input)  # type: ignore[arg-type]
                        break
                if payload is None:
                    last_error = "model returned no tool_use block"
                    sp["_ok"] = False
                    sp["validation_error"] = last_error
                else:
                    try:
                        return schema.model_validate(payload)
                    except ValidationError as exc:
                        last_error = exc.json(indent=None)
                        sp["_ok"] = False
                        sp["validation_error"] = last_error[:400]

            messages = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": json.dumps(payload or {}, default=str)},
                {
                    "role": "user",
                    "content": (
                        "Your previous output failed schema validation with these errors:\n"
                        f"{last_error}\n\nEmit a corrected object that satisfies every constraint."
                    ),
                },
            ]

        raise ValueError(f"{schema.__name__} validation failed after {retries + 1} attempts: {last_error}")


def _json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """Anthropic tool schemas must be self-contained: inline all ``$ref``/``$defs``."""
    raw = schema.model_json_schema()
    defs = raw.pop("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                key = str(node["$ref"]).rsplit("/", 1)[-1]
                merged = {k: v for k, v in node.items() if k != "$ref"}
                return resolve({**defs.get(key, {}), **merged})
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        return node

    resolved = resolve(raw)
    resolved.setdefault("type", "object")
    return resolved


async def gather_limited(coros: Sequence[Any], limit: int = 4) -> list[Any]:
    """Run coroutines concurrently with a cap, returning results/exceptions in order."""
    sem = asyncio.Semaphore(limit)

    async def runner(coro: Any) -> Any:
        async with sem:
            return await coro

    return await asyncio.gather(*(runner(c) for c in coros), return_exceptions=True)


_SHARED: dict[str, ClaudeClient] = {}


def get_llm(model: str | None = None) -> ClaudeClient:
    key = model or get_settings().model
    if key not in _SHARED:
        _SHARED[key] = ClaudeClient(key)
        log_event(log, "llm_client_created", model=key)
    return _SHARED[key]
