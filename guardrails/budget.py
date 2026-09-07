"""Per-session budgets: tool calls, retries, wall-clock and spend.

The orchestrator checks the budget before dispatching each agent, and the MCP hub
enforces the tool-call ceiling independently, so a runaway model cannot loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.config import get_settings


class BudgetExceeded(RuntimeError):
    """A hard session limit was hit."""


@dataclass
class SessionBudget:
    max_tool_calls: int = 0
    max_retries_per_agent: int = 0
    max_seconds: float = 0.0
    max_usd: float = 0.50

    tool_calls: int = 0
    retries: dict[str, int] = field(default_factory=dict)
    usd_spent: float = 0.0
    started_at: float = field(default_factory=time.perf_counter)
    events: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        settings = get_settings()
        self.max_tool_calls = self.max_tool_calls or settings.max_tool_calls_per_session
        self.max_retries_per_agent = self.max_retries_per_agent or settings.max_retries_per_agent
        self.max_seconds = self.max_seconds or settings.max_agent_seconds

    # ---- checks ----------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def tool_calls_left(self) -> int:
        return max(0, self.max_tool_calls - self.tool_calls)

    def check(self, what: str = "operation") -> None:
        if self.tool_calls >= self.max_tool_calls:
            raise BudgetExceeded(f"{what}: tool-call budget {self.max_tool_calls} exhausted")
        if self.elapsed > self.max_seconds:
            raise BudgetExceeded(f"{what}: time budget {self.max_seconds:.0f}s exhausted")
        if self.usd_spent > self.max_usd:
            raise BudgetExceeded(f"{what}: spend budget ${self.max_usd:.2f} exhausted")

    def allows(self, what: str = "operation") -> bool:
        try:
            self.check(what)
            return True
        except BudgetExceeded as exc:
            self.events.append(str(exc))
            return False

    # ---- accounting ------------------------------------------------------
    def record_tool_call(self, n: int = 1) -> None:
        self.tool_calls += n

    def record_spend(self, usd: float) -> None:
        self.usd_spent += usd

    def take_retry(self, agent: str) -> bool:
        used = self.retries.get(agent, 0)
        if used >= self.max_retries_per_agent:
            self.events.append(f"{agent}: retry limit {self.max_retries_per_agent} reached")
            return False
        self.retries[agent] = used + 1
        return True

    def report(self) -> dict[str, Any]:
        return {
            "tool_calls": self.tool_calls,
            "max_tool_calls": self.max_tool_calls,
            "tool_calls_left": self.tool_calls_left,
            "retries": dict(self.retries),
            "max_retries_per_agent": self.max_retries_per_agent,
            "elapsed_seconds": round(self.elapsed, 2),
            "max_seconds": self.max_seconds,
            "usd_spent": round(self.usd_spent, 6),
            "max_usd": self.max_usd,
            "events": self.events,
        }
