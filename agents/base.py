"""Base class for every AstralGraph domain agent.

Lifecycle per agent: **plan → execute (MCP only) → ingest into KG → synthesise →
guardrail check → optional repair**.

Each stage degrades gracefully: without an Anthropic key the agent uses a
deterministic heuristic planner and a template synthesiser, so the whole pipeline —
including MCP calls, the knowledge graph and the guardrails — remains testable and
demonstrable offline.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any, Sequence

from core.mcp_client import ToolResult
from core.telemetry import get_logger, log_event, span
from graph.builder import ingest_tool_result
from graph.schema import Fact
from guardrails.permissions import PermissionBroker
from guardrails.policy import GuardrailReport, run_guardrails
from guardrails.schemas import AgentFinding, Citation, Intent, ToolCall, ToolPlan
from agents import prompts
from agents.state import RunContext

log = get_logger("agents")


class BaseAgent:
    """Shared machinery. Subclasses provide a name, a role and a heuristic plan."""

    name: str = "base_agent"
    role: str = "Base Agent"
    max_calls: int = 3
    requires_citations: bool = False

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.broker = PermissionBroker(agent=self.name, hub=ctx.hub)
        self.results: list[ToolResult] = []
        self.facts: list[Fact] = []

    # ------------------------------------------------------------------ #
    # Planning
    # ------------------------------------------------------------------ #
    def heuristic_plan(self, intent: Intent) -> list[ToolCall]:
        """Deterministic fallback plan. Subclasses override."""
        return []

    def followups(self, intent: Intent) -> list[ToolCall]:
        """Second-round calls derived from data already retrieved.

        Used to chain deterministic ``astro_compute`` calls onto live catalogue
        values, so quantitative answers never depend on model arithmetic.
        """
        return []

    async def plan(self, intent: Intent) -> ToolPlan:
        heuristic = ToolPlan(calls=self.heuristic_plan(intent), reasoning="heuristic plan")
        if not self.ctx.llm_available or not self.broker.servers:
            return heuristic

        try:
            catalogue = await self.broker.catalogue()
            system = prompts.PLANNER.format(
                agent=self.role,
                catalogue=catalogue,
                max_calls=self.max_calls,
                today=date.today().isoformat(),
                question=self.ctx.question,
                intent=intent.model_dump_json(),
            )
            plan = await self.ctx.llm.structured(
                system=system,
                user=f"Plan the MCP calls for: {intent.normalized_question}",
                schema=ToolPlan,
                trace=self.ctx.trace,
                name=f"{self.name}.plan",
            )
        except Exception as exc:  # noqa: BLE001
            log_event(log, "plan_failed", agent=self.name, error=str(exc)[:200])
            return heuristic

        kept, rejected = self.broker.sanitise(plan.calls[: self.max_calls])
        if rejected:
            log_event(log, "plan_sanitised", agent=self.name, rejected=rejected)
        if not kept:
            return heuristic
        return ToolPlan(calls=kept, reasoning=plan.reasoning)

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    async def execute(self, plan: ToolPlan) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in plan.calls:
            if not self.ctx.budget.allows(f"{self.name}:{call.qualified}"):
                break
            try:
                result = await self.broker.call(call.server, call.tool, call.arguments)
            except Exception as exc:  # noqa: BLE001
                log_event(log, "tool_call_failed", agent=self.name, tool=call.qualified,
                          error=str(exc)[:200])
                continue
            self.ctx.budget.record_tool_call()
            results.append(result)
            if result.warnings:
                self.ctx.injection_warnings.extend(
                    f"{call.qualified}: {w}" for w in result.warnings
                )
            if result.ok:
                facts = ingest_tool_result(self.ctx.kg, result, self.name)
                self.facts.extend(facts)
                self._record_sources(result)
        self.results = results
        return results

    def _record_sources(self, result: ToolResult) -> None:
        payload = result.data
        if isinstance(payload, dict):
            source = payload.get("source") or {}
            for key in ("name", "url"):
                value = source.get(key)
                if value:
                    self.ctx.known_sources.append(str(value))
            data = payload.get("data")
            if isinstance(data, dict):
                for item in (data.get("results") or [])[:20]:
                    if isinstance(item, dict):
                        for key in ("title", "url", "source", "citation"):
                            if item.get(key):
                                self.ctx.known_sources.append(str(item[key]))

    # ------------------------------------------------------------------ #
    # Evidence rendering
    # ------------------------------------------------------------------ #
    def evidence_block(self, limit: int = 9000) -> str:
        blocks: list[str] = []
        for result in self.results:
            header = f"### TOOL {result.server}.{result.tool} — {'OK' if result.ok else 'FAILED'}"
            blocks.append(f"{header}\n{result.brief(limit // max(1, len(self.results)))}")
        text = "\n\n".join(blocks)
        return text[:limit] or "(no tool evidence collected)"

    def fact_lines(self, limit: int = 40) -> str:
        return "\n".join(f"- {f.to_sentence()}  {f.citation()}" for f in self.facts[:limit])

    def _auto_citations(self) -> list[Citation]:
        seen: set[str] = set()
        citations: list[Citation] = []
        for fact in self.facts:
            key = f"{fact.source_name}|{fact.source_url}"
            if fact.source_name and key not in seen:
                seen.add(key)
                citations.append(
                    Citation(
                        label=fact.source_name,
                        url=fact.source_url,
                        kind="computation" if fact.mcp_server == "astro_compute" else "dataset",
                    )
                )
        return citations[:10]

    # ------------------------------------------------------------------ #
    # Synthesis
    # ------------------------------------------------------------------ #
    def deterministic_finding(self, intent: Intent) -> AgentFinding:
        """Template answer built purely from graph facts (no LLM)."""
        successful = [r for r in self.results if r.ok]
        failed = [r for r in self.results if not r.ok]
        if not self.facts:
            summary = (
                f"{self.role}: no usable evidence was retrieved for "
                f"'{intent.normalized_question}'."
            )
            if failed:
                summary += " Tool failures: " + "; ".join(
                    f"{r.server}.{r.tool}: {r.error[:120]}" for r in failed[:3]
                )
            return AgentFinding(
                agent=self.name, summary=summary, confidence=0.1,
                limitations="no MCP evidence available",
                tools_used=[f"{r.server}.{r.tool}" for r in self.results],
            )

        lines = [f.to_sentence() for f in rank_facts(self.facts, self.ctx.question, limit=12)]
        summary = (
            f"{self.role} retrieved {len(self.facts)} facts from "
            f"{len(successful)} MCP tool call(s):\n- " + "\n- ".join(lines)
        )
        numbers = [v for v in (f.numeric_value for f in self.facts[:30]) if v is not None]
        return AgentFinding(
            agent=self.name,
            summary=summary[:2400],
            key_facts=lines[:15],
            numbers_used=numbers[:30],
            citations=self._auto_citations(),
            confidence=0.55 if successful else 0.2,
            limitations="deterministic summary (no LLM configured)" if not self.ctx.llm_available else "",
            tools_used=[f"{r.server}.{r.tool}" for r in self.results],
        )

    def system_prompt(self) -> str:
        return prompts.DOMAIN_AGENT.format(agent=self.role, shared_rules=prompts.SHARED_RULES)

    def user_prompt(self, intent: Intent) -> str:
        return (
            f"USER QUESTION: {self.ctx.question}\n"
            f"NORMALIZED: {intent.normalized_question}\n"
            f"ENTITIES: {', '.join(intent.entities) or 'none'}\n\n"
            f"TOOL EVIDENCE\n{self.evidence_block()}\n\n"
            f"KNOWLEDGE-GRAPH FACTS EXTRACTED FROM THAT EVIDENCE\n{self.fact_lines()}\n"
        )

    async def synthesize(self, intent: Intent) -> AgentFinding:
        if not self.ctx.llm_available:
            return self.deterministic_finding(intent)
        try:
            finding = await self.ctx.llm.structured(
                system=self.system_prompt(),
                user=self.user_prompt(intent),
                schema=AgentFinding,
                trace=self.ctx.trace,
                name=f"{self.name}.synthesize",
            )
        except Exception as exc:  # noqa: BLE001
            log_event(log, "synthesis_failed", agent=self.name, error=str(exc)[:200])
            return self.deterministic_finding(intent)
        finding.agent = self.name
        if not finding.citations:
            finding.citations = self._auto_citations()
        finding.tools_used = [f"{r.server}.{r.tool}" for r in self.results]
        return finding

    # ------------------------------------------------------------------ #
    # Guardrails + repair
    # ------------------------------------------------------------------ #
    def check(self, finding: AgentFinding) -> GuardrailReport:
        return run_guardrails(
            finding.summary,
            self.ctx.kg,
            finding.citations,
            extra_facts=self.facts,
            injection_warnings=self.ctx.injection_warnings,
            require_numeric=True,
            known_sources=self.ctx.known_sources,
        )

    async def repair(self, finding: AgentFinding, report: GuardrailReport,
                     intent: Intent) -> AgentFinding:
        if report.passed or not self.ctx.llm_available:
            return finding
        if not self.ctx.budget.take_retry(self.name):
            finding.limitations = (finding.limitations + " | guardrail retry budget exhausted").strip(" |")
            finding.confidence = min(finding.confidence, 0.35)
            return finding
        try:
            repaired = await self.ctx.llm.structured(
                system=self.system_prompt(),
                user=(
                    self.user_prompt(intent)
                    + "\n\nPREVIOUS ANSWER:\n"
                    + finding.summary
                    + "\n\n"
                    + prompts.REPAIR.format(feedback=report.feedback())
                ),
                schema=AgentFinding,
                trace=self.ctx.trace,
                name=f"{self.name}.repair",
            )
        except Exception as exc:  # noqa: BLE001
            log_event(log, "repair_failed", agent=self.name, error=str(exc)[:200])
            return finding
        repaired.agent = self.name
        repaired.tools_used = finding.tools_used
        if not repaired.citations:
            repaired.citations = finding.citations or self._auto_citations()
        return repaired

    # ------------------------------------------------------------------ #
    # Entry point used by the LangGraph node
    # ------------------------------------------------------------------ #
    async def run(self, intent: Intent) -> dict[str, Any]:
        with span(self.ctx.trace, self.name, "agent", agent=self.name) as sp:
            plan = await self.plan(intent)
            sp["planned_calls"] = [c.qualified for c in plan.calls]
            first_round = await self.execute(plan)

            extra, _ = self.broker.sanitise(self.followups(intent))
            if extra:
                follow_results = await self.execute(ToolPlan(calls=extra, reasoning="followup"))
                self.results = list(first_round) + list(follow_results)
                sp["followup_calls"] = [c.qualified for c in extra]

            finding = await self.synthesize(intent)
            report = self.check(finding)
            if not report.passed:
                finding = await self.repair(finding, report, intent)
                report = self.check(finding)
            sp["facts"] = len(self.facts)
            sp["guardrail_score"] = report.score
            sp["_ok"] = bool(self.facts) or bool(self.results)

        self.ctx.permission_reports.append(self.broker.report())
        return {
            "findings": [
                {
                    **finding.model_dump(),
                    "guardrail": report.to_dict(),
                    "tool_results": [
                        {"server": r.server, "tool": r.tool, "ok": r.ok, "error": r.error[:200]}
                        for r in self.results
                    ],
                }
            ],
            "agents_run": [self.name],
            "guardrail_reports": {self.name: report.to_dict()},
        }


def json_preview(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value, indent=2, default=str)
    return text[:limit] + ("..." if len(text) > limit else "")


_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how in into is it its many much of on"
    " or right that the their there these this to was were what when where which who why"
    " will with now today currently me my you your".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.\-]*")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def rank_facts(facts: Sequence[Fact], question: str, *, limit: int | None = None) -> list[Fact]:
    """Order facts by relevance to the question so summaries lead with the answer.

    Deterministic and cheap: lexical overlap plus small priors that favour computed
    results (which directly answer 'what is X' questions) and numeric facts.
    """
    wanted = _tokens(question)
    scored: list[tuple[float, int, Fact]] = []
    for index, fact in enumerate(facts):
        text = f"{fact.subject} {fact.predicate} {fact.value} {fact.unit or ''}"
        overlap = wanted & _tokens(text)
        score = 2.0 * len(overlap)
        if fact.mcp_server == "astro_compute":
            score += 1.5
        if fact.numeric_value is not None:
            score += 0.5
        if any(word in fact.predicate.lower() for word in ("total", "count", "number", "energy")):
            score += 0.75
        if fact.predicate.lower().endswith(("_planets", "_objects", "_names", "_list")):
            # Roll-up facts name the matches, which is usually what was asked for.
            score += 2.5
        scored.append((score, -index, fact))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    ordered = [fact for _, _, fact in scored]
    return ordered[:limit] if limit else ordered


def merge_findings(findings: Sequence[dict[str, Any]]) -> str:
    blocks = []
    for finding in findings:
        citations = "; ".join(
            f"{c.get('label', '')} {c.get('url', '')}".strip()
            for c in finding.get("citations", [])
        )
        blocks.append(
            f"### {finding.get('agent')} (confidence {finding.get('confidence', 0):.2f})\n"
            f"{finding.get('summary', '')}\n"
            f"Key facts: {'; '.join(finding.get('key_facts', [])[:10]) or 'none'}\n"
            f"Citations: {citations or 'none'}\n"
            f"Limitations: {finding.get('limitations') or 'none'}"
        )
    return "\n\n".join(blocks) or "(no findings)"
