"""Orchestrator Agent — final synthesis, guardrail gate and persistence.

MCP scope: ``filesystem`` (save the report) and ``memory`` (session knowledge graph).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from core.telemetry import get_logger, log_event, span
from graph.builder import link_answer, register_question
from guardrails.permissions import PermissionBroker
from guardrails.policy import run_guardrails
from guardrails.schemas import Citation, CriticVerdict, FinalAnswer
from agents import prompts
from agents.base import merge_findings, rank_facts
from agents.state import RunContext

log = get_logger("agents.orchestrator")


class OrchestratorAgent:
    name = "orchestrator"
    role = "Orchestrator"

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.broker = PermissionBroker(agent=self.name, hub=ctx.hub)

    async def run(self, findings: list[dict[str, Any]], critic: dict[str, Any]) -> dict[str, Any]:
        with span(self.ctx.trace, self.name, "agent", agent=self.name) as sp:
            verdict = _verdict(critic)
            answer = await self._synthesize(findings, verdict)
            report = self._check(answer)

            if not report.passed and self.ctx.llm_available and self.ctx.budget.take_retry(self.name):
                answer = await self._repair(findings, verdict, answer, report.feedback())
                report = self._check(answer)

            if not report.passed:
                answer = self._downgrade(answer, report)

            self._write_graph(answer)
            await self._persist(answer, report.to_dict())

            sp["verdict"] = verdict.verdict
            sp["guardrail_score"] = report.score
            sp["_ok"] = report.passed

        self.ctx.permission_reports.append(self.broker.report())
        return {
            "final": answer.model_dump(),
            "guardrails": {
                **report.to_dict(),
                "critic_verdict": verdict.verdict,
                "permissions": self.ctx.permission_reports,
                "budget": self.ctx.budget.report(),
                "mcp_servers_failed": dict(self.ctx.hub.failed_servers),
            },
            "agents_run": [self.name],
        }

    # ------------------------------------------------------------------ #
    async def _synthesize(self, findings: list[dict[str, Any]],
                          verdict: CriticVerdict) -> FinalAnswer:
        if not self.ctx.llm_available:
            return self._deterministic_answer(findings, verdict)
        try:
            return await self.ctx.llm.structured(
                system=prompts.ORCHESTRATOR.format(shared_rules=prompts.SHARED_RULES),
                user=self._user_prompt(findings, verdict),
                schema=FinalAnswer,
                trace=self.ctx.trace,
                name="orchestrator.synthesize",
                max_tokens=2048,
            )
        except Exception as exc:  # noqa: BLE001
            log_event(log, "synthesis_failed", error=str(exc)[:200])
            return self._deterministic_answer(findings, verdict)

    def _user_prompt(self, findings: list[dict[str, Any]], verdict: CriticVerdict) -> str:
        return (
            f"USER QUESTION: {self.ctx.question}\n\n"
            f"KNOWLEDGE-GRAPH EVIDENCE (the only permissible source of facts)\n"
            f"{self.ctx.kg.evidence_bundle()}\n\n"
            f"AGENT FINDINGS\n{merge_findings(findings)}\n\n"
            f"CRITIC VERDICT: {verdict.verdict}\n"
            f"CRITIC ASSESSMENT: {verdict.grounding_assessment}\n"
            f"CLAIMS TO REMOVE: {json.dumps(verdict.removed_claims[:10], default=str)}\n"
            f"REQUIRED FIXES: {json.dumps(verdict.required_fixes[:10], default=str)}\n"
        )

    async def _repair(self, findings: list[dict[str, Any]], verdict: CriticVerdict,
                      previous: FinalAnswer, feedback: str) -> FinalAnswer:
        try:
            return await self.ctx.llm.structured(
                system=prompts.ORCHESTRATOR.format(shared_rules=prompts.SHARED_RULES),
                user=(
                    self._user_prompt(findings, verdict)
                    + f"\n\nPREVIOUS ANSWER:\n{previous.answer}\n\n"
                    + prompts.REPAIR.format(feedback=feedback)
                ),
                schema=FinalAnswer,
                trace=self.ctx.trace,
                name="orchestrator.repair",
            )
        except Exception as exc:  # noqa: BLE001
            log_event(log, "repair_failed", error=str(exc)[:200])
            return previous

    def _deterministic_answer(self, findings: list[dict[str, Any]],
                              verdict: CriticVerdict) -> FinalAnswer:
        facts = self.ctx.kg.facts()
        if not facts:
            return FinalAnswer(
                answer=(
                    "I could not retrieve any evidence for this question. The MCP data "
                    "servers returned no usable results, so I will not guess. "
                    f"Servers that failed: {', '.join(self.ctx.hub.failed_servers) or 'none'}."
                ),
                confidence=0.05,
                caveats=["no MCP evidence retrieved"],
            )
        ranked = rank_facts(facts, self.ctx.question)
        lines = [f.to_sentence() for f in ranked[:10]]
        sources = sorted({f.source_name for f in ranked[:10] if f.source_name})
        best = _best_finding(findings)
        head = best[:600] if best else (
            f"Evidence retrieved from {', '.join(sources)} for: {self.ctx.question}"
        )
        body = f"{head}\n\nSupporting evidence (from MCP tool results only):\n- " + "\n- ".join(lines)
        return FinalAnswer(
            answer=body[:5900],
            key_points=lines[:8],
            citations=[
                Citation(label=f.source_name, url=f.source_url,
                         kind="computation" if f.mcp_server == "astro_compute" else "dataset")
                for f in _unique_sources(ranked)
            ][:10],
            confidence=0.5 if verdict.verdict == "accept" else 0.35,
            caveats=["deterministic synthesis (no LLM configured)"] if not self.ctx.llm_available else [],
            data_freshness=_freshness(facts),
        )

    # ------------------------------------------------------------------ #
    def _check(self, answer: FinalAnswer):
        return run_guardrails(
            answer.answer,
            self.ctx.kg,
            answer.citations,
            injection_warnings=self.ctx.injection_warnings,
            require_numeric=True,
            known_sources=self.ctx.known_sources,
        )

    def _downgrade(self, answer: FinalAnswer, report: Any) -> FinalAnswer:
        """Last-resort safety net: annotate rather than emit an ungrounded answer."""
        answer.confidence = min(answer.confidence, 0.3)
        notes = [f"Automated guardrail check did not fully pass: {'; '.join(report.failures)}"]
        if report.numeric.unverified:
            notes.append(
                "Unverified numbers: " + ", ".join(f"{n:g}" for n in report.numeric.unverified[:8])
            )
        answer.caveats = (answer.caveats + notes)[:6]
        return answer

    # ------------------------------------------------------------------ #
    def _write_graph(self, answer: FinalAnswer) -> None:
        question_id = register_question(self.ctx.kg, self.ctx.question, self.ctx.session_id)
        claim_ids = []
        for point in [answer.answer[:400], *answer.key_points[:5]]:
            supporting = [f.id for f in self.ctx.kg.facts()[:20]]
            claim_ids.append(
                self.ctx.kg.add_claim(point, self.name, supporting, grounded=True)
            )
        link_answer(self.ctx.kg, question_id, claim_ids)

    async def _persist(self, answer: FinalAnswer, guardrails: dict[str, Any]) -> None:
        payload = {
            "session_id": self.ctx.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "question": self.ctx.question,
            "answer": answer.model_dump(),
            "guardrails": guardrails,
            "graph_stats": self.ctx.kg.stats(),
            "trace": self.ctx.trace.summary(),
        }
        if self.broker.can("filesystem") and self.ctx.budget.allows("orchestrator:save"):
            try:
                await self.broker.call(
                    "filesystem", "write_file",
                    {"path": f"reports/{self.ctx.session_id}.json",
                     "content": json.dumps(payload, indent=2, default=str)},
                )
                self.ctx.budget.record_tool_call()
            except Exception as exc:  # noqa: BLE001
                log_event(log, "filesystem_save_failed", error=str(exc)[:160])

        # Always keep a local copy, even when the Node servers are unavailable.
        try:
            path = self.ctx.hub.settings.fs_root / "reports" / f"{self.ctx.session_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            self.ctx.kg.save(
                self.ctx.hub.settings.fs_root / "graphs" / f"{self.ctx.session_id}.json"
            )
        except Exception as exc:  # noqa: BLE001
            log_event(log, "local_save_failed", error=str(exc)[:160])

        if self.broker.can("memory") and self.ctx.budget.allows("orchestrator:memory"):
            try:
                await self.broker.call(
                    "memory", "create_entities",
                    {"entities": [{
                        "name": f"question:{self.ctx.session_id}",
                        "entityType": "ResearchSession",
                        "observations": [
                            self.ctx.question[:400],
                            f"confidence: {answer.confidence:.2f}",
                            f"sources: {', '.join(c.label for c in answer.citations[:5])}",
                        ],
                    }]},
                )
                self.ctx.budget.record_tool_call()
            except Exception as exc:  # noqa: BLE001
                log_event(log, "memory_save_failed", error=str(exc)[:160])


def _verdict(critic: dict[str, Any]) -> CriticVerdict:
    try:
        return CriticVerdict.model_validate(
            {k: v for k, v in critic.items()
             if k in CriticVerdict.model_fields}
        )
    except Exception:  # noqa: BLE001
        return CriticVerdict(verdict="revise", grounding_assessment="critic output unavailable")


def _best_finding(findings: list[dict[str, Any]]) -> str:
    """Lead with the most confident agent that actually retrieved evidence."""
    usable = [
        f for f in findings
        if str(f.get("summary", "")).strip() and f.get("key_facts")
    ]
    if not usable:
        return ""
    best = max(usable, key=lambda f: (float(f.get("confidence", 0.0)), len(f.get("key_facts", []))))
    return str(best.get("summary", "")).strip()


def _unique_sources(facts: list[Any]) -> list[Any]:
    seen, out = set(), []
    for fact in facts:
        key = f"{fact.source_name}|{fact.source_url}"
        if fact.source_name and key not in seen:
            seen.add(key)
            out.append(fact)
    return out


def _freshness(facts: list[Any]) -> str:
    stamps = sorted({f.retrieved_at for f in facts if f.retrieved_at})
    return f"data retrieved {stamps[-1]}" if stamps else ""
