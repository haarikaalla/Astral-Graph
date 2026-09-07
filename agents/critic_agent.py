"""Critic Agent — grounds or rejects every claim against the knowledge graph.

The Critic is deliberately split in two:

* a **deterministic layer** (Python) that computes grounding, numeric verification
  and citation status — this cannot hallucinate and is authoritative;
* an **LLM layer** that reads the deterministic report plus the evidence and issues
  a claim-by-claim judgement and repair instructions.

MCP scope: ``astro_compute`` (re-verify numbers) and ``memory`` (persist verdicts).
"""

from __future__ import annotations

from typing import Any

from core.telemetry import get_logger, log_event, span
from guardrails.permissions import PermissionBroker
from guardrails.policy import run_guardrails
from guardrails.schemas import ClaimJudgement, CriticVerdict
from agents import prompts
from agents.base import merge_findings
from agents.state import RunContext

log = get_logger("agents.critic")


class CriticAgent:
    name = "critic_agent"
    role = "Critic Agent"

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.broker = PermissionBroker(agent=self.name, hub=ctx.hub)

    async def run(self, findings: list[dict[str, Any]]) -> dict[str, Any]:
        with span(self.ctx.trace, self.name, "agent", agent=self.name) as sp:
            combined = "\n\n".join(f.get("summary", "") for f in findings)
            citations = [
                c for finding in findings for c in _citations(finding)
            ]
            report = run_guardrails(
                combined,
                self.ctx.kg,
                citations,
                injection_warnings=self.ctx.injection_warnings,
                require_numeric=True,
                known_sources=self.ctx.known_sources,
            )
            sp["guardrail_score"] = report.score
            sp["grounding"] = round(report.grounding.score, 3)

            verdict = await self._judge(findings, report.to_dict())
            await self._verify_numbers(report)
            await self._persist(verdict, report.score)

            sp["verdict"] = verdict.verdict
            sp["_ok"] = verdict.verdict != "reject"

        self.ctx.permission_reports.append(self.broker.report())
        return {
            "critic": {
                **verdict.model_dump(),
                "deterministic_report": report.to_dict(),
                "guardrail_feedback": report.feedback(),
            },
            "agents_run": [self.name],
            "guardrail_reports": {self.name: report.to_dict()},
        }

    # ------------------------------------------------------------------ #
    async def _judge(self, findings: list[dict[str, Any]],
                     deterministic: dict[str, Any]) -> CriticVerdict:
        fallback = self._deterministic_verdict(deterministic)
        if not self.ctx.llm_available:
            return fallback
        try:
            return await self.ctx.llm.structured(
                system=prompts.CRITIC,
                user=(
                    f"USER QUESTION: {self.ctx.question}\n\n"
                    f"KNOWLEDGE-GRAPH EVIDENCE\n{self.ctx.kg.evidence_bundle()}\n\n"
                    f"AGENT FINDINGS\n{merge_findings(findings)}\n\n"
                    f"DETERMINISTIC GUARDRAIL REPORT (authoritative)\n{deterministic}\n"
                ),
                schema=CriticVerdict,
                trace=self.ctx.trace,
                name="critic",
            )
        except Exception as exc:  # noqa: BLE001
            log_event(log, "critic_llm_failed", error=str(exc)[:200])
            return fallback

    def _deterministic_verdict(self, deterministic: dict[str, Any]) -> CriticVerdict:
        grounding = deterministic.get("grounding", {})
        numeric = deterministic.get("numeric", {})
        unsupported = grounding.get("unsupported_sentences", []) or []
        unverified = numeric.get("unverified_numbers", []) or []

        if deterministic.get("passed"):
            verdict = "accept"
        elif grounding.get("fact_count", 0) == 0:
            verdict = "reject"
        else:
            verdict = "revise"

        judgements = [
            ClaimJudgement(claim=str(s)[:600], status="unsupported",
                           evidence="not matched to any knowledge-graph fact")
            for s in unsupported[:15]
        ]
        fixes = []
        if unverified:
            fixes.append(
                "Remove or re-derive these numbers via astro_compute: "
                + ", ".join(f"{n:g}" for n in unverified[:10])
            )
        if not deterministic.get("citations", {}).get("ok", True):
            fixes.append("Add a citation from the evidence for every literature claim.")
        return CriticVerdict(
            verdict=verdict,  # type: ignore[arg-type]
            grounding_assessment=(
                f"Deterministic check: grounding={grounding.get('grounding_score')}, "
                f"numeric verification={numeric.get('verification_rate')}, "
                f"failures={deterministic.get('failures')}"
            )[:1200],
            judgements=judgements,
            removed_claims=[str(s)[:600] for s in unsupported[:15]],
            required_fixes=fixes[:10],
            confidence=float(deterministic.get("guardrail_score", 0.0)),
        )

    async def _verify_numbers(self, report: Any) -> None:
        """Spot-check the largest unverified number against the deterministic server."""
        if not report.numeric.unverified or not self.broker.can("astro_compute"):
            return
        if not self.ctx.budget.allows("critic:verify"):
            return
        candidate = max(report.numeric.unverified, key=abs)
        expected = max(
            (v for _, v in self.ctx.kg.numeric_facts()),
            key=lambda v: -abs(abs(v) - abs(candidate)),
            default=None,
        )
        if expected is None:
            return
        try:
            await self.broker.call(
                "astro_compute", "compare_values",
                {"actual": float(candidate), "expected": float(expected),
                 "tolerance_percent": 5.0},
            )
            self.ctx.budget.record_tool_call()
        except Exception as exc:  # noqa: BLE001
            log_event(log, "critic_verify_failed", error=str(exc)[:160])

    async def _persist(self, verdict: CriticVerdict, score: float) -> None:
        """Write the verdict into the official Memory MCP server (knowledge graph)."""
        if not self.broker.can("memory") or not self.ctx.budget.allows("critic:memory"):
            return
        try:
            await self.broker.call(
                "memory",
                "create_entities",
                {
                    "entities": [
                        {
                            "name": f"verdict:{self.ctx.session_id}",
                            "entityType": "CriticVerdict",
                            "observations": [
                                f"question: {self.ctx.question[:300]}",
                                f"verdict: {verdict.verdict}",
                                f"guardrail_score: {score:.3f}",
                                f"removed_claims: {len(verdict.removed_claims)}",
                            ],
                        }
                    ]
                },
            )
            self.ctx.budget.record_tool_call()
        except Exception as exc:  # noqa: BLE001
            log_event(log, "critic_memory_failed", error=str(exc)[:160])


def _citations(finding: dict[str, Any]) -> list[Any]:
    from guardrails.schemas import Citation

    out = []
    for citation in finding.get("citations", []) or []:
        try:
            out.append(Citation.model_validate(citation))
        except Exception:  # noqa: BLE001
            continue
    return out
