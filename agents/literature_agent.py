"""Literature / RAG Agent.

MCP scope: ``rag`` (local Chroma index) and ``brave_search`` (open web, optional).
Citations are mandatory for this agent: :class:`guardrails.citations.CitationReport`
is checked with ``strict=True`` and a failure forces a repair pass.
"""

from __future__ import annotations

from guardrails.schemas import AgentFinding, Intent, ToolCall
from agents import prompts
from agents.base import BaseAgent

NEWS_WORDS = ("latest", "recent news", "this week", "announced", "breaking", "news")


class LiteratureAgent(BaseAgent):
    name = "literature_agent"
    role = "Literature & RAG Agent"
    max_calls = 3
    requires_citations = True

    def heuristic_plan(self, intent: Intent) -> list[ToolCall]:
        question = intent.normalized_question or self.ctx.question
        lowered = question.lower()
        calls: list[ToolCall] = [
            ToolCall(server="rag", tool="literature_search",
                     arguments={"query": question, "top_k": 6},
                     rationale="semantic retrieval over the indexed corpus")
        ]
        if intent.entities:
            calls.append(
                ToolCall(server="rag", tool="literature_search",
                         arguments={"query": " ".join(intent.entities[:3]), "top_k": 4},
                         rationale="entity-focused retrieval")
            )
        if self.broker.can("brave_search") and any(w in lowered for w in NEWS_WORDS):
            calls.append(
                ToolCall(server="brave_search", tool="brave_web_search",
                         arguments={"query": question, "count": 5},
                         rationale="recent news beyond the local index")
            )
        return calls[: self.max_calls]

    def system_prompt(self) -> str:
        return prompts.LITERATURE_AGENT.format(shared_rules=prompts.SHARED_RULES)

    def user_prompt(self, intent: Intent) -> str:
        blocks = []
        for result in self.results:
            if not result.ok:
                continue
            data = (result.data or {}).get("data") if isinstance(result.data, dict) else None
            if isinstance(data, dict) and data.get("context_block"):
                blocks.append(str(data["context_block"]))
            else:
                blocks.append(result.brief(2500))
        evidence = "\n\n".join(blocks) or "(no passages retrieved)"
        return (
            f"USER QUESTION: {self.ctx.question}\n"
            f"NORMALIZED: {intent.normalized_question}\n\n"
            f"RETRIEVED PASSAGES (cite these as [1], [2], ...)\n{evidence}\n"
        )

    def deterministic_finding(self, intent: Intent) -> AgentFinding:
        finding = super().deterministic_finding(intent)
        excerpts = [f for f in self.facts if f.predicate == "excerpt"]
        if excerpts:
            lines = [f"[{i}] {f.subject}: {str(f.value)[:280]}" for i, f in enumerate(excerpts[:5], 1)]
            finding.summary = (
                "Retrieved literature passages relevant to "
                f"'{intent.normalized_question}':\n" + "\n".join(lines)
            )[:2400]
            finding.key_facts = [line[:200] for line in lines]
            finding.confidence = 0.5
        return finding
