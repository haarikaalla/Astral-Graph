"""Pydantic contracts for every LLM boundary.

Nothing produced by a model enters the pipeline without passing one of these
models. Validation failures trigger the retry loop in
:meth:`core.llm.ClaudeClient.structured`.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Domain = Literal["neo", "exoplanet", "events", "literature", "general"]


class Intent(BaseModel):
    """Structured reading of the user's question."""

    normalized_question: str = Field(..., min_length=3, max_length=600)
    domains: list[Domain] = Field(..., min_length=1, max_length=5)
    entities: list[str] = Field(default_factory=list, max_length=12)
    sub_questions: list[str] = Field(default_factory=list, max_length=6)
    time_reference: str = Field(default="", max_length=120)
    needs_computation: bool = False
    needs_literature: bool = False
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    reasoning: str = Field(default="", max_length=800)

    @field_validator("domains")
    @classmethod
    def _dedupe(cls, value: list[str]) -> list[str]:
        seen: list[str] = []
        for item in value:
            if item not in seen:
                seen.append(item)
        return seen  # type: ignore[return-value]


class ToolCall(BaseModel):
    """One planned MCP invocation."""

    server: str = Field(..., min_length=2, max_length=40)
    tool: str = Field(..., min_length=2, max_length=60)
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="", max_length=300)

    @property
    def qualified(self) -> str:
        return f"{self.server}.{self.tool}"


class ToolPlan(BaseModel):
    """An agent's plan of MCP calls, capped so a model cannot fan out unboundedly."""

    calls: list[ToolCall] = Field(default_factory=list, max_length=6)
    reasoning: str = Field(default="", max_length=600)


class Citation(BaseModel):
    label: str = Field(..., min_length=2, max_length=300)
    url: str = Field(default="", max_length=500)
    kind: Literal["dataset", "literature", "computation", "web"] = "dataset"


class AgentFinding(BaseModel):
    """What one domain agent concluded, with its evidence attached."""

    agent: str = Field(..., min_length=2, max_length=40)
    summary: str = Field(..., min_length=1, max_length=2500)
    key_facts: list[str] = Field(default_factory=list, max_length=15)
    numbers_used: list[float] = Field(default_factory=list, max_length=30)
    citations: list[Citation] = Field(default_factory=list, max_length=15)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    limitations: str = Field(default="", max_length=600)
    tools_used: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def _require_evidence(self) -> "AgentFinding":
        if self.confidence > 0.85 and not (self.key_facts or self.citations):
            self.confidence = 0.5
        return self


class ClaimJudgement(BaseModel):
    claim: str = Field(..., min_length=3, max_length=600)
    status: Literal["supported", "unsupported", "contradicted", "needs_citation"]
    evidence: str = Field(default="", max_length=400)


class EvidenceRequest(BaseModel):
    """The Critic asking for another retrieval round instead of just rejecting.

    This is what turns the pipeline from a single pass into a reasoning loop: the
    Critic may route a targeted sub-question back to a domain agent when the
    evidence is thin, bounded by the session budget and a hard round limit.
    """

    domain: Domain
    question: str = Field(..., min_length=3, max_length=400)
    reason: str = Field(default="", max_length=300)


class CriticVerdict(BaseModel):
    """The Critic's ruling on the assembled findings."""

    verdict: Literal["accept", "revise", "reject"]
    grounding_assessment: str = Field(..., min_length=3, max_length=1200)
    judgements: list[ClaimJudgement] = Field(default_factory=list, max_length=20)
    removed_claims: list[str] = Field(default_factory=list, max_length=15)
    required_fixes: list[str] = Field(default_factory=list, max_length=10)
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list, max_length=4)
    source_conflicts: list[str] = Field(default_factory=list, max_length=10)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class FinalAnswer(BaseModel):
    """The synthesised, guardrail-checked response returned by ``POST /ask``."""

    answer: str = Field(..., min_length=1, max_length=6000)
    key_points: list[str] = Field(default_factory=list, max_length=10)
    citations: list[Citation] = Field(default_factory=list, max_length=20)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    caveats: list[str] = Field(default_factory=list, max_length=6)
    data_freshness: str = Field(default="", max_length=200)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    session_id: str | None = Field(default=None, max_length=64)
    include_trace: bool = True
    include_graph: bool = False
    max_tool_calls: int | None = Field(default=None, ge=1, le=60)


class AskResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    key_points: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    caveats: list[str] = Field(default_factory=list)
    grounded: bool = False
    grounding_score: float = 0.0
    critic_verdict: str = "unknown"
    agents_run: list[str] = Field(default_factory=list)
    evidence_rounds: int = 1
    source_agreement: float = 1.0
    guardrails: dict[str, Any] = Field(default_factory=dict)
    knowledge_graph: dict[str, Any] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    usd_cost: float = 0.0
