"""Citation enforcement.

Rules
-----
* Any sentence that *sounds like* a literature claim must carry a citation.
* Every citation an agent emits must correspond to a source that actually appeared
  in retrieved evidence — fabricated references are rejected outright.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from graph.store import KnowledgeGraph
from guardrails.schemas import Citation

LITERATURE_TRIGGERS = (
    "research", "study", "studies", "paper", "papers", "publication", "preprint",
    "according to", "authors", "et al", "arxiv", "reported", "reports that",
    "was announced", "review", "analysis shows", "literature", "published",
    "scientists", "astronomers found", "survey found", "suggests that",
)

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


@dataclass
class CitationReport:
    ok: bool
    required: bool
    provided: int
    verified: int
    uncited_claims: list[str] = field(default_factory=list)
    fabricated: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "citation_required": self.required,
            "citations_provided": self.provided,
            "citations_verified": self.verified,
            "uncited_literature_claims": self.uncited_claims,
            "fabricated_citations": self.fabricated,
            "notes": self.notes,
        }

    def feedback(self) -> str:
        parts = []
        if self.uncited_claims:
            parts.append(
                "These literature claims need an explicit citation or must be removed:\n- "
                + "\n- ".join(c[:200] for c in self.uncited_claims[:5])
            )
        if self.fabricated:
            parts.append(
                "These citations do not match any retrieved source and must be removed: "
                + "; ".join(self.fabricated[:5])
            )
        return "\n".join(parts)


def _known_sources(kg: KnowledgeGraph, extra: Iterable[str] = ()) -> set[str]:
    known = {str(node.get("label", "")).lower() for node in kg.sources()}
    for node in kg.entities():
        props = node.get("properties") or {}
        for key in ("url", "source_url", "source_name"):
            value = props.get(key)
            if value:
                known.add(str(value).lower())
        if node.get("type") == "Document":
            known.add(str(node.get("label", "")).lower())
    known |= {str(e).lower() for e in extra if e}
    return {k for k in known if k}


def _matches_known(citation: Citation, known: set[str]) -> bool:
    label = citation.label.lower().strip()
    url = citation.url.lower().strip()
    for candidate in known:
        if not candidate:
            continue
        if label and (label in candidate or candidate in label):
            return True
        if url and (url in candidate or candidate in url):
            return True
    return False


def check_citations(
    text: str,
    citations: Sequence[Citation],
    kg: KnowledgeGraph,
    *,
    known_extra: Iterable[str] = (),
    strict: bool = True,
) -> CitationReport:
    lowered = text.lower()
    required = any(trigger in lowered for trigger in LITERATURE_TRIGGERS)
    known = _known_sources(kg, known_extra)

    verified, fabricated = 0, []
    for citation in citations:
        if _matches_known(citation, known) or citation.kind == "computation":
            verified += 1
        else:
            fabricated.append(f"{citation.label} {citation.url}".strip())

    uncited: list[str] = []
    if required and verified == 0:
        for sentence in SENTENCE_RE.split(text.strip()):
            if any(trigger in sentence.lower() for trigger in LITERATURE_TRIGGERS):
                uncited.append(sentence.strip())

    ok = (verified > 0 or not required) and (not fabricated if strict else True)
    notes = []
    if not required:
        notes.append("no literature-style claims detected; citation optional")
    if not citations:
        notes.append("agent returned no citations")
    return CitationReport(
        ok=ok,
        required=required,
        provided=len(citations),
        verified=verified,
        uncited_claims=uncited[:6],
        fabricated=fabricated,
        notes=notes,
    )
