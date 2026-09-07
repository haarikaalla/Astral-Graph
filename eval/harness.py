"""AstralGraph benchmark harness.

Runs the fixed evaluation set through the full multi-agent pipeline and reports:

* **accuracy** — fraction of questions whose factual checks all pass
* **hallucination rate** — confident-but-wrong answers, ungrounded numbers, or
  any confident answer to a deliberately false-premise trap question
* **latency** — mean / p50 / p95 wall-clock per query
* **cost** — mean USD per query (from token accounting)
* **per-MCP-server call success rate**

Usage::

    python -m eval.harness                       # full set, sequential
    python -m eval.harness --only q08 q15        # subset by id prefix
    python -m eval.harness --concurrency 3       # run 3 questions at a time
    python -m eval.harness --repeat 2            # average over 2 runs
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agents.workflow import answer_question
from core.config import REPO_ROOT, get_settings
from core.telemetry import configure_logging, get_logger
from eval.questions import QUESTIONS, EvalQuestion
from eval.scoring import Score, score_answer

log = get_logger("eval")

RESULTS_DIR = REPO_ROOT / "eval" / "results"


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
async def run_one(question: EvalQuestion, *, max_tool_calls: int | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await answer_question(
            question.question,
            session_id=f"eval-{question.id}",
            include_trace=True,
            include_graph=False,
            max_tool_calls=max_tool_calls,
        )
    except Exception as exc:  # noqa: BLE001 - a crash is a failed question, not a failed run
        latency = (time.perf_counter() - started) * 1000.0
        log.exception("eval question crashed: %s", question.id)
        return {
            "question_id": question.id,
            "question": question.question,
            "kind": question.kind,
            "error": f"{type(exc).__name__}: {exc}",
            "answer": "",
            "latency_ms": round(latency, 1),
            "usd_cost": 0.0,
            "score": Score(question.id, question.kind, False, True,
                           notes=[f"crashed: {type(exc).__name__}"]).as_dict(),
            "server_stats": {},
        }

    score = score_answer(question, response)
    trace = response.trace or {}
    record = {
        "question_id": question.id,
        "question": question.question,
        "kind": question.kind,
        "ground_truth": question.ground_truth,
        "answer": response.answer,
        "confidence": response.confidence,
        "grounded": response.grounded,
        "grounding_score": response.grounding_score,
        "critic_verdict": response.critic_verdict,
        "agents_run": response.agents_run,
        "citations": [c.model_dump() for c in response.citations],
        "latency_ms": response.latency_ms,
        "usd_cost": response.usd_cost,
        "tool_calls": trace.get("tool_calls", 0),
        "server_stats": trace.get("server_stats", {}),
        "guardrail_score": (response.guardrails or {}).get("guardrail_score", 0.0),
        "guardrail_failures": (response.guardrails or {}).get("failures", []),
        "score": score.as_dict(),
        "error": "",
    }
    status = "PASS" if score.correct else "FAIL"
    print(f"  [{status}] {question.id:<22} {response.latency_ms/1000:6.1f}s  "
          f"conf={response.confidence:.2f}  {score.passed_checks}/{len(score.checks)} checks")
    for check in score.checks:
        if not check.passed:
            print(f"           - {check.name}: {check.detail}")
    return record


async def run_suite(
    questions: list[EvalQuestion],
    *,
    concurrency: int = 1,
    max_tool_calls: int | None = None,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def guarded(q: EvalQuestion) -> dict[str, Any]:
        async with semaphore:
            return await run_one(q, max_tool_calls=max_tool_calls)

    return list(await asyncio.gather(*(guarded(q) for q in questions)))


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records) or 1
    correct = sum(1 for r in records if r["score"]["correct"])
    hallucinated = sum(1 for r in records if r["score"]["hallucinated"])
    latencies = sorted(float(r["latency_ms"]) for r in records)
    costs = [float(r["usd_cost"]) for r in records]

    servers: dict[str, dict[str, int]] = {}
    for record in records:
        for server, stats in (record.get("server_stats") or {}).items():
            bucket = servers.setdefault(server, {"calls": 0, "ok": 0, "failed": 0})
            for key in ("calls", "ok", "failed"):
                bucket[key] += int(stats.get(key, 0))
    for stats in servers.values():
        stats["success_rate"] = round(stats["ok"] / stats["calls"], 3) if stats["calls"] else 0.0

    by_kind: dict[str, dict[str, Any]] = {}
    for record in records:
        bucket = by_kind.setdefault(record["kind"], {"total": 0, "correct": 0})
        bucket["total"] += 1
        bucket["correct"] += int(record["score"]["correct"])
    for bucket in by_kind.values():
        bucket["accuracy"] = round(bucket["correct"] / bucket["total"], 3)

    traps = [r for r in records if r["kind"] == "trap"]
    trap_refusals = sum(1 for r in traps if r["score"]["correct"])

    def pct(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        index = min(len(values) - 1, int(round(q * (len(values) - 1))))
        return round(values[index], 1)

    return {
        "questions": total,
        "accuracy": round(correct / total, 3),
        "correct": correct,
        "hallucination_rate": round(hallucinated / total, 3),
        "hallucinated": hallucinated,
        "trap_refusal_rate": round(trap_refusals / len(traps), 3) if traps else None,
        "grounded_rate": round(sum(1 for r in records if r.get("grounded")) / total, 3),
        "mean_latency_ms": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        "p50_latency_ms": pct(latencies, 0.5),
        "p95_latency_ms": pct(latencies, 0.95),
        "mean_usd_per_query": round(statistics.fmean(costs), 6) if costs else 0.0,
        "total_usd": round(sum(costs), 6),
        "mean_tool_calls": round(statistics.fmean([r.get("tool_calls", 0) for r in records]), 2),
        "errors": sum(1 for r in records if r.get("error")),
        "accuracy_by_kind": by_kind,
        "mcp_server_stats": servers,
    }


def to_markdown(summary: dict[str, Any], records: list[dict[str, Any]], *, llm: bool) -> str:
    mode = "Claude (Anthropic API)" if llm else "deterministic fallback (no ANTHROPIC_API_KEY)"
    trap_rate = summary["trap_refusal_rate"]
    trap_cell = "n/a" if trap_rate is None else f"{trap_rate:.1%}"
    lines = [
        "## AstralGraph benchmark results",
        "",
        f"_Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} — "
        f"{summary['questions']} questions — synthesis mode: {mode}_",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Accuracy | **{summary['accuracy']:.1%}** ({summary['correct']}/{summary['questions']}) |",
        f"| Hallucination rate | **{summary['hallucination_rate']:.1%}** "
        f"({summary['hallucinated']}/{summary['questions']}) |",
        f"| False-premise refusal rate | {trap_cell} |",
        f"| Grounded answers | {summary['grounded_rate']:.1%} |",
        f"| Mean latency | {summary['mean_latency_ms']/1000:.2f} s |",
        f"| p50 / p95 latency | {summary['p50_latency_ms']/1000:.2f} s / "
        f"{summary['p95_latency_ms']/1000:.2f} s |",
        f"| Mean cost per query | ${summary['mean_usd_per_query']:.5f} |",
        f"| Mean MCP tool calls per query | {summary['mean_tool_calls']} |",
        "",
        "### Per-MCP-server call success rate",
        "",
        "| MCP server | Calls | OK | Failed | Success rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for server, stats in sorted(summary["mcp_server_stats"].items()):
        lines.append(
            f"| `{server}` | {stats['calls']} | {stats['ok']} | {stats['failed']} | "
            f"{stats['success_rate']:.1%} |"
        )

    lines += ["", "### Per-question outcome", "",
              "| # | Question | Kind | Result | Conf. | Latency |",
              "| --- | --- | --- | --- | ---: | ---: |"]
    for record in records:
        result = "pass" if record["score"]["correct"] else "fail"
        question = record["question"][:70] + ("..." if len(record["question"]) > 70 else "")
        lines.append(
            f"| {record['question_id']} | {question} | {record['kind']} | {result} | "
            f"{record.get('confidence', 0):.2f} | {record['latency_ms']/1000:.1f} s |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def select(only: list[str] | None, kinds: list[str] | None) -> list[EvalQuestion]:
    picked = QUESTIONS
    if only:
        picked = [q for q in picked if any(q.id.startswith(prefix) for prefix in only)]
    if kinds:
        picked = [q for q in picked if q.kind in kinds]
    if not picked:
        raise SystemExit("no questions matched the filter")
    return picked


async def main_async(args: argparse.Namespace) -> int:
    settings = get_settings()
    settings.ensure_dirs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = select(args.only, args.kind)

    print(f"AstralGraph eval — {len(questions)} question(s), "
          f"concurrency={args.concurrency}, repeat={args.repeat}")
    print(f"LLM synthesis: {'ON (Claude)' if settings.llm_enabled else 'OFF (deterministic fallback)'}\n")

    records: list[dict[str, Any]] = []
    for run_index in range(args.repeat):
        if args.repeat > 1:
            print(f"--- run {run_index + 1}/{args.repeat} ---")
        records += await run_suite(
            questions, concurrency=args.concurrency, max_tool_calls=args.max_tool_calls
        )

    summary = aggregate(records)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_enabled": settings.llm_enabled,
        "model": settings.model,
        "summary": summary,
        "records": records,
    }
    json_path = RESULTS_DIR / f"report_{stamp}.json"
    md_path = RESULTS_DIR / f"report_{stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    markdown = to_markdown(summary, records, llm=settings.llm_enabled)
    md_path.write_text(markdown, encoding="utf-8")
    (RESULTS_DIR / "latest.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (RESULTS_DIR / "latest.md").write_text(markdown, encoding="utf-8")

    print("\n" + markdown)
    print(f"saved: {json_path.relative_to(REPO_ROOT)}")
    print(f"saved: {md_path.relative_to(REPO_ROOT)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AstralGraph benchmark")
    parser.add_argument("--only", nargs="*", help="question id prefixes to run (e.g. q08 q15)")
    parser.add_argument("--kind", nargs="*",
                        choices=["factual", "live", "computed", "trap"],
                        help="restrict to question kinds")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="questions to run in parallel (default 1)")
    parser.add_argument("--repeat", type=int, default=1, help="repeat the suite N times")
    parser.add_argument("--max-tool-calls", type=int, default=None,
                        help="override the per-session MCP tool-call budget")
    return parser


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
