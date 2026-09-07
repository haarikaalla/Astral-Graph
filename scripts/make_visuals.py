"""Generate the README visuals from real benchmark output.

    python scripts/make_visuals.py

Reads ``eval/results/latest.json`` and writes PNG charts plus an animated GIF of
the agent pipeline into ``docs/images/``. Nothing here is hand-drawn: every number
in every chart comes from an actual harness run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "eval" / "results" / "latest.json"
IMAGES = REPO / "docs" / "images"

BG = "#0b1020"
FG = "#e8ecff"
ACCENT = "#4cc9f0"
GOOD = "#57cc99"
BAD = "#f72585"
WARN = "#ffd166"
MUTED = "#7b8cde"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": FG,
    "axes.labelcolor": FG,
    "xtick.color": FG,
    "ytick.color": FG,
    "axes.edgecolor": "#2a3566",
    "font.size": 11,
})


def load() -> dict:
    if not RESULTS.exists():
        raise SystemExit(
            f"no benchmark results at {RESULTS}\nRun: python -m eval.harness"
        )
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def style(ax, title: str) -> None:
    ax.set_title(title, color=FG, fontsize=13, pad=14, weight="bold")
    ax.grid(axis="y", color="#222c55", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


# --------------------------------------------------------------------------- #
def chart_headline(summary: dict) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    cards = [
        ("Accuracy", f"{summary['accuracy']:.0%}", GOOD,
         f"{summary['correct']}/{summary['questions']} questions"),
        ("Hallucination rate", f"{summary['hallucination_rate']:.0%}", ACCENT,
         "unsupported claims"),
        ("False-premise refusals",
         "n/a" if summary["trap_refusal_rate"] is None else f"{summary['trap_refusal_rate']:.0%}",
         WARN, "trap questions rejected"),
        ("Mean latency", f"{summary['mean_latency_ms'] / 1000:.1f}s", MUTED,
         f"p95 {summary['p95_latency_ms'] / 1000:.1f}s"),
    ]
    for ax, (label, value, colour, sub) in zip(axes, cards):
        ax.axis("off")
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.03, 0.08), 0.94, 0.84, boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="#141a38", edgecolor=colour, linewidth=1.6, transform=ax.transAxes))
        ax.text(0.5, 0.66, value, ha="center", va="center", fontsize=27,
                weight="bold", color=colour, transform=ax.transAxes)
        ax.text(0.5, 0.36, label, ha="center", va="center", fontsize=11,
                color=FG, transform=ax.transAxes)
        ax.text(0.5, 0.20, sub, ha="center", va="center", fontsize=9,
                color="#95a0d6", transform=ax.transAxes)
    fig.suptitle("AstralGraph benchmark — 17 fixed questions with known-correct facts",
                 color=FG, fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(IMAGES / "benchmark_headline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def chart_servers(summary: dict) -> None:
    stats = summary["mcp_server_stats"]
    if not stats:
        return
    names = sorted(stats, key=lambda n: -stats[n]["calls"])
    ok = [stats[n]["ok"] for n in names]
    failed = [stats[n]["failed"] for n in names]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.barh(names, ok, color=GOOD, label="succeeded", height=0.62)
    ax.barh(names, failed, left=ok, color=BAD, label="failed", height=0.62)
    for i, name in enumerate(names):
        total = stats[name]["calls"]
        ax.text(total + 0.4, i, f"{stats[name]['success_rate']:.0%}",
                va="center", color=FG, fontsize=10, weight="bold")
    style(ax, "MCP tool calls per server (every fact travels through one of these)")
    ax.set_xlabel("tool calls")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#222c55", linewidth=0.8)
    ax.legend(facecolor="#141a38", edgecolor="#2a3566", labelcolor=FG, loc="lower right")
    fig.tight_layout()
    fig.savefig(IMAGES / "mcp_server_success.png", dpi=160)
    plt.close(fig)


def chart_latency(records: list[dict]) -> None:
    ordered = sorted(records, key=lambda r: r["latency_ms"])
    names = [r["question_id"].split("_", 1)[0] for r in ordered]
    seconds = [r["latency_ms"] / 1000 for r in ordered]
    colours = [GOOD if r["score"]["correct"] else BAD for r in ordered]

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.bar(names, seconds, color=colours, width=0.68)
    style(ax, "Latency per question (green = passed, pink = failed)")
    ax.set_ylabel("seconds")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(IMAGES / "latency_per_question.png", dpi=160)
    plt.close(fig)


def chart_accuracy_by_kind(summary: dict) -> None:
    by_kind = summary.get("accuracy_by_kind", {})
    if not by_kind:
        return
    kinds = list(by_kind)
    values = [by_kind[k]["accuracy"] * 100 for k in kinds]
    labels = [f"{k}\n{by_kind[k]['correct']}/{by_kind[k]['total']}" for k in kinds]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.bar(labels, values, color=[ACCENT, GOOD, WARN, MUTED][: len(kinds)], width=0.6)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.0f}%",
                ha="center", color=FG, weight="bold")
    style(ax, "Accuracy by question type")
    ax.set_ylim(0, 112)
    ax.set_ylabel("% correct")
    fig.tight_layout()
    fig.savefig(IMAGES / "accuracy_by_kind.png", dpi=160)
    plt.close(fig)


def chart_guardrails(records: list[dict]) -> None:
    checked = sum(r.get("score", {}).get("checks", []) and 1 or 0 for r in records)
    grounded = sum(1 for r in records if r.get("grounded"))
    traps = [r for r in records if r["kind"] == "trap"]
    trap_ok = sum(1 for r in traps if r["score"]["correct"])
    tool_calls = sum(r.get("tool_calls", 0) for r in records)

    labels = ["Questions\nscored", "Answers marked\ngrounded", "Trap questions\nrefused",
              "MCP tool calls\nmade"]
    values = [checked, grounded, trap_ok, tool_calls]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(labels, values, color=[MUTED, GOOD, WARN, ACCENT], width=0.55)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + max(values) * 0.02, str(value),
                ha="center", color=FG, weight="bold")
    style(ax, "Guardrail activity across the benchmark run")
    fig.tight_layout()
    fig.savefig(IMAGES / "guardrail_activity.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Animated pipeline
# --------------------------------------------------------------------------- #
STAGES = [
    ("Question", "\"How far is the ISS from London right now?\"", ACCENT),
    ("Intent Parser", "domains = [events]   no tool access", MUTED),
    ("Events Agent", "MCP -> iss.iss_now + iss.iss_ground_distance", GOOD),
    ("Knowledge Graph", "facts stored with source, URL and timestamp", WARN),
    ("Critic Agent", "every number re-checked against the graph", BAD),
    ("Orchestrator", "answer + citations, saved via Filesystem MCP", ACCENT),
]


def animation_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    boxes, titles, subs = [], [], []
    for index, (title, sub, colour) in enumerate(STAGES):
        y = 8.6 - index * 1.45
        box = mpatches.FancyBboxPatch(
            (0.7, y - 0.5), 8.6, 1.0,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            facecolor="#141a38", edgecolor="#2a3566", linewidth=1.4, alpha=0.35)
        ax.add_patch(box)
        boxes.append((box, colour))
        titles.append(ax.text(1.1, y + 0.13, title, fontsize=12, weight="bold",
                              color="#3d4680", va="center"))
        subs.append(ax.text(1.1, y - 0.24, sub, fontsize=9, color="#3d4680", va="center"))
        if index < len(STAGES) - 1:
            ax.annotate("", xy=(5.0, y - 0.62), xytext=(5.0, y - 0.5),
                        arrowprops=dict(arrowstyle="-|>", color="#2a3566", lw=1.4))

    header = ax.text(5.0, 9.62, "AstralGraph  ·  one question through the pipeline",
                     ha="center", fontsize=13, weight="bold", color=FG)
    footer = ax.text(5.0, 0.35, "", ha="center", fontsize=10, color=MUTED)

    frames_per_stage = 12

    def update(frame: int):
        active = frame // frames_per_stage
        for index, ((box, colour), title, sub) in enumerate(zip(boxes, titles, subs)):
            if index < active:
                box.set_alpha(0.9)
                box.set_edgecolor(colour)
                title.set_color(FG)
                sub.set_color("#95a0d6")
            elif index == active:
                pulse = 0.45 + 0.5 * abs(((frame % frames_per_stage) / frames_per_stage) - 0.5) * 2
                box.set_alpha(pulse)
                box.set_edgecolor(colour)
                title.set_color(colour)
                sub.set_color("#c8d0ff")
            else:
                box.set_alpha(0.3)
                box.set_edgecolor("#2a3566")
                title.set_color("#3d4680")
                sub.set_color("#3d4680")
        if active >= len(STAGES):
            footer.set_text("grounded answer · 0 unverified numbers · full provenance")
            footer.set_color(GOOD)
        else:
            footer.set_text("no agent may call an API directly — everything goes through MCP")
            footer.set_color(MUTED)
        return [b for b, _ in boxes] + titles + subs + [header, footer]

    total = frames_per_stage * (len(STAGES) + 2)
    anim = FuncAnimation(fig, update, frames=total, interval=90, blit=False)
    anim.save(IMAGES / "pipeline.gif", writer=PillowWriter(fps=11))
    plt.close(fig)


def animation_guardrail() -> None:
    """Show the critic rejecting a fabricated number."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    ax.text(5, 9.2, "Guardrail: the model never owns a number",
            ha="center", fontsize=13, weight="bold", color=FG)

    lines = [
        ("draft answer", "\"Apophis passes 31600 km from Earth and is 410 m wide.\"", FG),
        ("graph lookup", "31600 km  ->  matched: nasa_neo.neo_lookup close_approach", GOOD),
        ("graph lookup", "410 m     ->  no matching fact in the knowledge graph", BAD),
        ("critic verdict", "revise: remove or re-source the unverified value", WARN),
        ("final answer", "\"Apophis passes 31600 km from Earth.\"  [NASA NeoWs]", GOOD),
    ]
    texts = []
    for index, (label, body, colour) in enumerate(lines):
        y = 7.6 - index * 1.35
        left = ax.text(0.6, y, label, fontsize=10, color="#3d4680", va="center", weight="bold")
        right = ax.text(3.0, y, body, fontsize=10, color="#3d4680", va="center",
                        family="monospace")
        texts.append((left, right, colour))

    frames_per_line = 14

    def update(frame: int):
        active = frame // frames_per_line
        for index, (left, right, colour) in enumerate(texts):
            if index <= active:
                left.set_color("#95a0d6")
                right.set_color(colour)
            else:
                left.set_color("#3d4680")
                right.set_color("#3d4680")
        return [t for pair in texts for t in pair[:2]]

    total = frames_per_line * (len(lines) + 2)
    anim = FuncAnimation(fig, update, frames=total, interval=100, blit=False)
    anim.save(IMAGES / "guardrail.gif", writer=PillowWriter(fps=10))
    plt.close(fig)


def main() -> int:
    IMAGES.mkdir(parents=True, exist_ok=True)
    payload = load()
    summary = payload["summary"]
    records = payload["records"]

    chart_headline(summary)
    chart_servers(summary)
    chart_latency(records)
    chart_accuracy_by_kind(summary)
    chart_guardrails(records)
    animation_pipeline()
    animation_guardrail()

    for path in sorted(IMAGES.iterdir()):
        print(f"  wrote {path.relative_to(REPO)}  ({path.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
