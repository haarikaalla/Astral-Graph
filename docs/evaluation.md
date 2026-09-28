# Evaluation

AstralGraph includes a benchmark harness in `eval/` for running a fixed question set
through the full pipeline.

## Metrics

| Metric | Meaning |
| --- | --- |
| Accuracy | Share of questions whose factual checks pass. |
| Hallucination rate | Confident unsupported answers or false-premise failures. |
| Grounding score | How well the final answer maps to graph evidence. |
| Latency | Per-question wall-clock timing. |
| Cost | Estimated LLM cost from token accounting. |
| MCP success rate | Calls, successes, and failures per MCP server. |

## Commands

```bash
python -m eval.harness
python -m eval.harness --only q08 q15
python -m eval.harness --concurrency 3
python -m eval.harness --repeat 2
```

Reports are written under `eval/results/`, which is ignored by git because it is
generated output.

## Test Strategy

Fast CI runs:

```bash
pytest -m "not integration"
```

Live integration tests are kept manual:

```bash
pytest -m integration
```

Those tests may spawn MCP subprocesses and call public APIs, so they are better suited
for release checks or a manually triggered workflow.
