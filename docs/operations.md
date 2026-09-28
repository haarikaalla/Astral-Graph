# Operations

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
uvicorn api.main:app --reload
```

## Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

The API listens on port `8000`; the Streamlit UI listens on port `8501`.

## Environment

Important variables:

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Enables full LLM reasoning. |
| `NASA_API_KEY` | NASA API access; `DEMO_KEY` is rate-limited. |
| `BRAVE_API_KEY` | Enables Brave Search MCP server. |
| `ASTRAL_CORS_ORIGINS` | Comma-separated allowed browser origins. |
| `NEO4J_URI` | Optional Neo4j graph backend. |
| `NEO4J_PASSWORD` | Required when starting the Neo4j compose profile. |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | Optional tracing. |

## CORS

The default CORS setting is local-only:

```env
ASTRAL_CORS_ORIGINS=http://localhost:8501,http://127.0.0.1:8501
```

Use `*` only for disposable local demos.

## Neo4j

NetworkX is the default in-process graph store. To start Neo4j:

```bash
NEO4J_PASSWORD='replace-with-a-real-password' docker compose --profile neo4j up --build
```

Then configure:

```env
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=replace-with-a-real-password
```

## Release Checklist

- Run `pytest -m "not integration"`.
- Run `pytest -m integration` when public APIs and optional keys are available.
- Run `python -m eval.harness --concurrency 2`.
- Build the Docker image.
- Check `/health`, `/mcp/servers`, and `/rag/stats`.
