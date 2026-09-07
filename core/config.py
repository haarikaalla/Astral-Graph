"""Central, validated configuration for AstralGraph.

Everything is read once from the environment (``.env``) and exposed as a frozen
singleton via :func:`get_settings`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration. Field names map to the env vars in ``.env.example``."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM ----
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    model: str = Field(default="claude-sonnet-4-20250514", alias="ASTRAL_MODEL")
    fast_model: str = Field(default="claude-3-5-haiku-20241022", alias="ASTRAL_FAST_MODEL")
    max_tokens: int = Field(default=2048, alias="ASTRAL_MAX_TOKENS")
    temperature: float = Field(default=0.0, alias="ASTRAL_TEMPERATURE")

    # ---- Upstream data providers (consumed only inside MCP servers) ----
    nasa_api_key: str = Field(default="DEMO_KEY", alias="NASA_API_KEY")
    brave_api_key: str = Field(default="", alias="BRAVE_API_KEY")

    # ---- MCP ----
    enable_node_mcp_servers: bool = Field(default=True, alias="ASTRAL_ENABLE_NODE_MCP_SERVERS")
    fs_root: Path = Field(default=REPO_ROOT / "workspace_data", alias="ASTRAL_FS_ROOT")

    # ---- Knowledge graph ----
    neo4j_uri: str = Field(default="", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="", alias="NEO4J_PASSWORD")

    # ---- RAG ----
    chroma_dir: Path = Field(default=REPO_ROOT / "storage" / "chroma", alias="ASTRAL_CHROMA_DIR")
    embed_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="ASTRAL_EMBED_MODEL"
    )
    rag_top_k: int = Field(default=5, alias="ASTRAL_RAG_TOP_K")

    # ---- Guardrails ----
    max_tool_calls_per_session: int = Field(default=24, alias="ASTRAL_MAX_TOOL_CALLS_PER_SESSION")
    max_retries_per_agent: int = Field(default=2, alias="ASTRAL_MAX_RETRIES_PER_AGENT")
    max_agent_seconds: float = Field(default=90.0, alias="ASTRAL_MAX_AGENT_SECONDS")
    grounding_threshold: float = Field(default=0.60, alias="ASTRAL_GROUNDING_THRESHOLD")

    # ---- Observability ----
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")
    log_level: str = Field(default="INFO", alias="ASTRAL_LOG_LEVEL")
    log_dir: Path = Field(default=REPO_ROOT / "storage" / "logs", alias="ASTRAL_LOG_DIR")

    @field_validator("fs_root", "chroma_dir", "log_dir", mode="after")
    @classmethod
    def _absolutise(cls, value: Path) -> Path:
        return value if value.is_absolute() else (REPO_ROOT / value).resolve()

    @property
    def llm_enabled(self) -> bool:
        """True when a real Claude key is present; otherwise the app runs deterministic-only."""
        return bool(self.anthropic_api_key and self.anthropic_api_key.startswith("sk-ant"))

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def neo4j_enabled(self) -> bool:
        return bool(self.neo4j_uri)

    def ensure_dirs(self) -> None:
        for path in (self.fs_root, self.chroma_dir, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()  # type: ignore[call-arg]
    settings.ensure_dirs()
    return settings
