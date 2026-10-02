"""Configuration. Environment only — no secret has a default, ever.

Every field that carries a credential is typed ``str | None`` with no default value in
code and is read from the process environment or a local ``.env`` that is gitignored.
Code that needs a credential calls ``require()``, which fails loudly and names the missing
variable rather than sending an empty Authorization header and getting a confusing 401.
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


# Credentials people reach for under their vendor name, and the prefixed name this project
# actually reads. `env_prefix="ARAG_"` means a correctly-obtained, correctly-pasted key
# under the vendor's own name is read by NOTHING.
#
# Twice now. First the key went into `.env.example` (committed, placeholders only); then it
# went into `.env` as `GEMINI_API_KEY`. The second cost a whole verification run: seven
# questions, zero network calls, every one reported as `generation_unavailable` - which is
# the SAME abstain reason a quota exhaustion produces, so the results table was
# indistinguishable from the rate-limited run it was supposed to be compared against.
#
# A missing credential is a deployment fault. It should be loud at startup, not inferred
# later from a table of abstentions.
UNPREFIXED_CREDENTIALS: dict[str, str] = {
    "GEMINI_API_KEY": "ARAG_GEMINI_API_KEY",
    "GOOGLE_API_KEY": "ARAG_GEMINI_API_KEY",
    "DATABASE_URL": "ARAG_DATABASE_URL",
    "LANGFUSE_PUBLIC_KEY": "ARAG_LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY": "ARAG_LANGFUSE_SECRET_KEY",
}


def misprefixed_credentials(
    environ: Mapping[str, str] | None = None,
    *,
    dotenv: pathlib.Path | None = pathlib.Path(".env"),
) -> dict[str, str]:
    """Credentials present under a vendor name while the prefixed name is unset.

    Returns ``{found_name: expected_name}``. Checks the process environment and `.env`
    together, because the mistake is equally easy in either.

    Both sources are injectable so the guard itself is testable. A guard that can only be
    exercised against the developer's own `.env` is one whose detection half is never
    proven - which is the failure this project has now hit three times in another form.
    """
    env: dict[str, str] = dict(environ if environ is not None else os.environ)
    if dotenv is not None and dotenv.exists():
        for line in dotenv.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, _, value = stripped.partition("=")
            env.setdefault(name.strip(), value.strip())

    return {
        found: expected
        for found, expected in UNPREFIXED_CREDENTIALS.items()
        if env.get(found) and not env.get(expected)
    }


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: Literal["local", "ci", "preview", "production"] = "local"
    log_level: str = "INFO"

    # --- credentials: no defaults on purpose ---
    gemini_api_key: str | None = None
    database_url: str | None = None
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    # --- local (offline) model endpoints ---
    ollama_base_url: str = "http://localhost:11434"
    judge_model: str = "qwen2.5:32b-instruct-q4_K_M"
    contextualiser_model: str = "qwen2.5:7b-instruct"

    # --- online models ---
    generator_model: str = "gemini-3.6-flash"
    generator_price_key: str = "gemini-flash-free"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- retrieval knobs; tuned by the eval suite, never guessed in a PR ---
    dense_top_k: int = 30
    lexical_top_k: int = 30
    lexical_prefilter_k: int = 200
    rrf_k: int = 60
    rerank_top_n: int = 6
    abstain_threshold: float = 0.35

    # --- eval ---
    golden_path: Path = Field(default=REPO_ROOT / "data" / "golden" / "v1.jsonl")
    cassette_dir: Path = Field(default=REPO_ROOT / "data" / "cassettes")
    thresholds_path: Path = Field(default=REPO_ROOT / "src" / "arag" / "eval" / "thresholds.yaml")
    eval_log_path: Path = Field(default=REPO_ROOT / "EVAL_LOG.md")

    def require(self, name: str) -> str:
        value = getattr(self, name, None)
        if not value:
            raise RuntimeError(
                f"missing required configuration: ARAG_{name.upper()}. "
                "Set it in the environment or .env (see .env.example). "
                "Secrets are never read from code."
            )
        return str(value)


_settings: Settings | None = None


def settings() -> Settings:
    """Lazy singleton so importing a module never requires the environment to be complete."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
