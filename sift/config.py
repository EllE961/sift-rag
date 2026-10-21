"""Filesystem layout and runtime settings.

Everything is overridable through environment variables so the same code runs
locally, in CI and inside the Docker image.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _path(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value).expanduser().resolve() if value else default


DATA_DIR = _path("SIFT_DATA_DIR", ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DIR = DATA_DIR / "indexes"
RESULTS_DIR = _path("SIFT_RESULTS_DIR", ROOT / "results")
ASSETS_DIR = ROOT / "docs" / "assets"
WEB_DIR = ROOT / "web"

# USD per million tokens (input, output). Used to report the cost of every answer.
PRICING_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}


DEFAULT_BLEND = 0.3  # cross-encoder blend weight for collections without a tuned value (e.g. the workspace)


def tuned_blends(results_dir: Path | None = None) -> dict[str, float]:
    """Per-collection cross-encoder blend weights tuned on train queries (`python -m sift.eval blend`)."""
    import json

    path = (results_dir or RESULTS_DIR) / "retrieval.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for name, d in data.get("datasets", {}).items():
        best = d.get("rerank_blend", {}).get("best")
        if best and best.get("method") == "score":
            out[name] = float(best["w"])
    return out


@dataclass
class Settings:
    """Runtime configuration for the service and the eval harness."""

    llm_model: str = field(default_factory=lambda: os.environ.get("SIFT_LLM_MODEL", "claude-opus-5"))
    llm_effort: str = field(default_factory=lambda: os.environ.get("SIFT_LLM_EFFORT", "medium"))
    llm_max_tokens: int = field(default_factory=lambda: int(os.environ.get("SIFT_LLM_MAX_TOKENS", "4096")))
    dense_model: str = field(default_factory=lambda: os.environ.get("SIFT_DENSE_MODEL", "BAAI/bge-small-en-v1.5"))
    rerank_model: str = field(
        default_factory=lambda: os.environ.get("SIFT_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    device: str = field(default_factory=lambda: os.environ.get("SIFT_DEVICE", "auto"))
    host: str = field(default_factory=lambda: os.environ.get("SIFT_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("SIFT_PORT", "8002")))


def resolve_device(device: str = "auto") -> str:
    """Map ``auto`` to ``cuda`` when a GPU is visible, otherwise ``cpu``."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - torch is a hard dependency
        return "cpu"


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of a request at list price (0.0 for unknown models)."""
    price_in, price_out = PRICING_PER_MTOK.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
