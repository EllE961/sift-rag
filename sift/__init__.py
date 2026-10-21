"""Sift — a retrieval-augmented generation engine built from first principles.

Lexical retrieval (BM25 + a from-scratch Porter stemmer), dense retrieval,
a from-scratch HNSW index, rank fusion, cross-encoder reranking, grounded
generation with exact-span citations, and an evaluation harness.
"""

from __future__ import annotations

import os

# transformers will import TensorFlow if it is installed, which costs ~10 s and
# a wall of log noise. Sift is PyTorch-only, so opt out before anything imports it.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"
