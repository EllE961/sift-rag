"""HTTP API + web UI.

    python -m sift.server            # http://127.0.0.1:8002

Everything heavy (encoders, reranker, LLM client, collections) lives in a
``Services`` object built once at startup and injected into the app, so tests
swap in fakes and run on CPU in milliseconds.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import mimetypes
import os
import re
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .chunking import CHUNKERS, make_chunker
from .collection import Collection, load_or_build_beir
from .config import DEFAULT_BLEND, INDEX_DIR, RESULTS_DIR, WEB_DIR, Settings, tuned_blends
from .embed import Embedder
from .generate import ExtractiveGenerator, Generator
from .ingest import SUPPORTED, load_bytes, to_passages
from .pipeline import MODES, Retriever, SearchConfig
from .projection import place_query, projection_path
from .rerank import Reranker
from .text import Analyzer

# Windows' registry often maps .js to text/plain, which browsers refuse for ES modules.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

BENCHMARKS = ("scifact", "nfcorpus")
WORKSPACE = "workspace"
MAX_UPLOAD_BYTES = 15 * 1024 * 1024


# --------------------------------------------------------------------- services
class CollectionRegistry:
    """Lazily loads benchmark collections; owns the mutable workspace."""

    def __init__(self, embedder: Embedder, names: tuple[str, ...] = BENCHMARKS, root: Path = INDEX_DIR,
                 loader: Any = None, preloaded: dict[str, Collection] | None = None) -> None:  # fmt: skip
        self.embedder = embedder
        self.names = names
        self.root = root
        self.loader = loader or (lambda name: load_or_build_beir(name, embedder, root))
        self._loaded: dict[str, Collection] = dict(preloaded or {})
        self._lock = threading.Lock()
        if WORKSPACE not in self._loaded:
            self._loaded[WORKSPACE] = (
                Collection.load(WORKSPACE, root) if Collection.exists(WORKSPACE, root) else Collection(WORKSPACE, [])
            )
        self._projections: dict[str, dict | None] = {}

    def get(self, name: str) -> Collection:
        if name not in (*self.names, WORKSPACE):
            raise KeyError(name)
        with self._lock:
            if name not in self._loaded:
                self._loaded[name] = self.loader(name)
            return self._loaded[name]

    @property
    def workspace(self) -> Collection:
        return self._loaded[WORKSPACE]

    def projection(self, name: str) -> dict | None:
        if name not in self._projections:
            path = projection_path(name, self.root)
            self._projections[name] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        return self._projections[name]

    def describe(self) -> list[dict[str, Any]]:
        out = []
        for name in (*self.names, WORKSPACE):
            loaded = self._loaded.get(name)
            meta_path = self.root / name / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            out.append({
                "name": name,
                "kind": "workspace" if name == WORKSPACE else "benchmark",
                "loaded": loaded is not None,
                "n_passages": len(loaded) if loaded is not None else meta.get("n_passages"),
                "n_docs": len(loaded.docs) if loaded is not None else meta.get("n_docs"),
                "has_projection": projection_path(name, self.root).exists(),
            })  # fmt: skip
        return out


@dataclass
class Services:
    embedder: Embedder
    reranker: Reranker | None
    generators: dict[str, Generator]
    registry: CollectionRegistry
    results_dir: Path = RESULTS_DIR
    settings: Settings = field(default_factory=Settings)
    device: str = "cpu"
    claude_status: str = "unknown"  # unknown | ready | unavailable | disabled
    blend_defaults: dict[str, float] = field(default_factory=dict)

    def blend_for(self, collection: str) -> float:
        return self.blend_defaults.get(collection, DEFAULT_BLEND)

    def generator_for(self, requested: str) -> tuple[Generator, bool]:
        """Returns (generator, may_fall_back)."""
        if requested == "extractive" or "claude" not in self.generators:
            return self.generators["extractive"], False
        if requested == "auto" and self.claude_status == "unavailable":
            return self.generators["extractive"], False
        return self.generators["claude"], requested == "auto"


def build_services(settings: Settings | None = None) -> Services:
    """Real components (downloads model weights on first run)."""
    from .config import resolve_device
    from .embed import TransformerEmbedder
    from .rerank import CrossEncoderReranker

    settings = settings or Settings()
    device = resolve_device(settings.device)
    embedder = TransformerEmbedder(settings.dense_model, device=device, batch_size=64)
    reranker = CrossEncoderReranker(settings.rerank_model, device=device)
    generators: dict[str, Generator] = {"extractive": ExtractiveGenerator()}
    status = "disabled"
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        from .generate import ClaudeGenerator

        generators["claude"] = ClaudeGenerator(settings.llm_model, settings.llm_effort, settings.llm_max_tokens)
        status = "unknown"
    return Services(embedder=embedder, reranker=reranker, generators=generators,
                    registry=CollectionRegistry(embedder), settings=settings, device=device,
                    claude_status=status, blend_defaults=tuned_blends())  # fmt: skip


# ----------------------------------------------------------------------- schemas
class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    collection: str = "scifact"
    mode: str = "hybrid_rerank"
    fusion: str = "rrf"
    alpha: float = Field(0.5, ge=0, le=1)
    k: int = Field(10, ge=1, le=50)
    depth: int = Field(100, ge=10, le=500)
    rerank_depth: int = Field(50, ge=1, le=200)
    ef: int = Field(128, ge=8, le=1024)
    blend: float | None = Field(None, ge=0, le=1)  # None -> the collection's train-tuned default


class AskRequest(SearchRequest):
    k: int = Field(5, ge=1, le=20)
    generator: str = "auto"  # auto | claude | extractive


class UploadFile(BaseModel):
    name: str
    content_b64: str


class IngestRequest(BaseModel):
    files: list[UploadFile] = Field(min_length=1, max_length=20)
    chunker: str = "recursive"
    chunk_size: int = Field(160, ge=20, le=1000)


class ChunkRequest(BaseModel):
    text: str | None = None
    doc_id: str | None = None
    chunk_size: int = Field(120, ge=20, le=1000)


# --------------------------------------------------------------------------- app
def _sse(event: dict[str, Any]) -> str:
    payload = {k: v for k, v in event.items() if k != "answer_obj"}
    return f"event: {payload['type']}\ndata: {json.dumps(payload)}\n\n"


_WORD = re.compile(r"[A-Za-z0-9]+")


def match_spans(analyzer: Analyzer, query: str, text: str, limit: int = 200) -> list[list[int]]:
    """Character spans in ``text`` whose analyzed form matches a query term."""
    terms = set(analyzer(query))
    if not terms:
        return []
    out = []
    for m in _WORD.finditer(text):
        toks = analyzer(m.group(0))
        if toks and toks[0] in terms:
            out.append([m.start(), m.end()])
            if len(out) >= limit:
                break
    return out


def create_app(services: Services) -> FastAPI:
    app = FastAPI(title="Sift", version="0.1.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
    analyzer = Analyzer()

    @app.middleware("http")
    async def no_stale_assets(request, call_next):
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"  # revalidate (ETag) instead of serving stale JS
        return response

    def retriever_for(name: str) -> Retriever:
        try:
            col = services.registry.get(name)
        except KeyError:
            raise HTTPException(404, f"unknown collection {name!r}") from None
        if len(col) == 0:
            raise HTTPException(409, f"collection {name!r} is empty — ingest documents first")
        return Retriever(col, services.embedder, services.reranker)

    def run_search(req: SearchRequest) -> tuple[Retriever, dict[str, Any]]:
        if req.mode not in MODES:
            raise HTTPException(422, f"mode must be one of {MODES}")
        retriever = retriever_for(req.collection)
        if req.mode not in retriever.available_modes():
            raise HTTPException(422, f"mode {req.mode!r} unavailable; try {retriever.available_modes()}")
        blend = req.blend if req.blend is not None else services.blend_for(req.collection)
        cfg = SearchConfig(mode=req.mode, fusion=req.fusion, alpha=req.alpha, k=req.k, depth=req.depth,
                           rerank_depth=req.rerank_depth, ef=req.ef, blend=blend)  # fmt: skip
        result = retriever.search(req.query, cfg)
        trace = result.to_dict(retriever.collection)
        trace["collection"] = req.collection
        trace["corpus_size"] = len(retriever.collection)
        for hit in trace["hits"]:
            hit["highlights"] = match_spans(analyzer, req.query, hit["passage"]["text"])
        proj = services.registry.projection(req.collection)
        if proj is not None:
            dense = next((s for s in result.stages if s.name == "dense"), None)
            pool = dense.hits[:15] if dense else result.hits[:10]
            trace["map"] = {
                "query_xy": place_query(proj, [h.pid for h in pool], [h.score for h in pool] if dense else
                                        [1.0 - 0.02 * i for i in range(len(pool))]),
                "hit_ids": [h.pid for h in result.hits],
            }  # fmt: skip
        return retriever, trace

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "device": services.device,
            "embedder": services.embedder.name,
            "reranker": services.reranker.name if services.reranker else None,
            "llm": {"model": services.settings.llm_model, "status": services.claude_status,
                    "available": "claude" in services.generators},  # fmt: skip
            "modes": list(MODES),
            "chunkers": list(CHUNKERS),
        }

    @app.get("/api/collections")
    def collections() -> list[dict[str, Any]]:
        return services.registry.describe()

    @app.post("/api/search")
    def search(req: SearchRequest) -> dict[str, Any]:
        return run_search(req)[1]

    @app.post("/api/ask")
    def ask(req: AskRequest) -> StreamingResponse:
        retriever, trace = run_search(req)
        passages = [retriever.collection.passage(h["pid"]) for h in trace["hits"]]
        generator, may_fall_back = services.generator_for(req.generator)

        def events() -> Iterator[str]:
            yield _sse({"type": "trace", "trace": trace})
            yield _sse({"type": "generator", "name": generator.name})
            if req.generator == "auto" and generator.name == "extractive" and "claude" in services.generators:
                msg = "Claude is unavailable (the API key was rejected) — showing the offline extractive answer."
                yield _sse({"type": "notice", "message": msg})
            emitted_text = False
            for ev in generator.stream(req.query, passages):
                if ev["type"] == "error" and may_fall_back and not emitted_text:
                    services.claude_status = "unavailable"
                    notice = f"{ev['message']} Falling back to offline extractive mode."
                    yield _sse({"type": "notice", "message": notice})
                    yield _sse({"type": "generator", "name": "extractive"})
                    for ev2 in services.generators["extractive"].stream(req.query, passages):
                        yield _sse(ev2)
                    return
                if ev["type"] == "delta":
                    emitted_text = True
                    if generator.name == "claude":
                        services.claude_status = "ready"
                yield _sse(ev)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})  # fmt: skip

    @app.get("/api/doc/{collection}/{doc_id:path}")
    def document(collection: str, doc_id: str) -> dict[str, Any]:
        try:
            col = services.registry.get(collection)
        except KeyError:
            raise HTTPException(404, "unknown collection") from None
        doc = col.docs.get(doc_id)
        if doc is None:
            raise HTTPException(404, "unknown document")
        chunks = [{"id": p.id, "start": p.start, "end": p.end} for p in col.passages.values() if p.doc_id == doc_id]
        return {"id": doc.id, "title": doc.title, "text": doc.text, "metadata": doc.metadata, "passages": chunks}

    @app.get("/api/projection/{collection}")
    def projection(collection: str) -> JSONResponse:
        proj = services.registry.projection(collection)
        if proj is None:
            raise HTTPException(404, "no projection for this collection (run: python -m sift.eval projection)")
        return JSONResponse(proj)

    @app.post("/api/chunk")
    def chunk(req: ChunkRequest) -> dict[str, Any]:
        text = req.text
        if text is None and req.doc_id:
            doc = services.registry.workspace.docs.get(req.doc_id)
            if doc is None:
                raise HTTPException(404, "unknown workspace document")
            text = doc.text
        if not text:
            raise HTTPException(422, "provide text or doc_id")
        text = text[:200_000]
        out: dict[str, Any] = {"text": text, "strategies": {}}
        for name in CHUNKERS:
            t = time.perf_counter()
            chunker = make_chunker(name, req.chunk_size, embedder=services.embedder)
            chunks = chunker.split(text)
            out["strategies"][name] = {
                "ms": round((time.perf_counter() - t) * 1000, 2),
                "chunks": [{"start": c.start, "end": c.end, "tokens": c.n_tokens, **c.meta} for c in chunks],
            }
        return out

    @app.post("/api/ingest")
    def ingest(req: IngestRequest) -> dict[str, Any]:
        if req.chunker not in CHUNKERS:
            raise HTTPException(422, f"chunker must be one of {CHUNKERS}")
        docs = []
        for f in req.files:
            if not f.name.lower().endswith(SUPPORTED):
                raise HTTPException(415, f"{f.name}: unsupported type (supported: {', '.join(SUPPORTED)})")
            try:
                data = base64.b64decode(f.content_b64, validate=True)
            except (binascii.Error, ValueError):
                raise HTTPException(422, f"{f.name}: invalid base64") from None
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{f.name}: larger than {MAX_UPLOAD_BYTES // 2**20} MB")
            try:
                doc = load_bytes(f.name, data)
            except Exception as exc:  # - surface parser errors to the client
                raise HTTPException(422, f"{f.name}: could not parse ({exc})") from None
            if not doc.text.strip():
                raise HTTPException(422, f"{f.name}: no extractable text")
            docs.append(doc)
        chunker = make_chunker(req.chunker, req.chunk_size, embedder=services.embedder)
        passages = to_passages(docs, chunker)
        ws = services.registry.workspace
        t = time.perf_counter()
        added = ws.add(docs, passages, services.embedder)
        index_ms = (time.perf_counter() - t) * 1000
        with contextlib.suppress(OSError):  # persistence is best-effort; the in-memory index is live
            ws.save(services.registry.root)
        return {
            "added_passages": added,
            "index_ms": round(index_ms, 1),
            "workspace": {"n_docs": len(ws.docs), "n_passages": len(ws)},
            "documents": [
                {"id": d.id, "title": d.title, "chars": len(d.text),
                 "chunks": [{"id": p.id, "start": p.start, "end": p.end} for p in passages if p.doc_id == d.id]}
                for d in docs
            ],  # fmt: skip
        }

    @app.get("/api/workspace")
    def workspace() -> dict[str, Any]:
        ws = services.registry.workspace
        return {"documents": [{"id": d.id, "title": d.title, "chars": len(d.text),
                               "n_chunks": sum(1 for p in ws.passages.values() if p.doc_id == d.id)}
                              for d in ws.docs.values()]}  # fmt: skip

    @app.get("/api/results")
    def results() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in ("retrieval", "hnsw", "claims_nli", "claims_claude"):
            path = services.results_dir / f"{name}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if name.startswith("claims"):
                    data.pop("per_claim", None)
                out[name] = data
        return out

    if WEB_DIR.exists():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

    return app


def probe_llm(services: Services) -> None:
    """Check the Anthropic credentials with a free Models API call (no tokens spent)."""
    gen = services.generators.get("claude")
    if gen is None:
        return
    import anthropic

    try:
        gen.client.models.retrieve(services.settings.llm_model)  # type: ignore[attr-defined]
        services.claude_status = "ready"
    except anthropic.AuthenticationError:
        services.claude_status = "unavailable"
        print("warning: Anthropic credentials rejected (401) - answers will use offline extractive mode")
    except anthropic.APIError as exc:
        print(f"warning: could not verify the Anthropic API ({exc.__class__.__name__}); will try per request")


def warm(services: Services) -> None:
    """Load collections and run one query each: JIT-compiles the HNSW kernels and warms CUDA."""
    for name in BENCHMARKS:
        try:
            col = services.registry.get(name)
            Retriever(col, services.embedder, services.reranker).search("warm up", SearchConfig(k=1, rerank_depth=4))
        except Exception as exc:  # a missing benchmark must not stop the service
            print(f"warning: could not load {name}: {exc}")


def main() -> None:
    import uvicorn

    settings = Settings()
    services = build_services(settings)
    threading.Thread(target=probe_llm, args=(services,), daemon=True).start()
    warm(services)
    uvicorn.run(create_app(services), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
