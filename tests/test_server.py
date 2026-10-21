import base64
import json

import pytest

from sift.collection import Collection
from sift.embed import HashingEmbedder
from sift.generate import ExtractiveGenerator
from sift.rerank import OverlapReranker
from sift.server import CollectionRegistry, Services, create_app
from sift.types import Document, Passage

from .asgi import SyncASGIClient

DOCS = [
    Document("s1", "Vitamin D deficiency is associated with an increased risk of multiple sclerosis.", "Vitamin D and MS"),
    Document("s2", "Statin use after diagnosis was associated with lower breast cancer mortality.", "Statins"),
    Document("s3", "Autophagy activity declines with age in many organisms.", "Autophagy and aging"),
] + [Document(f"f{i}", f"Filler abstract {i} about agriculture and soil chemistry.", f"Filler {i}") for i in range(70)]


class FailingGenerator:
    name = "claude"

    def stream(self, question, passages, instructions=None):
        yield {"type": "error", "message": "Anthropic API error 401: invalid x-api-key."}
        yield {"type": "done", "answer": {"segments": []}, "answer_obj": None}


@pytest.fixture()
def client(tmp_path):
    emb = HashingEmbedder(64)
    col = Collection("scifact", [Passage(d.id, d.id, d.text, d.title, 0, len(d.text)) for d in DOCS], {d.id: d for d in DOCS})
    col.build_lexical().build_dense(emb, ann="hnsw")
    registry = CollectionRegistry(emb, names=("scifact",), root=tmp_path, preloaded={"scifact": col})
    services = Services(embedder=emb, reranker=OverlapReranker(),
                        generators={"extractive": ExtractiveGenerator(), "claude": FailingGenerator()},
                        registry=registry, results_dir=tmp_path, claude_status="unknown")  # fmt: skip
    (tmp_path / "retrieval.json").write_text(json.dumps({"datasets": {}}))
    return SyncASGIClient(create_app(services)), services


def sse_events(resp):
    out = []
    for block in resp.text.split("\n\n"):
        data = [line[6:] for line in block.split("\n") if line.startswith("data: ")]
        if data:
            out.append(json.loads("\n".join(data)))
    return out


def test_health_and_collections(client):
    c, _ = client
    h = c.get("/api/health").json()
    assert h["ok"] and h["llm"]["available"] and "hybrid_rerank" in h["modes"]
    names = [x["name"] for x in c.get("/api/collections").json()]
    assert names == ["scifact", "workspace"]


def test_search_trace(client):
    c, _ = client
    r = c.post("/api/search", json={"query": "vitamin D multiple sclerosis", "mode": "hybrid_rerank", "k": 3})
    assert r.status_code == 200
    t = r.json()
    assert t["hits"][0]["pid"] == "s1"
    assert t["stage_order"] == ["bm25", "dense", "fusion", "rerank"]
    assert t["hits"][0]["highlights"], "query-term spans for the UI"
    a, b = t["hits"][0]["highlights"][0]
    assert t["hits"][0]["passage"]["text"][a:b].lower() in {"vitamin", "d", "multiple", "sclerosis"}


def test_search_validation(client):
    c, _ = client
    assert c.post("/api/search", json={"query": "x", "collection": "nope"}).status_code == 404
    assert c.post("/api/search", json={"query": "x", "mode": "bogus"}).status_code == 422
    assert c.post("/api/search", json={"query": ""}).status_code == 422
    assert c.post("/api/search", json={"query": "x", "collection": "workspace"}).status_code == 409


def test_ask_falls_back_to_extractive_when_claude_fails(client):
    c, services = client
    r = c.post("/api/ask", json={"query": "Does autophagy decline with age?", "generator": "auto"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = sse_events(r)
    types = [e["type"] for e in ev]
    assert types[0] == "trace" and "notice" in types and types[-1] == "done"
    assert ev[-1]["answer"]["mode"] == "extractive"
    assert any(e["type"] == "citation" for e in ev)
    assert services.claude_status == "unavailable"
    # subsequent "auto" requests go straight to the offline generator
    ev2 = sse_events(c.post("/api/ask", json={"query": "statins breast cancer", "generator": "auto"}))
    assert next(e for e in ev2 if e["type"] == "generator")["name"] == "extractive"


def test_ingest_chunk_and_search_workspace(client):
    c, _ = client
    md = b"# Sieve manual\n\nA sieve separates fine particles from coarse ones.\n\nMesh number sets the cut-off size.\n"
    r = c.post("/api/ingest", json={"files": [{"name": "sieve.md", "content_b64": base64.b64encode(md).decode()}],
                                    "chunker": "sentence", "chunk_size": 20})  # fmt: skip
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["added_passages"] >= 1 and body["workspace"]["n_docs"] == 1
    doc_id = body["documents"][0]["id"]
    assert c.get(f"/api/doc/workspace/{doc_id}").json()["title"] == "Sieve manual"
    hit = c.post("/api/search", json={"query": "mesh cut-off size", "collection": "workspace", "mode": "hybrid", "k": 1}).json()
    assert hit["hits"][0]["passage"]["doc_id"] == doc_id
    ch = c.post("/api/chunk", json={"doc_id": doc_id, "chunk_size": 30}).json()
    assert set(ch["strategies"]) == {"fixed", "recursive", "sentence", "semantic"}
    for strat in ch["strategies"].values():
        for chunk in strat["chunks"]:
            assert 0 <= chunk["start"] < chunk["end"] <= len(ch["text"])


def test_ingest_rejects_bad_input(client):
    c, _ = client
    bad_type = {"files": [{"name": "a.exe", "content_b64": "AAAA"}]}
    assert c.post("/api/ingest", json=bad_type).status_code == 415
    bad_b64 = {"files": [{"name": "a.txt", "content_b64": "not base64!!"}]}
    assert c.post("/api/ingest", json=bad_b64).status_code == 422
    assert c.post("/api/chunk", json={}).status_code == 422


def test_results_and_index(client):
    c, _ = client
    assert "retrieval" in c.get("/api/results").json()
    assert c.get("/").status_code == 200
    assert c.get("/static/app.js").status_code == 200


def test_static_assets_have_module_safe_mime_types(client):
    c, _ = client
    assert c.get("/static/app.js").headers["content-type"].startswith("text/javascript")
    assert c.get("/static/styles.css").headers["content-type"].startswith("text/css")
