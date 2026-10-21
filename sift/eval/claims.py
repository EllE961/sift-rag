"""End-to-end claim verification on SciFact (retrieve → read → verdict + cite).

    python -m sift.eval claims --verifier nli            # offline, all 300 dev claims
    python -m sift.eval claims --verifier claude --n 99  # Claude, balanced sample

Conditions (what evidence the verifier sees):

* ``no_evidence`` / ``closed_book`` — no documents. For NLI this is the
  always-NEI floor; for Claude it is parametric knowledge only.
* ``bm25@5``    — top-5 BM25 abstracts.
* ``sift@5``    — top-5 from the full pipeline (BM25 ∥ dense → RRF → rerank).
* ``oracle``    — the gold evidence abstracts (or, for NEI claims, the abstracts
  the claim was written from, which by construction lack evidence).

Metrics: label accuracy and macro-F1 over {SUPPORTED, REFUTED, NEI};
evidence recall of the retrieved set; doc-level citation precision/recall;
sentence-level *rationale precision* (does a cited span overlap a gold
rationale sentence?); NEI abstention rate.

The NLI verifier's thresholds are tuned per condition on the SciFact *train*
claims (809) and evaluated on *dev* (300) — dev is never used for tuning.
"""

from __future__ import annotations

import json
import pickle
import random
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..collection import load_or_build_beir
from ..config import CACHE_DIR, DEFAULT_BLEND, RESULTS_DIR, Settings, tuned_blends
from ..datasets import LABELS, Claim, load_scifact_claims
from ..embed import TransformerEmbedder
from ..generate import (
    CLOSED_BOOK_INSTRUCTIONS,
    VERIFY_INSTRUCTIONS,
    Answer,
    Citation,
    ClaudeGenerator,
    ResponseCache,
    parse_verdict,
)
from ..pipeline import Retriever, SearchConfig
from ..rerank import CrossEncoderReranker
from ..types import Passage

K = 5


# ------------------------------------------------------------------- metrics
def macro_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] = LABELS) -> float:
    return sum(per_class_f1(gold, pred, labels).values()) / len(labels)


def per_class_f1(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] = LABELS) -> dict[str, float]:
    out = {}
    for lab in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[lab] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return out


def confusion(gold: Sequence[str], pred: Sequence[str], labels: Sequence[str] = LABELS) -> list[list[int]]:
    idx = {lab: i for i, lab in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        m[idx[g]][idx[p]] += 1
    return m


def cited_sentences(citation: Citation, passage: Passage) -> set[int]:
    """Gold-segmentation sentence indices overlapped by a citation span."""
    spans = passage.metadata.get("sentence_spans", [])
    out = set()
    for i, (a, b) in enumerate(spans):
        if min(b, citation.end) - max(a, citation.start) > 0:
            out.add(i)
    return out


def score_condition(claims: Sequence[Claim], records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    gold = [c.label for c in claims]
    pred = [r["label"] for r in records]
    n_cited = n_cited_gold = 0
    n_gold_docs = n_gold_docs_cited = 0
    n_sent = n_sent_gold = 0
    ev_recall: list[float] = []
    for c, r in zip(claims, records):
        if not c.evidence_doc_ids:
            continue
        ev = set(c.evidence_doc_ids)
        if r["retrieved"] is not None:
            ev_recall.append(len(ev.intersection(r["retrieved"])) / len(ev))
        cited_docs = set(r["cited_docs"])
        n_gold_docs += len(ev)
        n_gold_docs_cited += len(ev & cited_docs)
        if r["label"] != "NOT ENOUGH INFO":
            n_cited += len(cited_docs)
            n_cited_gold += len(cited_docs & ev)
            for doc_id, sents in r["cited_sentences"].items():
                gold_sents = set(c.rationales.get(doc_id, []))
                n_sent += len(sents)
                n_sent_gold += len(set(sents) & gold_sents)
    nei = [r["label"] == "NOT ENOUGH INFO" for c, r in zip(claims, records) if c.label == "NOT ENOUGH INFO"]
    return {
        "n": len(claims),
        "accuracy": round(sum(g == p for g, p in zip(gold, pred)) / len(gold), 4),
        "macro_f1": round(macro_f1(gold, pred), 4),
        "f1_per_class": {k: round(v, 4) for k, v in per_class_f1(gold, pred).items()},
        "confusion": {"labels": list(LABELS), "matrix": confusion(gold, pred)},
        "evidence_recall": round(sum(ev_recall) / len(ev_recall), 4) if ev_recall else None,
        "citation_precision": round(n_cited_gold / n_cited, 4) if n_cited else None,
        "citation_recall": round(n_gold_docs_cited / n_gold_docs, 4) if n_gold_docs else None,
        "rationale_precision": round(n_sent_gold / n_sent, 4) if n_sent else None,
        "nei_abstention": round(sum(nei) / len(nei), 4) if nei else None,
        "pred_distribution": dict(Counter(pred)),
    }


# ------------------------------------------------------------------ evidence
def release_passages() -> dict[str, Passage]:
    """SciFact abstracts with the release's gold sentence segmentation."""
    from ..datasets import download_scifact_release

    base = download_scifact_release()
    out: dict[str, Passage] = {}
    with open(base / "corpus.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            spans, pos, parts = [], 0, []
            for s in row["abstract"]:
                s = s.strip()
                spans.append((pos, pos + len(s)))
                parts.append(s)
                pos += len(s) + 1
            text = " ".join(parts)
            did = str(row["doc_id"])
            out[did] = Passage(id=did, doc_id=did, text=text, title=row["title"], start=0, end=len(text),
                               metadata={"sentence_spans": spans})  # fmt: skip
    return out


def production_config() -> SearchConfig:
    """The configuration the service ships: hybrid RRF + cross-encoder blended with the
    train-tuned weight for SciFact (see `python -m sift.eval blend`)."""
    return SearchConfig(mode="hybrid_rerank", k=K, rerank_depth=50, depth=100, ef=128,
                        blend=tuned_blends().get("scifact", DEFAULT_BLEND))  # fmt: skip


def evidence_sets(claims: Sequence[Claim], retriever: Retriever, split: str = "") -> dict[str, list[list[str] | None]]:
    """condition -> per-claim list of doc ids (None = no retrieval). Cached on disk."""
    cfg = production_config()
    cache = CACHE_DIR / f"claims_evidence_{split}_{cfg.blend}.json"
    if split and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    bm25 = retriever.collection.bm25
    assert bm25 is not None
    out: dict[str, list[list[str] | None]] = {"no_evidence": [], "bm25@5": [], "sift@5": [], "oracle": []}
    for c in claims:
        out["no_evidence"].append(None)
        out["bm25@5"].append([pid for pid, _ in bm25.search(c.claim, K)])
        out["sift@5"].append(retriever.search(c.claim, cfg).ranking())
        out["oracle"].append(c.evidence_doc_ids or c.cited_doc_ids)
    if split:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def sample_balanced(claims: Sequence[Claim], n: int, seed: int = 13) -> list[Claim]:
    rng = random.Random(seed)
    by_label = {lab: [c for c in claims if c.label == lab] for lab in LABELS}
    per = n // len(LABELS)
    out = []
    for lab in LABELS:
        pool = by_label[lab][:]
        rng.shuffle(pool)
        out.extend(pool[:per])
    return sorted(out, key=lambda c: int(c.id))


def _record(label: str, citations: Sequence[Citation], retrieved: list[str] | None,
            by_id: dict[str, Passage]) -> dict[str, Any]:  # fmt: skip
    cited_docs: list[str] = []
    sents: dict[str, list[int]] = {}
    for cit in citations:
        if cit.doc_id not in cited_docs:
            cited_docs.append(cit.doc_id)
        p = by_id.get(cit.doc_id)
        if p is not None:
            sents.setdefault(cit.doc_id, [])
            sents[cit.doc_id] = sorted(set(sents[cit.doc_id]) | cited_sentences(cit, p))
    return {"label": label, "retrieved": retrieved, "cited_docs": cited_docs, "cited_sentences": sents}


# ----------------------------------------------------------------------- NLI
def run_nli(retriever: Retriever, by_id: dict[str, Passage], n: int | None = None) -> dict[str, Any]:
    from ..verify import NLIVerifier, decide, tune_thresholds

    verifier = NLIVerifier(embedder=retriever.embedder)
    train, dev = load_scifact_claims("train"), load_scifact_claims("dev")
    if n:
        dev = sample_balanced(dev, n)
    t0 = time.time()
    ev_train = evidence_sets(train, retriever, "train")
    ev_dev = evidence_sets(dev, retriever, f"dev{len(dev)}")
    retrieval_s = time.time() - t0

    def score_with_retry(claim: str, passages: list[Passage]):
        # The GPU is shared; a transient cuBLAS failure must not cost the whole run.
        for attempt in range(4):
            try:
                return verifier.score(claim, passages)
            except RuntimeError as exc:
                if "CUDA" not in str(exc) or attempt == 3:
                    raise
                import torch

                torch.cuda.empty_cache()
                time.sleep(5 * (attempt + 1))
        raise AssertionError("unreachable")

    def scores(claims: Sequence[Claim], ev: list[list[str] | None], key: str):
        path = CACHE_DIR / f"nli_scores_{key}.pkl"
        if path.exists():
            cached = pickle.loads(path.read_bytes())
            return [([by_id[d] for d in (docs or []) if d in by_id], sc) for docs, sc in zip(ev, cached)]
        out, raw = [], []
        for c, docs in zip(claims, ev):
            passages = [by_id[d] for d in (docs or []) if d in by_id]
            sc = score_with_retry(c.claim, passages)
            out.append((passages, sc))
            raw.append(sc)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(raw))
        return out

    results: dict[str, Any] = {}
    per_claim: dict[str, list[dict[str, Any]]] = {}
    for cond in ("no_evidence", "bm25@5", "sift@5", "oracle"):
        t = time.time()
        tag = cond.replace("@", "at")
        tr = scores(train, ev_train[cond], f"train_{tag}_{production_config().blend}")
        dv = scores(dev, ev_dev[cond], f"dev{len(dev)}_{tag}_{production_config().blend}")
        best = tune_thresholds([x[1] for x in tr], [c.label for c in train])
        te, tc, ts = best["tau_entail"], best["tau_contradict"], best["tau_sim"]
        records = []
        for c, docs, (passages, s) in zip(dev, ev_dev[cond], dv):
            e, cc = verifier.aggregate(s, ts)
            label = decide(e, cc, te, tc) if passages else "NOT ENOUGH INFO"
            cits = verifier.explain(label, s, passages, tau_s=ts)
            rec = _record(label, cits, docs, by_id)
            rec.update({"claim_id": c.id, "gold": c.label, "entail": round(e, 4), "contradict": round(cc, 4)})
            records.append(rec)
        metrics = score_condition(dev, records)
        metrics.update({"tau_entail": te, "tau_contradict": tc, "tau_sim": ts,
                        "train_macro_f1": round(best["macro_f1"], 4),
                        "seconds": round(time.time() - t, 1)})  # fmt: skip
        results[cond] = metrics
        per_claim[cond] = records
        print(f"  [nli] {cond:12s} acc={metrics['accuracy']:.3f} F1={metrics['macro_f1']:.3f} "
              f"cite-P={metrics['citation_precision']} rat-P={metrics['rationale_precision']}")  # fmt: skip
    return {
        "verifier": "nli",
        "model": verifier.model_name,
        "n_dev": len(dev),
        "n_train": len(train),
        "dev_label_distribution": dict(Counter(c.label for c in dev)),
        "retrieval_seconds": round(retrieval_s, 1),
        "pipeline_config": production_config().__dict__,
        "conditions": results,
        "per_claim": per_claim,
    }


# -------------------------------------------------------------------- Claude
def run_claude(retriever: Retriever, by_id: dict[str, Passage], n: int, model: str | None = None,
               workers: int = 4) -> dict[str, Any]:  # fmt: skip
    settings = Settings()
    gen = ClaudeGenerator(model=model or settings.llm_model, effort=settings.llm_effort, max_tokens=4096)
    cache = ResponseCache(CACHE_DIR / "claude_responses.jsonl")
    dev = sample_balanced(load_scifact_claims("dev"), n)
    ev = evidence_sets(dev, retriever, f"claude{n}")

    def call(claim: Claim, docs: list[str] | None) -> Answer:
        passages = [by_id[d] for d in (docs or []) if d in by_id]
        instr = (VERIFY_INSTRUCTIONS if docs is not None else CLOSED_BOOK_INSTRUCTIONS).format(claim=claim.claim)
        params = gen.build_request(claim.claim, passages, instr)
        key = ResponseCache.key(params)
        hit = cache.get(key)
        if hit is not None and not hit.error:
            return hit
        ans = gen.answer(claim.claim, passages, instr)
        if not ans.error:
            cache.put(key, ans)
        return ans

    # Fail fast (and cheaply) on bad credentials before fanning out.
    probe = call(dev[0], None)
    if probe.error:
        raise SystemExit(f"Claude probe failed: {probe.error}")

    results: dict[str, Any] = {}
    per_claim: dict[str, list[dict[str, Any]]] = {}
    totals = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    for cond, docs_list in (("closed_book", ev["no_evidence"]), ("bm25@5", ev["bm25@5"]), ("sift@5", ev["sift@5"]),
                            ("oracle", ev["oracle"])):  # fmt: skip
        t = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answers = list(pool.map(lambda cd: call(*cd), zip(dev, docs_list)))
        records = []
        for c, docs, ans in zip(dev, docs_list, answers):
            label = parse_verdict(ans.text) or "NOT ENOUGH INFO"
            rec = _record(label, ans.citations, docs, by_id)
            rec.update({"claim_id": c.id, "gold": c.label, "parsed": parse_verdict(ans.text) is not None,
                        "error": ans.error, "answer": ans.text[-1200:]})  # fmt: skip
            records.append(rec)
            totals["input_tokens"] += ans.input_tokens
            totals["output_tokens"] += ans.output_tokens
            totals["cost_usd"] += ans.cost_usd
        metrics = score_condition(dev, records)
        lat = sorted(a.latency_ms for a in answers if a.latency_ms)
        metrics.update({
            "parse_failures": sum(1 for r in records if not r["parsed"]),
            "errors": sum(1 for r in records if r["error"]),
            "latency_ms_p50": lat[len(lat) // 2] if lat else None,
            "seconds": round(time.time() - t, 1),
        })
        results[cond] = metrics
        per_claim[cond] = records
        print(f"  [claude] {cond:12s} acc={metrics['accuracy']:.3f} F1={metrics['macro_f1']:.3f} "
              f"cite-P={metrics['citation_precision']} rat-P={metrics['rationale_precision']}")  # fmt: skip
    totals["cost_usd"] = round(totals["cost_usd"], 4)
    return {
        "verifier": "claude",
        "model": gen.model,
        "effort": gen.effort,
        "n_dev": len(dev),
        "dev_label_distribution": dict(Counter(c.label for c in dev)),
        "usage": totals,
        "conditions": results,
        "per_claim": per_claim,
    }


def main(verifier: str = "nli", n: int | None = None, model: str | None = None) -> dict[str, Any]:
    bge = TransformerEmbedder("BAAI/bge-small-en-v1.5", batch_size=64)
    col = load_or_build_beir("scifact", bge)
    retriever = Retriever(col, bge, CrossEncoderReranker())
    by_id = release_passages()
    if verifier == "nli":
        out = run_nli(retriever, by_id, n=n)
    else:
        out = run_claude(retriever, by_id, n=n or 99, model=model)
    out["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path: Path = RESULTS_DIR / f"claims_{verifier}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return out
