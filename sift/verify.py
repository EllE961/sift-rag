"""Offline claim verification with a small NLI cross-encoder.

Classic SciFact-style pipeline (cf. VeriSci, Wadden et al. 2020) — no LLM:

1. Split each retrieved abstract into sentences.
2. Score every (sentence ⇒ claim) pair with an NLI model, giving
   P(entailment), P(contradiction), P(neutral).
3. Gate by relevance: NLI models label *unrelated* pairs as contradictions
   with high confidence ("The moon is made of cheese" vs. any abstract ->
   P(contradiction) ~ 0.99). Only sentences whose embedding similarity to the
   claim clears τ_s may support or refute.
4. Aggregate: E = max entailment, C = max contradiction over gated sentences.
5. Decide with thresholds tuned on the *train* claims::

       SUPPORTED        if E >= τ_e and E >= C
       REFUTED          elif C >= τ_c
       NOT ENOUGH INFO  otherwise

6. Cite the sentence(s) that drove the decision, as exact character spans.

It serves two purposes: a fully offline end-to-end baseline, and a cheap
"second opinion" against which the LLM's verdicts can be compared.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .config import resolve_device
from .generate import Citation
from .text import split_sentences
from .types import Passage

LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO")


@dataclass
class SentenceScore:
    passage_index: int
    start: int
    end: int
    entail: float
    contradict: float
    neutral: float
    sim: float = 1.0  # cosine(claim, sentence) when an embedder is available


@dataclass
class Verdict:
    label: str
    entail: float
    contradict: float
    citations: list[Citation] = field(default_factory=list)


def decide(entail: float, contradict: float, tau_e: float, tau_c: float) -> str:
    if entail >= tau_e and entail >= contradict:
        return "SUPPORTED"
    if contradict >= tau_c:
        return "REFUTED"
    return "NOT ENOUGH INFO"


def sentence_spans(p: Passage) -> list[tuple[int, int]]:
    spans = p.metadata.get("sentence_spans") if p.metadata else None
    return [tuple(s) for s in spans] if spans else split_sentences(p.text)


class NLIVerifier:
    name = "nli"

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-MiniLM2-L6-H768",
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 256,
        embedder: object | None = None,
    ) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        # The slow (pure-Python BPE) tokenizer avoids depending on the exact `tokenizers` version
        # that serialized tokenizer.json.
        self.model = CrossEncoder(model_name, device=resolve_device(device), max_length=max_length,
                                  tokenizer_args={"use_fast": False})  # fmt: skip
        id2label = {int(k): v.lower() for k, v in self.model.model.config.id2label.items()}
        lookup = {v: k for k, v in id2label.items()}
        self._idx = (lookup["entailment"], lookup["contradiction"], lookup["neutral"])
        self.batch_size = batch_size
        self.embedder = embedder

    def score(self, claim: str, passages: Sequence[Passage]) -> list[SentenceScore]:
        units: list[tuple[int, int, int]] = []
        for pi, p in enumerate(passages):
            for a, b in sentence_spans(p):
                units.append((pi, a, b))
        if not units:
            return []
        pairs = [(passages[pi].text[a:b], claim) for pi, a, b in units]
        probs = self.model.predict(pairs, batch_size=self.batch_size, apply_softmax=True, show_progress_bar=False)
        probs = np.asarray(probs, dtype=np.float32)
        if self.embedder is not None:
            q = self.embedder.encode_queries([claim])[0]
            sims = self.embedder.encode_docs([t for t, _ in pairs]) @ q
        else:
            sims = np.ones(len(units), dtype=np.float32)
        ie, ic, inn = self._idx
        return [
            SentenceScore(pi, a, b, float(pr[ie]), float(pr[ic]), float(pr[inn]), float(sim))
            for (pi, a, b), pr, sim in zip(units, probs, sims)
        ]

    @staticmethod
    def aggregate(scores: Sequence[SentenceScore], tau_s: float = 0.0) -> tuple[float, float]:
        gated = [s for s in scores if s.sim >= tau_s]
        if not gated:
            return 0.0, 0.0
        return max(s.entail for s in gated), max(s.contradict for s in gated)

    @staticmethod
    def explain(label: str, scores: Sequence[SentenceScore], passages: Sequence[Passage], top: int = 2,
                floor: float = 0.5, tau_s: float = 0.0) -> list[Citation]:  # fmt: skip
        scores = [s for s in scores if s.sim >= tau_s]
        if label == "NOT ENOUGH INFO" or not scores:
            return []
        key = (lambda s: s.entail) if label == "SUPPORTED" else (lambda s: s.contradict)
        best = sorted(scores, key=key, reverse=True)[:top]
        out = []
        for i, s in enumerate(best):
            if i > 0 and key(s) < floor:
                break
            p = passages[s.passage_index]
            out.append(Citation(n=s.passage_index + 1, passage_id=p.id, doc_id=p.doc_id, title=p.title,
                                cited_text=p.text[s.start:s.end], start=s.start, end=s.end,
                                doc_start=p.start + s.start, doc_end=p.start + s.end))  # fmt: skip
        return out

    def verify(self, claim: str, passages: Sequence[Passage], tau_e: float = 0.5, tau_c: float = 0.5,
               tau_s: float = 0.0) -> Verdict:  # fmt: skip
        scores = self.score(claim, passages)
        e, c = self.aggregate(scores, tau_s)
        label = decide(e, c, tau_e, tau_c)
        return Verdict(label, e, c, self.explain(label, scores, passages, tau_s=tau_s))


PROB_GRID = (*np.round(np.arange(0.05, 1.0, 0.05), 2).tolist(), 0.97, 0.98, 0.99, 0.995, 0.998)
SIM_GRID = (0.0, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)


def tune_thresholds(
    scores: Sequence[Sequence[SentenceScore]], gold: Sequence[str],
    prob_grid: Sequence[float] = PROB_GRID, sim_grid: Sequence[float] = SIM_GRID,
) -> dict[str, float]:  # fmt: skip
    """Grid-search (τ_s, τ_e, τ_c) maximizing macro-F1 on labelled claims."""
    from .eval.claims import macro_f1

    best = {"tau_sim": 0.0, "tau_entail": 0.5, "tau_contradict": 0.5, "macro_f1": -1.0}
    for ts in sim_grid:
        agg = [NLIVerifier.aggregate(sc, ts) for sc in scores]
        for te in prob_grid:
            for tc in prob_grid:
                pred = [decide(e, c, te, tc) for e, c in agg]
                f1 = macro_f1(gold, pred)
                if f1 > best["macro_f1"]:
                    best = {"tau_sim": float(ts), "tau_entail": float(te), "tau_contradict": float(tc), "macro_f1": f1}
    return best
