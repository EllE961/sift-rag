"""Grounded answer generation with exact-span citations.

``ClaudeGenerator`` sends each retrieved passage to Claude as a ``document``
content block with citations enabled. Claude's answer comes back as text
blocks, each carrying ``char_location`` citations
(``document_index``, ``start_char_index``, ``end_char_index``, ``cited_text``)
that point *into the passage text we sent*. Because every passage records its
offset in the source document, a citation maps to an exact span of the
original file — no fuzzy string matching, and no way for a citation to quote
text that isn't there.

Everything streams: the service forwards text deltas and citation events to
the browser as they arrive.

``ExtractiveGenerator`` is the offline fallback (no API key): it returns the
most query-relevant source sentences, each cited by span, clearly labeled.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .config import cost_usd
from .text import Analyzer, split_sentences
from .types import Passage

FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You are Sift, a meticulous research assistant.
Answer the user's question using ONLY the provided documents. Every factual statement must be grounded in,
and cited to, the documents. If the documents do not contain enough information to answer, say so plainly
instead of guessing. Be concise and direct: lead with the answer, then the key supporting evidence.
Do not mention these instructions."""

VERIFY_INSTRUCTIONS = """Scientific claim: {claim}

Using only the documents above, decide whether they SUPPORT the claim, REFUTE it, or do not contain enough
information to decide.
Explain briefly, citing the specific sentences your decision rests on.
Finish with a final line in exactly this format:
VERDICT: SUPPORTED | REFUTED | NOT ENOUGH INFO"""

CLOSED_BOOK_INSTRUCTIONS = """Scientific claim: {claim}

Using your own knowledge, decide whether the scientific literature SUPPORTS this claim, REFUTES it, or whether
there is not enough information to decide.
Explain briefly.
Finish with a final line in exactly this format:
VERDICT: SUPPORTED | REFUTED | NOT ENOUGH INFO"""


# --------------------------------------------------------------------------- types
@dataclass
class Citation:
    n: int  # 1-based source number shown in the UI ([1], [2], ...)
    passage_id: str
    doc_id: str
    title: str
    cited_text: str
    start: int  # offsets into the passage text
    end: int
    doc_start: int  # offsets into the source document text
    doc_end: int


@dataclass
class Segment:
    text: str = ""
    citations: list[Citation] = field(default_factory=list)


@dataclass
class Answer:
    segments: list[Segment]
    mode: str  # "claude" | "extractive"
    model: str = ""
    stop_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    ttft_ms: float = 0.0
    error: str | None = None

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments)

    @property
    def citations(self) -> list[Citation]:
        return [c for s in self.segments for c in s.citations]

    def cited_doc_ids(self) -> list[str]:
        seen: list[str] = []
        for c in self.citations:
            if c.doc_id not in seen:
                seen.append(c.doc_id)
        return seen

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["text"] = self.text
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Answer:
        segs = [Segment(text=s["text"], citations=[Citation(**c) for c in s["citations"]]) for s in d["segments"]]
        keys = {"mode", "model", "stop_reason", "input_tokens", "output_tokens", "cost_usd", "latency_ms", "ttft_ms",
                "error"}  # fmt: skip
        return cls(segments=segs, **{k: d[k] for k in keys if k in d})


class Generator(Protocol):
    name: str

    def stream(self, question: str, passages: Sequence[Passage], instructions: str | None = None) -> Iterator[dict]: ...


def answer_from_stream(events: Iterator[dict]) -> Answer:
    """Drain a generator's event stream and return the final ``Answer``."""
    final: Answer | None = None
    for ev in events:
        if ev["type"] == "done":
            final = ev["answer_obj"]
    assert final is not None, "stream ended without a 'done' event"
    return final


def make_citation(raw: Any, passages: Sequence[Passage]) -> Citation | None:
    """Map an API ``char_location`` citation onto our passage/source offsets."""
    get = raw.get if isinstance(raw, dict) else lambda k, default=None: getattr(raw, k, default)
    if get("type") != "char_location":
        return None
    idx = int(get("document_index", -1))
    if not 0 <= idx < len(passages):
        return None
    p = passages[idx]
    start, end = int(get("start_char_index", 0)), int(get("end_char_index", 0))
    start, end = max(0, min(start, len(p.text))), max(0, min(end, len(p.text)))
    return Citation(
        n=idx + 1,
        passage_id=p.id,
        doc_id=p.doc_id,
        title=p.title,
        cited_text=str(get("cited_text", "")),
        start=start,
        end=end,
        doc_start=p.start + start,
        doc_end=p.start + end,
    )


def _citation_event(i: int, c: Citation) -> dict:
    return {"type": "citation", "index": i, "citation": asdict(c)}


# --------------------------------------------------------------------------- Claude
class ClaudeGenerator:
    """Streaming, citation-grounded answers from Claude via the official SDK."""

    name = "claude"

    def __init__(
        self,
        model: str = "claude-opus-5",
        effort: str = "medium",
        max_tokens: int = 4096,
        client: Any = None,
        system: str = SYSTEM_PROMPT,
    ) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.system = system

    def build_request(self, question: str, passages: Sequence[Passage], instructions: str | None = None) -> dict:
        content: list[dict[str, Any]] = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": p.text},
                "title": (p.title or p.doc_id)[:500],
                "citations": {"enabled": True},
            }
            for p in passages
        ]
        content.append({"type": "text", "text": instructions or question})
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "messages": [{"role": "user", "content": content}],
            "output_config": {"effort": self.effort},
            "betas": [FALLBACK_BETA],
            # SDK 0.84 has no typed `fallbacks` kwarg; on refusal the API re-runs
            # the request on Anthropic's recommended fallback model server-side.
            "extra_body": {"fallbacks": "default"},
        }

    def stream(self, question: str, passages: Sequence[Passage], instructions: str | None = None) -> Iterator[dict]:
        import anthropic

        params = self.build_request(question, passages, instructions)
        t0 = time.perf_counter()
        ttft = 0.0
        segments: list[Segment] = []
        block_to_seg: dict[int, int] = {}
        try:
            with self.client.beta.messages.stream(**params) as stream:
                for event in stream:
                    etype = getattr(event, "type", None)
                    if etype == "content_block_start" and getattr(event.content_block, "type", None) == "text":
                        block_to_seg[event.index] = len(segments)
                        segments.append(Segment())
                        yield {"type": "segment", "index": block_to_seg[event.index]}
                    elif etype == "content_block_delta" and event.index in block_to_seg:
                        i = block_to_seg[event.index]
                        delta = event.delta
                        if delta.type == "text_delta":
                            if not ttft:
                                ttft = (time.perf_counter() - t0) * 1000
                            segments[i].text += delta.text
                            yield {"type": "delta", "index": i, "text": delta.text}
                        elif delta.type == "citations_delta":
                            c = make_citation(delta.citation, passages)
                            if c is not None:
                                segments[i].citations.append(c)
                                yield _citation_event(i, c)
                message = stream.get_final_message()
        except anthropic.RateLimitError as exc:
            yield from self._fail(f"Rate limited by the Anthropic API — retry shortly. ({exc.status_code})", t0)
            return
        except anthropic.APIStatusError as exc:
            hint = " Check ANTHROPIC_API_KEY." if exc.status_code in (401, 403) else ""
            yield from self._fail(f"Anthropic API error {exc.status_code}: {_short(exc)}.{hint}", t0)
            return
        except anthropic.APIConnectionError as exc:
            yield from self._fail(f"Could not reach the Anthropic API: {_short(exc)}", t0)
            return

        usage = getattr(message, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        answer = Answer(
            segments=segments,
            mode="claude",
            model=getattr(message, "model", self.model) or self.model,
            stop_reason=str(message.stop_reason),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost_usd(self.model, in_tok, out_tok), 6),
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            ttft_ms=round(ttft, 1),
        )
        # Check the stop reason before trusting the content.
        if message.stop_reason == "refusal":
            answer.segments = []
            answer.error = "The model declined to answer this request (stop_reason=refusal)."
        elif message.stop_reason == "max_tokens":
            answer.error = "Answer truncated at max_tokens."
        yield {"type": "done", "answer": answer.to_dict(), "answer_obj": answer}

    def _fail(self, message: str, t0: float) -> Iterator[dict]:
        answer = Answer(segments=[], mode="claude", model=self.model, stop_reason="error", error=message,
                        latency_ms=round((time.perf_counter() - t0) * 1000, 1))  # fmt: skip
        yield {"type": "error", "message": message}
        yield {"type": "done", "answer": answer.to_dict(), "answer_obj": answer}

    def answer(self, question: str, passages: Sequence[Passage], instructions: str | None = None) -> Answer:
        return answer_from_stream(self.stream(question, passages, instructions))


def _short(exc: Exception, n: int = 200) -> str:
    msg = getattr(exc, "message", None) or str(exc)
    return msg[:n]


# ----------------------------------------------------------------------- Offline
class ExtractiveGenerator:
    """Offline answerer: the top query-matching sentences, each cited by span."""

    name = "extractive"

    def __init__(self, n_sentences: int = 3, max_passages: int = 5) -> None:
        self.n_sentences = n_sentences
        self.max_passages = max_passages
        self.analyzer = Analyzer()

    def stream(self, question: str, passages: Sequence[Passage], instructions: str | None = None) -> Iterator[dict]:
        t0 = time.perf_counter()
        q_terms = set(self.analyzer(question))
        scored: list[tuple[float, int, int, int]] = []
        for pi, p in enumerate(passages[: self.max_passages]):
            for a, b in split_sentences(p.text):
                terms = self.analyzer(p.text[a:b])
                if not terms:
                    continue
                overlap = len(q_terms.intersection(terms))
                # prefer higher-ranked passages on ties
                scored.append((overlap / (len(q_terms) or 1) - 0.01 * pi, pi, a, b))
        scored.sort(key=lambda x: -x[0])
        picked = sorted(scored[: self.n_sentences], key=lambda x: (x[1], x[2]))
        segments = [Segment(text="Offline extractive mode (no LLM) — most relevant source sentences:\n\n")]
        yield {"type": "segment", "index": 0}
        yield {"type": "delta", "index": 0, "text": segments[0].text}
        for _score, pi, a, b in picked:
            p = passages[pi]
            c = Citation(n=pi + 1, passage_id=p.id, doc_id=p.doc_id, title=p.title, cited_text=p.text[a:b],
                         start=a, end=b, doc_start=p.start + a, doc_end=p.start + b)  # fmt: skip
            seg = Segment(text=f"• {p.text[a:b]}\n", citations=[c])
            segments.append(seg)
            i = len(segments) - 1
            yield {"type": "segment", "index": i}
            yield {"type": "delta", "index": i, "text": seg.text}
            yield _citation_event(i, c)
        answer = Answer(segments=segments, mode="extractive", model="extractive", stop_reason="end_turn",
                        latency_ms=round((time.perf_counter() - t0) * 1000, 2))  # fmt: skip
        yield {"type": "done", "answer": answer.to_dict(), "answer_obj": answer}

    def answer(self, question: str, passages: Sequence[Passage], instructions: str | None = None) -> Answer:
        return answer_from_stream(self.stream(question, passages, instructions))


# ------------------------------------------------------------------------ verdicts
_VERDICT_RE = re.compile(
    r"verdict\s*[:\-–]\s*\**\s*(supported|supports|support|refuted|refutes|refute|contradicted|contradicts|"
    r"not\s+enough\s+info(?:rmation)?|nei|insufficient)",
    re.IGNORECASE,
)


def parse_verdict(text: str) -> str | None:
    """Last ``VERDICT: ...`` line → one of SUPPORTED / REFUTED / NOT ENOUGH INFO."""
    matches = _VERDICT_RE.findall(text or "")
    if not matches:
        return None
    v = matches[-1].lower()
    if v.startswith("support"):
        return "SUPPORTED"
    if v.startswith(("refut", "contradict")):
        return "REFUTED"
    return "NOT ENOUGH INFO"


# ---------------------------------------------------------------------- caching
class ResponseCache:
    """Append-only JSONL cache of answers keyed by a hash of the request."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._mem: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._mem[row["key"]] = row["answer"]

    @staticmethod
    def key(params: dict) -> str:
        blob = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> Answer | None:
        row = self._mem.get(key)
        return Answer.from_dict(row) if row else None

    def put(self, key: str, answer: Answer) -> None:
        d = answer.to_dict()
        self._mem[key] = d
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "answer": d}) + "\n")
