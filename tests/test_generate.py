"""Generation tests. The Claude path is exercised end-to-end through the real
``anthropic`` SDK against an in-process mock HTTP transport that replays a
recorded-shape SSE stream — no network, no API key."""

import json

import pytest

try:  # anthropic>=1.0 is built on httpx2; older releases use httpx
    import httpx2 as httpx
except ImportError:
    import httpx

from sift.generate import (
    FALLBACK_BETA,
    Answer,
    ClaudeGenerator,
    ExtractiveGenerator,
    ResponseCache,
    answer_from_stream,
    make_citation,
    parse_verdict,
)
from sift.types import Passage

anthropic = pytest.importorskip("anthropic")

PASSAGES = [
    Passage(id="eiffel#0", doc_id="eiffel", title="Eiffel Tower",
            text="The Eiffel Tower is 330 metres tall. It was completed in 1889.", start=100, end=163),
    Passage(id="paris#0", doc_id="paris", title="Paris", text="Paris is the capital of France.", start=0, end=31),
]  # fmt: skip


def sse(events: list[dict]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


def stream_events(stop_reason: str = "end_turn") -> list[dict]:
    cit = {"type": "char_location", "cited_text": "The Eiffel Tower is 330 metres tall.", "document_index": 0,
           "document_title": "Eiffel Tower", "start_char_index": 0, "end_char_index": 36}  # fmt: skip
    return [
        {"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
                                              "model": "claude-opus-5", "content": [], "stop_reason": None,
                                              "stop_sequence": None, "usage": {"input_tokens": 1200, "output_tokens": 1}}},  # fmt: skip
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "It is "}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": "", "citations": []}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "citations_delta", "citation": cit}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "330 metres tall"}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": ".\n\nVERDICT: SUPPORTED"}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": {"output_tokens": 40}},
        {"type": "message_stop"},
    ]


def mock_client(handler):
    return anthropic.Anthropic(api_key="test-key", base_url="http://mock.local",
                               http_client=httpx.Client(transport=httpx.MockTransport(handler)), max_retries=0)  # fmt: skip


def test_claude_stream_maps_citations_to_source_spans():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["beta"] = request.headers.get("anthropic-beta", "")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(stream_events()))

    gen = ClaudeGenerator(model="claude-opus-5", client=mock_client(handler))
    events = list(gen.stream("How tall is the Eiffel Tower?", PASSAGES))
    answer = events[-1]["answer_obj"]

    # request shape: documents with citations, question last, fallbacks opted in, no sampling params
    body = seen["body"]
    content = body["messages"][0]["content"]
    assert [c["type"] for c in content] == ["document", "document", "text"]
    assert all(c["citations"] == {"enabled": True} for c in content[:2])
    assert content[0]["source"]["data"] == PASSAGES[0].text
    assert body["fallbacks"] == "default" and FALLBACK_BETA in seen["beta"]
    assert "temperature" not in body and "top_p" not in body
    assert body["output_config"] == {"effort": "medium"}

    # streamed events and the final answer
    assert [e["type"] for e in events if e["type"] in ("delta", "citation")] == ["delta", "citation", "delta", "delta"]
    assert answer.text == "It is 330 metres tall.\n\nVERDICT: SUPPORTED"
    (c,) = answer.citations
    assert (c.n, c.passage_id, c.doc_id) == (1, "eiffel#0", "eiffel")
    assert PASSAGES[0].text[c.start : c.end] == c.cited_text
    assert (c.doc_start, c.doc_end) == (100, 136)  # offset into the *source document*
    assert answer.input_tokens == 1200 and answer.output_tokens == 40
    assert answer.cost_usd == pytest.approx((1200 * 5 + 40 * 25) / 1e6)
    assert parse_verdict(answer.text) == "SUPPORTED"
    assert answer.cited_doc_ids() == ["eiffel"]


def test_refusal_is_checked_before_reading_content():
    def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse(stream_events("refusal")))

    answer = ClaudeGenerator(client=mock_client(handler)).answer("q", PASSAGES)
    assert answer.segments == [] and "refusal" in answer.error


@pytest.mark.parametrize("status,needle", [(401, "ANTHROPIC_API_KEY"), (429, "Rate limited"), (500, "error 500")])
def test_api_errors_become_error_events(status, needle):
    def handler(request):
        return httpx.Response(status, json={"type": "error", "error": {"type": "x", "message": "boom"}})

    events = list(ClaudeGenerator(client=mock_client(handler)).stream("q", PASSAGES))
    assert events[0]["type"] == "error" and needle in events[0]["message"]
    assert events[-1]["type"] == "done" and events[-1]["answer_obj"].error


def test_connection_error():
    def handler(request):
        raise httpx.ConnectError("no route to host")

    answer = ClaudeGenerator(client=mock_client(handler)).answer("q", PASSAGES)
    assert "Could not reach" in answer.error


def test_make_citation_rejects_bad_indices_and_clamps():
    assert make_citation({"type": "page_location", "document_index": 0}, PASSAGES) is None
    assert make_citation({"type": "char_location", "document_index": 9}, PASSAGES) is None
    c = make_citation({"type": "char_location", "document_index": 1, "start_char_index": 5, "end_char_index": 999,
                       "cited_text": "x"}, PASSAGES)  # fmt: skip
    assert c.end == len(PASSAGES[1].text)


@pytest.mark.parametrize(
    "text,label",
    [
        ("blah\nVERDICT: SUPPORTED", "SUPPORTED"),
        ("Verdict - refuted.", "REFUTED"),
        ("**VERDICT:** NOT ENOUGH INFO", "NOT ENOUGH INFO"),
        ("VERDICT: SUPPORTED ... on reflection VERDICT: REFUTED", "REFUTED"),
        ("VERDICT: contradicted", "REFUTED"),
        ("no verdict line", None),
    ],
)
def test_parse_verdict(text, label):
    assert parse_verdict(text) == label


def test_extractive_generator_cites_exact_sentences():
    answer = ExtractiveGenerator(n_sentences=2).answer("When was the tower completed?", PASSAGES)
    assert answer.mode == "extractive"
    assert answer.citations
    for c in answer.citations:
        p = next(p for p in PASSAGES if p.id == c.passage_id)
        assert p.text[c.start : c.end] == c.cited_text
    assert "1889" in answer.text


def test_response_cache_roundtrip(tmp_path):
    cache = ResponseCache(tmp_path / "c.jsonl")
    ans = answer_from_stream(ExtractiveGenerator().stream("tower", PASSAGES))
    key = ResponseCache.key({"a": 1})
    cache.put(key, ans)
    again = ResponseCache(tmp_path / "c.jsonl").get(key)
    assert isinstance(again, Answer) and again.text == ans.text
    assert [c.cited_text for c in again.citations] == [c.cited_text for c in ans.citations]
