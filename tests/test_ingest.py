import pytest

from sift.chunking import SentenceChunker
from sift.ingest import load_bytes, to_passages


def minimal_pdf(text: str) -> bytes:
    """A hand-assembled one-page PDF with a single text object."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return out


def test_markdown_title_and_cleanup():
    doc = load_bytes("my_notes.md", b"# Real Title\r\n\r\n\r\n\r\nBody   text\twith  spaces.\n")
    assert doc.title == "Real Title"
    assert doc.text == "# Real Title\n\nBody text with spaces."
    assert doc.id.startswith("my_notes-")


def test_html_strips_scripts_and_keeps_blocks():
    html = b"<html><head><title>Page</title><style>p{}</style></head><body><h1>Head</h1><p>One.</p><script>x()</script><p>Two.</p></body></html>"
    doc = load_bytes("page.html", html)
    assert doc.title == "Page"
    assert "x()" not in doc.text and "p{}" not in doc.text
    assert doc.text.split() == ["Head", "One.", "Two."]


def test_pdf_text_and_page_offsets():
    doc = load_bytes("paper.pdf", minimal_pdf("Hello from a PDF"))
    assert "Hello from a PDF" in doc.text
    page = doc.metadata["pages"][0]
    assert doc.text[page["start"] : page["end"]].strip() == doc.text.strip()


def test_unsupported_type():
    with pytest.raises(ValueError):
        load_bytes("x.docx", b"...")


def test_to_passages_offsets():
    doc = load_bytes("a.txt", b"First sentence here. Second one follows. A third closes it.")
    ps = to_passages([doc], SentenceChunker(4, 0))
    assert len(ps) == 3
    for p in ps:
        assert doc.text[p.start : p.end] == p.text and p.doc_id == doc.id
    whole = to_passages([doc])
    assert len(whole) == 1 and whole[0].text == doc.text
