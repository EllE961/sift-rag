"""Document loaders (.txt / .md / .html / .pdf) and passage construction."""

from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Iterable
from html.parser import HTMLParser
from pathlib import Path

from .chunking import Chunker
from .types import Document, Passage

SUPPORTED = (".txt", ".md", ".markdown", ".html", ".htm", ".pdf")


class _HTMLText(HTMLParser):
    _BLOCK = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article",
              "pre", "blockquote", "table"}  # fmt: skip
    _SKIP = {"script", "style", "noscript", "head", "svg"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK:
            self.parts.append("\n\n" if tag.startswith("h") or tag == "p" else "\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _doc_id(name: str, data: bytes) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(name).stem)[:40].strip("-") or "doc"
    return f"{stem}-{hashlib.sha1(data).hexdigest()[:8]}"


def load_bytes(name: str, data: bytes) -> Document:
    """Parse an uploaded file into a ``Document``."""
    suffix = Path(name).suffix.lower()
    title = Path(name).stem.replace("_", " ").replace("-", " ")
    metadata: dict = {"filename": name, "bytes": len(data)}
    if suffix in (".txt", ".md", ".markdown"):
        text = _clean(data.decode("utf-8", errors="replace"))
        if suffix != ".txt":
            heading = re.search(r"^#\s+(.+)$", text, flags=re.M)
            title = heading.group(1).strip() if heading else title
    elif suffix in (".html", ".htm"):
        parser = _HTMLText()
        parser.feed(data.decode("utf-8", errors="replace"))
        text = _clean("".join(parser.parts))
        title = _clean(parser.title) or title
    elif suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages, offsets, pos = [], [], 0
        for page_no, page in enumerate(reader.pages, start=1):
            page_text = _clean(page.extract_text() or "")
            pages.append(page_text)
            offsets.append({"page": page_no, "start": pos, "end": pos + len(page_text)})
            pos += len(page_text) + 2
        text = "\n\n".join(pages)
        metadata["pages"] = offsets
        info_title = (reader.metadata or {}).get("/Title") if reader.metadata else None
        title = str(info_title) if info_title else title
    else:
        raise ValueError(f"unsupported file type {suffix!r}; expected one of {SUPPORTED}")
    return Document(id=_doc_id(name, data), title=title.strip(), text=text, metadata=metadata)


def load_path(path: Path) -> Document:
    return load_bytes(path.name, path.read_bytes())


def to_passages(docs: Iterable[Document], chunker: Chunker | None = None) -> list[Passage]:
    """One passage per chunk (or per document when ``chunker`` is None)."""
    out: list[Passage] = []
    for doc in docs:
        if chunker is None:
            out.append(Passage(id=doc.id, doc_id=doc.id, text=doc.text, title=doc.title, start=0, end=len(doc.text)))
            continue
        for chunk in chunker.split(doc.text):
            out.append(
                Passage(
                    id=f"{doc.id}#{chunk.index}",
                    doc_id=doc.id,
                    text=chunk.text,
                    title=doc.title,
                    start=chunk.start,
                    end=chunk.end,
                    metadata={"chunker": getattr(chunker, "name", "?"), **chunk.meta},
                )
            )
    return out
