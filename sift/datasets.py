"""Benchmark datasets: BEIR (SciFact, NFCorpus, FiQA) and the SciFact claim release.

BEIR layout::

    corpus.jsonl    {"_id", "title", "text"}
    queries.jsonl   {"_id", "text"}
    qrels/<split>.tsv   query-id <TAB> corpus-id <TAB> score   (with a header row)

The original SciFact release adds gold veracity labels (SUPPORT / CONTRADICT)
and rationale sentences, which we use for the end-to-end claim-verification eval.

CLI::

    python -m sift.datasets download            # scifact, nfcorpus, scifact claims
    python -m sift.datasets download --fiqa     # + fiqa (17 MB zip, 57k docs)
"""

from __future__ import annotations

import argparse
import csv
import json
import tarfile
import urllib.request
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .config import RAW_DIR
from .types import Document

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
SCIFACT_RELEASE_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

LABELS = ("SUPPORTED", "REFUTED", "NOT ENOUGH INFO")
_SCIFACT_TO_LABEL = {"SUPPORT": "SUPPORTED", "CONTRADICT": "REFUTED"}


@dataclass
class BeirDataset:
    name: str
    corpus: dict[str, Document]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    split: str = "test"

    @property
    def eval_queries(self) -> dict[str, str]:
        """Queries that have at least one relevance judgement in this split."""
        return {qid: self.queries[qid] for qid in self.qrels if qid in self.queries}


@dataclass
class Claim:
    """A SciFact claim with its gold label and evidence."""

    id: str
    claim: str
    label: str  # one of LABELS
    evidence_doc_ids: list[str] = field(default_factory=list)
    cited_doc_ids: list[str] = field(default_factory=list)
    rationales: dict[str, list[int]] = field(default_factory=dict)  # doc_id -> sentence indices


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url} -> {dest}")
    with urllib.request.urlopen(url, timeout=120) as resp, open(tmp, "wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    tmp.replace(dest)
    return dest


def download_beir(name: str, root: Path = RAW_DIR) -> Path:
    target = root / name
    if (target / "corpus.jsonl").exists():
        return target
    archive = _download(BEIR_URL.format(name=name), root / f"{name}.zip")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    return target


def download_scifact_release(root: Path = RAW_DIR) -> Path:
    target = root / "scifact_release" / "data"
    if (target / "claims_dev.jsonl").exists():
        return target
    archive = _download(SCIFACT_RELEASE_URL, root / "scifact_release.tar.gz")
    with tarfile.open(archive) as tf:
        tf.extractall(root / "scifact_release")  # trusted, pinned source
    return target


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with open(path, encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if len(row) < 3:
                continue
            qid, did, score = row[0], row[1], int(row[2])
            qrels[qid][did] = score
    return dict(qrels)


def load_corpus(name: str, root: Path = RAW_DIR) -> dict[str, Document]:
    path = download_beir(name, root) / "corpus.jsonl"
    return {
        str(row["_id"]): Document(id=str(row["_id"]), title=row.get("title", "") or "", text=row.get("text", "") or "")
        for row in _read_jsonl(path)
    }


def load_beir(name: str, split: str = "test", root: Path = RAW_DIR) -> BeirDataset:
    base = download_beir(name, root)
    corpus = load_corpus(name, root)
    queries = {str(r["_id"]): r["text"] for r in _read_jsonl(base / "queries.jsonl")}
    qrels_path = base / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        raise FileNotFoundError(f"{name} has no '{split}' split ({qrels_path})")
    return BeirDataset(name=name, corpus=corpus, queries=queries, qrels=load_qrels(qrels_path), split=split)


def load_scifact_claims(split: str = "dev", root: Path = RAW_DIR) -> list[Claim]:
    """Load SciFact claims with a single gold label each.

    A claim with no evidence is NOT ENOUGH INFO. SciFact guarantees all
    evidence sets for one claim share a label, so taking the first is safe; we
    still assert it to fail loudly if the data ever changes.
    """
    base = download_scifact_release(root)
    claims: list[Claim] = []
    for row in _read_jsonl(base / f"claims_{split}.jsonl"):
        evidence: dict[str, list[dict]] = row.get("evidence", {}) or {}
        labels = {e["label"] for sets in evidence.values() for e in sets}
        if not labels:
            label = "NOT ENOUGH INFO"
        else:
            assert len(labels) == 1, f"claim {row['id']} has mixed labels {labels}"
            label = _SCIFACT_TO_LABEL[labels.pop()]
        rationales = {
            str(doc): sorted({s for e in sets for s in e["sentences"]}) for doc, sets in evidence.items()
        }
        claims.append(
            Claim(
                id=str(row["id"]),
                claim=row["claim"],
                label=label,
                evidence_doc_ids=[str(d) for d in evidence],
                cited_doc_ids=[str(d) for d in row.get("cited_doc_ids", [])],
                rationales=rationales,
            )
        )
    return claims


def load_scifact_sentences(root: Path = RAW_DIR) -> dict[str, list[str]]:
    """doc_id -> list of abstract sentences (the release's own segmentation)."""
    base = download_scifact_release(root)
    return {str(r["doc_id"]): list(r["abstract"]) for r in _read_jsonl(base / "corpus.jsonl")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    sub = parser.add_subparsers(dest="cmd", required=True)
    dl = sub.add_parser("download")
    dl.add_argument("--fiqa", action="store_true", help="also fetch FiQA-2018 (57k docs)")
    args = parser.parse_args()
    if args.cmd == "download":
        names = ["scifact", "nfcorpus"] + (["fiqa"] if args.fiqa else [])
        for name in names:
            ds = load_beir(name)
            print(f"{name:10s} docs={len(ds.corpus):>6d}  test queries={len(ds.eval_queries)}")
        claims = load_scifact_claims("dev")
        print(f"scifact claims (dev) = {len(claims)}")


if __name__ == "__main__":
    main()
