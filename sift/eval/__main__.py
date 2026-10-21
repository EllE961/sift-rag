"""Evaluation CLI.

    python -m sift.eval retrieval [--datasets scifact nfcorpus]
    python -m sift.eval hnsw
    python -m sift.eval claims --verifier nli|claude [--n 100]
    python -m sift.eval figures
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(prog="python -m sift.eval")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval", help="BEIR retrieval ablation")
    r.add_argument("--datasets", nargs="+", default=["scifact", "nfcorpus"])
    sub.add_parser("blend", help="tune a first-stage/cross-encoder score blend (needs retrieval.json)")
    sub.add_parser("hnsw", help="HNSW recall/QPS benchmark")
    c = sub.add_parser("claims", help="SciFact end-to-end claim verification")
    c.add_argument("--verifier", choices=["nli", "claude"], default="nli")
    c.add_argument("--n", type=int, default=None, help="claims to sample (nli: all 300; claude: 99)")
    c.add_argument("--model", default=None, help="Claude model id (claude verifier)")
    sub.add_parser("figures", help="render README figures from results/*.json")
    sub.add_parser("projection", help="precompute 2D embedding maps for the UI")
    args = parser.parse_args()

    if args.cmd == "retrieval":
        from .retrieval import main as run

        run(args.datasets)
    elif args.cmd == "blend":
        from .retrieval import blend_study

        blend_study()
    elif args.cmd == "hnsw":
        from .hnsw_bench import main as run

        run()
    elif args.cmd == "claims":
        from .claims import main as run

        run(verifier=args.verifier, n=args.n, model=args.model)
    elif args.cmd == "figures":
        from .figures import main as run

        run()
    elif args.cmd == "projection":
        from ..projection import main as run

        run()


if __name__ == "__main__":
    main()
