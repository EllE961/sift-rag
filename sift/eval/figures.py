"""README figures, rendered from ``results/*.json`` in the UI's visual language.

    python -m sift.eval figures

Same tokens as the web UI: graphite surfaces, warm-paper ink, the four
CVD-validated stage colors (lexical, dense, fusion, rerank), hairline grids,
direct labels instead of legends wherever the marks allow.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch, Patch

from ..config import ASSETS_DIR, RESULTS_DIR, ROOT

BG = "#0f1216"
PANEL = "#0f1216"
INK1, INK2, INK3, INK4 = "#ecebe6", "#b9b8b1", "#85857f", "#5b5c59"
RULE = "#22262d"
RULE2 = "#2c3139"
LEX, DENSE, FUSION, RERANK = "#d95926", "#3987e5", "#199e70", "#9085e9"
MARKER = "#e8b44c"
RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#1c5cab"]
GOOD, BAD = "#0ca30c", "#d03b3b"

FONT_DIR = ROOT / "docs" / "fonts"


def _fonts() -> dict[str, fm.FontProperties]:
    """Register the bundled IBM Plex files; DejaVu Sans is the glyph fallback (Greek, symbols)."""
    have = False
    for path in FONT_DIR.glob("*.ttf"):
        fm.fontManager.addfont(str(path))
        have = True
    sans = ["IBM Plex Sans", "DejaVu Sans"] if have else ["DejaVu Sans"]
    mono = ["IBM Plex Mono", "DejaVu Sans Mono"] if have else ["DejaVu Sans Mono"]
    return {
        "sans": fm.FontProperties(family=sans, weight=400),
        "sans-medium": fm.FontProperties(family=sans, weight=500),
        "sans-semibold": fm.FontProperties(family=sans, weight=600),
        "mono": fm.FontProperties(family=mono, weight=400),
        "mono-medium": fm.FontProperties(family=mono, weight=500),
    }


F = _fonts()


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": RULE2, "axes.labelcolor": INK2, "text.color": INK1,
        "xtick.color": INK3, "ytick.color": INK3, "axes.grid": False,
        "font.family": ["IBM Plex Sans", "DejaVu Sans"],
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
    })  # fmt: skip


def family(method: str) -> str:
    if "rerank" in method:
        return RERANK
    if method.startswith("Hybrid"):
        return FUSION
    if method.startswith("Dense"):
        return DENSE
    return LEX


def frame(fig: plt.Figure, title: str, subtitle: str = "") -> None:
    card = FancyBboxPatch((0.004, 0.004), 0.992, 0.992, boxstyle="round,pad=0,rounding_size=0.012",
                          transform=fig.transFigure, facecolor=BG, edgecolor=RULE2, lw=1, zorder=-10)
    fig.patches.append(card)
    h = fig.get_figheight()  # place text at fixed physical offsets so short figures don't collide
    fig.text(0.035, 1 - 0.22 / h, title, fontproperties=F["sans-semibold"], fontsize=14, color=INK1, va="top")
    if subtitle:
        fig.text(0.035, 1 - 0.52 / h, subtitle, fontproperties=F["sans"], fontsize=9.5, color=INK3, va="top")


def axis(ax: plt.Axes, grid: str = "x") -> None:
    ax.spines["left"].set_color(RULE2)
    ax.spines["bottom"].set_color(RULE2)
    ax.tick_params(length=0, pad=6)
    if grid:
        ax.grid(axis=grid, color=RULE, lw=0.8)
        ax.set_axisbelow(True)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(F["mono"])
        lab.set_fontsize(8.5)


def clipped_texts(fig: plt.Figure) -> list[str]:
    """Visible text artists that extend past the figure edge (i.e. would be cut off)."""
    from matplotlib.text import Text

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w, h = fig.bbox.width, fig.bbox.height
    # Tick labels outside the view interval exist as Text objects but are never drawn.
    undrawn: set[int] = set()
    for ax in fig.axes:
        for axis_ in (ax.xaxis, ax.yaxis):
            drawn = {id(t) for t in axis_._update_ticks()}
            for tick in axis_.get_major_ticks() + axis_.get_minor_ticks():
                if id(tick) not in drawn:
                    undrawn.update({id(tick.label1), id(tick.label2)})
    out = []
    for t in fig.findobj(Text):
        if id(t) in undrawn or not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer)
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > w + 0.5 or bb.y1 > h + 0.5:
            out.append(t.get_text())
    return out


def fit_left(fig: plt.Figure, ax: plt.Axes, pad_in: float = 0.3) -> None:
    """Set the left margin so the longest y tick label fits, plus padding."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    widest = max((lab.get_window_extent(renderer).width for lab in ax.get_yticklabels()), default=0.0)
    fig.subplots_adjust(left=(widest + (pad_in + 0.1) * fig.dpi) / fig.bbox.width)


def save(fig: plt.Figure, name: str) -> Path:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    path = ASSETS_DIR / name
    clipped = clipped_texts(fig)
    if clipped:
        print(f"WARNING {name}: clipped text {clipped}")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def load(name: str) -> dict[str, Any] | None:
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------- figures
PUBLISHED_BM25 = {"scifact": 0.665, "nfcorpus": 0.325}  # BEIR paper (Elasticsearch)
ORDER = [
    "BM25", "Dense · MiniLM-L6", "Dense · bge-small", "Dense · bge-small (HNSW ef=128)",
    "Hybrid · RRF", "Hybrid · convex (train-tuned)", "BM25 → rerank", "Hybrid RRF → rerank",
    "Hybrid convex → rerank", "Hybrid RRF → rerank, blended (train-tuned)",
]  # fmt: skip
SHORT = {"Hybrid RRF → rerank, blended (train-tuned)": "Hybrid RRF → blended rerank (tuned)",
         "Hybrid · convex (train-tuned)": "Hybrid · convex (tuned)"}  # fmt: skip


def fig_ablation(R: dict[str, Any]) -> None:
    ds = R["datasets"]
    names = [n for n in ("scifact", "nfcorpus") if n in ds]
    common = [m for m in ds[names[0]]["metrics"] if all(m in ds[n]["metrics"] for n in names)]
    methods = [m for m in ORDER if m in common] + [m for m in common if m not in ORDER]
    fig, axes = plt.subplots(1, len(names), figsize=(12.5, 0.42 * len(methods) + 2.0))
    axes = [axes] if len(names) == 1 else axes
    frame(fig, "Retrieval quality on BEIR (nDCG@10, test split)",
          "Colour = method family · bars ordered as in the ablation · "
          "dashed line = published BM25 (BEIR, Elasticsearch)")
    for ax, n in zip(axes, names):
        vals = [ds[n]["metrics"][m]["ndcg@10"] for m in methods]
        y = list(range(len(methods)))[::-1]
        best = max(vals)
        ax.barh(y, vals, height=0.56, color=[family(m) for m in methods], alpha=0.95, zorder=2)
        for yi, v in zip(y, vals):
            weight = "mono-medium" if v == best else "mono"
            ax.text(v + best * 0.012, yi, f"{v:.3f}", va="center", fontproperties=F[weight],
                    fontsize=8.5, color=INK1 if v == best else INK2, zorder=5,
                    bbox={"facecolor": BG, "edgecolor": "none", "pad": 1.0})  # fmt: skip
        ax.axvline(PUBLISHED_BM25.get(n, 0), color=INK3, lw=1, ls=(0, (2, 2)), zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([SHORT.get(m, m) for m in methods] if ax is axes[0] else [""] * len(methods))
        ax.set_xlim(0, best * 1.16)
        ax.set_title(f"{n}  ·  {ds[n]['n_test_queries']} queries  ·  {ds[n]['n_docs']:,} docs", loc="left",
                     fontproperties=F["sans-medium"], fontsize=10.5, color=INK2, pad=10)  # fmt: skip
        axis(ax, "x")
        for lab in ax.get_yticklabels():
            lab.set_fontproperties(F["sans"])
            lab.set_fontsize(9.5)
            lab.set_color(INK2)
    fig.subplots_adjust(left=0.2, right=0.97, top=0.8, bottom=0.08, wspace=0.08)
    fit_left(fig, axes[0])
    save(fig, "ablation.png")


def fig_rerank(R: dict[str, Any]) -> None:
    ds = R["datasets"]
    names = [n for n in ("scifact", "nfcorpus") if n in ds]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    frame(fig, "An off-domain cross-encoder hurts a strong hybrid first stage",
          "ms-marco-MiniLM-L6 reranking hybrid (RRF) candidates · left: change vs. no reranking (test) · "
          "right: blending CE and first-stage scores instead, weight tuned on train")  # fmt: skip
    shades = {"scifact": "#c4bdf6", "nfcorpus": RERANK}
    for n in names:
        sw = ds[n]["rerank_depth_sweep"]
        base = ds[n]["metrics"]["Hybrid · RRF"]["ndcg@10"]
        axes[0].plot([r["depth"] for r in sw], [r["ndcg@10"] - base for r in sw], color=shades[n], marker="o", ms=5,
                     mec=BG, mew=1.5)  # fmt: skip
        axes[0].text(sw[-1]["depth"] + 3, sw[-1]["ndcg@10"] - base, n, va="center", fontsize=9, color=INK2)
        axes[1].plot([r["depth"] for r in sw], [r["latency_ms_p50"] for r in sw], color=shades[n], marker="o", ms=5,
                     mec=BG, mew=1.5)  # fmt: skip
        axes[1].text(sw[-1]["depth"] + 3, sw[-1]["latency_ms_p50"], n, va="center", fontsize=9, color=INK2)
        bl = ds[n].get("rerank_blend")
        if bl:
            pts = [r for r in bl["sweep"] if r["method"] == bl["best"]["method"]]
            w0 = pts[0]["ndcg@10_train"]
            axes[2].plot([r["w"] for r in pts], [r["ndcg@10_train"] - w0 for r in pts], color=shades[n])
            axes[2].scatter([bl["best"]["w"]], [bl["best"]["ndcg@10_train"] - w0], s=60, color=MARKER, edgecolor=BG,
                            lw=1.5, zorder=4)  # fmt: skip
            axes[2].text(1.03, pts[-1]["ndcg@10_train"] - w0, n, va="center", fontsize=9, color=INK2)
    axes[0].axhline(0, color=INK3, lw=1)
    axes[0].set_title("Δ nDCG@10 vs. no rerank", loc="left", fontproperties=F["sans-medium"], color=INK2)
    axes[0].set_xlabel("candidates reranked")
    axes[1].set_title("p50 rerank latency (ms)", loc="left", fontproperties=F["sans-medium"], color=INK2)
    axes[1].set_xlabel("candidates reranked")
    axes[1].set_ylim(bottom=0)
    axes[2].axhline(0, color=INK3, lw=1)
    axes[2].set_title("Δ nDCG@10 vs. weight on CE score (train)", loc="left", fontproperties=F["sans-medium"],
                      color=INK2)  # fmt: skip
    axes[2].set_xlabel("w   (0 = first stage only, 1 = cross-encoder only)")
    for ax in axes:
        axis(ax, "y")
        ax.set_xmargin(0.08)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.74, bottom=0.16, wspace=0.32)
    save(fig, "rerank.png")


def fig_fusion(R: dict[str, Any]) -> None:
    ds = R["datasets"]
    names = [n for n in ("scifact", "nfcorpus") if n in ds]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
    frame(fig, "Hybrid fusion: tuned on train, frozen for test",
          "Convex combination of min-max/z-scored BM25 and dense scores · solid = train (used to pick α) · "
          "faint = test, shown for reference only · ◆ = chosen α")  # fmt: skip
    for ax, n in zip(axes, names):
        tun = ds[n]["convex_tuning"]
        best = tun["best"]
        tr = [r for r in tun["sweep"] if r["norm"] == best["norm"]]
        te = ds[n]["convex_alpha_sweep_test_reference_only"]
        ax.plot([r["alpha"] for r in tr], [r["ndcg@10"] for r in tr], color=FUSION)
        ax.plot([r["alpha"] for r in te], [r["ndcg@10"] for r in te], color=FUSION, alpha=0.35, lw=1.5)
        ax.scatter([best["alpha"]], [best["ndcg@10"]], marker="D", s=46, color=MARKER, edgecolor=BG, lw=1.5, zorder=4)
        rrf = ds[n]["metrics"]["Hybrid · RRF"]["ndcg@10"]
        ax.axhline(rrf, color=INK3, lw=1, ls=(0, (2, 2)))
        ax.text(0.99, rrf, "RRF on test ", va="top", ha="right", fontsize=8.5, color=INK3)
        y_tr, y_te = tr[-1]["ndcg@10"], te[-1]["ndcg@10"]
        lo, hi = ax.get_ylim()
        gap = 0.035 * (hi - lo)
        if abs(y_tr - y_te) < gap:  # nudge converging end labels apart
            mid = (y_tr + y_te) / 2
            y_tr, y_te = (mid + gap / 2, mid - gap / 2) if y_tr >= y_te else (mid - gap / 2, mid + gap / 2)
        ax.text(1.02, y_tr, "train", va="center", fontsize=8.5, color=INK2)
        ax.text(1.02, y_te, "test", va="center", fontsize=8.5, color=INK3)
        ax.set_title(f"{n}  ·  α* = {best['alpha']:.2f} ({best['norm']})", loc="left",
                     fontproperties=F["sans-medium"], color=INK2)  # fmt: skip
        ax.set_xlabel("α  (0 = BM25 only, 1 = dense only)")
        axis(ax, "y")
    fig.subplots_adjust(left=0.06, right=0.95, top=0.74, bottom=0.16, wspace=0.18)
    save(fig, "fusion.png")


def fig_latency(R: dict[str, Any]) -> None:
    ds = R["datasets"]
    names = [n for n in ("scifact", "nfcorpus") if n in ds]
    stages = [("bm25", "BM25", LEX), ("embed", "Encode query", DENSE), ("dense", "HNSW search", DENSE),
              ("fusion", "RRF fusion", FUSION), ("rerank", "Cross-encoder @50", RERANK)]  # fmt: skip
    fig, ax = plt.subplots(figsize=(12, 3.7))
    frame(fig, "Where a query's time goes",
          "p50 (marker) to p95 (bar end) per stage · serving path, one query at a time · log scale · "
          "shared RTX 3080")  # fmt: skip
    marks = {"scifact": ("o", INK1), "nfcorpus": ("D", INK3)}
    for yi, (key, _label, color) in enumerate(stages[::-1]):
        ax.add_patch(plt.Rectangle((1e-3, yi - 0.38), 1e4, 0.76, color=PANEL, zorder=0))
        for j, n in enumerate(names):
            st = ds[n]["latency_hybrid_rerank"]["stages"].get(key)
            if not st:
                continue
            off = (j - (len(names) - 1) / 2) * 0.26
            ax.plot([st["p50"], st["p95"]], [yi + off, yi + off], color=color, lw=3, solid_capstyle="round", zorder=2)
            m, c = marks[n]
            ax.scatter([st["p50"]], [yi + off], marker=m, s=34, color=c, edgecolor=BG, lw=1.2, zorder=3)
            ax.text(st["p95"] * 1.12, yi + off, f"{st['p50']:.2f} ms" if st["p50"] < 10 else f"{st['p50']:.0f} ms",
                    va="center", fontproperties=F["mono"], fontsize=8, color=INK2)  # fmt: skip
    ax.set_xscale("log")
    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels([s_[1] for s_ in stages[::-1]])
    ax.set_xlim(0.1, 3000)
    ax.set_ylim(-0.6, len(stages) - 0.4)
    ax.set_xlabel("milliseconds (log)")
    axis(ax, "x")
    for lab in ax.get_yticklabels():
        lab.set_fontproperties(F["sans"])
        lab.set_fontsize(9.5)
        lab.set_color(INK2)
    totals = " · ".join(
        f"{n}: {ds[n]['latency_hybrid_rerank']['total_ms']['p50']:.0f} ms p50 total" for n in names
    )
    for n in names:
        m, c = marks[n]
        ax.scatter([], [], marker=m, s=34, color=c, label=n)
    leg = fig.legend(loc="upper right", bbox_to_anchor=(0.97, 1 - 0.14 / fig.get_figheight()), frameon=False,
                     fontsize=9, ncol=len(names), handletextpad=0.4, columnspacing=1.2)  # fmt: skip
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.text(0.97, 1 - 0.56 / fig.get_figheight(), totals, ha="right", va="top", fontproperties=F["mono"],
             fontsize=8.5, color=INK3)  # fmt: skip
    fig.subplots_adjust(left=0.14, right=0.97, top=0.78, bottom=0.17)
    save(fig, "latency.png")


def fig_hnsw(H: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    frame(fig, f"HNSW from scratch: recall vs throughput on {H['n_total']:,} real sentence embeddings",
          f"bge-small (384-d) · {H['n_queries']:,} real BEIR queries · single thread · "
          "each line sweeps ef · ◆ = exact brute force (recall 1.0) at the same N")  # fmt: skip
    ax = axes[0]
    for i, row in enumerate(H["scaling"]):
        c = RAMP[min(i, len(RAMP) - 1)]
        xs = [p["qps"] for p in row["curve"]]
        ys = [p["recall@10"] for p in row["curve"]]
        ax.plot(xs, ys, color=c, marker="o", ms=3.5, mec=BG, mew=1, label=f"N = {row['n']:,}")
        ax.scatter([row["flat_qps_1thread"]], [1.0], marker="D", s=40, color=c, edgecolor=BG, lw=1.2, zorder=4)
    leg = ax.legend(loc="lower left", frameon=False, fontsize=8.5, handlelength=1.4)
    for t in leg.get_texts():
        t.set_color(INK2)
    ax.set_xscale("log")
    ax.set_xlabel("queries / second (log)")
    ax.set_ylabel("recall@10")
    ax.set_title("Recall–throughput trade-off by corpus size", loc="left", fontproperties=F["sans-medium"], color=INK2)
    axis(ax, "both")

    ax = axes[1]
    ns = [r["n"] for r in H["scaling"]]
    exact = [r["flat_qps_1thread"] for r in H["scaling"]]
    for target, c in ((0.95, RAMP[2]), (0.99, RAMP[4])):
        q = []
        for r in H["scaling"]:
            ok = [p for p in r["curve"] if p["recall@10"] >= target]
            q.append(max(p["qps"] for p in ok) if ok else float("nan"))
        ax.plot(ns, q, color=c, marker="o", ms=5, mec=BG, mew=1.5)
        ax.text(ns[-1] * 1.06, q[-1], f"HNSW @ R≥{target}", va="center", fontsize=8.5, color=INK2)
    ax.plot(ns, exact, color=INK3, marker="D", ms=4.5, mec=BG, mew=1.2)
    ax.text(ns[-1] * 1.06, exact[-1], "exact (1 thread)", va="center", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("corpus size N (log)")
    ax.set_ylabel("queries / second (log)")
    ax.set_title("Exact search slows linearly with N; HNSW sub-linearly", loc="left", fontproperties=F["sans-medium"],
                 color=INK2)  # fmt: skip
    ax.set_xlim(ns[0] * 0.8, ns[-1] * 2.6)
    ticks = [100, 200, 500, 1000, 2000, 5000]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:,}" for t in ticks])
    ax.set_xticks(ns)
    ax.set_xticklabels([f"{n // 1000}k" for n in ns])
    ax.minorticks_off()
    first, last = H["scaling"][0], H["scaling"][-1]
    ev0 = next(p for p in first["curve"] if p["recall@10"] >= 0.95)["distance_evals_per_query"]
    ev1 = next(p for p in last["curve"] if p["recall@10"] >= 0.95)["distance_evals_per_query"]
    note = (f"distance evaluations / query at R@10 ≥ 0.95\n"
            f"{ev0:,.0f} → {ev1:,.0f} for {last['n'] / first['n']:.0f}× more data")  # fmt: skip
    ax.text(0.98, 0.96, note, transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK3,
            linespacing=1.5)  # fmt: skip
    axis(ax, "both")
    fig.subplots_adjust(left=0.07, right=0.97, top=0.76, bottom=0.14, wspace=0.22)
    save(fig, "hnsw.png")


COND = {"no_evidence": ("No evidence", INK4), "closed_book": ("Closed book", INK4), "bm25@5": ("BM25 top-5", LEX),
        "sift@5": ("Sift top-5", RERANK), "oracle": ("Oracle evidence", INK2)}  # fmt: skip


def fig_claims(C: dict[str, Any], name: str) -> None:
    conds = list(C["conditions"])
    metrics = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("citation_precision", "Citation precision"),
               ("rationale_precision", "Rationale precision"), ("evidence_recall", "Evidence recall")]  # fmt: skip
    fig = plt.figure(figsize=(13, 4.6))
    who = C["model"] if C["verifier"] == "claude" else "offline NLI verifier (" + C["model"].split("/")[-1] + ")"
    frame(fig, "End-to-end claim verification on SciFact (dev)",
          f"{C['n_dev']} claims · {who} · what the verifier is shown decides what it can conclude")  # fmt: skip
    ax = fig.add_axes([0.06, 0.16, 0.56, 0.6])
    w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        label, color = COND.get(c, (c, INK3))
        vals = [C["conditions"][c].get(k) for k, _ in metrics]
        xs = [j + (i - (len(conds) - 1) / 2) * w for j in range(len(metrics))]
        for x, v in zip(xs, vals):
            if v is None:
                continue
            ax.bar(x, v, width=w * 0.86, color=color, zorder=2)
            ax.text(x, v + 0.015, f"{v * 100:.0f}", ha="center", fontsize=7.5, color=INK2, fontproperties=F["mono"])
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m[1] for m in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    axis(ax, "y")
    for lab in ax.get_xticklabels():
        lab.set_fontproperties(F["sans"])
        lab.set_fontsize(9)
        lab.set_color(INK2)
    handles = [Patch(facecolor=COND.get(c, (c, INK3))[1], label=COND.get(c, (c, INK3))[0]) for c in conds]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1.13), ncol=len(conds), frameon=False,
                    fontsize=8.5, handlelength=1, handleheight=0.8, columnspacing=1.4)  # fmt: skip
    for t in leg.get_texts():
        t.set_color(INK2)
    # confusion matrix for the full pipeline
    key = "sift@5"
    m = C["conditions"][key]["confusion"]["matrix"]
    labs = ["SUP", "REF", "NEI"]
    ax2 = fig.add_axes([0.71, 0.16, 0.22, 0.6])
    vmax = max(max(r) for r in m)
    ax2.imshow(m, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#141a24", "#1c5cab", "#86b6ef"]),
               vmin=0, vmax=vmax)  # fmt: skip
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, str(m[i][j]), ha="center", va="center", fontproperties=F["mono-medium"], fontsize=10,
                     color="#0b0d10" if m[i][j] > vmax * 0.6 else INK1)  # fmt: skip
    ax2.set_xticks(range(3))
    ax2.set_yticks(range(3))
    ax2.set_xticklabels(labs)
    ax2.set_yticklabels(labs)
    ax2.set_xlabel("predicted")
    ax2.set_ylabel("gold")
    ax2.set_title(f"Sift top-5 · acc {C['conditions'][key]['accuracy'] * 100:.1f}%", loc="left",
                  fontproperties=F["sans-medium"], color=INK2, fontsize=10)  # fmt: skip
    for s in ax2.spines.values():
        s.set_visible(False)
    ax2.tick_params(length=0)
    save(fig, f"{name}.png")


def main() -> None:
    style()
    R = load("retrieval")
    if R:
        fig_ablation(R)
        fig_rerank(R)
        fig_fusion(R)
        fig_latency(R)
    H = load("hnsw")
    if H:
        fig_hnsw(H)
    for name in ("claims_nli", "claims_claude"):
        C = load(name)
        if C:
            fig_claims(C, name)


if __name__ == "__main__":
    main()
