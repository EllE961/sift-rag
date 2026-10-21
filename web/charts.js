// Hand-rolled SVG charts. Marks follow one spec everywhere: 2px lines,
// <=16px bars with 4px rounded data-ends, hairline recessive grids, text in
// ink tokens (never series colors), a shared hover tooltip.

const NS = "http://www.w3.org/2000/svg";

export const STAGE = {
  bm25: { label: "BM25", color: "var(--stage-lexical)", cls: "lexical" },
  embed: { label: "Encode query", color: "var(--stage-dense)", cls: "dense" },
  dense: { label: "Dense ANN", color: "var(--stage-dense)", cls: "dense" },
  fusion: { label: "Fusion", color: "var(--stage-fusion)", cls: "fusion" },
  rerank: { label: "Cross-encoder", color: "var(--stage-rerank)", cls: "rerank" },
};

export function s(tag, attrs = {}, children = []) {
  const el = document.createElementNS(NS, tag);
  let style = attrs.style || "";
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null || v === false || k === "style") continue;
    // typography as inline style: presentation attributes lose to the stylesheet's `.chart text` rule
    if (k === "font-family" || k === "font-size" || k === "font-weight") style += `${k}:${typeof v === "number" && k === "font-size" ? v + "px" : v};`;
    else el.setAttribute(k, v);
  }
  if (style) el.setAttribute("style", style);
  for (const c of [].concat(children)) {
    if (c == null) continue;
    el.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
  }
  return el;
}

export function esc(str) {
  return String(str ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

export const fmt = {
  ms: (v) => (v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2)) + " ms",
  n: (v) => Number(v).toLocaleString("en-US"),
  p: (v, d = 1) => (v * 100).toFixed(d) + "%",
  f: (v, d = 3) => Number(v).toFixed(d),
  compact: (v) => (v >= 1e6 ? (v / 1e6).toFixed(1) + "M" : v >= 1e3 ? (v / 1e3).toFixed(v >= 1e4 ? 0 : 1) + "k" : String(Math.round(v))),
};

function truncate(text, n) {
  return text.length > n ? text.slice(0, n - 1).trimEnd() + "…" : text;
}

// ----------------------------------------------------------------- tooltip
let tipEl = null;
export function tip(html, evt) {
  if (!tipEl) {
    tipEl = document.createElement("div");
    tipEl.className = "chart-tip";
    document.body.appendChild(tipEl);
  }
  if (html == null) {
    tipEl.hidden = true;
    return;
  }
  tipEl.innerHTML = html;
  tipEl.hidden = false;
  const pad = 14;
  const r = tipEl.getBoundingClientRect();
  let x = evt.clientX + pad;
  let y = evt.clientY + pad;
  if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - pad;
  if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - pad;
  tipEl.style.left = `${x}px`;
  tipEl.style.top = `${y}px`;
}

function width(el, fallback = 640) {
  return Math.max(320, Math.floor(el.clientWidth || fallback) - 0);
}

function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const step0 = span / count;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const out = [];
  for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

function roundedBar(x, y, w, h, r = 4) {
  // rounded data-end on the right, square at the baseline (left)
  r = Math.min(r, h / 2, Math.max(0, w));
  if (w <= 0) return "";
  return `M${x},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y + h - r}Q${x + w},${y + h} ${x + w - r},${y + h}H${x}Z`;
}

function roundedCol(x, y0, w, h, r = 4) {
  // vertical column growing up from baseline y0
  r = Math.min(r, w / 2, Math.max(0, h));
  if (h <= 0) return "";
  const y = y0 - h;
  return `M${x},${y0}V${y + r}Q${x},${y} ${x + r},${y}H${x + w - r}Q${x + w},${y} ${x + w},${y + r}V${y0}Z`;
}

// --------------------------------------------------------------- rank flow
export function rankFlow(el, trace, { maxRank = 15 } = {}) {
  el.innerHTML = "";
  const stages = trace.stage_order;
  if (stages.length < 2) {
    el.innerHTML = `<p class="muted" style="padding:8px 0">Rank flow needs at least two ranked stages — switch the mode to <b>Hybrid</b> or <b>Hybrid + rerank</b>.</p>`;
    return;
  }
  const finalIds = trace.hits.map((h) => h.pid);
  const finalRank = Object.fromEntries(finalIds.map((p, i) => [p, i + 1]));
  const W = width(el);
  const labelW = Math.min(300, W * 0.38);
  const m = { t: 34, r: labelW + 12, b: 8, l: 34 };
  const rowH = 22;
  const H = m.t + (maxRank + 1.6) * rowH + m.b;
  const iw = W - m.l - m.r;
  const xs = stages.map((_, i) => m.l + (iw * i) / (stages.length - 1));
  const yOf = (r) => (r == null || r > maxRank ? m.t + (maxRank + 1.1) * rowH : m.t + (r - 0.5) * rowH);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Rank flow across pipeline stages" });

  // rank gridlines + labels
  const grid = s("g", { class: "grid" });
  for (const r of [1, 3, 5, 10, 15]) {
    if (r > maxRank) continue;
    grid.appendChild(s("line", { x1: m.l, x2: m.l + iw, y1: yOf(r), y2: yOf(r) }));
    svg.appendChild(s("text", { x: m.l - 10, y: yOf(r) + 4, "text-anchor": "end" }, String(r)));
  }
  svg.appendChild(s("text", { x: m.l - 10, y: yOf(null) + 4, "text-anchor": "end" }, "out"));
  grid.appendChild(s("line", { x1: m.l, x2: m.l + iw, y1: yOf(null), y2: yOf(null), "stroke-opacity": 0.5 }));
  svg.insertBefore(grid, svg.firstChild);

  // stage columns
  stages.forEach((st, i) => {
    const meta = STAGE[st] || { label: st, color: "var(--ink-3)" };
    svg.appendChild(s("line", { x1: xs[i], x2: xs[i], y1: m.t - 6, y2: yOf(null) + 8, stroke: "var(--rule-2)" }));
    const g = s("g", { transform: `translate(${xs[i]},${m.t - 18})` });
    g.appendChild(s("rect", { x: -40, y: -9, width: 8, height: 8, rx: 2, fill: meta.color, transform: "translate(0,0)" }));
    g.appendChild(s("text", { x: -28, y: -1, class: "lbl-2", "font-family": "IBM Plex Sans", "font-size": 12 }, meta.label));
    svg.appendChild(g);
  });

  const docs = trace.rank_flow.slice().sort((a, b) => (finalRank[b.pid] ? 1 : 0) - (finalRank[a.pid] ? 1 : 0));
  const threads = s("g");
  const hits = s("g");
  const nodes = s("g");
  const byPid = {};
  for (const d of docs.reverse()) {
    const pts = stages.map((st, i) => [xs[i], yOf(d.ranks[st])]);
    let path = `M${pts[0][0]},${pts[0][1]}`;
    for (let i = 1; i < pts.length; i++) {
      const [x0, y0] = pts[i - 1];
      const [x1, y1] = pts[i];
      const dx = (x1 - x0) * 0.5;
      path += ` C${x0 + dx},${y0} ${x1 - dx},${y1} ${x1},${y1}`;
    }
    const isFinal = !!finalRank[d.pid];
    const line = s("path", {
      d: path, class: "thread",
      stroke: isFinal ? "var(--ink-2)" : "var(--ink-4)",
      "stroke-opacity": isFinal ? 0.9 : 0.55,
      "stroke-width": isFinal ? 2 : 1.5,
    });
    threads.appendChild(line);
    const dots = pts.map(([x, y], i) =>
      s("circle", { cx: x, cy: y, r: 4, fill: d.ranks[stages[i]] != null && d.ranks[stages[i]] <= maxRank ? (isFinal ? "var(--ink-1)" : "var(--ink-4)") : "var(--bench-3)", stroke: "var(--bench-1)", "stroke-width": 2 }),
    );
    dots.forEach((c) => nodes.appendChild(c));
    const hit = s("path", { d: path, class: "thread-hit" });
    hits.appendChild(hit);
    byPid[d.pid] = { line, dots, isFinal };
    const html = () => {
      const rows = stages.map((st) => `<div class="row"><span>${STAGE[st]?.label || st}</span><b>${d.ranks[st] ?? "—"}</b></div>`).join("");
      return `<div class="title">${esc(d.title || d.pid)}</div>${rows}`;
    };
    const on = (evt) => {
      for (const v of Object.values(byPid)) v.line.setAttribute("stroke-opacity", v === byPid[d.pid] ? 1 : 0.18);
      line.setAttribute("stroke", "var(--marker)");
      line.setAttribute("stroke-width", 2.5);
      threads.appendChild(line);
      tip(html(), evt);
    };
    const off = () => {
      for (const v of Object.values(byPid)) {
        v.line.setAttribute("stroke-opacity", v.isFinal ? 0.9 : 0.55);
        v.line.setAttribute("stroke", v.isFinal ? "var(--ink-2)" : "var(--ink-4)");
        v.line.setAttribute("stroke-width", v.isFinal ? 2 : 1.5);
      }
      tip(null);
    };
    hit.addEventListener("mousemove", on);
    hit.addEventListener("mouseleave", off);
  }
  svg.appendChild(threads);
  svg.appendChild(nodes);
  svg.appendChild(hits);

  // final-rank end labels
  const last = stages[stages.length - 1];
  for (const d of docs) {
    const fr = finalRank[d.pid];
    if (!fr) continue;
    const y = yOf(d.ranks[last]);
    if (d.ranks[last] == null || d.ranks[last] > maxRank) continue;
    const g = s("g", { transform: `translate(${xs[xs.length - 1] + 14},${y + 4})` });
    g.appendChild(s("text", { x: 0, y: 0, class: "lbl-1", "font-weight": 500 }, String(fr)));
    g.appendChild(s("text", { x: 18, y: 0, class: "lbl-2", "font-family": "Newsreader, Georgia, serif", "font-size": 13 }, truncate(d.title || d.pid, Math.floor((labelW - 24) / 6.9))));
    svg.appendChild(g);
  }
  el.appendChild(svg);
}

// --------------------------------------------------------------- waterfall
export function waterfall(el, trace) {
  el.innerHTML = "";
  const stages = trace.stages;
  const total = trace.total_ms;
  const W = width(el, 480);
  const rowH = 30;
  const m = { t: 8, r: 72, b: 26, l: 112 };
  const H = m.t + rowH * (stages.length + 1) + m.b;
  const iw = W - m.l - m.r;
  const x = (v) => m.l + (v / Math.max(total, 1e-6)) * iw;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Latency waterfall" });
  const ticks = niceTicks(0, total, 4);
  for (const t of ticks) {
    svg.appendChild(s("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, stroke: "var(--rule-1)" }));
    svg.appendChild(s("text", { x: x(t), y: H - 8, "text-anchor": "middle" }, t === 0 ? "0" : `${t} ms`));
  }
  let start = 0;
  stages.forEach((st, i) => {
    const meta = STAGE[st.name] || { label: st.name, color: "var(--ink-3)" };
    const y = m.t + i * rowH + 7;
    svg.appendChild(s("rect", { x: 0, y: y + 2, width: 8, height: 8, rx: 2, fill: meta.color, opacity: st.name === "embed" ? 0.55 : 1 }));
    svg.appendChild(s("text", { x: 16, y: y + 10, class: "lbl-2", "font-family": "IBM Plex Sans", "font-size": 12 }, meta.label));
    const w = Math.max(2, x(start + st.latency_ms) - x(start));
    const bar = s("path", { d: roundedBar(x(start), y, w, 12), fill: meta.color, opacity: st.name === "embed" ? 0.55 : 0.95 });
    bar.addEventListener("mousemove", (e) => tip(`<b>${meta.label}</b><div class="row"><span>latency</span><b>${fmt.ms(st.latency_ms)}</b></div><div class="row"><span>starts at</span><span>${fmt.ms(start)}</span></div>`, e));
    bar.addEventListener("mouseleave", () => tip(null));
    svg.appendChild(bar);
    svg.appendChild(s("text", { x: x(start) + w + 6, y: y + 10, class: "lbl-1" }, fmt.ms(st.latency_ms)));
    start += st.latency_ms;
  });
  const yT = m.t + stages.length * rowH + 7;
  svg.appendChild(s("text", { x: 16, y: yT + 10, class: "lbl-1", "font-family": "IBM Plex Sans", "font-size": 12, "font-weight": 600 }, "Total"));
  svg.appendChild(s("path", { d: roundedBar(x(0), yT + 3, Math.max(2, x(total) - x(0)), 6, 3), fill: "var(--ink-3)" }));
  svg.appendChild(s("text", { x: x(total) + 6, y: yT + 10, class: "lbl-1", "font-weight": 500 }, fmt.ms(total)));
  el.appendChild(svg);
}

// --------------------------------------------------------- score histograms
export function scoreDistributions(el, trace) {
  el.innerHTML = "";
  const finalIds = new Set(trace.hits.map((h) => h.pid));
  for (const st of trace.stages.filter((x) => x.hits.length)) {
    const meta = STAGE[st.name] || { label: st.name, color: "var(--ink-3)", cls: "" };
    const wrap = document.createElement("div");
    wrap.className = "mini";
    wrap.innerHTML = `<h5><i class="sw ${meta.cls}"></i>${meta.label}<span class="muted mono" style="margin-left:auto;font-size:11px">${st.hits.length} cand.</span></h5>`;
    const scores = st.hits.map((h) => h.score);
    const lo = Math.min(...scores), hi = Math.max(...scores);
    const bins = 16;
    const counts = new Array(bins).fill(0);
    const bi = (v) => Math.min(bins - 1, Math.floor(((v - lo) / (hi - lo || 1)) * bins));
    scores.forEach((v) => counts[bi(v)]++);
    const W = 260, H = 74, m = { t: 4, b: 18, l: 2, r: 2 };
    const bw = (W - m.l - m.r) / bins;
    const cmax = Math.max(...counts, 1);
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}` });
    svg.appendChild(s("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: "var(--rule-2)" }));
    counts.forEach((c, i) => {
      const h = ((H - m.t - m.b - 10) * c) / cmax;
      if (c) svg.appendChild(s("path", { d: roundedCol(m.l + i * bw + 1, H - m.b, bw - 2, h, 2), fill: meta.color, opacity: 0.8 }));
    });
    st.hits.forEach((hit) => {
      if (!finalIds.has(hit.pid)) return;
      const xx = m.l + ((hit.score - lo) / (hi - lo || 1)) * (W - m.l - m.r);
      svg.appendChild(s("line", { x1: xx, x2: xx, y1: H - m.b + 2, y2: H - m.b + 8, stroke: "var(--marker)", "stroke-width": 2, "stroke-linecap": "round" }));
    });
    svg.appendChild(s("text", { x: m.l, y: H - 2 }, lo.toFixed(st.name === "bm25" ? 1 : 3)));
    svg.appendChild(s("text", { x: W - m.r, y: H - 2, "text-anchor": "end" }, hi.toFixed(st.name === "bm25" ? 1 : 3)));
    wrap.appendChild(svg);
    el.appendChild(wrap);
  }
}

// ------------------------------------------------------- horizontal bars
// rows: [{label, value, color, note}]
export function hbars(el, rows, { max, format = (v) => v.toFixed(3), labelW = 200, title } = {}) {
  el.innerHTML = "";
  const W = width(el, 520);
  const rowH = 26;
  const m = { t: title ? 22 : 4, r: 60, b: 22, l: labelW };
  const H = m.t + rows.length * rowH + m.b;
  const vmax = max ?? Math.max(...rows.map((r) => r.value)) * 1.05;
  const iw = W - m.l - m.r;
  const x = (v) => m.l + (v / vmax) * iw;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}` });
  if (title) svg.appendChild(s("text", { x: 0, y: 12, class: "lbl-2", "font-family": "IBM Plex Sans", "font-size": 12, "font-weight": 500 }, title));
  for (const t of niceTicks(0, vmax, 4)) {
    svg.appendChild(s("line", { x1: x(t), x2: x(t), y1: m.t, y2: H - m.b, stroke: "var(--rule-1)" }));
    svg.appendChild(s("text", { x: x(t), y: H - 6, "text-anchor": "middle" }, String(t)));
  }
  rows.forEach((r, i) => {
    const y = m.t + i * rowH + 6;
    svg.appendChild(s("rect", { x: 0, y: y + 3, width: 8, height: 8, rx: 2, fill: r.color }));
    svg.appendChild(s("text", { x: 16, y: y + 11, class: r.emph ? "lbl-1" : "lbl-2", "font-family": "IBM Plex Sans", "font-size": 12 }, truncate(r.label, Math.floor((labelW - 20) / 6.4))));
    const bar = s("path", { d: roundedBar(x(0), y, Math.max(1, x(r.value) - x(0)), 14), fill: r.color, opacity: r.emph ? 1 : 0.85 });
    bar.addEventListener("mousemove", (e) => tip(`<b>${esc(r.label)}</b><div class="row"><span>${esc(r.metric || "value")}</span><b>${format(r.value)}</b></div>${r.note ? `<div style="margin-top:4px">${esc(r.note)}</div>` : ""}`, e));
    bar.addEventListener("mouseleave", () => tip(null));
    svg.appendChild(bar);
    svg.appendChild(s("text", { x: x(r.value) + 6, y: y + 11, class: r.emph ? "lbl-1" : "lbl-2", "font-weight": r.emph ? 500 : 400 }, format(r.value)));
  });
  svg.appendChild(s("line", { x1: x(0), x2: x(0), y1: m.t, y2: H - m.b, stroke: "var(--rule-3)" }));
  el.appendChild(svg);
}

// --------------------------------------------------------------- lines
// series: [{name, color, points: [{x, y, ...}], dashed?: false, width?}]
export function lines(el, series, opts = {}) {
  el.innerHTML = "";
  const { xlog = false, xLabel = "", yLabel = "", yMin, yMax, xFormat = fmt.compact, yFormat = (v) => v.toFixed(2), height = 260, refs = [], markers = [], labelAt = "last" } = opts;
  const W = width(el, 520);
  const m = { t: 14, r: 92, b: 40, l: yLabel ? 66 : 48 };
  const H = height;
  const all = series.flatMap((sr) => sr.points);
  const endLabels = [];
  const xs = all.map((p) => p.x).concat(markers.map((p) => p.x));
  const ys = all.map((p) => p.y).concat(markers.map((p) => p.y));
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  if (xlog) { x0 = Math.log10(x0) - 0.05; x1 = Math.log10(x1) + 0.05; }
  const y0 = yMin ?? Math.min(...ys), y1 = yMax ?? Math.max(...ys);
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const X = (v) => m.l + (((xlog ? Math.log10(v) : v) - x0) / (x1 - x0 || 1)) * iw;
  const Y = (v) => m.t + ih - ((v - y0) / (y1 - y0 || 1)) * ih;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}` });
  for (const t of niceTicks(y0, y1, 4)) {
    svg.appendChild(s("line", { x1: m.l, x2: m.l + iw, y1: Y(t), y2: Y(t), stroke: "var(--rule-1)" }));
    svg.appendChild(s("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, yFormat(t)));
  }
  const xt = xlog
    ? [1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5, 2e5].filter((v) => Math.log10(v) >= x0 && Math.log10(v) <= x1)
    : niceTicks(x0, x1, 5);
  for (const t of xt) {
    svg.appendChild(s("line", { x1: X(t), x2: X(t), y1: m.t + ih, y2: m.t + ih + 4, stroke: "var(--rule-3)" }));
    svg.appendChild(s("text", { x: X(t), y: m.t + ih + 17, "text-anchor": "middle" }, xFormat(t)));
  }
  svg.appendChild(s("line", { x1: m.l, x2: m.l + iw, y1: m.t + ih, y2: m.t + ih, stroke: "var(--rule-3)" }));
  if (xLabel) svg.appendChild(s("text", { x: m.l + iw / 2, y: H - 4, "text-anchor": "middle", class: "lbl-2", "font-family": "IBM Plex Sans", "font-size": 11.5 }, xLabel));
  if (yLabel) svg.appendChild(s("text", { x: 12, y: m.t + ih / 2, transform: `rotate(-90 12 ${m.t + ih / 2})`, "text-anchor": "middle", class: "lbl-2", "font-family": "IBM Plex Sans", "font-size": 11.5 }, yLabel));
  for (const r of refs) {
    svg.appendChild(s("line", { x1: m.l, x2: m.l + iw, y1: Y(r.y), y2: Y(r.y), stroke: "var(--ink-3)", "stroke-dasharray": "3 3" }));
    svg.appendChild(s("text", { x: m.l + iw - 4, y: Y(r.y) - 6, "text-anchor": "end" }, r.label));
  }
  for (const sr of series) {
    const d = sr.points.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(p.y)}`).join("");
    svg.appendChild(s("path", { d, fill: "none", stroke: sr.color, "stroke-width": sr.width || 2, "stroke-linejoin": "round", "stroke-linecap": "round", opacity: sr.faint ? 0.45 : 1 }));
    for (const p of sr.points) {
      const c = s("circle", { cx: X(p.x), cy: Y(p.y), r: sr.faint ? 3 : 4, fill: sr.color, stroke: "var(--bench-1)", "stroke-width": 2, opacity: sr.faint ? 0.6 : 1 });
      const hit = s("circle", { cx: X(p.x), cy: Y(p.y), r: 10, fill: "transparent" });
      hit.addEventListener("mousemove", (e) => tip(`<b>${esc(sr.name)}</b>${(p.tip || [[xLabel || "x", xFormat(p.x)], [yLabel || "y", yFormat(p.y)]]).map(([k, v]) => `<div class="row"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("")}`, e));
      hit.addEventListener("mouseleave", () => tip(null));
      svg.appendChild(c);
      svg.appendChild(hit);
    }
    if (sr.endLabel !== false && sr.points.length) {
      const p = labelAt === "first" ? sr.points[0] : sr.points[sr.points.length - 1];
      endLabels.push({ x: X(p.x) + 8, y: Y(p.y) + (labelAt === "first" ? 14 : 4), text: sr.short || sr.name });
    }
  }
  // de-collide end labels: keep order, enforce a minimum vertical gap
  endLabels.sort((a, b) => a.y - b.y);
  for (let i = 1; i < endLabels.length; i++) {
    if (Math.abs(endLabels[i].x - endLabels[i - 1].x) < 60 && endLabels[i].y - endLabels[i - 1].y < 13) endLabels[i].y = endLabels[i - 1].y + 13;
  }
  for (const l of endLabels) svg.appendChild(s("text", { x: l.x, y: l.y, class: "lbl-2", "font-family": "IBM Plex Sans", "font-size": 11.5 }, l.text));
  for (const mk of markers) {
    const g = s("rect", { x: X(mk.x) - 4.5, y: Y(mk.y) - 4.5, width: 9, height: 9, fill: mk.color, stroke: "var(--bench-1)", "stroke-width": 2, transform: `rotate(45 ${X(mk.x)} ${Y(mk.y)})` });
    const hit = s("circle", { cx: X(mk.x), cy: Y(mk.y), r: 10, fill: "transparent" });
    hit.addEventListener("mousemove", (e) => tip(`<b>${esc(mk.name)}</b>${mk.tip.map(([k, v]) => `<div class="row"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join("")}`, e));
    hit.addEventListener("mouseleave", () => tip(null));
    svg.appendChild(g);
    svg.appendChild(hit);
  }
  el.appendChild(svg);
}

// ---------------------------------------------------------- column groups
// groups: [{label, bars: [{name, value, color}]}]
export function columns(el, groups, { yMax = 1, format = (v) => fmt.p(v, 0), height = 230 } = {}) {
  el.innerHTML = "";
  const W = width(el, 520);
  const m = { t: 18, r: 8, b: 36, l: 40 };
  const H = height;
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const Y = (v) => m.t + ih - (v / yMax) * ih;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}` });
  for (const t of niceTicks(0, yMax, 4)) {
    svg.appendChild(s("line", { x1: m.l, x2: m.l + iw, y1: Y(t), y2: Y(t), stroke: "var(--rule-1)" }));
    svg.appendChild(s("text", { x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, format(t)));
  }
  const gw = iw / groups.length;
  groups.forEach((g, gi) => {
    const n = g.bars.length;
    const bw = Math.min(22, (gw * 0.7) / n);
    const gx = m.l + gi * gw + (gw - (bw + 2) * n) / 2;
    g.bars.forEach((b, i) => {
      if (b.value == null) return;
      const xx = gx + i * (bw + 2);
      const path = s("path", { d: roundedCol(xx, Y(0), bw, Y(0) - Y(b.value)), fill: b.color, opacity: 0.95 });
      path.addEventListener("mousemove", (e) => tip(`<b>${esc(b.name)}</b><div class="row"><span>${esc(g.label)}</span><b>${format(b.value)}</b></div>`, e));
      path.addEventListener("mouseleave", () => tip(null));
      svg.appendChild(path);
      if (b.label) svg.appendChild(s("text", { x: xx + bw / 2, y: Y(b.value) - 5, "text-anchor": "middle", class: "lbl-2", "font-size": 10 }, b.label));
    });
    svg.appendChild(s("text", { x: m.l + gi * gw + gw / 2, y: H - 14, "text-anchor": "middle", class: "lbl-2", "font-family": "IBM Plex Sans", "font-size": 12 }, g.label));
  });
  svg.appendChild(s("line", { x1: m.l, x2: m.l + iw, y1: Y(0), y2: Y(0), stroke: "var(--rule-3)" }));
  el.appendChild(svg);
}

// ------------------------------------------------------- confusion matrix
const SEQ = ["#14243b", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7"];
export function confusion(el, matrix, labels, title) {
  el.innerHTML = "";
  const short = { SUPPORTED: "SUP", REFUTED: "REF", "NOT ENOUGH INFO": "NEI" };
  const n = labels.length;
  const cell = 46, m = { t: 34, l: 40 };
  const W = m.l + n * cell + 4, H = m.t + n * cell + 4;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, style: `max-width:${W * 1.25}px` });
  const max = Math.max(1, ...matrix.flat());
  if (title) el.insertAdjacentHTML("beforeend", `<h5>${esc(title)}</h5>`);
  labels.forEach((lab, i) => {
    svg.appendChild(s("text", { x: m.l + i * cell + cell / 2, y: m.t - 8, "text-anchor": "middle" }, short[lab] || lab));
    svg.appendChild(s("text", { x: m.l - 8, y: m.t + i * cell + cell / 2 + 4, "text-anchor": "end" }, short[lab] || lab));
  });
  matrix.forEach((row, i) =>
    row.forEach((v, j) => {
      const t = v / max;
      const fill = v === 0 ? "var(--bench-2)" : SEQ[Math.min(SEQ.length - 1, Math.floor(t * (SEQ.length - 1)) + (t > 0 ? 0 : 0))];
      const r = s("rect", { x: m.l + j * cell + 1, y: m.t + i * cell + 1, width: cell - 2, height: cell - 2, rx: 4, fill, stroke: i === j ? "var(--rule-4)" : "none" });
      r.addEventListener("mousemove", (e) => tip(`<div class="row"><span>gold</span><b>${labels[i]}</b></div><div class="row"><span>predicted</span><b>${labels[j]}</b></div><div class="row"><span>claims</span><b>${v}</b></div>`, e));
      r.addEventListener("mouseleave", () => tip(null));
      svg.appendChild(r);
      svg.appendChild(s("text", { x: m.l + j * cell + cell / 2, y: m.t + i * cell + cell / 2 + 4, "text-anchor": "middle", fill: t > 0.45 ? "#fff" : "var(--ink-2)", style: "pointer-events:none" }, String(v)));
    }),
  );
  el.appendChild(svg);
  el.insertAdjacentHTML("beforeend", `<p class="muted" style="margin:6px 0 0;font-size:11px">rows = gold · columns = predicted</p>`);
}

// ------------------------------------------------------ latency dot plot
// rows: [{label, color, values: [{series, p50, p95}]}], series: [{name, shape, color}]
export function dotplot(el, rows, series, { height } = {}) {
  el.innerHTML = "";
  const W = width(el, 520);
  const rowH = 34;
  const m = { t: 8, r: 70, b: 32, l: 132 };
  const H = height || m.t + rows.length * rowH + m.b;
  const all = rows.flatMap((r) => r.values.flatMap((v) => [v.p50, v.p95])).filter((v) => v > 0);
  const lo = Math.log10(Math.min(...all) * 0.6), hi = Math.log10(Math.max(...all) * 1.6);
  const iw = W - m.l - m.r;
  const X = (v) => m.l + ((Math.log10(v) - lo) / (hi - lo)) * iw;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}` });
  for (const t of [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]) {
    if (Math.log10(t) < lo || Math.log10(t) > hi) continue;
    svg.appendChild(s("line", { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b, stroke: "var(--rule-1)" }));
    svg.appendChild(s("text", { x: X(t), y: H - 12, "text-anchor": "middle" }, t >= 1 ? `${t} ms` : `${t}`));
  }
  rows.forEach((r, i) => {
    const y = m.t + i * rowH + rowH / 2;
    svg.appendChild(s("rect", { x: 0, y: y - 4, width: 8, height: 8, rx: 2, fill: r.color }));
    svg.appendChild(s("text", { x: 16, y: y + 4, class: "lbl-2", "font-family": "IBM Plex Sans", "font-size": 12 }, r.label));
    r.values.forEach((v, j) => {
      const off = (j - (r.values.length - 1) / 2) * 9;
      const sr = series.find((x) => x.name === v.series) || series[0];
      svg.appendChild(s("line", { x1: X(v.p50), x2: X(v.p95), y1: y + off, y2: y + off, stroke: r.color, "stroke-width": 3, "stroke-linecap": "round" }));
      const mk = sr.shape === "diamond"
        ? s("rect", { x: X(v.p50) - 4, y: y + off - 4, width: 8, height: 8, fill: sr.color, stroke: "var(--bench-1)", "stroke-width": 1.5, transform: `rotate(45 ${X(v.p50)} ${y + off})` })
        : s("circle", { cx: X(v.p50), cy: y + off, r: 4.5, fill: sr.color, stroke: "var(--bench-1)", "stroke-width": 1.5 });
      const hit = s("circle", { cx: X(v.p50), cy: y + off, r: 10, fill: "transparent" });
      hit.addEventListener("mousemove", (e) => tip(`<b>${esc(r.label)} · ${esc(v.series)}</b><div class="row"><span>p50</span><b>${fmt.ms(v.p50)}</b></div><div class="row"><span>p95</span><b>${fmt.ms(v.p95)}</b></div>`, e));
      hit.addEventListener("mouseleave", () => tip(null));
      svg.appendChild(mk);
      svg.appendChild(hit);
    });
    const p50s = r.values.map((q) => fmt.ms(q.p50)).join(" / ");
    svg.appendChild(s("text", { x: X(Math.max(...r.values.map((q) => q.p95))) + 8, y: y + 4, class: "lbl-2" }, p50s));
  });
  el.appendChild(svg);
}
