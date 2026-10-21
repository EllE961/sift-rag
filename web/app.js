import { STAGE, columns, confusion, dotplot, esc, fmt, hbars, lines, rankFlow, scoreDistributions, tip, waterfall } from "./charts.js";
import { EmbeddingMap } from "./map.js";

// ------------------------------------------------------------------- state
const state = {
  view: "ask",
  collection: localStorage.getItem("sift.collection") || "scifact",
  mode: localStorage.getItem("sift.mode") || "hybrid_rerank",
  gen: localStorage.getItem("sift.gen") || "auto",
  chunker: "recursive",
  chunkSize: 160,
  collections: [],
  health: null,
  trace: null,
  segments: [],
  answerDone: false,
  results: null,
  projections: {},
  selectedDoc: null,
  chunkData: null,
  chunkStrategy: "recursive",
  abort: null,
};

const SUGGESTIONS = {
  scifact: [
    ["SUPPORTED in SciFact", "ALDH1 expression is associated with poorer prognosis in breast cancer."],
    ["REFUTED in SciFact", "Anthrax spores can be disposed of easily after they are dispersed."],
    ["SUPPORTED in SciFact", "Autophagy declines in aged organisms."],
    ["SUPPORTED in SciFact", "Bariatric surgery has a positive impact on mental health."],
    ["REFUTED in SciFact", "Chenodeoxycholic acid treatment reduces whole-body energy expenditure."],
    ["SUPPORTED in SciFact", "Birth-weight is positively associated with breast cancer."],
  ],
  nfcorpus: [
    ["NFCorpus", "Do statins reduce the risk of dying from breast cancer?"],
    ["NFCorpus", "Can diet help treat asthma and eczema?"],
    ["NFCorpus", "Is cholesterol a fuel source for breast cancer cells?"],
    ["NFCorpus", "Does eating fish expose people to mercury?"],
    ["NFCorpus", "What foods lower blood pressure?"],
    ["NFCorpus", "Are nuts linked to lower mortality?"],
  ],
  workspace: [
    ["Workspace", "What are the key findings?"],
    ["Workspace", "Summarize the methodology."],
    ["Workspace", "What limitations do the authors mention?"],
  ],
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function toast(msg, ms = 2600) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms);
}

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

// ------------------------------------------------------------------ router
const VIEWS = ["ask", "trace", "map", "eval", "ingest"];
function route() {
  const v = location.hash.replace("#", "") || "ask";
  state.view = VIEWS.includes(v) ? v : "ask";
  for (const sec of $$(".view")) sec.hidden = sec.dataset.view !== state.view;
  for (const a of $$(".tabs a")) a.setAttribute("aria-selected", String(a.dataset.view === state.view));
  moveTabInk();
  if (state.view === "trace") renderTrace();
  if (state.view === "map") showMap();
  if (state.view === "eval") renderEval();
  if (state.view === "ingest") refreshWorkspace();
}
function moveTabInk() {
  const a = $(`.tabs a[data-view="${state.view}"]`);
  const ink = $(".tab-ink");
  if (!a || !ink) return;
  const pr = a.parentElement.getBoundingClientRect();
  const r = a.getBoundingClientRect();
  ink.style.width = `${r.width - 24}px`;
  ink.style.transform = `translateX(${r.left - pr.left + 12}px)`;
}
window.addEventListener("hashchange", route);
window.addEventListener("resize", () => {
  moveTabInk();
  if (state.view === "trace") renderTrace();
});

// ---------------------------------------------------------------- controls
function segmented(root, attr, value, onChange) {
  const buttons = $$("button", root);
  const sync = (v) => buttons.forEach((b) => b.setAttribute("aria-checked", String(b.dataset[attr] === v)));
  sync(value);
  buttons.forEach((b) =>
    b.addEventListener("click", () => {
      if (b.disabled) return;
      sync(b.dataset[attr]);
      onChange(b.dataset[attr]);
    }),
  );
  return sync;
}

function initPicker() {
  const root = $("#collection-picker");
  const trigger = $(".picker-trigger", root);
  const menu = $(".picker-menu", root);
  const close = () => {
    menu.hidden = true;
    trigger.setAttribute("aria-expanded", "false");
  };
  trigger.addEventListener("click", (e) => {
    e.stopPropagation();
    const open = menu.hidden;
    menu.hidden = !open;
    trigger.setAttribute("aria-expanded", String(open));
  });
  document.addEventListener("click", close);
  document.addEventListener("keydown", (e) => e.key === "Escape" && close());
  menu.addEventListener("click", (e) => {
    const opt = e.target.closest(".picker-option");
    if (!opt) return;
    setCollection(opt.dataset.name);
    close();
  });
}

function renderPicker() {
  $("[data-value]").textContent = state.collection;
  const kinds = { benchmark: "BEIR benchmark · doc-level", workspace: "Your uploaded documents · chunked" };
  $(".picker-menu").innerHTML = state.collections
    .map(
      (c) => `<button class="picker-option" role="option" data-name="${c.name}" aria-selected="${c.name === state.collection}">
        <span class="name">${c.name}</span><span class="count">${c.n_passages != null ? fmt.n(c.n_passages) + " passages" : "not built"}</span>
        <span class="kind">${kinds[c.kind] || c.kind}</span></button>`,
    )
    .join("");
}

function setCollection(name) {
  if (name === state.collection) return;
  state.collection = name;
  localStorage.setItem("sift.collection", name);
  renderPicker();
  renderSuggestions();
  renderAnatomy();
  resetAsk();
  if (state.view === "map") showMap();
}

async function loadMeta() {
  try {
    const [health, cols] = await Promise.all([api("/api/health"), api("/api/collections")]);
    state.health = health;
    state.collections = cols;
    renderPicker();
    renderStatus();
  } catch (e) {
    toast(`Server unreachable: ${e.message}`);
  }
}

function renderStatus() {
  const h = state.health;
  if (!h) return;
  $("[data-device]").textContent = `${h.device} · ${h.embedder} · ${h.reranker || "no reranker"}`;
  const llm = $("[data-llm]");
  const st = h.llm.status;
  const cls = st === "ready" ? "ok" : st === "unknown" ? "warn" : "off";
  const label = !h.llm.available ? "LLM off · extractive" : st === "unavailable" ? `${h.llm.model} · unavailable` : h.llm.model;
  llm.innerHTML = `<i class="dot ${cls}"></i><span>${esc(label)}</span>`;
  llm.title = st === "unknown" ? "Claude configured — key not verified yet" : st;
}

// --------------------------------------------------------------------- ask
function renderSuggestions() {
  const list = SUGGESTIONS[state.collection] || SUGGESTIONS.scifact;
  $("#suggestions").innerHTML = list
    .map(([tag, q]) => `<button class="suggestion" type="button" data-q="${esc(q)}"><span class="tag">${esc(tag)}</span>${esc(q)}</button>`)
    .join("");
}

async function renderAnatomy() {
  if (!state.results) {
    try {
      state.results = await api("/api/results");
    } catch {
      return;
    }
  }
  const R = state.results;
  const d = R.retrieval?.datasets?.[state.collection];
  const set = (stage, html) => {
    const em = $(`#anatomy-steps li[data-stage="${stage}"] em`);
    if (em) em.innerHTML = html;
  };
  ["bm25", "dense", "fusion", "rerank", "answer"].forEach((s) => set(s, ""));
  $("#anatomy-note").textContent = "";
  if (d) {
    const m = d.metrics;
    const v = (k) => m[k]?.["ndcg@10"];
    set("bm25", `${fmt.f(v("BM25"))}<small>nDCG@10</small>`);
    set("dense", `${fmt.f(v("Dense · bge-small (HNSW ef=128)") ?? v("Dense · bge-small"))}<small>nDCG@10</small>`);
    set("fusion", `${fmt.f(v("Hybrid · RRF"))}<small>nDCG@10</small>`);
    const blended = v("Hybrid RRF → rerank, blended (train-tuned)");
    if (blended != null) set("rerank", `${fmt.f(blended)}<small>nDCG@10</small>`);
    $("#anatomy-note").textContent = `· measured on ${state.collection} test queries`;
  }
  const C = R.claims_claude || R.claims_nli;
  if (C && state.collection === "scifact") {
    const c = C.conditions["sift@5"];
    set("answer", `${fmt.p(c.accuracy, 0)}<small>claim accuracy (${C.verifier === "claude" ? "Claude" : "NLI"})</small>`);
  }
}

function resetAsk() {
  state.trace = null;
  state.segments = [];
  $("#ask-grid").hidden = true;
  $("#sieve-strip").hidden = true;
  $("#ask-empty").hidden = false;
}

function renderSieve(trace, gen) {
  const el = $("#sieve-strip");
  const st = Object.fromEntries(trace.stages.map((s) => [s.name, s]));
  const steps = [{ cls: "corpus", label: "Corpus", val: fmt.n(trace.corpus_size), sub: "passages" }];
  if (st.bm25) steps.push({ cls: "lexical", label: "BM25", val: st.bm25.n, sub: fmt.ms(st.bm25.latency_ms) });
  if (st.dense) steps.push({ cls: "dense", label: st.dense.meta.index === "hnsw" ? "Dense · HNSW" : "Dense · exact", val: st.dense.n, sub: fmt.ms(st.dense.latency_ms + (st.embed?.latency_ms || 0)) });
  if (st.fusion) steps.push({ cls: "fusion", label: st.fusion.meta.method === "rrf" ? "Fusion · RRF" : "Fusion · convex", val: st.fusion.n, sub: fmt.ms(st.fusion.latency_ms) });
  if (st.rerank) steps.push({ cls: "rerank", label: "Rerank", val: `${st.rerank.meta.depth}→${trace.hits.length}`, sub: fmt.ms(st.rerank.latency_ms) });
  steps.push({ cls: "answer", label: gen?.label || "Answer", val: gen?.val || "…", sub: gen?.sub || "", pending: !gen?.done });
  el.innerHTML = steps
    .map(
      (s) => `<a class="sieve-step ${s.pending ? "pending" : ""}" href="#trace" title="Open the full trace">
        <div class="lbl"><i class="sw ${s.cls}"></i>${esc(s.label)}</div>
        <div class="val">${esc(String(s.val))}<small>${esc(s.sub)}</small></div>
        <i class="mesh ${s.cls}"></i></a>`,
    )
    .join("");
  el.hidden = false;
}

function citationsFor(n) {
  const out = [];
  for (const seg of state.segments) for (const c of seg.citations) if (c.n === n) out.push(c);
  return out;
}

function highlightText(text, terms, cites) {
  // boundaries from both span sets, then emit pieces with flags
  const pts = new Set([0, text.length]);
  for (const [a, b] of terms) { pts.add(a); pts.add(b); }
  for (const c of cites) { pts.add(c.start); pts.add(c.end); }
  const cuts = [...pts].filter((p) => p >= 0 && p <= text.length).sort((a, b) => a - b);
  let html = "";
  for (let i = 0; i < cuts.length - 1; i++) {
    const a = cuts[i], b = cuts[i + 1];
    if (a === b) continue;
    const piece = esc(text.slice(a, b));
    const isTerm = terms.some(([x, y]) => a >= x && b <= y);
    const cited = cites.some((c) => a >= c.start && b <= c.end);
    let h = isTerm ? `<span class="term">${piece}</span>` : piece;
    if (cited) h = `<mark data-cite>${h}</mark>`;
    html += h;
  }
  return html;
}

function renderSources(trace) {
  const list = $("#source-list");
  list.innerHTML = "";
  const order = trace.stage_order;
  trace.hits.forEach((hit, i) => {
    const n = i + 1;
    const li = document.createElement("li");
    li.className = "source";
    li.id = `src-${n}`;
    li.style.animationDelay = `${i * 45}ms`;
    const chips = order
      .map((st) => {
        const r = hit.stage_ranks[st];
        const meta = STAGE[st];
        return `<span class="rank-chip ${r ? "" : "absent"}" title="${meta.label} rank"><i class="sw ${meta.cls}"></i>${meta.label.split(" ")[0]} <b>${r ? "#" + r : "—"}</b></span>`;
      })
      .join("");
    li.innerHTML = `<span class="num">${n}</span>
      <h3>${esc(hit.passage.title || hit.passage.doc_id)}<span class="doc-id">${esc(hit.passage.doc_id)}</span></h3>
      <div class="ranks">${chips}</div>
      <p class="source-text clamp" data-text></p>
      <button class="expand" type="button">Show full passage</button>`;
    li._hit = hit;
    $(".expand", li).addEventListener("click", (e) => {
      const p = $("[data-text]", li);
      p.classList.toggle("clamp");
      e.target.textContent = p.classList.contains("clamp") ? "Show full passage" : "Collapse";
    });
    list.appendChild(li);
    paintSource(n);
  });
  const st = Object.fromEntries(trace.stages.map((s) => [s.name, s]));
  $("#sources-meta").textContent = `top ${trace.hits.length} of ${fmt.n(trace.corpus_size)} · ${fmt.ms(trace.total_ms)}${st.dense?.meta?.distance_evals ? ` · HNSW scanned ${fmt.n(st.dense.meta.distance_evals)} vectors` : ""}`;
}

function paintSource(n) {
  const li = $(`#src-${n}`);
  if (!li) return;
  const cites = citationsFor(n);
  const p = $("[data-text]", li);
  p.innerHTML = highlightText(li._hit.passage.text, li._hit.highlights || [], cites);
  li.classList.toggle("cited", cites.length > 0);
  $(".expand", li).hidden = li._hit.passage.text.length < 520;
}

function focusSource(n) {
  const li = $(`#src-${n}`);
  if (!li) return;
  li.scrollIntoView({ behavior: "smooth", block: "center" });
  li.classList.add("focus");
  const p = $("[data-text]", li);
  p.classList.remove("clamp");
  $(".expand", li).textContent = "Collapse";
  const marks = $$("mark", li);
  marks.forEach((m) => m.classList.add("flash"));
  setTimeout(() => {
    li.classList.remove("focus");
    marks.forEach((m) => m.classList.remove("flash"));
  }, 1600);
}

// --- answer rendering: segments -> markdown-lite with citation chips
let renderQueued = false;
function queueAnswerRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    renderAnswer();
  });
}

function renderAnswer() {
  const CHIP = "\u0001";
  let raw = "";
  const chipList = [];
  for (const seg of state.segments) {
    if (!seg) continue;
    // chips attach to the end of the cited text, before any trailing newline
    const body = seg.text.replace(/\s+$/, "");
    raw += body;
    const ns = [...new Set(seg.citations.map((c) => c.n))];
    for (const n of ns) {
      raw += `${CHIP}${chipList.length}${CHIP}`;
      chipList.push(n);
    }
    raw += seg.text.slice(body.length);
  }
  let html = esc(raw)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|\n)#{1,4}\s*(.+)/g, "$1<h4>$2</h4>");
  const blocks = html.split(/\n{2,}/).map((block) => {
    if (/^<h4>/.test(block.trim())) return block;
    const linesArr = block.split("\n");
    if (linesArr.every((l) => /^\s*([-•*]|\d+\.)\s+/.test(l) || !l.trim())) {
      return `<ul>${linesArr.filter((l) => l.trim()).map((l) => `<li>${l.replace(/^\s*([-•*]|\d+\.)\s+/, "")}</li>`).join("")}</ul>`;
    }
    return `<p>${block.replace(/\n/g, "<br>")}</p>`;
  });
  html = blocks.join("");
  html = html.replace(new RegExp(`${CHIP}(\\d+)${CHIP}`, "g"), (_, i) => {
    const n = chipList[Number(i)];
    return `<button class="cite" type="button" data-n="${n}" aria-label="Source ${n}">${n}</button>`;
  });
  if (!state.answerDone) html = html.replace(/(<\/p>|<\/ul>)?$/, (m) => `<span class="caret"></span>${m || ""}`);
  const body = $("#answer-body");
  body.innerHTML = html || (state.answerDone ? `<p class="muted">No answer.</p>` : `<span class="caret"></span>`);
}

$("#answer-body").addEventListener("click", (e) => {
  const chip = e.target.closest(".cite");
  if (chip) focusSource(Number(chip.dataset.n));
});
$("#answer-body").addEventListener("mouseover", (e) => {
  const chip = e.target.closest(".cite");
  if (!chip) return;
  const n = Number(chip.dataset.n);
  const cites = citationsFor(n);
  const hit = state.trace?.hits[n - 1];
  if (hit) tip(`<div class="title">${esc(hit.passage.title)}</div>${cites.slice(0, 2).map((c) => `<div style="margin-top:4px;color:var(--marker-ink)">“${esc(c.cited_text.slice(0, 220))}${c.cited_text.length > 220 ? "…" : ""}”</div>`).join("")}`, e);
});
$("#answer-body").addEventListener("mouseout", (e) => e.target.closest(".cite") && tip(null));

function answerSkeleton() {
  $("#answer-body").innerHTML = `<div class="answer-skel"><i style="width:92%"></i><i style="width:84%"></i><i style="width:88%"></i><i style="width:46%"></i></div>`;
}

async function ask(query) {
  if (!query.trim()) return;
  state.abort?.abort();
  const ctrl = new AbortController();
  state.abort = ctrl;
  state.segments = [];
  state.answerDone = false;
  $("#ask-empty").hidden = true;
  $("#ask-grid").hidden = false;
  $("#answer-notice").hidden = true;
  $("#answer-meta").textContent = "retrieving…";
  $("#source-list").innerHTML = "";
  answerSkeleton();
  let genInfo = { label: "Answer", val: "…", sub: "" };
  const t0 = performance.now();
  let firstText = 0;
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, collection: state.collection, mode: state.mode, generator: state.gen, k: 5 }),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      let d = res.statusText;
      try { d = (await res.json()).detail; } catch {}
      throw new Error(d);
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const data = block.split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n");
        if (!data) continue;
        const ev = JSON.parse(data);
        switch (ev.type) {
          case "trace":
            state.trace = ev.trace;
            renderSources(ev.trace);
            renderSieve(ev.trace, genInfo);
            $("#answer-meta").textContent = "generating…";
            mapInstance?.setResult(ev.trace);
            if (state.view === "trace") renderTrace();
            if (state.view === "map") showMap();
            break;
          case "generator":
            genInfo = ev.name === "claude" ? { label: "Claude", val: "…", sub: "" } : { label: "Extractive", val: "…", sub: "offline" };
            renderSieve(state.trace, genInfo);
            if (!$("#answer-body .answer-skel")) answerSkeleton();
            state.segments = [];
            break;
          case "notice":
            $("#answer-notice").textContent = ev.message;
            $("#answer-notice").hidden = false;
            break;
          case "segment":
            state.segments[ev.index] = state.segments[ev.index] || { text: "", citations: [] };
            break;
          case "delta":
            if (!firstText) {
              firstText = performance.now();
              genInfo.val = "streaming";
              genInfo.sub = "";
              renderSieve(state.trace, genInfo);
            }
            (state.segments[ev.index] ||= { text: "", citations: [] }).text += ev.text;
            queueAnswerRender();
            break;
          case "citation":
            (state.segments[ev.index] ||= { text: "", citations: [] }).citations.push(ev.citation);
            paintSource(ev.citation.n);
            queueAnswerRender();
            break;
          case "error":
            $("#answer-notice").textContent = ev.message;
            $("#answer-notice").hidden = false;
            break;
          case "done": {
            state.answerDone = true;
            const a = ev.answer;
            state.segments = a.segments.length ? a.segments : state.segments;
            renderAnswer();
            for (let n = 1; n <= (state.trace?.hits.length || 0); n++) paintSource(n);
            const parts = [a.mode === "claude" ? a.model : "offline extractive", fmt.ms(a.latency_ms || performance.now() - t0)];
            if (a.output_tokens) parts.push(`${fmt.n(a.input_tokens)} in · ${fmt.n(a.output_tokens)} out`);
            if (a.cost_usd) parts.push(`$${a.cost_usd.toFixed(4)}`);
            $("#answer-meta").textContent = parts.join(" · ");
            if (a.error) {
              $("#answer-notice").textContent = a.error;
              $("#answer-notice").hidden = false;
            }
            genInfo = a.ttft_ms
              ? { ...genInfo, val: fmt.ms(a.ttft_ms), sub: `TTFT · ${fmt.ms(a.latency_ms)} total`, done: true }
              : { ...genInfo, val: fmt.ms(a.latency_ms), sub: a.mode === "extractive" ? "offline" : "total", done: true };
            if (state.trace) renderSieve(state.trace, genInfo);
            loadMeta();
            break;
          }
        }
      }
    }
  } catch (e) {
    if (e.name === "AbortError") return;
    state.answerDone = true;
    $("#answer-body").innerHTML = "";
    $("#answer-meta").textContent = "";
    $("#answer-notice").textContent = e.message;
    $("#answer-notice").hidden = false;
  }
}

// ------------------------------------------------------------------ trace
function renderTrace() {
  const t = state.trace;
  const root = $("#view-trace");
  $("[data-empty]", root).hidden = !!t;
  $("[data-content]", root).hidden = !t;
  if (!t) return;
  $("#trace-query").textContent = t.query;
  $("#trace-total").innerHTML = `<div class="big">${t.total_ms.toFixed(1)}<small>ms</small></div><div class="cap">retrieval wall-clock · ${esc(t.collection)} · mode ${esc(t.config.mode)}</div>`;
  const st = Object.fromEntries(t.stages.map((s) => [s.name, s]));
  const N = t.corpus_size;
  const cells = [{ cls: "corpus", label: "Corpus", n: N, sub: `${esc(t.collection)}` }];
  if (st.bm25) cells.push({ cls: "lexical", label: "BM25", n: st.bm25.n, sub: `k1=${st.bm25.meta.k1} b=${st.bm25.meta.b} · ${fmt.ms(st.bm25.latency_ms)}` });
  if (st.dense) cells.push({ cls: "dense", label: "Dense", n: st.dense.n, sub: `${st.dense.meta.index}${st.dense.meta.ef ? " ef=" + st.dense.meta.ef : ""}${st.dense.meta.distance_evals ? " · " + fmt.compact(st.dense.meta.distance_evals) + " dists" : ""}` });
  if (st.fusion) cells.push({ cls: "fusion", label: "Fusion", n: st.fusion.n, sub: st.fusion.meta.method === "rrf" ? `RRF k=${st.fusion.meta.k}` : `convex α=${st.fusion.meta.alpha}` });
  if (st.rerank) cells.push({ cls: "rerank", label: "Rerank", n: st.rerank.meta.depth, sub: `${st.rerank.meta.blend != null ? `blend w=${st.rerank.meta.blend}` : "pure CE"} · ${fmt.ms(st.rerank.latency_ms)}` });
  cells.push({ cls: "answer", label: "Results", n: t.hits.length, sub: "passed to the reader" });
  $("#funnel").innerHTML = cells
    .map((c, i) => {
      const w = Math.max(2, (Math.log(c.n + 1) / Math.log(N + 1)) * 100);
      return `<div class="funnel-stage"><div class="lbl"><i class="sw ${c.cls}"></i>${c.label}</div>
        <div class="n">${fmt.n(c.n)}</div><div class="sub">${c.sub}</div>
        <div class="bar"><i class="mesh ${c.cls}" style="width:${w}%;animation-delay:${i * 70}ms"></i></div>
        ${i < cells.length - 1 ? '<span class="arrow"></span>' : ""}</div>`;
    })
    .join("");
  rankFlow($("#rankflow"), t);
  waterfall($("#waterfall"), t);
  scoreDistributions($("#scoredist"), t);
}

// -------------------------------------------------------------------- map
let mapInstance = null;
async function showMap() {
  const canvas = $("#map-canvas");
  if (!mapInstance) {
    mapInstance = new EmbeddingMap(canvas, $("#map-tooltip"));
    $("#map-reset").addEventListener("click", () => mapInstance.reset());
  }
  const name = state.collection;
  const empty = $("#map-empty");
  if (!(name in state.projections)) {
    try {
      state.projections[name] = await api(`/api/projection/${name}`);
    } catch {
      state.projections[name] = null;
    }
  }
  const proj = state.projections[name];
  if (!proj) {
    empty.hidden = false;
    empty.innerHTML = `<div><p class="eyebrow">No map for “${esc(name)}”</p><p>Maps are precomputed for the benchmark collections<br><span class="mono">python -m sift.eval projection</span></p></div>`;
    mapInstance.data = null;
    mapInstance.draw();
    $("#cluster-list").innerHTML = "";
    return;
  }
  empty.hidden = true;
  if (mapInstance.data !== proj) mapInstance.setData(proj);
  mapInstance.resize();
  $("#cluster-list").innerHTML = proj.clusters
    .slice()
    .sort((a, b) => b.size - a.size)
    .map((c) => `<li><button type="button" data-c="${c.id}"><span><i class="cl-sw" style="background:${c.color || "var(--ink-3)"}"></i>${esc(c.label)}</span><span class="n">${fmt.n(c.size)}</span></button></li>`)
    .join("");
  $$("#cluster-list button").forEach((b) => {
    const id = Number(b.dataset.c);
    b.addEventListener("click", () => mapInstance.focusCluster(id));
    b.addEventListener("mouseenter", () => { mapInstance.hoverCluster = id; mapInstance.draw(); });
    b.addEventListener("mouseleave", () => { mapInstance.hoverCluster = -1; mapInstance.draw(); });
  });
  const t = state.trace;
  if (t && t.collection === name) {
    mapInstance.setResult(t);
    $("#map-hits").innerHTML = t.hits.map((h, i) => `<li><span class="n">${i + 1}</span><span>${esc(h.passage.title)}</span></li>`).join("");
    setTimeout(() => mapInstance.focusResult(), 120);
  }
}

// ------------------------------------------------------------------- eval
const FAMILY = {
  BM25: "var(--stage-lexical)",
  "Dense · MiniLM-L6": "var(--stage-dense)",
  "Dense · bge-small": "var(--stage-dense)",
  "Dense · bge-small (HNSW ef=128)": "var(--stage-dense)",
  "Hybrid · RRF": "var(--stage-fusion)",
  "Hybrid · convex (train-tuned)": "var(--stage-fusion)",
  "BM25 → rerank": "var(--stage-rerank)",
  "Hybrid RRF → rerank": "var(--stage-rerank)",
  "Hybrid convex → rerank": "var(--stage-rerank)",
};
const shortName = (m) => m.replace("Hybrid RRF → rerank, blended (train-tuned)", "Hybrid → blended rerank").replace(" (train-tuned)", " (tuned)");
const familyOf = (m) => FAMILY[m] || (m.includes("rerank") ? "var(--stage-rerank)" : m.startsWith("Hybrid") ? "var(--stage-fusion)" : m.startsWith("Dense") ? "var(--stage-dense)" : "var(--stage-lexical)");
const COND = {
  no_evidence: ["No evidence", "var(--ink-4)"],
  closed_book: ["Closed book", "var(--ink-4)"],
  "bm25@5": ["BM25 top-5", "var(--stage-lexical)"],
  "sift@5": ["Sift top-5", "var(--stage-rerank)"],
  oracle: ["Oracle evidence", "var(--ink-2)"],
};
const RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#1c5cab"];

async function renderEval() {
  const root = $("#eval-root");
  if (!state.results) {
    try {
      state.results = await api("/api/results");
    } catch (e) {
      root.innerHTML = `<div class="view-empty"><p class="eyebrow">Results unavailable</p><p>${esc(e.message)}</p></div>`;
      return;
    }
  }
  const R = state.results;
  if (!R.retrieval && !R.hnsw && !R.claims_nli) {
    root.innerHTML = `<div class="view-empty"><p class="eyebrow">No results yet</p><p>Run <span class="mono">python -m sift.eval retrieval</span>, <span class="mono">hnsw</span> and <span class="mono">claims</span>.</p></div>`;
    return;
  }
  if (root.dataset.rendered) return;
  root.dataset.rendered = "1";
  root.innerHTML = "";
  const ds = R.retrieval?.datasets || {};
  const names = Object.keys(ds);

  // hero numbers — the four things worth remembering
  const hero = document.createElement("div");
  hero.className = "eval-hero";
  const cells = [];
  if (ds.scifact) {
    const m = ds.scifact.metrics;
    const best = Object.entries(m).sort((a, b) => b[1]["ndcg@10"] - a[1]["ndcg@10"])[0];
    cells.push(["Best nDCG@10 · SciFact", fmt.f(best[1]["ndcg@10"]), `${best[0]} · BM25 ${fmt.f(m.BM25["ndcg@10"])}`]);
  }
  if (ds.nfcorpus) {
    const m = ds.nfcorpus.metrics;
    const best = Object.entries(m).sort((a, b) => b[1]["ndcg@10"] - a[1]["ndcg@10"])[0];
    cells.push(["Best nDCG@10 · NFCorpus", fmt.f(best[1]["ndcg@10"]), `${best[0]} · BM25 ${fmt.f(m.BM25["ndcg@10"])}`]);
  }
  if (R.hnsw) {
    const big = R.hnsw.scaling[R.hnsw.scaling.length - 1];
    const pt = big.curve.find((p) => p["recall@10"] >= 0.95) || big.curve[big.curve.length - 1];
    cells.push(["HNSW speed-up @ recall≥0.95", `${(pt.qps / big.flat_qps_1thread).toFixed(1)}×`, `${fmt.n(big.n)} vectors · ${fmt.n(Math.round(pt.qps))} QPS vs ${fmt.n(Math.round(big.flat_qps_1thread))} exact`]);
  }
  const claims = R.claims_claude || R.claims_nli;
  if (claims) {
    const c = claims.conditions["sift@5"];
    cells.push([`Claim verification · ${claims.verifier === "claude" ? claims.model : "NLI"}`, fmt.p(c.accuracy, 1), `macro-F1 ${fmt.f(c.macro_f1)} · rationale precision ${c.rationale_precision != null ? fmt.p(c.rationale_precision, 0) : "—"}`]);
  }
  hero.innerHTML = cells.map(([k, v, d]) => `<div class="hero-cell"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div><div class="d">${esc(d)}</div></div>`).join("");
  root.appendChild(hero);

  // retrieval table
  if (names.length) {
    const sec = section(root, "Retrieval quality", "BEIR test splits · exact dense search unless noted · convex weights tuned on train, never test");
    const methods = Object.keys(ds[names[0]].metrics);
    const cols = ["ndcg@10", "recall@100", "mrr@10"];
    const best = {};
    for (const n of names) for (const c of cols) best[`${n}${c}`] = Math.max(...methods.map((m) => ds[n].metrics[m][c]));
    const head = `<tr><th></th>${names.map((n) => `<th class="group" colspan="${cols.length}">${n}</th>`).join("")}</tr>
      <tr><th>Method</th>${names.map(() => cols.map((c) => `<th>${c.replace("ndcg", "nDCG").replace("recall", "Recall").replace("mrr", "MRR")}</th>`).join("")).join("")}</tr>`;
    const body = methods
      .map((m) => {
        const tds = names
          .map((n) =>
            cols
              .map((c) => {
                const v = ds[n].metrics[m][c];
                const isBest = Math.abs(v - best[`${n}${c}`]) < 1e-9;
                return `<td class="${isBest ? "best" : ""}"><span class="cell-bar"><i style="width:${Math.round(v * 56)}px"></i>${v.toFixed(3)}</span></td>`;
              })
              .join(""),
          )
          .join("");
        return `<tr><td class="method"><i class="method-dot" style="background:${familyOf(m)}"></i>${esc(m)}</td>${tds}</tr>`;
      })
      .join("");
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<div class="legend"><span><i class="sq" style="background:var(--stage-lexical)"></i>Lexical</span><span><i class="sq" style="background:var(--stage-dense)"></i>Dense</span><span><i class="sq" style="background:var(--stage-fusion)"></i>Hybrid</span><span><i class="sq" style="background:var(--stage-rerank)"></i>+ cross-encoder</span></div>
      <div class="table-wrap"><table class="results"><thead>${head}</thead><tbody>${body}</tbody></table></div>
      <p class="footnote">Published BM25 for reference: SciFact 0.665 (BEIR, Elasticsearch) / 0.679 (Anserini) · NFCorpus 0.325 / 0.322. Bold = column best.</p>`;
    sec.appendChild(panel);

    const grid = document.createElement("div");
    grid.className = "eval-grid";
    sec.appendChild(grid);
    for (const n of names) {
      const p = chartPanel(grid, `nDCG@10 · ${n}`, `${ds[n].n_test_queries} test queries · ${fmt.n(ds[n].n_docs)} docs`);
      const top = Math.max(...methods.map((m) => ds[n].metrics[m]["ndcg@10"]));
      hbars(p, methods.map((m) => ({ label: shortName(m), value: ds[n].metrics[m]["ndcg@10"], color: familyOf(m), metric: "nDCG@10", emph: ds[n].metrics[m]["ndcg@10"] === top })), { labelW: 220 });
    }
    // rerank: quality change vs depth (a delta keeps both datasets readable on one axis), blending, fusion
    const grid2 = document.createElement("div");
    grid2.className = "eval-grid";
    sec.appendChild(grid2);
    const shade = (i) => (i ? "var(--stage-rerank)" : "#c4bdf6");
    const pd = chartPanel(grid2, "Reranking depth: quality change", "Δ nDCG@10 vs. hybrid RRF without reranking · pure cross-encoder order");
    lines(pd, names.map((n, i) => {
      const base = ds[n].metrics["Hybrid · RRF"]["ndcg@10"];
      return { name: n, short: n, color: shade(i), points: ds[n].rerank_depth_sweep.map((r) => ({ x: r.depth, y: r["ndcg@10"] - base, tip: [["depth", r.depth], ["nDCG@10", r["ndcg@10"].toFixed(4)], ["Δ vs no rerank", (r["ndcg@10"] - base).toFixed(4)], ["p50 latency", fmt.ms(r.latency_ms_p50)]] })) };
    }), { xLabel: "candidates reranked (N)", yLabel: "Δ nDCG@10", yFormat: (v) => (v > 0 ? "+" : "") + v.toFixed(3), xFormat: (v) => String(v), height: 240, refs: [{ y: 0, label: "no rerank" }] });
    const pb = chartPanel(grid2, "Blending instead of replacing", "Weight on the cross-encoder score, tuned on train · Δ vs first stage only");
    lines(pb, names.filter((n) => ds[n].rerank_blend).map((n, i) => {
      const bl = ds[n].rerank_blend;
      const pts = bl.sweep.filter((r) => r.method === bl.best.method);
      const w0 = pts[0]["ndcg@10_train"];
      return { name: `${n} (best w=${bl.best.w})`, short: n, color: shade(i), points: pts.map((r) => ({ x: r.w, y: r["ndcg@10_train"] - w0, tip: [["w", r.w], ["nDCG@10 (train)", r["ndcg@10_train"].toFixed(4)], ["test @ chosen w", bl.test_metrics["ndcg@10"].toFixed(4)]] })) };
    }), { xLabel: "w (0 = first stage only · 1 = cross-encoder only)", yLabel: "Δ nDCG@10 (train)", yFormat: (v) => (v > 0 ? "+" : "") + v.toFixed(3), xFormat: (v) => v.toFixed(1), height: 240, refs: [{ y: 0, label: "first stage only" }] });
    const grid3 = document.createElement("div");
    grid3.className = "eval-grid";
    sec.appendChild(grid3);
    for (const n of names) {
      const tun = ds[n].convex_tuning;
      const pa = chartPanel(grid3, `Convex fusion weight · ${n}`, `α* = ${tun.best.alpha} (${tun.best.norm}) chosen on train · test shown faintly`);
      lines(pa, [
        { name: "train", short: "train", color: "var(--stage-fusion)", points: tun.sweep.filter((r) => r.norm === tun.best.norm).map((r) => ({ x: r.alpha, y: r["ndcg@10"], tip: [["α", r.alpha], ["nDCG@10 (train)", r["ndcg@10"].toFixed(4)]] })) },
        { name: "test (reference only)", short: "test", color: "var(--stage-fusion)", faint: true, width: 1.5, points: ds[n].convex_alpha_sweep_test_reference_only.map((r) => ({ x: r.alpha, y: r["ndcg@10"], tip: [["α", r.alpha], ["nDCG@10 (test)", r["ndcg@10"].toFixed(4)]] })) },
      ], { xLabel: "α (0 = BM25 only · 1 = dense only)", yLabel: "nDCG@10", xFormat: (v) => v.toFixed(1), yFormat: (v) => v.toFixed(2), height: 220, refs: [{ y: ds[n].metrics["Hybrid · RRF"]["ndcg@10"], label: "RRF (test)" }] });
    }
    const pt = chartPanel(sec, "Serving latency per stage", "p50 (marker) → p95 (bar end) · one query at a time · log scale · shared GPU");
    const stageNames = ["bm25", "embed", "dense", "fusion", "rerank"];
    const series = [{ name: names[0], shape: "circle", color: "var(--ink-1)" }, { name: names[1], shape: "diamond", color: "var(--ink-3)" }].filter((x) => x.name);
    dotplot(pt, stageNames.map((sn) => ({ label: sn === "rerank" ? "Cross-encoder @50" : STAGE[sn].label, color: STAGE[sn].color, values: names.filter((n) => ds[n].latency_hybrid_rerank.stages[sn]).map((n) => ({ series: n, ...ds[n].latency_hybrid_rerank.stages[sn] })) })), series);
    pt.insertAdjacentHTML("afterbegin", `<div class="legend" style="padding:0 0 6px">${series.map((x) => `<span><i class="sq" style="background:${x.color};${x.shape === "diamond" ? "transform:rotate(45deg)" : "border-radius:50%"}"></i>${x.name} · ${fmt.ms(ds[x.name].latency_hybrid_rerank.total_ms.p50)} p50 total</span>`).join("")}</div>`);
  }

  // HNSW
  if (R.hnsw) {
    const H = R.hnsw;
    const sec = section(root, "Approximate nearest neighbours", `HNSW from scratch (numba) on ${fmt.n(H.n_total)} real sentence embeddings · ${H.n_queries} real queries · single thread`);
    const grid = document.createElement("div");
    grid.className = "eval-grid";
    sec.appendChild(grid);
    const p = chartPanel(grid, "Recall@10 vs queries/second", "Each line sweeps ef; diamonds = exact search at the same N (recall 1.0)");
    const sizes = H.scaling;
    p.insertAdjacentHTML("beforebegin", `<div class="legend">${sizes.map((r, i) => `<span><i style="background:${RAMP[Math.min(i, RAMP.length - 1)]}"></i>N = ${fmt.n(r.n)}</span>`).join("")}</div>`);
    lines(p, sizes.map((r, i) => ({ name: `N = ${fmt.n(r.n)}`, short: `N=${fmt.compact(r.n)}`, endLabel: false, color: RAMP[Math.min(i, RAMP.length - 1)], points: r.curve.map((c) => ({ x: c.qps, y: c["recall@10"], tip: [["ef", c.ef], ["recall@10", c["recall@10"].toFixed(4)], ["QPS", fmt.n(Math.round(c.qps))], ["dist. evals / query", fmt.n(Math.round(c.distance_evals_per_query))]] })) })), {
      xlog: true, labelAt: "first", xLabel: "queries / second (log)", yLabel: "recall@10", yMin: Math.min(...sizes.flatMap((r) => r.curve.map((c) => c["recall@10"]))) - 0.02, yMax: 1.005, yFormat: (v) => v.toFixed(2), height: 300,
      markers: sizes.map((r, i) => ({ x: r.flat_qps_1thread, y: 1, color: RAMP[Math.min(i, RAMP.length - 1)], name: `Exact · N = ${fmt.n(r.n)}`, tip: [["QPS (1 thread)", fmt.n(Math.round(r.flat_qps_1thread))], ["QPS (BLAS threads)", fmt.n(Math.round(r.flat_qps_blas))]] })),
    });
    const p2 = chartPanel(grid, "Graph degree M", `N = ${fmt.n(H.n_total)} · higher M = denser graph, more memory, better recall per hop`);
    if (H.m_sweep.length) lines(p2, H.m_sweep.map((r, i) => ({ name: `M = ${r.M}`, short: `M=${r.M}`, color: RAMP[[0, 2, 4][i] ?? i], points: r.curve.map((c) => ({ x: c.qps, y: c["recall@10"], tip: [["ef", c.ef], ["recall@10", c["recall@10"].toFixed(4)], ["QPS", fmt.n(Math.round(c.qps))]] })) })), { xlog: true, labelAt: "first", xLabel: "queries / second (log)", yLabel: "recall@10", yMax: 1.005, yFormat: (v) => v.toFixed(2), height: 300 });
    else p2.innerHTML = `<p class="muted">M sweep not run yet.</p>`;
  }

  // claims
  for (const key of ["claims_claude", "claims_nli"]) {
    const C = R[key];
    if (!C) continue;
    const who = C.verifier === "claude" ? `${C.model} (effort ${C.effort})` : `NLI cross-encoder (${C.model.split("/").pop()})`;
    const sec = section(root, `Claim verification · ${C.verifier === "claude" ? "Claude" : "offline NLI"}`, `SciFact dev · ${C.n_dev} claims · ${who}${C.verifier === "nli" ? " · thresholds tuned on 809 train claims" : ""}`);
    const conds = Object.keys(C.conditions);
    const grid = document.createElement("div");
    grid.className = "eval-grid";
    sec.appendChild(grid);
    const p = chartPanel(grid, "What the verifier sees changes what it can conclude", "Accuracy · macro-F1 · citation precision · rationale precision");
    const metricsList = [["accuracy", "Accuracy"], ["macro_f1", "Macro-F1"], ["citation_precision", "Citation P"], ["rationale_precision", "Rationale P"]];
    columns(p, metricsList.map(([k, label]) => ({ label, bars: conds.map((c) => ({ name: COND[c]?.[0] || c, value: C.conditions[c][k], color: COND[c]?.[1] || "var(--ink-3)", label: C.conditions[c][k] != null ? Math.round(C.conditions[c][k] * 100) + "" : "" })) })), { height: 260 });
    p.insertAdjacentHTML("afterbegin", `<div class="legend" style="padding:0 0 8px">${conds.map((c) => `<span><i class="sq" style="background:${COND[c]?.[1]}"></i>${COND[c]?.[0] || c}</span>`).join("")}</div>`);
    const p2 = chartPanel(grid, "Evidence recall vs verdict accuracy", "Does the retrieved set contain the gold evidence abstracts?");
    hbars(p2, conds.filter((c) => C.conditions[c].evidence_recall != null).map((c) => ({ label: COND[c]?.[0] || c, value: C.conditions[c].evidence_recall, color: COND[c]?.[1], metric: "evidence recall", note: `accuracy ${fmt.p(C.conditions[c].accuracy)}` })), { max: 1, format: (v) => fmt.p(v, 0), labelW: 150 });
    const conf = document.createElement("div");
    conf.className = "panel";
    conf.innerHTML = `<header class="panel-head"><h3>Confusion matrices</h3><span class="meta">rows = gold label</span></header><div class="confusions"></div>`;
    sec.appendChild(conf);
    for (const c of conds) {
      const cell = document.createElement("div");
      cell.className = "confusion";
      $(".confusions", conf).appendChild(cell);
      confusion(cell, C.conditions[c].confusion.matrix, C.conditions[c].confusion.labels, `${COND[c]?.[0] || c} · acc ${fmt.p(C.conditions[c].accuracy, 0)}`);
    }
    if (C.usage) sec.insertAdjacentHTML("beforeend", `<p class="footnote" style="padding:0">Spend: ${fmt.n(C.usage.input_tokens)} input + ${fmt.n(C.usage.output_tokens)} output tokens · $${C.usage.cost_usd.toFixed(2)}</p>`);
  }
}

function section(root, title, sub) {
  const sec = document.createElement("section");
  sec.className = "eval-section";
  sec.innerHTML = `<header><h2>${esc(title)}</h2><p>${esc(sub)}</p></header>`;
  root.appendChild(sec);
  return sec;
}
function chartPanel(parent, title, meta) {
  const p = document.createElement("div");
  p.className = "panel";
  p.innerHTML = `<header class="panel-head"><h3>${esc(title)}</h3><span class="meta">${esc(meta)}</span></header><div class="chart"></div>`;
  parent.appendChild(p);
  return $(".chart", p);
}

// ----------------------------------------------------------------- ingest
async function refreshWorkspace() {
  try {
    const ws = await api("/api/workspace");
    const list = $("#ws-list");
    $("#ws-meta").textContent = ws.documents.length ? `${ws.documents.length} docs` : "";
    if (!ws.documents.length) {
      list.innerHTML = `<li class="muted" style="padding:8px 12px">Nothing ingested yet.</li>`;
      return;
    }
    list.innerHTML = ws.documents
      .map((d) => `<li><button type="button" data-id="${esc(d.id)}" class="${d.id === state.selectedDoc ? "on" : ""}"><span class="t">${esc(d.title)}</span><span class="m">${d.n_chunks} chunks</span><span class="m" style="grid-column:1/-1">${fmt.n(d.chars)} chars · ${esc(d.id)}</span></button></li>`)
      .join("");
    $$("#ws-list button").forEach((b) => b.addEventListener("click", () => compareChunks({ doc_id: b.dataset.id })));
  } catch {}
}

function fileToB64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).split(",")[1] || "");
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

async function ingestFiles(files) {
  if (!files.length) return;
  const dz = $("#dropzone");
  dz.classList.add("busy");
  try {
    const payload = { chunker: state.chunker, chunk_size: state.chunkSize, files: await Promise.all([...files].map(async (f) => ({ name: f.name, content_b64: await fileToB64(f) }))) };
    const res = await api("/api/ingest", { method: "POST", body: JSON.stringify(payload) });
    toast(`Indexed ${res.added_passages} chunks from ${res.documents.length} document${res.documents.length > 1 ? "s" : ""} in ${res.index_ms.toFixed(0)} ms`);
    await refreshWorkspace();
    await loadMeta();
    if (res.documents[0]) compareChunks({ doc_id: res.documents[0].id });
  } catch (e) {
    toast(e.message, 4200);
  } finally {
    dz.classList.remove("busy");
  }
}

async function compareChunks(src) {
  state.selectedDoc = src.doc_id || null;
  $$("#ws-list button").forEach((b) => b.classList.toggle("on", b.dataset.id === state.selectedDoc));
  const box = $("#chunk-compare");
  box.innerHTML = `<div class="skeleton-block" style="height:240px"></div>`;
  try {
    state.chunkData = await api("/api/chunk", { method: "POST", body: JSON.stringify({ ...src, chunk_size: Math.min(state.chunkSize, 400) }) });
    renderChunks();
  } catch (e) {
    box.innerHTML = `<div class="view-empty inline"><p>${esc(e.message)}</p></div>`;
  }
}

function renderChunks() {
  const d = state.chunkData;
  const box = $("#chunk-compare");
  const total = d.text.length || 1;
  const names = { fixed: "Fixed window", recursive: "Recursive", sentence: "Sentence", semantic: "Semantic" };
  const rows = Object.entries(d.strategies)
    .map(([k, v]) => {
      const sizes = v.chunks.map((c) => c.tokens);
      const mean = sizes.reduce((a, b) => a + b, 0) / (sizes.length || 1);
      const segs = v.chunks.map((c, i) => `<i data-i="${i}" style="flex:${Math.max(1, c.end - c.start)} 1 0" title="chunk ${i + 1} · ${c.tokens} words"></i>`).join("");
      return `<div class="ribbon-row ${k === state.chunkStrategy ? "on" : ""}" data-k="${k}"><span class="name">${names[k]}</span><div class="ribbon">${segs}</div><span class="stats">${v.chunks.length} chunks · μ ${mean.toFixed(0)}w</span></div>`;
    })
    .join("");
  box.innerHTML = `<div class="ribbons">${rows}</div><div class="chunk-text" id="chunk-text-view"></div><p class="footnote" style="padding:8px 0 0">Ribbon width ∝ characters · hover a segment to locate it · ${fmt.n(total)} characters</p>`;
  $$(".ribbon-row", box).forEach((row) => {
    row.addEventListener("click", () => {
      state.chunkStrategy = row.dataset.k;
      renderChunks();
    });
    $$(".ribbon i", row).forEach((seg) =>
      seg.addEventListener("mouseenter", () => {
        if (row.dataset.k !== state.chunkStrategy) return;
        hotChunk(Number(seg.dataset.i));
      }),
    );
  });
  const strat = d.strategies[state.chunkStrategy];
  let html = "";
  let pos = 0;
  strat.chunks.forEach((c, i) => {
    if (c.start > pos) html += esc(d.text.slice(pos, c.start));
    const brk = state.chunkStrategy === "semantic" && i > 0 ? `<span class="brk" style="opacity:${Math.min(1, 0.25 + (c.break_score || 0) * 2.5)}" title="boundary distance ${(c.break_score || 0).toFixed(3)}"></span>` : "";
    html += `${brk}<span class="tok">${i + 1}·${c.tokens}w</span><span class="c ${i % 2 ? "odd" : ""}" data-i="${i}">${esc(d.text.slice(Math.max(c.start, pos), c.end))}</span>`;
    pos = Math.max(pos, c.end);
  });
  html += esc(d.text.slice(pos));
  $("#chunk-text-view").innerHTML = html;
  $$("#chunk-text-view .c").forEach((sp) => sp.addEventListener("mouseenter", () => hotChunk(Number(sp.dataset.i))));
}

function hotChunk(i) {
  $$("#chunk-text-view .c").forEach((sp) => sp.classList.toggle("hot", Number(sp.dataset.i) === i));
  const row = $(`.ribbon-row[data-k="${state.chunkStrategy}"]`);
  if (row) $$(".ribbon i", row).forEach((seg) => seg.classList.toggle("hot", Number(seg.dataset.i) === i));
}

function initIngest() {
  const dz = $("#dropzone");
  const input = $("#file-input");
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("over");
    ingestFiles(e.dataTransfer.files);
  });
  input.addEventListener("change", () => ingestFiles(input.files));
  segmented($("#chunker-control"), "chunker", state.chunker, (v) => (state.chunker = v));
  const range = $("#chunk-size");
  range.addEventListener("input", () => {
    state.chunkSize = Number(range.value);
    $("#chunk-size-out").textContent = range.value;
  });
  range.addEventListener("change", () => {
    if (state.chunkData) compareChunks(state.selectedDoc ? { doc_id: state.selectedDoc } : { text: state.chunkData.text });
  });
  $("#chunk-preview").addEventListener("click", () => {
    const text = $("#chunk-text").value.trim();
    if (text) compareChunks({ text });
  });
}

// ------------------------------------------------------------------- boot
function init() {
  const boot = new URLSearchParams(location.search);
  if (["auto", "extractive"].includes(boot.get("gen"))) state.gen = boot.get("gen");
  if (boot.get("mode")) state.mode = boot.get("mode");
  if (boot.get("collection")) state.collection = boot.get("collection");
  initPicker();
  initIngest();
  segmented($("#mode-control"), "mode", state.mode, (v) => {
    state.mode = v;
    localStorage.setItem("sift.mode", v);
  });
  segmented($("#gen-control"), "gen", state.gen, (v) => {
    state.gen = v;
    localStorage.setItem("sift.gen", v);
  });
  $("#query-form").addEventListener("submit", (e) => {
    e.preventDefault();
    ask($("#query-input").value);
  });
  $("#suggestions").addEventListener("click", (e) => {
    const b = e.target.closest(".suggestion");
    if (!b) return;
    $("#query-input").value = b.dataset.q;
    ask(b.dataset.q);
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && document.activeElement.tagName !== "INPUT" && document.activeElement.tagName !== "TEXTAREA") {
      e.preventDefault();
      location.hash = "#ask";
      $("#query-input").focus();
    }
  });
  renderSuggestions();
  renderAnatomy();
  loadMeta();
  route();
  const params = new URLSearchParams(location.search);
  if (params.get("doc")) compareChunks({ doc_id: params.get("doc") });
  if (params.get("q")) {
    $("#query-input").value = params.get("q");
    if (params.get("collection")) setCollection(params.get("collection"));
    ask(params.get("q"));
  }
  requestAnimationFrame(moveTabInk);
  document.fonts?.ready.then(moveTabInk);
}

init();
