// Canvas embedding map: thousands of points, pan/zoom, topic labels,
// the current query and its retrieved neighbours.

const ease = (t) => 1 - Math.pow(1 - t, 3);

export class EmbeddingMap {
  constructor(canvas, tooltip) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.tooltip = tooltip;
    this.data = null;
    this.result = null;
    this.view = { k: 1, tx: 0, ty: 0 };
    this.hover = -1;
    this.hoverCluster = -1;
    this.anim = null;
    this.onHoverDoc = null;
    this._bind();
    new ResizeObserver(() => this.resize()).observe(canvas);
  }

  setData(proj) {
    this.data = proj;
    this.idIndex = new Map(proj.ids.map((id, i) => [id, i]));
    // points grouped by topic so each topic is one path / one fill
    this.byCluster = proj.clusters.map((c) => ({ id: c.id, rgb: hexToRgb(c.color || "#85857f"), idx: [] }));
    proj.cluster.forEach((c, i) => this.byCluster[c]?.idx.push(i));
    this.view = { k: 1, tx: 0, ty: 0 };
    this.resize();
  }

  setResult(trace) {
    if (!this.data || !trace?.map) {
      this.result = null;
    } else {
      const ids = trace.map.hit_ids.filter((id) => this.idIndex.has(id));
      this.result = { q: trace.map.query_xy, hits: ids.map((id) => this.idIndex.get(id)) };
    }
    this.pulse = performance.now();
    this.draw();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect();
    if (!r.width) return;
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width;
    this.h = r.height;
    this.draw();
  }

  // world [0,1]^2 -> screen
  base() {
    const size = Math.min(this.w, this.h) * 0.86;
    return { size, ox: (this.w - size) / 2, oy: (this.h - size) / 2 };
  }
  toScreen(x, y, v = this.view) {
    const b = this.base();
    return [b.ox + (x * b.size) * v.k + v.tx, b.oy + (y * b.size) * v.k + v.ty];
  }
  toWorld(sx, sy) {
    const b = this.base();
    return [(sx - b.ox - this.view.tx) / (b.size * this.view.k), (sy - b.oy - this.view.ty) / (b.size * this.view.k)];
  }

  animateTo(target, ms = 480) {
    const from = { ...this.view };
    const t0 = performance.now();
    const step = () => {
      // rAF timestamps can predate t0; a negative t through the cubic ease would overshoot backwards
      const t = Math.min(1, Math.max(0, (performance.now() - t0) / ms));
      const e = ease(t);
      this.view = { k: from.k + (target.k - from.k) * e, tx: from.tx + (target.tx - from.tx) * e, ty: from.ty + (target.ty - from.ty) * e };
      this.draw();
      if (t < 1) this.anim = requestAnimationFrame(step);
    };
    cancelAnimationFrame(this.anim);
    this.anim = requestAnimationFrame(step);
  }

  reset() {
    this.animateTo({ k: 1, tx: 0, ty: 0 });
  }

  focusCluster(cid) {
    if (!this.data) return;
    const pts = this.data.xy.filter((_, i) => this.data.cluster[i] === cid);
    const xs = pts.map((p) => p[0]).sort((a, b) => a - b);
    const ys = pts.map((p) => p[1]).sort((a, b) => a - b);
    const q = (arr, f) => arr[Math.floor(f * (arr.length - 1))];
    const x0 = q(xs, 0.05), x1 = q(xs, 0.95), y0 = q(ys, 0.05), y1 = q(ys, 0.95);
    const span = Math.max(x1 - x0, y1 - y0, 0.05);
    const k = Math.min(8, 0.8 / span);
    const b = this.base();
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    this.animateTo({ k, tx: this.w / 2 - b.ox - cx * b.size * k, ty: this.h / 2 - b.oy - cy * b.size * k });
  }

  focusResult() {
    if (!this.result) return;
    const pts = this.result.hits.map((i) => this.data.xy[i]).concat([this.result.q]);
    const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const span = Math.max(x1 - x0, y1 - y0, 0.08);
    const k = Math.max(1, Math.min(2.5, 0.55 / span)); // keep enough context to read the neighbourhood
    const b = this.base();
    const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
    this.animateTo({ k, tx: this.w / 2 - b.ox - cx * b.size * k, ty: this.h / 2 - b.oy - cy * b.size * k }, 700);
  }

  draw() {
    const { ctx } = this;
    if (!this.w) return;
    ctx.clearRect(0, 0, this.w, this.h);
    if (!this.data) return;
    const { xy, clusters } = this.data;
    const k = this.view.k;
    const r = Math.min(3, 1.35 + Math.sqrt(k) * 0.45);
    const hc = this.hoverCluster;
    // Points: topic colour at moderate alpha, quieter when a result is on the map so markers lead.
    const base = this.result ? 0.5 : 0.68;
    for (const group of this.byCluster) {
      const a = hc < 0 ? base : group.id === hc ? 0.9 : 0.1;
      ctx.fillStyle = `rgba(${group.rgb.join(",")},${a})`;
      ctx.beginPath();
      for (const i of group.idx) {
        const [sx, sy] = this.toScreen(xy[i][0], xy[i][1]);
        if (sx < -4 || sy < -4 || sx > this.w + 4 || sy > this.h + 4) continue;
        ctx.moveTo(sx + r, sy);
        ctx.arc(sx, sy, r, 0, Math.PI * 2);
      }
      ctx.fill();
    }

    // Retrieved markers are obstacles that topic labels must not cover.
    const markers = [];
    let q = null;
    if (this.result) {
      q = this.toScreen(this.result.q[0], this.result.q[1]);
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(232, 180, 76, 0.45)";
      for (const i of this.result.hits) {
        const [hx, hy] = this.toScreen(xy[i][0], xy[i][1]);
        markers.push([hx, hy]);
        ctx.beginPath();
        ctx.moveTo(q[0], q[1]);
        ctx.lineTo(hx, hy);
        ctx.stroke();
      }
    }
    this.drawLabels(clusters, markers, q);

    if (this.result) {
      const since = (performance.now() - (this.pulse || 0)) / 1000;
      markers.forEach(([hx, hy], n) => {
        if (since < 1.2) {
          ctx.beginPath();
          ctx.arc(hx, hy, 6 + since * 16, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(232, 180, 76, ${0.5 * (1 - since / 1.2)})`;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.arc(hx, hy, 6, 0, Math.PI * 2);
        ctx.fillStyle = "#e8b44c";
        ctx.fill();
        ctx.lineWidth = 2.5;
        ctx.strokeStyle = "#0b0d10";
        ctx.stroke();
        ctx.font = "600 11px 'IBM Plex Mono', monospace";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.lineWidth = 3;
        ctx.strokeText(String(n + 1), hx + 9, hy - 8);
        ctx.fillStyle = "#f3cd7d";
        ctx.fillText(String(n + 1), hx + 9, hy - 8);
      });
      if (since < 1.2) requestAnimationFrame(() => this.draw());
      ctx.save();
      ctx.translate(q[0], q[1]);
      ctx.rotate(Math.PI / 4);
      ctx.fillStyle = "#ecebe6";
      ctx.strokeStyle = "#0b0d10";
      ctx.lineWidth = 2.5;
      ctx.fillRect(-5.5, -5.5, 11, 11);
      ctx.strokeRect(-5.5, -5.5, 11, 11);
      ctx.restore();
    }

    if (this.hover >= 0) {
      const [sx, sy] = this.toScreen(xy[this.hover][0], xy[this.hover][1]);
      ctx.beginPath();
      ctx.arc(sx, sy, 5, 0, Math.PI * 2);
      ctx.strokeStyle = "#ecebe6";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  // Topic labels: kept inside the canvas, nudged off retrieved markers and each other,
  // faded where no free spot exists. Larger topics are placed first.
  drawLabels(clusters, markers, q) {
    const { ctx } = this;
    ctx.font = "500 12px 'IBM Plex Sans', system-ui, sans-serif";
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.lineJoin = "round";
    const pad = 10;
    const obstacles = markers.map(([x, y]) => [x - 10, y - 18, x + 24, y + 10]); // marker + its number
    if (q) obstacles.push([q[0] - 10, q[1] - 10, q[0] + 10, q[1] + 10]);
    const placed = [];
    const hit = (b) => obstacles.concat(placed).some((o) => b[0] < o[2] && b[2] > o[0] && b[1] < o[3] && b[3] > o[1]);
    const order = clusters.slice().sort((a, b) => b.size - a.size);
    for (const c of order) {
      const [cx, cy] = this.toScreen(c.center[0], c.center[1]);
      if (cx < -60 || cy < -30 || cx > this.w + 60 || cy > this.h + 30) continue; // topic is off-screen
      const tw = ctx.measureText(c.label).width + 14; // swatch dot + gap
      const clampX = (x) => Math.min(Math.max(x, pad + tw / 2), this.w - pad - tw / 2);
      const clampY = (y) => Math.min(Math.max(y, pad + 8), this.h - pad - 8);
      let spot = null;
      for (const [dx, dy] of [[0, 0], [0, -20], [0, 20], [0, -40], [0, 40], [-tw / 2, 0], [tw / 2, 0]]) {
        const x = clampX(cx + dx);
        const y = clampY(cy + dy);
        const box = [x - tw / 2 - 4, y - 10, x + tw / 2 + 4, y + 10];
        if (!hit(box)) {
          spot = { x, y, box, alpha: 1 };
          break;
        }
      }
      if (!spot) {
        const x = clampX(cx);
        const y = clampY(cy);
        spot = { x, y, box: [x - tw / 2, y - 10, x + tw / 2, y + 10], alpha: 0.28 };
      }
      placed.push(spot.box);
      const focus = this.hoverCluster < 0 || this.hoverCluster === c.id;
      ctx.globalAlpha = spot.alpha * (focus ? 1 : 0.35);
      const x0 = spot.x - tw / 2;
      ctx.beginPath();
      ctx.arc(x0 + 4, spot.y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = c.color || "#85857f";
      ctx.fill();
      ctx.lineWidth = 4;
      ctx.strokeStyle = "rgba(11, 13, 16, 0.92)";
      ctx.strokeText(c.label, x0 + 14, spot.y);
      ctx.fillStyle = c.id === this.hoverCluster ? "#ffffff" : "#d6d5cf";
      ctx.fillText(c.label, x0 + 14, spot.y);
      ctx.globalAlpha = 1;
    }
  }

  nearest(sx, sy, maxPx = 9) {
    if (!this.data) return -1;
    let best = -1, bd = maxPx * maxPx;
    const { xy } = this.data;
    for (let i = 0; i < xy.length; i++) {
      const [px, py] = this.toScreen(xy[i][0], xy[i][1]);
      const d = (px - sx) ** 2 + (py - sy) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  _bind() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener("mousedown", (e) => {
      drag = { x: e.clientX, y: e.clientY, tx: this.view.tx, ty: this.view.ty, moved: false };
      c.classList.add("dragging");
    });
    window.addEventListener("mouseup", () => {
      drag = null;
      c.classList.remove("dragging");
    });
    window.addEventListener("mousemove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
      this.view.tx = drag.tx + dx;
      this.view.ty = drag.ty + dy;
      this.draw();
    });
    c.addEventListener("mousemove", (e) => {
      if (drag) return;
      const r = c.getBoundingClientRect();
      const sx = e.clientX - r.left, sy = e.clientY - r.top;
      const i = this.nearest(sx, sy);
      if (i !== this.hover) {
        this.hover = i;
        this.draw();
      }
      const tt = this.tooltip;
      if (i >= 0) {
        const cl = this.data.clusters[this.data.cluster[i]];
        tt.innerHTML = `${escapeHtml(this.data.titles[i])}<span class="mono">${escapeHtml(this.data.ids[i])} · ${escapeHtml(cl ? cl.label : "")}</span>`;
        tt.hidden = false;
        const tw = tt.offsetWidth, th = tt.offsetHeight;
        tt.style.left = `${Math.min(sx + 14, this.w - tw - 8)}px`;
        tt.style.top = `${Math.min(sy + 14, this.h - th - 8)}px`;
      } else {
        tt.hidden = true;
      }
    });
    c.addEventListener("mouseleave", () => {
      this.hover = -1;
      this.tooltip.hidden = true;
      this.draw();
    });
    c.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const r = c.getBoundingClientRect();
        const sx = e.clientX - r.left, sy = e.clientY - r.top;
        const factor = Math.exp(-e.deltaY * 0.0015);
        const k = Math.min(40, Math.max(0.6, this.view.k * factor));
        const f = k / this.view.k;
        const b = this.base();
        // keep the world point under the cursor fixed: sx = ox + x*size*k + tx
        this.view.tx = sx - b.ox - (sx - b.ox - this.view.tx) * f;
        this.view.ty = sy - b.oy - (sy - b.oy - this.view.ty) * f;
        this.view.k = k;
        this.draw();
      },
      { passive: false },
    );
  }
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}
