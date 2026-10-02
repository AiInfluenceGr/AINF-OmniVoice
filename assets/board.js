/* Live board background for the OmniVoice panel.
 * Port of the AIINFLUENCE website's electric.js: current pulses running along
 * the real copper traces of the board photo toward the chip.
 * Light by design: one 2D canvas, ~12 short strokes + sprites per frame, no
 * filters/shadowBlur, DPR capped at 1.5, a slow-frame watchdog that halves the
 * load, rAF pauses when the tab is hidden, prefers-reduced-motion = static.
 * __BOARD_SRC__ and __TRACES__ are filled in by app.py. */
(() => {
  const BOARD_SRC = "__BOARD_SRC__";
  const data = __TRACES__;
  // Pulse colour: soft green current, kept subtle on the dark board.
  const PULSES = 12, SPEED = [0.22, 0.42], TAIL = [80, 170], BURST = 0.2, POS_Y = 0.5;
  const HALO = "122,205,146", CORE = "219,246,224";
  const KEY = "ov-fx";

  const mount = () => {
    if (document.querySelector(".board-bg")) return;
    const bg = document.createElement("div");
    bg.className = "board-bg";
    bg.setAttribute("aria-hidden", "true");
    bg.innerHTML = '<img class="board-img" alt="" src="' + BOARD_SRC + '"><div class="board-veil"></div><canvas class="board-electric"></canvas>';
    document.body.prepend(bg);
    const canvas = bg.querySelector("canvas");
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let enabled = localStorage.getItem(KEY) !== "off";
    document.documentElement.classList.toggle("ov-fx-off", !enabled);
    const syncBtn = () => document.querySelectorAll(".ov-fx").forEach((b) => {
      const label = enabled ? "FX ON" : "FX OFF";
      if (b.textContent !== label) b.textContent = label;
      if (b.classList.contains("is-on") !== enabled) b.classList.toggle("is-on", enabled);
    });
    document.addEventListener("click", (e) => {
      if (!e.target.closest(".ov-fx")) return;
      enabled = !enabled;
      localStorage.setItem(KEY, enabled ? "on" : "off");
      document.documentElement.classList.toggle("ov-fx-off", !enabled);
      syncBtn();
      if (enabled && window.__ovFxKick) window.__ovFxKick();
    });
    // Gradio re-renders HTML blocks; re-sync the footer button only when one is (re)added.
    let pending = false;
    new MutationObserver(() => {
      if (pending) return;
      pending = true;
      requestAnimationFrame(() => { pending = false; syncBtn(); });
    }).observe(document.body, { childList: true, subtree: true });
    syncBtn();
    if (reduced || !data || !data.paths || !data.paths.length) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const [IW, IH] = data.size;
    const paths = data.paths.map((pts) => {
      const cum = [0];
      for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
      return { pts, cum, len: cum[cum.length - 1], end: pts[pts.length - 1] };
    });
    const [CX, CY] = data.chip || [IW / 2, IH / 2];
    const order = paths.map((p, i) => ({ i, key: Math.atan2(p.end[1] - CY, p.end[0] - CX) }))
      .sort((a, b) => a.key - b.key).map((o) => o.i);
    // longer traces get picked more often (they carry the motion), short ones still fire
    const weights = order.map((pi) => Math.sqrt(paths[pi].len));
    const wsum = weights.reduce((x, y) => x + y, 0);
    const pickK = () => { let r = Math.random() * wsum; for (let k = 0; k < weights.length; k++) { r -= weights[k]; if (r <= 0) return k; } return weights.length - 1; };

    const sprite = document.createElement("canvas");
    sprite.width = sprite.height = 64;
    const sctx = sprite.getContext("2d");
    const g = sctx.createRadialGradient(32, 32, 0, 32, 32, 32);
    g.addColorStop(0, "rgba(" + CORE + ",0.95)");
    g.addColorStop(0.16, "rgba(178,230,188,0.72)");
    g.addColorStop(0.42, "rgba(" + HALO + ",0.22)");
    g.addColorStop(1, "rgba(" + HALO + ",0)");
    sctx.fillStyle = g;
    sctx.fillRect(0, 0, 64, 64);

    const lowEnd = (navigator.hardwareConcurrency || 8) <= 4;
    let target = lowEnd ? 5 : PULSES;
    let dprCap = 1.5;
    let scale = 1;
    const resize = () => {
      const w = canvas.clientWidth, h = canvas.clientHeight;
      const dpr = Math.min(window.devicePixelRatio || 1, dprCap);
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      scale = Math.max(w / IW, h / IH);
      const ox = (w - IW * scale) / 2, oy = (h - IH * scale) * POS_Y;
      ctx.setTransform(dpr * scale, 0, 0, dpr * scale, dpr * ox, dpr * oy);
    };
    resize();
    new ResizeObserver(resize).observe(canvas);

    const rand = (a, b) => a + Math.random() * (b - a);
    const pointAt = (p, s) => {
      if (s <= 0) return p.pts[0];
      if (s >= p.len) return p.end;
      let i = 1;
      while (p.cum[i] < s) i++;
      const t = (s - p.cum[i - 1]) / (p.cum[i] - p.cum[i - 1]);
      const a = p.pts[i - 1], b = p.pts[i];
      return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
    };
    const pulses = [], flashes = [];
    const spawn = (pi, delay = 0) => {
      const speed = rand(SPEED[0], SPEED[1]);
      pulses.push({ p: paths[pi], s: -delay * speed, speed, tail: rand(TAIL[0], TAIL[1]) });
    };
    const spawnSome = () => {
      const k = pickK();
      if (Math.random() < BURST && target - pulses.length >= 3) for (let j = 0; j < 3; j++) spawn(order[(k + j) % order.length], j * 90);
      else spawn(order[k]);
    };

    let last = performance.now(), slowTime = 0, checkedFor = 0, running = false;
    const clear = () => { ctx.save(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.restore(); };
    const frame = (now) => {
      if (!enabled) { running = false; clear(); pulses.length = 0; flashes.length = 0; return; }
      const dt = Math.min(now - last, 50);
      last = now;
      if (dt > 26) slowTime += dt;
      checkedFor += dt;
      if (checkedFor > 2000) {
        if (slowTime > 900 && (target > 4 || dprCap > 1)) { target = Math.max(4, Math.floor(target / 2)); dprCap = 1; resize(); }
        slowTime = 0; checkedFor = 0;
      }
      while (pulses.length < target && Math.random() < 0.08) spawnSome();
      clear();
      ctx.globalCompositeOperation = "lighter";
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      const core = 1.6 / scale, halo = 7 / scale, headSize = 40 / scale;
      for (let n = pulses.length - 1; n >= 0; n--) {
        const pl = pulses[n];
        pl.s += pl.speed * dt;
        const p = pl.p;
        if (pl.s - pl.tail > p.len) { pulses.splice(n, 1); continue; }
        if (pl.s <= 0) continue;
        const head = Math.min(pl.s, p.len), s0 = Math.max(0, pl.s - pl.tail);
        const a = pointAt(p, s0), b = pointAt(p, head);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        for (let i = 1; i < p.pts.length; i++) if (p.cum[i] > s0 && p.cum[i] < head) ctx.lineTo(p.pts[i][0], p.pts[i][1]);
        ctx.lineTo(b[0], b[1]);
        const g1 = ctx.createLinearGradient(a[0], a[1], b[0], b[1]);
        g1.addColorStop(0, "rgba(" + HALO + ",0)"); g1.addColorStop(1, "rgba(" + HALO + ",0.3)");
        ctx.strokeStyle = g1; ctx.lineWidth = halo; ctx.stroke();
        const g2 = ctx.createLinearGradient(a[0], a[1], b[0], b[1]);
        g2.addColorStop(0, "rgba(" + HALO + ",0)"); g2.addColorStop(1, "rgba(" + CORE + ",0.85)");
        ctx.strokeStyle = g2; ctx.lineWidth = core; ctx.stroke();
        if (pl.s < p.len) {
          ctx.globalAlpha = 0.7;
          ctx.drawImage(sprite, b[0] - headSize / 2, b[1] - headSize / 2, headSize, headSize);
          ctx.globalAlpha = 1;
        } else if (!pl.flashed) { pl.flashed = true; flashes.push({ x: p.end[0], y: p.end[1], t: 0 }); }
      }
      for (let n = flashes.length - 1; n >= 0; n--) {
        const f = flashes[n];
        f.t += dt;
        const k = 1 - f.t / 420;
        if (k <= 0) { flashes.splice(n, 1); continue; }
        const size = (18 + 22 * (1 - k)) / scale;
        ctx.globalAlpha = 0.5 * k;
        ctx.drawImage(sprite, f.x - size / 2, f.y - size / 2, size, size);
      }
      ctx.globalAlpha = 1;
      requestAnimationFrame(frame);
    };
    const kick = () => { if (running) return; running = true; last = performance.now(); requestAnimationFrame(frame); };
    window.__ovFxKick = kick;
    if (enabled) kick();
  };
  if (document.body) mount(); else document.addEventListener("DOMContentLoaded", mount);
})();
