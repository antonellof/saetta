// Hero background: ASCII 3D. Saetta's bolt (the logo's outline, extruded) turns slowly, shaded
// with a character ramp like donut.c: sampled surface points with normals, rotated, projected
// into a grid of character cells with a depth buffer, each cell's character picked by the light
// it catches. A faint field of drifting ASCII noise fills the whole header. Every few seconds
// the bolt flashes. The pointer tilts it. Plain 2D canvas text, no library. Pauses offscreen and
// in hidden tabs, one still frame with reduced motion, lighter on phones, follows data-theme.
(() => {
  const canvas = document.getElementById("hero-bg");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- the bolt: the favicon path (32-unit box), centred, y up, extruded to +-D
  const P = [[18.5, 3], [7, 18], [14.5, 18], [12, 29], [25, 13], [17.2, 13]].map(([x, y]) => [(x - 16) / 13, (16 - y) / 13]);
  const D = 0.13;
  const pts = []; // x, y, z, nx, ny, nz
  const inside = (x, y) => {
    let c = false;
    for (let i = 0, j = P.length - 1; i < P.length; j = i++) {
      const [xi, yi] = P[i], [xj, yj] = P[j];
      if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
    }
    return c;
  };
  for (let x = -1.3; x <= 1.3; x += 0.018)
    for (let y = -1.3; y <= 1.3; y += 0.018)
      if (inside(x, y)) { pts.push(x, y, -D, 0, 0, -1); pts.push(x, y, D, 0, 0, 1); }
  let area = 0;
  for (let i = 0, j = P.length - 1; i < P.length; j = i++) area += P[j][0] * P[i][1] - P[i][0] * P[j][1];
  const orient = area > 0 ? 1 : -1;
  for (let i = 0; i < P.length; i++) {
    const [ax, ay] = P[i], [bx, by] = P[(i + 1) % P.length];
    const len = Math.hypot(bx - ax, by - ay);
    const nx = (orient * (by - ay)) / len, ny = (orient * -(bx - ax)) / len;
    for (let s = 0; s <= len; s += 0.012)
      for (let z = -D; z <= D; z += 0.02) pts.push(ax + ((bx - ax) * s) / len, ay + ((by - ay) * s) / len, z, nx, ny, 0);
  }
  const NPTS = pts.length / 6;

  // ---- value noise for the field
  const perm = new Uint8Array(512);
  {
    let seed = 9;
    const p = Array.from({ length: 256 }, (_, i) => i);
    for (let i = 255; i > 0; i--) { seed = (seed * 16807) % 2147483647; const j = seed % (i + 1); [p[i], p[j]] = [p[j], p[i]]; }
    for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
  }
  const hash = (x, y) => perm[(perm[x & 255] + y) & 511] / 255;
  const noise = (x, y) => {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
    const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
    const a = hash(xi, yi), b = hash(xi + 1, yi), c = hash(xi, yi + 1), d = hash(xi + 1, yi + 1);
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
  };

  const RAMP = ".,-~:;=!*#$@";
  const FIELD = " .:-=+";
  const NB = 5; // colour bands for the bolt, dark to bright

  // ---- theme
  let C;
  const theme = () => {
    const light = document.documentElement.dataset.theme === "light";
    C = light
      ? { field: "rgba(23,23,23,", fieldA: [0.07, 0.12, 0.2], fieldHi: "rgba(42,120,214,", bands: ["#8250df", "#6a5fdc", "#4f6fd9", "#2a78d6", "#123a75"], flash: "#2a78d6" }
      : { field: "rgba(190,200,220,", fieldA: [0.08, 0.14, 0.24], fieldHi: "rgba(88,166,255,", bands: ["#7b55c7", "#9b74f0", "#6f8ff5", "#58a6ff", "#e6f0ff"], flash: "#ffffff" };
  };
  theme();
  new MutationObserver(() => { theme(); if (!raf) draw(lastT); }).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

  // ---- layout
  let W = 0, H = 0, dpr = 1, cw = 8.4, ch = 16, cols = 0, rows = 0, small = false, fontPx = 14;
  let zbuf = new Float32Array(0), lum = new Float32Array(0);
  function resize() {
    const r = canvas.getBoundingClientRect();
    small = r.width < 700;
    dpr = Math.min(devicePixelRatio || 1, 2);
    W = r.width; H = r.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    fontPx = small ? 11 : 13;
    ctx.font = `${fontPx}px "JetBrains Mono", ui-monospace, Menlo, monospace`;
    ctx.textBaseline = "top";
    cw = ctx.measureText("M").width; ch = Math.round(fontPx * 1.25);
    cols = Math.ceil(W / cw); rows = Math.ceil(H / ch);
    zbuf = new Float32Array(cols * rows); lum = new Float32Array(cols * rows);
    if (!raf) draw(lastT);
  }

  const mouse = { x: 0, y: 0, sx: 0, sy: 0 };
  addEventListener("pointermove", (e) => { mouse.x = e.clientX / innerWidth - 0.5; mouse.y = e.clientY / innerHeight - 0.5; }, { passive: true });

  // a point light at the upper left, on the viewer's side (the viewer looks along +z): flat faces
  // get a gradient across them, so the whole ramp shows
  const LP = [-1.9, 1.7, -2.4];

  let lastT = 0, flash = 0, nextFlash = 2.5;
  const rowBuf = Array.from({ length: NB + 1 }, () => []);
  const fieldBuf = [[], [], [], []]; // the field in three brightness levels, then strikes

  // ---- field patterns: each gives a cell a value (0..1) and its characters; they change at
  // random and cross-fade
  const BIN = " .01";
  const hash2 = (a, b) => hash(a & 255, b & 255);
  const MODES = [
    // ripples spreading from the bolt
    (c, r, t, n, dist) => [n * 0.72 + ((0.5 + 0.5 * Math.sin(dist * 0.042 - t * 2.1)) / (1 + dist / 700)) * 0.42 + 0.12 / (1 + dist / 260), FIELD],
    // binary rain: each column falls at its own speed
    (c, r, t, n) => {
      const sp = 3 + hash2(c, 7) * 9, len = 6 + hash2(c, 13) * 14, span = rows + len + 8;
      const head = (t * sp + hash2(c, 29) * span) % span;
      const d = head - r;
      const v = d >= 0 && d < len ? 0.46 + 0.36 * (1 - d / len) : n * 0.55;
      return [v, BIN];
    },
    // a vortex of spiral arms around the bolt
    (c, r, t, n, dist, ang) => [n * 0.5 + (0.5 + 0.5 * Math.sin(ang * 3 + dist * 0.028 - t * 1.4)) * 0.5 / (1 + dist / 1200) + 0.1, FIELD],
    // compute sweep: diagonal bands, tiles flickering on as they are computed
    (c, r, t, n) => {
      const band = 0.5 + 0.5 * Math.sin((c * 0.5 + r * 1.1) * 0.22 - t * 2.6);
      const tc = Math.floor(c / 6), tr = Math.floor(r / 3);
      const on = hash2(tc * 7 + Math.floor(t * 1.5), tr) > 0.86 ? 0.35 : 0;
      return [n * 0.45 + band * 0.32 + on, FIELD];
    },
    // deep drifting clouds
    (c, r, t, n) => [0.12 + n * 0.95, FIELD],
  ];
  // a random pattern to start; ?mode=N pins one (0 ripples, 1 rain, 2 vortex, 3 sweep, 4 drift)
  const pinned = new URLSearchParams(location.search).get("mode");
  let mode = pinned !== null ? Math.max(0, Math.min(MODES.length - 1, +pinned | 0)) : Math.floor(Math.random() * MODES.length);
  let prevMode = mode, mixK = 1, nextSwitch = pinned !== null ? Infinity : 11;
  function switchMode(t) {
    prevMode = mode;
    do mode = Math.floor(Math.random() * MODES.length); while (mode === prevMode);
    mixK = 0;
    nextSwitch = t + 9 + Math.random() * 5;
  }

  // ---- ASCII lightning across the field, on each flash
  let strike = new Map(), strikeT = 9;
  function newStrike() {
    strike = new Map();
    const walk = (c, r, depth, branchy) => {
      for (let i = 0; i < depth && r < rows; i++) {
        const dx = Math.random() < 0.34 ? -1 : Math.random() < 0.5 ? 1 : 0;
        c += dx;
        if (c < 0 || c >= cols) return;
        strike.set(r * cols + c, dx < 0 ? "/" : dx > 0 ? "\\" : "|");
        if (branchy && Math.random() < 0.07) walk(c, r + 1, Math.floor(depth * 0.4), false);
        r++;
      }
    };
    walk(Math.floor(cols * (0.08 + Math.random() * 0.84)), 0, Math.floor(rows * (0.45 + Math.random() * 0.35)), true);
    strikeT = 0;
  }

  function draw(t) {
    lastT = t;
    mouse.sx += (mouse.x - mouse.sx) * 0.05;
    mouse.sy += (mouse.y - mouse.sy) * 0.05;
    ctx.clearRect(0, 0, W, H);
    ctx.font = `${fontPx}px "JetBrains Mono", ui-monospace, Menlo, monospace`;
    ctx.textBaseline = "top";

    // project the bolt into the cell grid
    zbuf.fill(0); lum.fill(-1);
    // spin and tilt wander, so the motion never repeats exactly
    const A = t * 0.5 + Math.sin(t * 0.17) * 0.9 + mouse.sx * 1.2;
    const B = 0.18 + Math.sin(t * 0.29) * 0.12 + Math.sin(t * 0.71) * 0.05 + mouse.sy * 0.5;
    const cA = Math.cos(A), sA = Math.sin(A), cB = Math.cos(B), sB = Math.sin(B);
    const K2 = 4.2;
    const boltH = small ? Math.min(H * 0.32, 260) : Math.min(H * 0.62, 560);
    const K1 = (boltH * K2) / 2.3;
    const cx = W * (small ? 0.76 : 0.74), cy = small ? H * 0.26 : Math.min(H * 0.33, 360);
    for (let i = 0; i < NPTS; i++) {
      const o = i * 6;
      const x = pts[o], y = pts[o + 1], z = pts[o + 2];
      // rotate around y, then x
      const x1 = x * cA + z * sA, z1 = -x * sA + z * cA;
      const y2 = y * cB - z1 * sB, z2 = y * sB + z1 * cB;
      const nx = pts[o + 3], ny = pts[o + 4], nz = pts[o + 5];
      const nx1 = nx * cA + nz * sA, nz1 = -nx * sA + nz * cA;
      const ny2 = ny * cB - nz1 * sB, nz2 = ny * sB + nz1 * cB;
      const ooz = 1 / (K2 + z2);
      const col = Math.floor((cx + K1 * x1 * ooz) / cw), row = Math.floor((cy - K1 * y2 * ooz) / ch);
      if (col < 0 || row < 0 || col >= cols || row >= rows) continue;
      const k = row * cols + col;
      if (ooz > zbuf[k]) {
        zbuf[k] = ooz;
        const lx = LP[0] - x1, ly = LP[1] - y2, lz = LP[2] - z2, ln = Math.hypot(lx, ly, lz);
        const d = (nx1 * lx + ny2 * ly + nz2 * lz) / ln;
        lum[k] = Math.max(0, d) * Math.min(1.7, 11 / (ln * ln)) - 0.04;
      }
    }

    // field, then bolt bands; one string per row and band
    const fk = (small ? 0.75 : 1) * (1 + flash * 1.2);
    // a strike flickers: on, off, on, then fades
    const strikeVis = strikeT < 0.07 ? 1 : strikeT < 0.13 ? 0 : strikeT < 0.2 ? 0.9 : Math.max(0, 1 - (strikeT - 0.2) / 0.45);
    for (let r = 0; r < rows; r++) {
      for (const b of rowBuf) b.length = 0;
      for (const b of fieldBuf) b.length = 0;
      for (let c = 0; c < cols; c++) {
        const k = r * cols + c;
        const l = lum[k];
        if (l > -1) {
          const v = Math.max(0, Math.min(0.999, l));
          const ch2 = RAMP[Math.floor(v * RAMP.length)];
          const band = Math.min(NB - 1, Math.floor(v * NB));
          for (let q = 0; q <= NB; q++) rowBuf[q].push(q === band + 1 ? ch2 : " ");
          for (let q = 0; q < 4; q++) fieldBuf[q].push(" ");
        } else {
          const sc = strike.size && strikeVis > 0 ? strike.get(k) : undefined;
          if (sc) {
            for (let q = 0; q < 4; q++) fieldBuf[q].push(q === 3 ? sc : " ");
            for (let q = 0; q <= NB; q++) rowBuf[q].push(" ");
            continue;
          }
          // the current pattern, cross-faded from the previous one
          const n = noise(c * 0.11 + t * 0.12, r * 0.22 - t * 0.05) * 0.7 + noise(c * 0.31 - t * 0.08, r * 0.6) * 0.3;
          const dx = c * cw - cx, dy = (r * ch - cy) * 1.15, dist = Math.hypot(dx, dy), ang = Math.atan2(dy, dx);
          const [va, seta] = MODES[mode](c, r, t, n, dist, ang);
          let v = va, set = seta;
          if (mixK < 1) {
            const [vb, setb] = MODES[prevMode](c, r, t, n, dist, ang);
            v = vb + (va - vb) * mixK;
            if (mixK < 0.5) set = setb;
          }
          let lvl = -1, fch = " ";
          if (v > 0.42) {
            lvl = v < 0.56 ? 0 : v < 0.7 ? 1 : 2;
            fch = set[Math.min(set.length - 1, 1 + Math.floor((v - 0.42) * 2.2 * (set.length - 1)))];
          }
          for (let q = 0; q < 4; q++) fieldBuf[q].push(q === lvl ? fch : " ");
          for (let q = 0; q <= NB; q++) rowBuf[q].push(" ");
        }
      }
      const y = r * ch;
      for (let q = 0; q < 3; q++) {
        ctx.fillStyle = `${q === 2 ? C.fieldHi : C.field}${C.fieldA[q] * fk})`;
        ctx.fillText(fieldBuf[q].join(""), 0, y);
      }
      if (strikeVis > 0) {
        const s3 = fieldBuf[3].join("");
        if (s3.trim()) { ctx.fillStyle = `${C.fieldHi}${0.95 * strikeVis * (small ? 0.6 : 1)})`; ctx.fillText(s3, 0, y); }
      }
      for (let q = 1; q <= NB; q++) {
        const s = rowBuf[q].join("");
        if (!s.trim()) continue;
        ctx.fillStyle = flash > 0.05 && q >= NB - 1 ? C.flash : C.bands[q - 1];
        // on phones the text spans the width: the bolt is a ghost behind it
        ctx.globalAlpha = (0.55 + 0.45 * ((q - 1) / (NB - 1)) + flash * 0.3) * (small ? 0.32 : 1);
        ctx.fillText(s, 0, y);
        ctx.globalAlpha = 1;
      }
    }
  }

  function step(dt, t) {
    if (t > nextFlash) { flash = 1; nextFlash = t + 4 + Math.random() * 4; if (cols) newStrike(); }
    flash = Math.max(0, flash - dt * 2.2);
    strikeT += dt;
    if (t > nextSwitch) switchMode(t);
    mixK = Math.min(1, mixK + dt / 1.6);
  }

  // run only while visible
  let raf = 0, visible = true, prev = 0;
  const loop = (now) => {
    const dt = Math.min(0.05, (now - prev) / 1000);
    prev = now;
    step(dt, lastT + dt);
    draw(lastT + dt);
    raf = requestAnimationFrame(loop);
  };
  const play = () => { if (reduced || raf || !visible || document.hidden) return; prev = performance.now(); raf = requestAnimationFrame(loop); };
  const stop = () => { cancelAnimationFrame(raf); raf = 0; };
  new IntersectionObserver(([e]) => { visible = e.isIntersecting; visible ? play() : stop(); }).observe(canvas);
  document.addEventListener("visibilitychange", () => (document.hidden ? stop() : play()));
  new ResizeObserver(resize).observe(canvas);

  lastT = 0.9; // a three-quarter view for the first (and reduced-motion) frame
  resize();
  // the cell size depends on the web font: measure again once it has loaded
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(resize);
  play();
})();
