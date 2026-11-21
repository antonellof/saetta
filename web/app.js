// saetta.dev: theme toggle, copy buttons, and the terminal replays (what `saetta auto` and the
// local API print; simulated, labelled as such on the page).

const root = document.documentElement;
document.getElementById("theme").addEventListener("click", () => {
  const t = root.dataset.theme === "light" ? "dark" : "light";
  root.dataset.theme = t;
  try { localStorage.setItem("theme", t); } catch (e) {}
});

for (const b of document.querySelectorAll(".copy")) {
  b.addEventListener("click", async () => {
    const text = document.getElementById(b.dataset.copy).innerText.split("\n").filter((l) => !l.startsWith("#")).join("\n");
    try {
      await navigator.clipboard.writeText(text);
      b.textContent = "Copied";
      setTimeout(() => (b.textContent = "Copy"), 1500);
    } catch (e) { b.textContent = "Select it"; }
  });
}

const term = document.getElementById("term");
const metaL = document.getElementById("meta-l");
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
const pad = (s, n) => s + " ".repeat(Math.max(0, n - s.length));
let run = 0; // bumped on tab change: an older replay stops at its next step

const models = [["coder", "downloaded, recommended"], ["full", "105 GB"], ["27b-turbo", "20 GB"], ["27b-8bit", "29 GB"], ["27b-4bit", "16 GB"]];
const agents = [["Claude Code", "installed"], ["pi", "installed"], ["opencode", "not installed"]];
const menu = (title, rows, sel) =>
  `${title}\n` + rows.map(([a, b], i) => i === sel
    ? `  <span class="sel"><span class="a">&gt;</span> ${pad(a, 16)}${b} </span>`
    : `    ${pad(a, 16)}<span class="d">${b}</span>`).join("\n");

async function type(id, prefix, text) {
  for (let i = 0; i <= text.length; i++) {
    if (id !== run) return false;
    term.innerHTML = prefix + esc(text.slice(0, i)) + '<span class="cursor"></span>';
    await sleep(45 + Math.random() * 40);
  }
  return true;
}
async function show(id, html, ms) {
  if (id !== run) return false;
  term.innerHTML = html;
  await sleep(ms);
  return id === run;
}

async function auto(id) {
  const p = '<span class="d">~/project</span> <span class="a">$</span> ';
  if (!(await type(id, p, "saetta auto"))) return;
  const intro = p + "saetta auto\nsaetta: local coding models for this Mac\n\n" +
    '<span class="d">  Apple M5 Max, 128 GB memory, 297 GB free disk</span>\n\n';
  for (const s of [0, 1, 2, 1, 0]) if (!(await show(id, intro + menu("Choose a model", models, s), s ? 380 : 1100))) return;
  const m = intro + 'Choose a model: <span class="g">coder</span> (downloaded)\n\n';
  if (!(await show(id, m + menu("Choose a coding agent", agents, 0), 1300))) return;
  const s = m + "Starting Claude Code on coder (next time: saetta start claude --pack coder)\n\n" +
    "saetta: loading Qwen3.8-Flash-Next-GSQ-RCO-Coder-MLX (context 204800 tokens)...\n";
  if (!(await show(id, s + '<span class="cursor"></span>', 1400))) return;
  await show(id, s + '<span class="g">saetta: ready at http://127.0.0.1:8080 (6.8 s)</span>\n\n' +
    '<span class="a">&gt;</span> fix the failing test<span class="cursor"></span>', 4000);
}

async function api(id) {
  const p = '<span class="d">~/project</span> <span class="a">$</span> ';
  const cmd = 'curl -sN localhost:8080/v1/chat/completions -d \'{"messages":[{"role":"user","content":"Write is_prime in Python"}],"stream":true}\'';
  if (!(await type(id, p, cmd))) return;
  const head = p + esc(cmd) + "\n";
  const words = ["def", " is_prime", "(n", "):", "\\n", "    if", " n", " <", " 2", ":", " return", " False", "\\n", "    i", " =", " 2", "\\n", "    while", " i", " *", " i", " <=", " n", ":"];
  let out = "";
  for (const w of words) {
    out += `<span class="d">data:</span> {"choices":[{"delta":{"content":"<span class="a">${esc(w)}</span>"}}]}\n`;
    const lines = out.split("\n");
    if (!(await show(id, head + lines.slice(-14).join("\n") + '<span class="cursor"></span>', 70))) return;
  }
  await show(id, head + out.split("\n").slice(-13).join("\n") + '<span class="d">data:</span> [DONE]\n\n' +
    '<span class="d"># the same server answers Anthropic\'s /v1/messages: Claude Code uses that one</span>', 4200);
}

const tabs = { auto: [auto, "Qwen3.8-Flash-Next Coder on an M5 Max, 128 GB"], api: [api, "the local server, streaming, on 127.0.0.1:8080"] };
let current = "auto";

async function loop() {
  const id = ++run;
  if (reduce) {
    term.innerHTML = '<span class="a">$</span> saetta auto\n\n' + menu("Choose a model", models, 0);
    return;
  }
  while (id === run) await tabs[current][0](id);
}

for (const t of document.querySelectorAll(".tab")) {
  t.addEventListener("click", () => {
    for (const o of document.querySelectorAll(".tab")) o.setAttribute("aria-selected", o === t ? "true" : "false");
    current = t.dataset.tab;
    metaL.textContent = tabs[current][1];
    loop();
  });
}
loop();
