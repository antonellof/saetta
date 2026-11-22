// saetta.dev/docs: theme toggle (as on the main page), copy buttons, the sidebar filter, the
// phone-width menu, and the "On this page" highlight.

const root = document.documentElement;
document.getElementById("theme")?.addEventListener("click", () => {
  const t = root.dataset.theme === "light" ? "dark" : "light";
  root.dataset.theme = t;
  try { localStorage.setItem("theme", t); } catch (e) {}
});

for (const b of document.querySelectorAll(".doc-code .copy")) {
  b.addEventListener("click", async () => {
    const text = b.closest(".doc-code").querySelector("pre").innerText.replace(/\n$/, "");
    try {
      await navigator.clipboard.writeText(text);
      b.textContent = "Copied";
    } catch (e) {
      b.textContent = "Select it";
    }
    setTimeout(() => (b.textContent = "Copy"), 1500);
  });
}

// sidebar filter: pages whose title or section matches
const filter = document.querySelector(".ds-filter");
const none = document.querySelector(".ds-none");
filter?.addEventListener("input", () => {
  const q = filter.value.trim().toLowerCase();
  let shown = 0;
  for (const g of document.querySelectorAll(".ds-group")) {
    const sec = g.querySelector(".ds-title").textContent.toLowerCase();
    let n = 0;
    for (const li of g.querySelectorAll("li")) {
      const on = !q || sec.includes(q) || li.textContent.toLowerCase().includes(q);
      li.hidden = !on;
      n += on;
    }
    g.hidden = n === 0;
    shown += n;
  }
  none.hidden = shown > 0;
});

// the drawer at phone width
const ds = document.getElementById("ds-nav");
const scrim = document.querySelector(".ds-scrim");
const opener = document.querySelector(".ds-open");
function drawer(open) {
  ds.classList.toggle("open", open);
  scrim.hidden = !open;
  opener.setAttribute("aria-expanded", open ? "true" : "false");
  if (open) ds.querySelector("[aria-current]")?.scrollIntoView({ block: "center" });
}
opener?.addEventListener("click", () => drawer(true));
document.querySelector(".ds-close")?.addEventListener("click", () => drawer(false));
scrim?.addEventListener("click", () => drawer(false));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && ds.classList.contains("open")) drawer(false); });

// keep the current page in view in the desktop sidebar
const cur = ds?.querySelector("[aria-current]");
if (cur && getComputedStyle(ds).position === "sticky") {
  const r = cur.getBoundingClientRect();
  if (r.bottom > ds.clientHeight) ds.scrollTop = cur.offsetTop - ds.clientHeight / 3;
}

// "On this page": the last heading above the top third of the window
const links = [...document.querySelectorAll(".doc-toc a")];
const heads = links.map((a) => document.getElementById(a.hash.slice(1))).filter(Boolean);
if (heads.length) {
  let ticking = false;
  const mark = () => {
    ticking = false;
    const y = window.innerHeight / 3;
    let at = 0;
    heads.forEach((h, i) => { if (h.getBoundingClientRect().top < y) at = i; });
    if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 2) at = heads.length - 1;
    links.forEach((a, i) => a.classList.toggle("on", i === at));
  };
  window.addEventListener("scroll", () => { if (!ticking) { ticking = true; requestAnimationFrame(mark); } }, { passive: true });
  mark();
}
