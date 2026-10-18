/* Base vs fine-tuned voice comparison. Labels are visible on purpose: the
   question is which model keeps the reference voice, not a blind preference. */

const $ = (id) => document.getElementById(id);
const state = { secret: localStorage.getItem("qc_secret") || "", data: null };

function headers() { return { "Authorization": "Bearer " + state.secret, "Content-Type": "application/json" }; }

async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: { ...headers(), ...(opts.headers || {}) } });
  if (res.status === 401) { localStorage.removeItem("qc_secret"); state.secret = ""; $("gate").classList.remove("hidden"); throw new Error("unauthorized"); }
  if (!res.ok) throw new Error(path + " -> " + res.status);
  return res.json();
}

function toast(m) {
  const t = $("toast"); t.textContent = m; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 1400);
}

function url(itemId, variant) {
  return `api/ft/audio/${encodeURIComponent(itemId)}/${variant}?key=${encodeURIComponent(state.secret)}`;
}

function render() {
  const card = $("pairs");
  card.innerHTML = "";
  for (const it of state.data.items) {
    const label = state.data.labels[it.id] || {};
    const wrap = document.createElement("div");
    wrap.className = "pair";
    wrap.innerHTML = `
      <div class="labelrow"><span class="tag">${it.id}</span>
        <span class="mono dim">${it.seconds.base}s · ${it.steps} steps</span></div>
      <div class="text">${escapeHtml(it.text)}</div>
      <div class="players">
        <div class="player base">
          <div class="labelrow"><span class="tag">base (pretrained)</span></div>
          <audio controls preload="metadata" src="${url(it.id, "base")}"></audio>
        </div>
        <div class="player ft">
          <div class="labelrow"><span class="tag">fine-tuned (ours)</span></div>
          <audio controls preload="metadata" src="${url(it.id, "finetuned")}"></audio>
        </div>
      </div>
      <div class="choice">
        <button data-c="base" data-id="${it.id}">base sounds more like 莼儿</button>
        <button data-c="finetuned" data-id="${it.id}">fine-tuned sounds more like 莼儿</button>
        <button data-c="same" data-id="${it.id}">same</button>
        <input class="note" data-id="${it.id}" placeholder="note" value="${escapeAttr(label.notes || "")}"/>
      </div>`;
    card.appendChild(wrap);
  }
  card.querySelectorAll(".choice button").forEach((b) => {
    const id = b.dataset.id;
    const label = state.data.labels[id];
    b.classList.toggle("on", !!label && label.choice === b.dataset.c);
    b.onclick = async () => {
      const note = card.querySelector(`.note[data-id="${CSS.escape(id)}"]`).value;
      try {
        const res = await api("api/ft/label", { method: "POST",
          body: JSON.stringify({ id, choice: b.dataset.c, notes: note }) });
        state.data.labels[id] = res.saved;
        $("savestate").textContent = "saved";
        render();
      } catch (e) { toast("save failed: " + e.message); }
    };
  });
  const rows = Object.values(state.data.labels);
  const c = { base: 0, finetuned: 0, same: 0 };
  rows.forEach((r) => { c[r.choice] = (c[r.choice] || 0) + 1; });
  $("sum").textContent = rows.length
    ? `base ${c.base} · fine-tuned ${c.finetuned} · same ${c.same}`
    : "no choices yet";
}

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }
function escapeAttr(s) { return escapeHtml(s).replace(/"/g, "&quot;"); }

$("unlock").onclick = async () => {
  const secret = $("secret").value.trim();
  if (!secret) return;
  state.secret = secret; localStorage.setItem("qc_secret", secret);
  try {
    state.data = await api("api/ft/items");
    $("gate").classList.add("hidden"); $("main").hidden = false;
    render();
  } catch (e) { $("gaterr").textContent = String(e.message).includes("unauthorized") ? "wrong secret" : e.message; }
};
$("secret").addEventListener("keydown", (e) => { if (e.key === "Enter") $("unlock").click(); });

$("export").onclick = async () => {
  try {
    const res = await fetch("api/ft/export.csv?key=" + encodeURIComponent(state.secret), { headers: headers() });
    if (!res.ok) throw new Error("export " + res.status);
    const url = URL.createObjectURL(new Blob([await res.text()], { type: "text/csv" }));
    const a = document.createElement("a"); a.href = url; a.download = "finetune_labels.csv"; a.click();
    URL.revokeObjectURL(url);
  } catch (e) { toast("export failed: " + e.message); }
};

if (state.secret) $("unlock").click();
