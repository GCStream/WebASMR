/* Quantization A/B page. Relative URLs keep it working under Caddy's /review/
   prefix; audio elements authenticate with ?key= because they cannot send an
   Authorization header. */

const $ = (id) => document.getElementById(id);
const state = {
  secret: localStorage.getItem("qc_secret") || "",
  items: [],
  labels: {},
  queue: [],
  pos: 0,
  artifacts: new Set(),
  playing: null,
};

const ARTIFACTS = ["metallic", "muffled", "clicks", "breath_loss", "unstable", "other"];
const ART_LABEL = {
  metallic: "metallic / buzz", muffled: "muffled", clicks: "clicks / dropouts",
  breath_loss: "breath/whisper lost", unstable: "unstable / drifting",
  other: "other",
};

function headers() { return { "Authorization": "Bearer " + state.secret, "Content-Type": "application/json" }; }

async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: { ...headers(), ...(opts.headers || {}) } });
  if (res.status === 401) { logout(); throw new Error("unauthorized"); }
  if (!res.ok) throw new Error(path + " -> " + res.status);
  return res.json();
}

function toast(msg) {
  const t = $("qtoast");
  t.textContent = msg; t.classList.add("show");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove("show"), 1400);
}

function logout() {
  localStorage.removeItem("qc_secret");
  state.secret = "";
  $("gate").classList.remove("hidden");
  $("qmain").hidden = true;
}

async function boot() {
  if (!state.secret) { $("gate").classList.remove("hidden"); return; }
  try {
    const data = await api("api/quant/items");
    state.items = data.items || [];
    state.labels = data.labels || {};
    state.voice = data.voice || "";
    $("gate").classList.add("hidden");
    $("qmain").hidden = false;
    buildArtifacts();
    applyQueue(true);
    render();
  } catch (e) {
    if (String(e.message).includes("unauthorized")) return;
    toast("load failed: " + e.message);
  }
}

function buildArtifacts() {
  const box = $("qartifacts");
  box.innerHTML = "";
  for (const key of ARTIFACTS) {
    const b = document.createElement("button");
    b.dataset.a = key; b.textContent = ART_LABEL[key];
    b.onclick = () => {
      if (state.artifacts.has(key)) state.artifacts.delete(key); else state.artifacts.add(key);
      b.classList.toggle("on", state.artifacts.has(key));
    };
    box.appendChild(b);
  }
}

function applyQueue(reset) {
  const all = $("qall").checked;
  state.queue = state.items.filter((it) => all || !state.labels[it.id]);
  if (reset) state.pos = 0;
  if (state.pos >= state.queue.length) state.pos = Math.max(0, state.queue.length - 1);
}

function item() { return state.queue[state.pos]; }

function audioUrl(it, variant) {
  return `api/quant/audio/${encodeURIComponent(it.id)}/${variant}?key=${encodeURIComponent(state.secret)}`;
}

function render() {
  const it = item();
  const total = state.items.length;
  const judged = Object.keys(state.labels).length;
  $("qprog").textContent = `${judged} / ${total}`;
  $("qbar").style.width = (100 * judged / Math.max(1, total)) + "%";
  if (!it) {
    $("qtext").textContent = "nothing left to judge";
    $("qid").textContent = "";
    $("audioA").removeAttribute("src"); $("audioB").removeAttribute("src");
    $("qsummary").textContent = "all items judged";
    return;
  }
  $("qid").textContent = `${it.id} · ${it.voice || state.voice} · ${it.steps} steps`;
  $("qmeta").textContent = `${it.steps} steps · cfg ${it.cfg} · target ${it.target_seconds}s`;
  $("qtext").textContent = it.text;
  $("audioA").src = audioUrl(it, it.a);
  $("audioB").src = audioUrl(it, it.b);
  $("qnotes").value = "";
  state.artifacts.clear();
  document.querySelectorAll("#qartifacts button").forEach((b) => b.classList.remove("on"));
  const label = state.labels[it.id];
  document.querySelectorAll(".accuracy button").forEach((b) => {
    b.classList.toggle("on", !!label && label.preference === b.dataset.p);
  });
  $("qreveal").textContent = label
    ? `revealed: A = ${label.a_variant}, B = ${label.b_variant} — you chose ${label.preference}`
    : "";
  $("qsavestate").textContent = label ? "saved" : "";
  renderSummary();
}

function renderSummary() {
  const rows = Object.values(state.labels);
  if (!rows.length) { $("qsummary").textContent = "no judgements yet"; return; }
  const c = { a: 0, b: 0, same: 0, bad: 0 };
  let int4Wins = 0, fp16Wins = 0;
  for (const r of rows) {
    c[r.preference] = (c[r.preference] || 0) + 1;
    if (r.preference === "a" && r.a_variant === "int4") int4Wins++;
    if (r.preference === "b" && r.b_variant === "int4") int4Wins++;
    if (r.preference === "a" && r.a_variant === "fp16") fp16Wins++;
    if (r.preference === "b" && r.b_variant === "fp16") fp16Wins++;
  }
  const n = rows.length;
  $("qsummary").textContent =
    `judged     ${n}\n` +
    `int4 wins  ${int4Wins}\n` +
    `fp16 wins  ${fp16Wins}\n` +
    `same       ${c.same || 0}\n` +
    `both bad   ${c.bad || 0}`;
}

async function save(preference) {
  const it = item();
  if (!it) return;
  const payload = {
    id: it.id, preference,
    artifacts: [...state.artifacts],
    notes: $("qnotes").value,
    a_variant: it.a, b_variant: it.b,
  };
  try {
    const res = await api("api/quant/label", { method: "POST", body: JSON.stringify(payload) });
    state.labels[it.id] = res.saved;
    $("qsavestate").textContent = "saved";
    renderSummary();
    $("qreveal").textContent = `revealed: A = ${it.a}, B = ${it.b} — you chose ${preference}`;
    if ($("qautonext").checked) setTimeout(() => next(1), 900);
  } catch (e) {
    toast("save failed: " + e.message);
  }
}

function next(delta) {
  if (!state.queue.length) return;
  state.pos = (state.pos + delta + state.queue.length) % state.queue.length;
  stopAll();
  render();
}

function stopAll() {
  for (const a of [$("audioA"), $("audioB")]) { a.pause(); a.currentTime = 0; }
  state.playing = null;
}

function playOne(which) {
  stopAll();
  const a = which === "a" ? $("audioA") : $("audioB");
  a.play();
  state.playing = which;
}

async function playBoth() {
  stopAll();
  const a = $("audioA");
  await a.play();
  a.onended = () => { a.onended = null; $("audioB").play(); };
}

async function exportCsv() {
  try {
    const res = await fetch("api/quant/export.csv?key=" + encodeURIComponent(state.secret), { headers: headers() });
    if (!res.ok) throw new Error("export " + res.status);
    const text = await res.text();
    const url = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
    const link = document.createElement("a");
    link.href = url; link.download = "quant_ab_labels.csv"; link.click();
    URL.revokeObjectURL(url);
  } catch (e) { toast("export failed: " + e.message); }
}

$("unlock").onclick = async () => {
  const secret = $("secret").value.trim();
  if (!secret) return;
  state.secret = secret;
  localStorage.setItem("qc_secret", secret);
  $("gaterr").textContent = "";
  await boot();
  if ($("qmain").hidden) { $("gaterr").textContent = "wrong secret"; logout(); }
};
$("secret").addEventListener("keydown", (e) => { if (e.key === "Enter") $("unlock").click(); });

$("qall").onchange = () => { applyQueue(true); render(); };
$("qprev").onclick = () => next(-1);
$("qnext").onclick = () => next(1);
$("playA").onclick = () => playOne("a");
$("playB").onclick = () => playOne("b");
$("playAB").onclick = playBoth;
$("qexport").onclick = exportCsv;
document.querySelectorAll(".accuracy button").forEach((b) => {
  b.onclick = () => save(b.dataset.p);
});

document.addEventListener("keydown", (e) => {
  const typing = e.target.matches("textarea, select, input[type=search], input[type=text], input[type=password]");
  if (typing) {
    if (e.key === "Escape") e.target.blur();
    return;
  }
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); playBoth(); return; }
  if (k === "1") { save("a"); return; }
  if (k === "2") { save("b"); return; }
  if (k === "3") { save("same"); return; }
  if (k === "4") { save("bad"); return; }
  if (k === "e") { e.preventDefault(); $("qnotes").focus(); return; }
  if (k === "arrowright") { next(1); return; }
  if (k === "arrowleft") { next(-1); return; }
});

$("audioA").onended = () => { if (state.playing === "a") state.playing = null; };
$("audioB").onended = () => { if (state.playing === "b") state.playing = null; };

boot();