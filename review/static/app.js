/* Accuracy review console. All requests are relative so the app works under
   Caddy's /review/ prefix, where the prefix is stripped before it reaches us. */

const $ = (id) => document.getElementById(id);
const state = {
  secret: localStorage.getItem("qc_secret") || "",
  items: [],
  labels: {},
  queue: [],
  pos: 0,
  item: null,
  buffer: null,
  blobUrl: null,
  audio: new Audio(),
  ctx: null,
  playing: false,
  dirty: false,
  meta: null,
};

const ACC = ["correct", "minor", "major", "unusable"];
const ERRS = ["wrong_words", "missing_words", "hallucinated", "bad_boundary",
              "non_speech", "other"];
const ERR_KEY = { w: "wrong_words", m: "missing_words", h: "hallucinated",
                  b: "bad_boundary", n: "non_speech", o: "other" };

function headers() {
  return { "Authorization": "Bearer " + state.secret, "Content-Type": "application/json" };
}

async function api(path, opts = {}) {
  const res = await fetch(path, { ...opts, headers: { ...headers(), ...(opts.headers || {}) } });
  if (res.status === 401) { logout(); throw new Error("unauthorized"); }
  if (!res.ok) throw new Error(path + " -> " + res.status);
  return res.json();
}

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 1400);
}

function logout() {
  localStorage.removeItem("qc_secret");
  state.secret = "";
  $("gate").classList.remove("hidden");
  $("main").hidden = true;
}

async function boot() {
  if (!state.secret) { $("gate").classList.remove("hidden"); return; }
  try {
    state.meta = await api("api/meta");
    const idx = await api("api/index");
    state.items = idx.items;
    state.labels = (await api("api/labels")).labels;
    $("gate").classList.add("hidden");
    $("main").hidden = false;
    buildPerformerSelect();
    buildErrorButtons();
    renderMeta();
    applyFilters(true);
    const hash = decodeURIComponent(location.hash.replace(/^#/, ""));
    if (hash) { const i = state.queue.findIndex((x) => x.id === hash); if (i >= 0) state.pos = i; }
    loadCurrent();
  } catch (e) {
    if (String(e.message).includes("unauthorized")) return;
    toast("failed to load: " + e.message);
  }
}

function buildPerformerSelect() {
  const sel = $("performer");
  const performers = Object.entries(state.meta.performers);
  for (const [name, n] of performers) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = `${name} (${n})`;
    sel.appendChild(opt);
  }
}

function buildErrorButtons() {
  const box = $("errors");
  box.innerHTML = "";
  const labels = {
    wrong_words: "wrong words", missing_words: "missing words",
    hallucinated: "hallucinated", bad_boundary: "bad boundary",
    non_speech: "non-speech", other: "other",
  };
  for (const key of ERRS) {
    const b = document.createElement("button");
    b.dataset.e = key;
    b.textContent = labels[key];
    b.onclick = () => toggleError(key);
    box.appendChild(b);
  }
}

function isSuspicious(it) {
  return it.cps < 2.0 || it.cps > 6.5 || it.duration < 3 || it.duration > 12 ||
         (it.tags && it.tags.length > 0) || it.n_speakers > 1;
}

function applyFilters(resetPos) {
  const mode = $("mode").value;
  const performer = $("performer").value;
  const q = $("search").value.trim().toLowerCase();
  const q_items = state.items.filter((it) => {
    if (performer && it.performer !== performer) return false;
    if (q && !(it.text || "").toLowerCase().includes(q)) return false;
    const label = state.labels[it.id];
    if (mode === "unlabeled" && label) return false;
    if (mode === "labeled" && !label) return false;
    if (mode === "suspicious" && !isSuspicious(it)) return false;
    if (mode === "tags" && !(it.tags && it.tags.length)) return false;
    if (mode === "multi" && !(it.n_speakers > 1)) return false;
    if (mode === "short" && !(it.duration < 3)) return false;
    if (mode === "long" && !(it.duration > 12)) return false;
    return true;
  });
  const sample = parseInt($("sample").value || "0", 10);
  if (sample > 0 && sample < q_items.length) {
    // deterministic sample so the same set can be resumed after a reload
    let seed = 42;
    const rand = () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };
    const pool = q_items.slice();
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    const order = new Map(state.items.map((it, i) => [it.id, i]));
    q_items.splice(0, q_items.length, ...pool.slice(0, sample));
    q_items.sort((a, b) => order.get(a.id) - order.get(b.id));
  } else if ($("shuffle").checked) {
    for (let i = q_items.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [q_items[i], q_items[j]] = [q_items[j], q_items[i]];
    }
  }
  state.queue = q_items;
  if (resetPos) state.pos = 0;
  if (state.pos >= state.queue.length) state.pos = Math.max(0, state.queue.length - 1);
  renderQueueStats();
}

function renderMeta() {
  const m = state.meta;
  const labeled = Object.keys(state.labels).length;
  $("progtext").textContent = `${labeled} / ${m.n_items}`;
  $("progbar").style.width = (100 * labeled / Math.max(1, m.n_items)) + "%";
  const c = m.counts || {};
  $("counts").textContent = ACC.filter((a) => c[a]).map((a) => `${a}:${c[a]}`).join("  ");
}

function renderQueueStats() {
  const total = state.queue.length;
  const labeled = state.queue.filter((it) => state.labels[it.id]).length;
  $("queuestats").textContent =
    `position   ${state.pos + 1} / ${total}\n` +
    `labeled    ${labeled}\n` +
    `remaining  ${total - labeled}`;
  renderSummary();
}

function wilson(p, n) {
  if (!n) return null;
  const z = 1.96, z2 = z * z;
  const denom = 1 + z2 / n;
  const centre = (p + z2 / (2 * n)) / denom;
  const half = z * Math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

function renderSummary() {
  const rows = Object.values(state.labels);
  const n = rows.length;
  if (!n) { $("summary").textContent = "no labels yet"; return; }
  const c = {};
  for (const r of rows) c[r.accuracy] = (c[r.accuracy] || 0) + 1;
  const correct = (c.correct || 0) / n;
  const ok = ((c.correct || 0) + (c.minor || 0)) / n;
  const bad = ((c.major || 0) + (c.unusable || 0)) / n;
  const ci = wilson(correct, n);
  const ci2 = wilson(ok, n);
  $("summary").textContent =
    `labeled    ${n}\n` +
    `correct    ${(100 * correct).toFixed(1)}%  (95% CI ${(100 * ci[0]).toFixed(1)}-${(100 * ci[1]).toFixed(1)}%)\n` +
    `acceptable ${(100 * ok).toFixed(1)}%  (95% CI ${(100 * ci2[0]).toFixed(1)}-${(100 * ci2[1]).toFixed(1)}%)\n` +
    `major+bad  ${(100 * bad).toFixed(1)}%\n` +
    `minor ${c.minor || 0} · major ${c.major || 0} · unusable ${c.unusable || 0}`;
}

function current() { return state.queue[state.pos]; }

async function loadCurrent() {
  const it = current();
  if (!it) { $("mtext").textContent = "queue is empty"; return; }
  history.replaceState(null, "", "#" + encodeURIComponent(it.id));
  let detail;
  try { detail = await api("api/item/" + encodeURIComponent(it.id)); }
  catch (e) { toast("load failed: " + e.message); return; }
  state.item = detail;
  state.dirty = false;
  renderItem(detail);
  await loadAudio(it.id);
}

async function loadAudio(id) {
  stopAudio();
  const useContext = $("context").checked;
  const url = (useContext ? "api/context/" : "api/audio/") + encodeURIComponent(id);
  const res = await fetch(url, { headers: headers() });
  if (!res.ok) { toast("audio missing"); drawWave(0); return; }
  const buf = await res.arrayBuffer();
  if (state.blobUrl) URL.revokeObjectURL(state.blobUrl);
  state.blobUrl = URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
  state.audio.src = state.blobUrl;
  state.audio.playbackRate = parseFloat($("rate").value);
  state.audio.loop = $("loop").checked;
  try {
    state.ctx = state.ctx || new (window.AudioContext || window.webkitAudioContext)();
    state.buffer = await state.ctx.decodeAudioData(buf.slice(0));
    state.padS = useContext ? 1.0 : 0.0;
    drawWave(0);
  } catch (e) {
    state.buffer = null;
    state.padS = 0;
    drawWave(0);
  }
}

function renderItem(it) {
  $("mfile").textContent = it.source_file + "  #" + it.segment_index;
  $("mtime").textContent =
    `${it.start_s.toFixed(2)}s → ${it.end_s.toFixed(2)}s · ${it.duration.toFixed(2)}s · ` +
    `${it.cps.toFixed(2)} chars/s · ${it.chars} chars`;
  const badges = [];
  if (it.tags) badges.push(`<span class="badge">${escapeHtml(it.tags)}</span>`);
  if (it.n_speakers > 1) badges.push(`<span class="badge soft">${it.n_speakers} speakers</span>`);
  if (it.cps < 2.0) badges.push(`<span class="badge warn">low cps</span>`);
  if (it.cps > 6.5) badges.push(`<span class="badge warn">high cps</span>`);
  $("mbadges").innerHTML = badges.join("");

  $("mtext").textContent = it.text;
  $("mchars").textContent = `${it.duration.toFixed(2)}s`;
  const base = (it.baseline || "").trim();
  $("baselinewrap").classList.toggle("hidden", !base);
  $("mbaseline").textContent = base;

  const label = it.label || {};
  document.querySelectorAll(".accuracy button").forEach((b) => {
    b.classList.toggle("on", b.dataset.a === label.accuracy);
  });
  document.querySelectorAll(".errors button").forEach((b) => {
    b.classList.toggle("on", (label.errors || []).includes(b.dataset.e));
  });
  $("corrected").value = label.corrected || "";
  $("notes").value = label.notes || "";
  $("clock").textContent = `0.00 / ${it.duration.toFixed(2)}`;
  $("savestate").textContent = label.accuracy ? `saved: ${label.accuracy}` : "";
  renderQueueStats();
}

/* ---------------- audio ---------------- */

function stopAudio() {
  state.audio.pause();
  state.playing = false;
  $("play").textContent = "▶";
}

function togglePlay() {
  if (!state.audio.src) return;
  if (state.audio.paused) {
    state.audio.play();
    state.playing = true;
    $("play").textContent = "❚❚";
    raf();
  } else {
    stopAudio();
    drawWave(state.audio.currentTime / (state.item?.duration || 1));
  }
}

function replay() {
  if (!state.audio.src) return;
  state.audio.currentTime = 0;
  state.audio.play();
  state.playing = true;
  $("play").textContent = "❚❚";
  raf();
}

function raf() {
  if (!state.playing) return;
  const d = state.audio.duration || state.item?.duration || 1;
  const t = state.audio.currentTime;
  drawWave(t / d);
  $("clock").textContent = `${t.toFixed(2)} / ${d.toFixed(2)}`;
  const progress = Object.assign({}, state.labels[state.item?.id] || {});
  if (state.dirty) progress.accuracy = progress.accuracy || undefined;
  requestAnimationFrame(raf);
}

function drawWave(progress) {
  const canvas = $("wave");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr; canvas.height = h * dpr;
  }
  const c = canvas.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  c.fillStyle = "#e6e1d1"; c.fillRect(0, 0, w, h);
  if (!state.buffer) {
    c.fillStyle = "#8f8f8f"; c.font = "12px monospace";
    c.fillText("no waveform", 12, h / 2);
    return;
  }
  const data = state.buffer.getChannelData(0);
  const pad = state.padS || 0;
  const totalS = state.buffer.duration;
  const clipStartFrac = pad / totalS;
  const clipEndFrac = Math.min(1, 1 - pad / totalS);
  if (pad > 0) {
    c.fillStyle = "#d8d2bf";
    c.fillRect(0, 0, w * clipStartFrac, h);
    c.fillRect(w * clipEndFrac, 0, w * (1 - clipEndFrac), h);
  }
  const step = Math.max(1, Math.floor(data.length / w));
  const mid = h / 2;
  c.fillStyle = "#101010";
  for (let x = 0; x < w; x++) {
    let min = 1, max = -1;
    for (let i = 0; i < step; i++) {
      const v = data[x * step + i] || 0;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const y1 = mid - max * mid * 0.95;
    const y2 = mid - min * mid * 0.95;
    c.fillRect(x, y1, 1, Math.max(1, y2 - y1));
  }
  if (progress > 0) {
    c.fillStyle = "rgba(0,0,0,0.18)";
    c.fillRect(0, 0, w * Math.min(1, progress), h);
  }
  if (pad > 0) {
    c.strokeStyle = "#000"; c.lineWidth = 1;
    c.strokeRect(w * clipStartFrac, 0, w * (clipEndFrac - clipStartFrac), h - 1);
  }
}

/* ---------------- labels ---------------- */

function setAccuracy(a) {
  if (!state.item) return;
  state.item.label = { ...(state.item.label || {}), accuracy: a };
  markDirty();
  save().then(() => {
    renderMeta(); renderQueueStats();
    if ($("autonext").checked) setTimeout(() => next(1), 220);
  });
}

function toggleError(key) {
  if (!state.item) return;
  const label = state.item.label = { ...(state.item.label || {}) };
  const set = new Set(label.errors || []);
  set.has(key) ? set.delete(key) : set.add(key);
  label.errors = [...set];
  document.querySelectorAll(".errors button").forEach((b) => {
    b.classList.toggle("on", label.errors.includes(b.dataset.e));
  });
  markDirty();
  save();
}

function markDirty() {
  state.dirty = true;
  $("savestate").textContent = "editing…";
}

let saveTimer = null;
async function save() {
  if (!state.item) return;
  const payload = {
    id: state.item.id,
    accuracy: (state.item.label || {}).accuracy || null,
    errors: (state.item.label || {}).errors || [],
    corrected: $("corrected").value,
    notes: $("notes").value,
  };
  try {
    const res = await api("api/label", { method: "POST", body: JSON.stringify(payload) });
    state.labels[state.item.id] = res.saved;
    state.dirty = false;
    $("savestate").textContent = res.saved.accuracy ? `saved: ${res.saved.accuracy}` : "saved";
    if (state.queue[state.pos]) state.queue[state.pos].label = res.saved.accuracy || null;
  } catch (e) {
    $("savestate").textContent = "save failed";
    toast("save failed: " + e.message);
  }
}

function debounceSave() {
  markDirty();
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, 600);
}

function next(delta) {
  if (!state.queue.length) return;
  state.pos = (state.pos + delta + state.queue.length) % state.queue.length;
  stopAudio();
  loadCurrent();
}

/* ---------------- events ---------------- */

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

$("unlock").onclick = async () => {
  const secret = $("secret").value.trim();
  if (!secret) return;
  state.secret = secret;
  localStorage.setItem("qc_secret", secret);
  $("gaterr").textContent = "";
  await boot();
  if ($("main").hidden) { $("gaterr").textContent = "wrong secret"; logout(); }
};
$("secret").addEventListener("keydown", (e) => { if (e.key === "Enter") $("unlock").click(); });

for (const id of ["mode", "performer"]) $(id).onchange = (e) => {
  applyFilters(true); loadCurrent(); e.target.blur();
};
$("search").oninput = () => { applyFilters(true); loadCurrent(); };
$("shuffle").onchange = () => applyFilters(true);
$("sample").onchange = (e) => { applyFilters(true); loadCurrent(); e.target.blur(); };
$("export").onclick = () => {
  window.open("api/export.csv?key=" + encodeURIComponent(state.secret), "_blank");
};

$("play").onclick = togglePlay;
$("replay").onclick = replay;
$("prev").onclick = () => next(-1);
$("next").onclick = () => next(1);
$("skip").onclick = () => next(1);
$("rate").onchange = () => { state.audio.playbackRate = parseFloat($("rate").value); };
$("loop").onchange = () => { state.audio.loop = $("loop").checked; };
$("context").onchange = (e) => { if (state.item) loadAudio(state.item.id); e.target.blur(); };
$("usebaseline").onclick = () => {
  const base = (state.item?.baseline || "").trim();
  if (!base) return;
  $("corrected").value = base;
  debounceSave();
};
$("corrected").oninput = debounceSave;
$("notes").oninput = debounceSave;
document.querySelectorAll(".accuracy button").forEach((b) => {
  b.onclick = () => setAccuracy(b.dataset.a);
});
$("wave").onclick = (e) => {
  if (!state.item) return;
  const rect = $("wave").getBoundingClientRect();
  const frac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  const d = state.audio.duration || state.item.duration;
  state.audio.currentTime = frac * d;
  drawWave(frac);
  $("clock").textContent = `${(frac * d).toFixed(2)} / ${d.toFixed(2)}`;
};

document.addEventListener("keydown", (e) => {
  const typing = e.target.matches(
    'textarea, select, input[type=search], input[type=text], ' +
    'input[type=number], input[type=password]');
  if (typing) {
    if (e.key === "Escape") e.target.blur();
    return;
  }
  const k = e.key.toLowerCase();
  if (k === " ") { e.preventDefault(); togglePlay(); return; }
  if (k === "r") { replay(); return; }
  if (k === "s") { next(1); return; }
  if (k === "e") { e.preventDefault(); $("corrected").focus(); return; }
  if (k === "arrowright") { next(1); return; }
  if (k === "arrowleft") { next(-1); return; }
  if (["1", "2", "3", "4"].includes(k)) { setAccuracy(ACC[Number(k) - 1]); return; }
  if (ERR_KEY[k]) { toggleError(ERR_KEY[k]); return; }
  if (k === "?") { document.querySelector(".side").scrollIntoView({ behavior: "smooth" }); }
});

state.audio.addEventListener("ended", () => {
  state.playing = false;
  $("play").textContent = "▶";
  drawWave(1);
});

window.addEventListener("resize", () => drawWave(state.audio.currentTime / (state.item?.duration || 1)));
boot();