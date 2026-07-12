"use strict";

// ---- state ----
let sessionsList = [];
let current = null;        // sessiedata van /api/sessions/<id>
let ctx = null;            // AudioContext (lazy, na user-gesture)
let bufA = null, bufB = null, bufR = null;
let rawA = null, rawB = null, rawR = null;  // ArrayBuffers (gefetcht bij sessie-load)
let srcA = null, srcB = null, srcR = null;
let gainA = null, gainB = null, gainR = null;
let playing = false, startedAt = 0, offset = 0;
let listen = "b";  // 'a' origineel | 'b' verbeterd | 'r' residu (verschil)
let trimA = 1;  // monitoring-trim voor originelen die boven 0 dBFS pieken (32-bit float)

const $ = (id) => document.getElementById(id);

const METRIC_LABELS = {
  lufs_integrated: ["Loudness (LUFS)", null],
  lufs_short_term_max: ["Short-term max (LUFS)", null],
  lufs_momentary_max: ["Momentary max (LUFS)", null],
  true_peak_dbtp: ["True peak (dBTP)", null],
  plr_db: ["PLR (dB)", null],
  rms_db: ["RMS (dB)", null],
  noise_floor_db: ["Ruisvloer (dB)", "down"],
  snr_db: ["SNR (dB)", "up"],
  crest_factor_db: ["Crest factor (dB)", null],
  lra_db: ["Loudness range (dB)", null],
  silence_pct: ["Stilte (%)", null],
  clip_events: ["Clip-momenten", "down"],
};

const SEG_LABELS = { speech: "spraak", music: "muziek", silence: "stilte" };
const REG_LABELS = { hum: "netbrom", noise: "ruis", clip: "clipping",
                     boom: "dreun", repair: "painting", track: "spoor",
                     filler: "vulwoord", repeat: "herhaling", pause: "pauze",
                     text: "tekst weg", unselected: "buiten selectie",
                     bleep: "bleep", tone: "toon", pop: "two-pop" };

// ---- sessielijst ----
function splitLabel(label) {
  const m = (label || "").match(/^(.*?)\s+—\s+(.+)$/);
  return m ? { base: m[1], tag: m[2] } : { base: label, tag: null };
}

function renderList() {
  const ul = $("session-list");
  ul.innerHTML = "";
  for (const s of sessionsList) {
    const li = document.createElement("li");
    li.dataset.id = s.session_id;
    if (current && s.session_id === current.session_id) li.classList.add("active");
    const { base, tag } = splitLabel(s.label);
    li.innerHTML = `<div>${base}${tag ? `<span class="chip">${tag}</span>` : ""}</div>` +
      `<div class="d">${s.created} · ${fmtTime(s.duration_s)}` +
      (s.has_processed ? "" : " · alleen analyse") + `</div>`;
    li.onclick = () => { location.hash = "#/session/" + s.session_id; };
    ul.appendChild(li);
  }
}

async function loadList() {
  sessionsList = await (await fetch("/api/sessions")).json();
  renderList();
}

// push: nieuwe sessies verschijnen vanzelf, met een klikbare melding
function showNewSessionToast(s) {
  let t = $("toast");
  if (!t) {
    t = document.createElement("div");
    t.id = "toast";
    document.body.appendChild(t);
  }
  t.innerHTML = `🔔 Nieuw: <b>${s.label}</b> — klik om te openen`;
  t.onclick = () => { location.hash = "#/session/" + s.session_id; t.classList.remove("show"); };
  t.classList.add("show");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.remove("show"), 20000);
}

setInterval(async () => {
  try {
    const list = await (await fetch("/api/sessions")).json();
    if (!Array.isArray(list) || !list.length) return;
    const known = sessionsList.length ? sessionsList[0].session_id : null;
    if (list[0].session_id !== known) {
      sessionsList = list;
      renderList();
      showNewSessionToast(list[0]);
    }
  } catch { /* viewer offline of herstart: stil doorproberen */ }
}, 3000);

function fmtTime(t) {
  t = Math.max(0, t || 0);
  return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
}

// ---- sessie laden ----
async function openSession(id) {
  stop();
  bufA = bufB = bufR = rawA = rawB = rawR = null; offset = 0;
  const r = await fetch("/api/sessions/" + id);
  if (!r.ok) { alert("Sessie niet gevonden"); return; }
  current = await r.json();

  document.querySelectorAll("#session-list li").forEach(
    (li) => li.classList.toggle("active", li.dataset.id === id));
  $("empty").hidden = true;
  $("detail").hidden = false;
  $("s-label").textContent = current.label;
  $("s-meta").textContent =
    `${current.created} · ${fmtTime(current.duration_s)} · ${current.sample_rate} Hz` +
    (current.profile ? ` · profiel: ${current.profile === "speech" ? "spraak" : "muziek"}` : "");
  renderScoreBadges();

  const hasB = current.has_processed;
  listen = hasB ? "b" : "a";
  $("row-b").style.display = hasB ? "" : "none";
  $("btn-b").disabled = !hasB;
  $("btn-r").disabled = !hasB;
  setListenButtons();

  // Monitoring-trim voor A: loudness-matched met B (eerlijk vergelijk van kárakter,
  // niet van volume), en nooit boven full scale afspelen.
  const pkA = wfPeak(current.waveform_original);
  const lA = current.original?.metrics?.lufs_integrated;
  const lB = current.processed?.metrics?.lufs_integrated;
  trimA = 1;
  if (hasB && lA != null && lB != null) trimA = Math.pow(10, (lB - lA) / 20);
  else if (pkA > 1) trimA = 0.891 / pkA;
  if (pkA * trimA > 0.98) trimA = 0.98 / pkA;
  const tagA = document.querySelector("#wave-a").parentElement.querySelector(".tag");
  tagA.textContent = Math.abs(20 * Math.log10(trimA)) > 0.5
    ? `A · origineel (${(20 * Math.log10(trimA)).toFixed(1)} dB, loudness-matched)`
    : "A · origineel";

  redrawWaves(null);
  renderTimeline();
  renderMetrics();
  renderCompliance();
  renderChain();
  setSpec("original");

  const dln = (suffix) => `${current.label.replace(/[^\w\d-]+/g, "_")}_${suffix}.wav`;
  $("dl-a").href = `/files/${id}/original.wav`;
  $("dl-a").setAttribute("download", dln("A_origineel"));
  $("dl-b").hidden = !hasB;
  $("dl-b").href = `/files/${id}/processed.wav`;
  $("dl-b").setAttribute("download", dln("B_verbeterd"));
  $("dl-r").hidden = true;

  rawA = await (await fetch(`/files/${id}/original.wav`)).arrayBuffer();
  if (hasB) {
    rawB = await (await fetch(`/files/${id}/processed.wav`)).arrayBuffer();
    const rr = await fetch(`/files/${id}/residual.wav`);
    rawR = rr.ok ? await rr.arrayBuffer() : null;  // oudere sessies hebben geen residu
    $("btn-r").disabled = !rawR;
    if (rawR) {
      $("dl-r").hidden = false;
      $("dl-r").href = `/files/${id}/residual.wav`;
      $("dl-r").setAttribute("download", dln("R_verschil"));
    }
  }
  updateTime();
}

// ---- panelen ----
function renderScoreBadges() {
  const el = $("score-badges");
  const sA = current.original?.scores, sB = current.processed?.scores;
  if (!sA) { el.innerHTML = ""; return; }
  let html = `<span class="badge" title="kwaliteitsscore origineel (0-100)">score ${sA.overall}</span>`;
  if (sB) html += `<span class="badge-arrow">→</span>` +
    `<span class="badge after" title="kwaliteitsscore verbeterd (0-100)">${sB.overall}</span>`;
  el.innerHTML = html;
}

function renderTimeline() {
  const box = $("timeline");
  const tl = current.timeline || {};
  const segs = tl.segments || [], regs = tl.regions || [];
  if (!segs.length && !regs.length) { box.hidden = true; return; }
  box.hidden = false;
  const dur = current.duration_s || 1;
  const segLane = $("tl-segments"), regLane = $("tl-regions");
  const trackBox = $("tl-tracks");
  segLane.innerHTML = ""; regLane.innerHTML = ""; trackBox.innerHTML = "";

  const add = (lane, item, cls, label) => {
    const el = document.createElement("div");
    el.className = "tl-item " + cls;
    el.style.left = (item.start_s / dur * 100) + "%";
    el.style.width = (Math.max(item.end_s - item.start_s, dur * 0.005) / dur * 100) + "%";
    el.title = `${label} · ${fmtTime(item.start_s)}–${fmtTime(item.end_s)}`;
    el.onclick = (e) => { e.stopPropagation(); seekTo(item.start_s / dur); };
    lane.appendChild(el);
  };

  // sync-sessies: elke track zijn eigen lane, als een echte 32-sporenrecorder
  const isMultitrack = regs.length > 0 && regs.every((r) => r.kind === "track");
  if (isMultitrack) {
    $("tl-regions-row").style.display = "none";
    for (const r of regs) {
      const row = document.createElement("div");
      row.className = "tl-row";
      const name = document.createElement("span");
      name.className = "tl-name";
      name.textContent = (r.label || "spoor").replace(/\s*\(.*$/, "");
      name.title = r.label || "";
      const lane = document.createElement("div");
      lane.className = "tl-lane";
      row.appendChild(name);
      row.appendChild(lane);
      trackBox.appendChild(row);
      add(lane, r, "reg-track", r.label || "spoor");
    }
  } else {
    for (const r of regs) add(regLane, r, "reg-" + r.kind,
                              r.label || REG_LABELS[r.kind] || r.kind);
    $("tl-regions-row").style.display = regs.length ? "" : "none";
  }
  for (const s of segs) add(segLane, s, "seg-" + s.kind, SEG_LABELS[s.kind] || s.kind);
  $("tl-segments-row").style.display = segs.length ? "" : "none";

  const legend = [];
  for (const kind of Object.keys(SEG_LABELS))
    if (segs.some((s) => s.kind === kind))
      legend.push(`<span><span class="sw seg-${kind}"></span>${SEG_LABELS[kind]}</span>`);
  if (isMultitrack) {
    legend.push(`<span><span class="sw reg-track"></span>${regs.length} sporen · ` +
                "klik een spoor om ernaartoe te springen</span>");
  } else {
    for (const kind of Object.keys(REG_LABELS)) {
      const n = regs.filter((r) => r.kind === kind).length;
      if (n) legend.push(`<span><span class="sw reg-${kind}"></span>${REG_LABELS[kind]} (${n})</span>`);
    }
  }
  $("tl-legend").innerHTML = legend.join("");
}

function renderMetrics() {
  const mA = current.original?.metrics || {};
  const mB = current.processed?.metrics || null;
  const deltas = current.deltas || {};
  let html = mB
    ? "<tr><th>Meting</th><th>A</th><th>B</th><th>Δ</th></tr>"
    : "<tr><th>Meting</th><th>Waarde</th></tr>";
  for (const [key, [label, goodDir]] of Object.entries(METRIC_LABELS)) {
    const a = mA[key];
    if (a === undefined || a === null) continue;
    if (mB) {
      const b = mB[key];
      const d = deltas[key];
      let cls = "", txt = "";
      if (typeof d === "number" && d !== 0) {
        txt = (d > 0 ? "+" : "") + d;
        if (goodDir) cls = (d > 0) === (goodDir === "up") ? "delta-good" : "delta-bad";
      }
      html += `<tr><td>${label}</td><td>${a}</td><td>${b ?? "—"}</td><td class="${cls}">${txt}</td></tr>`;
    } else {
      html += `<tr><td>${label}</td><td>${a}</td></tr>`;
    }
  }
  $("metrics").innerHTML = html;
}

function renderCompliance() {
  const panel = $("compliance-panel");
  const rep = current.compliance;
  if (!rep || !rep.checks) { panel.hidden = true; return; }
  panel.hidden = false;
  $("comp-spec").textContent = rep.spec_name || rep.spec;
  const v = $("comp-verdict");
  v.textContent = rep.passed ? "GESLAAGD" : "NIET GESLAAGD";
  v.className = "badge " + (rep.passed ? "pass" : "fail");
  let html = "<tr><th>Criterium</th><th>Gemeten</th><th>Vereist</th><th></th></tr>";
  for (const c of rep.checks) {
    const mark = c.passed ? "✓" : (c.advisory ? "△" : "✗");
    const cls = c.passed ? "delta-good" : (c.advisory ? "" : "delta-bad");
    html += `<tr><td>${c.name}${c.advisory ? " <i>(richtlijn)</i>" : ""}</td>` +
      `<td>${c.measured ?? "—"}</td><td>${c.requirement}</td>` +
      `<td class="${cls}">${mark}</td></tr>`;
  }
  $("compliance").innerHTML = html;
}

function renderChain() {
  const rat = current.chain?.rationale || [];
  $("rationale").innerHTML = rat.map((r) => `<li>${r}</li>`).join("") ||
    "<li>Alleen analyse — nog geen bewerking.</li>";
  $("chain").textContent = JSON.stringify(current.chain?.steps || [], null, 2);
  const log = $("log-link");
  log.hidden = !current.has_log;
  if (current.has_log) log.href = `/files/${current.session_id}/log.md`;
}

function setSpec(which) {
  document.querySelectorAll(".spec-tabs button").forEach(
    (b) => b.classList.toggle("active", b.dataset.spec === which));
  $("spec-img").src = `/files/${current.session_id}/spectrogram_${which}.png`;
}

// ---- waveform ----
function wfPeak(wf, gain = 1) {
  if (!wf) return 0.01;
  let p = 0.01;
  for (let i = 0; i < wf.min.length; i++)
    p = Math.max(p, Math.abs(wf.min[i] * gain), Math.abs(wf.max[i] * gain));
  return p;
}

function drawWave(canvas, wf, playPos = null, gain = 1, ref = null) {
  if (!wf) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  const h = canvas.getAttribute("height") * 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const n = wf.min.length, mid = h / 2;
  const peak = ref || wfPeak(wf, gain);
  g.fillStyle = "rgba(76,194,255,.14)";
  g.fillRect(0, mid - 0.5, w, 1);
  g.fillStyle = "#4a8fc0";
  for (let i = 0; i < n; i++) {
    const x = (i / n) * w;
    const y1 = mid - (wf.max[i] * gain / peak) * (mid - 4);
    const y2 = mid - (wf.min[i] * gain / peak) * (mid - 4);
    g.fillRect(x, y1, Math.max(w / n - 0.3, 0.7), Math.max(y2 - y1, 1));
  }
  if (playPos !== null && current) {
    const x = (playPos / current.duration_s) * w;
    g.fillStyle = "#ffd166";
    g.fillRect(x, 0, 1.5, h);
  }
}

function redrawWaves(pos) {
  // Gedeelde schaal op afspeelniveau: verschillen in dynamiek/gate/level blijven zichtbaar.
  const wA = current.waveform_original, wB = current.waveform_processed;
  const ref = current.has_processed
    ? Math.max(wfPeak(wA, trimA), wfPeak(wB))
    : wfPeak(wA, trimA);
  drawWave($("wave-a"), wA, pos, trimA, ref);
  if (current.has_processed) drawWave($("wave-b"), wB, pos, 1, ref);
}

// ---- audio: gesynchroniseerde A/B ----
async function ensureBuffers() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === "suspended") await ctx.resume();
  const id = current.session_id;
  if (!rawA) rawA = await (await fetch(`/files/${id}/original.wav`)).arrayBuffer();
  if (!rawB && current.has_processed)
    rawB = await (await fetch(`/files/${id}/processed.wav`)).arrayBuffer();
  if (!rawR && current.has_processed) {
    const rr = await fetch(`/files/${id}/residual.wav`);
    rawR = rr.ok ? await rr.arrayBuffer() : null;
  }
  if (!bufA && rawA) bufA = await ctx.decodeAudioData(rawA.slice(0));
  if (!bufB && rawB) bufB = await ctx.decodeAudioData(rawB.slice(0));
  if (!bufR && rawR) bufR = await ctx.decodeAudioData(rawR.slice(0));
}

function _gainFor(which) {
  if (which === "a") return listen === "a" || !bufB ? trimA : 0;
  if (which === "b") return listen === "b" && bufB ? 1 : 0;
  return listen === "r" && bufR ? 1 : 0;
}

async function play() {
  if (!current || playing) return;
  await ensureBuffers();
  if (!bufA) return;
  gainA = ctx.createGain(); gainB = ctx.createGain(); gainR = ctx.createGain();
  for (const g of [gainA, gainB, gainR]) g.connect(ctx.destination);
  gainA.gain.value = _gainFor("a");
  gainB.gain.value = _gainFor("b");
  gainR.gain.value = _gainFor("r");
  srcA = ctx.createBufferSource(); srcA.buffer = bufA; srcA.connect(gainA);
  if (bufB) { srcB = ctx.createBufferSource(); srcB.buffer = bufB; srcB.connect(gainB); }
  if (bufR) { srcR = ctx.createBufferSource(); srcR.buffer = bufR; srcR.connect(gainR); }
  const t0 = ctx.currentTime + 0.03;
  if (offset >= bufA.duration) offset = 0;
  srcA.start(t0, offset);
  if (srcB) srcB.start(t0, offset);
  if (srcR) srcR.start(t0, offset);
  srcA.onended = () => { if (playing) { stop(); offset = 0; updateTime(); } };
  startedAt = t0; playing = true;
  setPlayIcon(true);
  tick();
}

function setPlayIcon(isPlaying) {
  $("ic-play").hidden = isPlaying;
  $("ic-pause").hidden = !isPlaying;
}

function stop(keepOffset = false) {
  if (srcA) { srcA.onended = null; try { srcA.stop(); } catch {} }
  for (const s of [srcB, srcR]) if (s) { try { s.stop(); } catch {} }
  srcA = srcB = srcR = null;
  if (playing && keepOffset && ctx) offset += Math.max(0, ctx.currentTime - startedAt);
  playing = false;
  setPlayIcon(false);
}

function togglePlay() { playing ? (stop(true), updateTime()) : play(); }

function seekTo(frac) {
  const was = playing;
  stop();
  offset = frac * (current?.duration_s || 0);
  updateTime();
  if (was) play();
}

function setListen(which) {
  if (which !== "a" && !current?.has_processed) return;
  if (which === "r" && playing && !bufR) return;
  listen = which;
  setListenButtons();
  if (playing && gainA && gainB && gainR) {
    const t = ctx.currentTime;
    gainA.gain.setTargetAtTime(_gainFor("a"), t, 0.005);
    gainB.gain.setTargetAtTime(_gainFor("b"), t, 0.005);
    gainR.gain.setTargetAtTime(_gainFor("r"), t, 0.005);
  }
}

function setListenButtons() {
  $("btn-a").classList.toggle("active", listen === "a");
  $("btn-b").classList.toggle("active", listen === "b");
  $("btn-r").classList.toggle("active", listen === "r");
}

function pos() {
  return playing && ctx ? offset + (ctx.currentTime - startedAt) : offset;
}

function updateTime() {
  $("time").textContent = `${fmtTime(pos())} / ${fmtTime(current?.duration_s || 0)}`;
  redrawWaves(pos());
}

function tick() {
  if (!playing) return;
  updateTime();
  requestAnimationFrame(tick);
}

// ---- events ----
$("btn-play").onclick = togglePlay;
$("btn-a").onclick = () => setListen("a");
$("btn-b").onclick = () => setListen("b");
$("btn-r").onclick = () => setListen("r");
document.querySelectorAll(".spec-tabs button").forEach(
  (b) => (b.onclick = () => setSpec(b.dataset.spec)));
for (const id of ["wave-a", "wave-b"]) {
  $(id).addEventListener("click", (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    seekTo((e.clientX - rect.left) / rect.width);
  });
}
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  if (e.key === "a") setListen("a");
  if (e.key === "b") setListen(listen === "b" ? "a" : "b");
  if (e.key === "r") setListen("r");
});
window.addEventListener("resize", () => current && redrawWaves(pos()));
window.addEventListener("hashchange", route);

function route() {
  const m = location.hash.match(/^#\/session\/(.+)$/);
  if (m) openSession(m[1]);
}

// editorkeuze ("Open in ..."): instelbaar, onthouden in localStorage
async function initEditors() {
  try {
    const eds = await (await fetch("/api/editors")).json();
    const sel = $("editor-select");
    sel.innerHTML = eds.map((e) =>
      `<option value="${e.key}" ${e.installed ? "" : "disabled"}>` +
      `${e.name}${e.installed ? "" : " (niet gevonden)"}</option>`).join("");
    const saved = localStorage.getItem("ait-editor");
    const ok = [...sel.options].find((o) => o.value === saved && !o.disabled);
    if (ok) sel.value = saved;
    else {
      const first = [...sel.options].find((o) => !o.disabled);
      if (first) sel.value = first.value;
    }
    updateEditorButton();
    sel.onchange = () => { localStorage.setItem("ait-editor", sel.value); updateEditorButton(); };
  } catch { /* endpoint ontbreekt in oudere server */ }
}
function updateEditorButton() {
  const o = $("editor-select").selectedOptions[0];
  $("btn-editor").textContent = "Open in " + (o ? o.textContent.replace(" (niet gevonden)", "") : "…");
}
$("btn-editor").onclick = async () => {
  if (!current) return;
  const res = await fetch("/api/open-in-editor", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: current.session_id, editor: $("editor-select").value }),
  });
  const j = await res.json().catch(() => ({}));
  if (!res.ok) alert(j.error || "Openen mislukt");
};

initEditors();
loadList().then(route);
