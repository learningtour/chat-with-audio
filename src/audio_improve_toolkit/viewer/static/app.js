"use strict";

// ---- state ----
let sessionsList = [];
let current = null;        // sessiedata van /api/sessions/<id>
let ctx = null;            // AudioContext (lazy, na user-gesture)
let bufA = null, bufB = null;
let rawA = null, rawB = null;  // ArrayBuffers (gefetcht bij sessie-load)
let srcA = null, srcB = null, gainA = null, gainB = null;
let playing = false, startedAt = 0, offset = 0, listenB = true;
let trimA = 1;  // monitoring-trim voor originelen die boven 0 dBFS pieken (32-bit float)

const $ = (id) => document.getElementById(id);

const METRIC_LABELS = {
  lufs_integrated: ["Loudness (LUFS)", null],
  true_peak_dbtp: ["True peak (dBTP)", null],
  rms_db: ["RMS (dB)", null],
  noise_floor_db: ["Ruisvloer (dB)", "down"],
  snr_db: ["SNR (dB)", "up"],
  crest_factor_db: ["Crest factor (dB)", null],
  lra_db: ["Loudness range (dB)", null],
  silence_pct: ["Stilte (%)", null],
  clip_events: ["Clip-momenten", "down"],
};

// ---- sessielijst ----
async function loadList() {
  sessionsList = await (await fetch("/api/sessions")).json();
  const ul = $("session-list");
  ul.innerHTML = "";
  for (const s of sessionsList) {
    const li = document.createElement("li");
    li.dataset.id = s.session_id;
    li.innerHTML = `<div>${s.label}</div><div class="d">${s.created} · ${fmtTime(s.duration_s)}` +
      (s.has_processed ? "" : " · alleen analyse") + `</div>`;
    li.onclick = () => { location.hash = "#/session/" + s.session_id; };
    ul.appendChild(li);
  }
}

function fmtTime(t) {
  t = Math.max(0, t || 0);
  return `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
}

// ---- sessie laden ----
async function openSession(id) {
  stop();
  bufA = bufB = rawA = rawB = null; offset = 0;
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

  const hasB = current.has_processed;
  listenB = hasB;
  $("row-b").style.display = hasB ? "" : "none";
  $("btn-b").disabled = !hasB;
  setABButtons();

  drawWave($("wave-a"), current.waveform_original);
  if (hasB) drawWave($("wave-b"), current.waveform_processed);
  renderMetrics();
  renderChain();
  setSpec("original");

  rawA = await (await fetch(`/files/${id}/original.wav`)).arrayBuffer();
  if (hasB) rawB = await (await fetch(`/files/${id}/processed.wav`)).arrayBuffer();
  updateTime();
}

// ---- tabellen & panelen ----
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
  const sA = current.original?.scores, sB = current.processed?.scores;
  if (sA) {
    const b = sB ? `<td>${sB.overall}</td><td></td>` : "";
    html += `<tr><td>Score (0-100)</td><td>${sA.overall}</td>${b}</tr>`;
  }
  $("metrics").innerHTML = html;
}

function renderChain() {
  const rat = current.chain?.rationale || [];
  $("rationale").innerHTML = rat.map((r) => `<li>${r}</li>`).join("") ||
    "<li>Alleen analyse — nog geen bewerking.</li>";
  $("chain").textContent = JSON.stringify(current.chain?.steps || [], null, 2);
}

function setSpec(which) {
  document.querySelectorAll(".spec-tabs button").forEach(
    (b) => b.classList.toggle("active", b.dataset.spec === which));
  $("spec-img").src = `/files/${current.session_id}/spectrogram_${which}.png`;
}

// ---- waveform ----
function drawWave(canvas, wf, playPos = null) {
  if (!wf) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  const h = canvas.getAttribute("height") * 1;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  const n = wf.min.length, mid = h / 2;
  let peak = 0.01;
  for (let i = 0; i < n; i++) peak = Math.max(peak, Math.abs(wf.min[i]), Math.abs(wf.max[i]));
  g.fillStyle = "#3f7fa8";
  for (let i = 0; i < n; i++) {
    const x = (i / n) * w;
    const y1 = mid - (wf.max[i] / peak) * (mid - 4);
    const y2 = mid - (wf.min[i] / peak) * (mid - 4);
    g.fillRect(x, y1, Math.max(w / n - 0.3, 0.7), Math.max(y2 - y1, 1));
  }
  if (playPos !== null && current) {
    const x = (playPos / current.duration_s) * w;
    g.fillStyle = "#ffd166";
    g.fillRect(x, 0, 1.5, h);
  }
}

function redrawWaves(pos) {
  drawWave($("wave-a"), current.waveform_original, pos);
  if (current.has_processed) drawWave($("wave-b"), current.waveform_processed, pos);
}

// ---- audio: gesynchroniseerde A/B ----
async function ensureBuffers() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === "suspended") await ctx.resume();
  const id = current.session_id;
  if (!rawA) rawA = await (await fetch(`/files/${id}/original.wav`)).arrayBuffer();
  if (!rawB && current.has_processed)
    rawB = await (await fetch(`/files/${id}/processed.wav`)).arrayBuffer();
  if (!bufA && rawA) {
    bufA = await ctx.decodeAudioData(rawA.slice(0));
    let peak = 0;
    for (let c = 0; c < bufA.numberOfChannels; c++) {
      const d = bufA.getChannelData(c);
      for (let i = 0; i < d.length; i++) {
        const v = Math.abs(d[i]);
        if (v > peak) peak = v;
      }
    }
    trimA = peak > 1 ? 0.891 / peak : 1;  // naar -1 dBFS voor eerlijke monitoring
    if (trimA < 1) {
      const db = (20 * Math.log10(trimA)).toFixed(1);
      document.querySelector("#wave-a").parentElement.querySelector(".tag").textContent =
        `A · origineel (${db} dB monitoring-trim)`;
    }
  }
  if (!bufB && rawB) bufB = await ctx.decodeAudioData(rawB.slice(0));
}

async function play() {
  if (!current || playing) return;
  await ensureBuffers();
  if (!bufA) return;
  gainA = ctx.createGain(); gainB = ctx.createGain();
  gainA.connect(ctx.destination); gainB.connect(ctx.destination);
  gainA.gain.value = listenB && bufB ? 0 : trimA;
  gainB.gain.value = listenB && bufB ? 1 : 0;
  srcA = ctx.createBufferSource(); srcA.buffer = bufA; srcA.connect(gainA);
  if (bufB) { srcB = ctx.createBufferSource(); srcB.buffer = bufB; srcB.connect(gainB); }
  const t0 = ctx.currentTime + 0.03;
  if (offset >= bufA.duration) offset = 0;
  srcA.start(t0, offset);
  if (srcB) srcB.start(t0, offset);
  srcA.onended = () => { if (playing) { stop(); offset = 0; updateTime(); } };
  startedAt = t0; playing = true;
  $("btn-play").textContent = "❚❚";
  tick();
}

function stop(keepOffset = false) {
  if (srcA) { srcA.onended = null; try { srcA.stop(); } catch {} }
  if (srcB) { try { srcB.stop(); } catch {} }
  srcA = srcB = null;
  if (playing && keepOffset && ctx) offset += Math.max(0, ctx.currentTime - startedAt);
  playing = false;
  $("btn-play").textContent = "▶";
}

function togglePlay() { playing ? (stop(true), updateTime()) : play(); }

function seekTo(frac) {
  const was = playing;
  stop();
  offset = frac * (current?.duration_s || 0);
  updateTime();
  if (was) play();
}

function setAB(toB) {
  if (toB && !current?.has_processed) return;
  listenB = toB;
  setABButtons();
  if (playing && gainA && gainB) {
    const t = ctx.currentTime;
    gainA.gain.setTargetAtTime(toB ? 0 : trimA, t, 0.005);
    gainB.gain.setTargetAtTime(toB ? 1 : 0, t, 0.005);
  }
}

function setABButtons() {
  $("btn-a").classList.toggle("active", !listenB);
  $("btn-b").classList.toggle("active", listenB);
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
$("btn-a").onclick = () => setAB(false);
$("btn-b").onclick = () => setAB(true);
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
  if (e.key === "b") setAB(!listenB);
  if (e.key === "a") setAB(false);
});
window.addEventListener("resize", () => current && redrawWaves(pos()));
window.addEventListener("hashchange", route);

function route() {
  const m = location.hash.match(/^#\/session\/(.+)$/);
  if (m) openSession(m[1]);
}

loadList().then(route);
