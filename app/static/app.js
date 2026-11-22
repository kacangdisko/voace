const $ = (id) => document.getElementById(id);
let ws, ctx, mic, band = [130, 160], targetS = 300, fillerSet = new Set(), finalHTML = "", nudgeTimer;

init();

async function init() {
  const defaults = await (await fetch("/api/defaults")).json();
  band = defaults.pace_band;
  const setFillers = () => ($("fillers").value = defaults.fillers[$("lang").value].join(", "));
  $("lang").onchange = setFillers;
  setFillers();
  $("start").onclick = start;
  $("stop").onclick = stop;
}

async function start() {
  const fillers = $("fillers").value.split(",").map((s) => s.trim().toLowerCase()).filter(Boolean);
  fillerSet = new Set(fillers.filter((f) => !f.includes(" ")));
  targetS = Number($("target").value) * 60;
  try {
    // Raw mic: browser noise suppression / AGC would hide what the speaker really sounds like.
    mic = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
    ctx = new AudioContext({ sampleRate: 16000 });
    await ctx.audioWorklet.addModule("worklet.js");
  } catch (e) {
    return fail(`Microphone unavailable: ${e.message}`);
  }

  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => {
    ws.send(JSON.stringify({ type: "start", lang: $("lang").value, fillers }));
    const node = new AudioWorkletNode(ctx, "pcm16");
    node.port.onmessage = (e) => ws.readyState === WebSocket.OPEN && ws.send(e.data);
    ctx.createMediaStreamSource(mic).connect(node).connect(ctx.destination); // node outputs silence
  };
  ws.onmessage = (e) => handle(JSON.parse(e.data));
  ws.onerror = () => fail("Lost connection to the server");

  finalHTML = "";
  $("final").innerHTML = $("partial").innerHTML = "";
  $("error").hidden = true;
  document.body.dataset.level = $("level").value;
  show("live");
  setStatus("Listening");
}

function stop() {
  mic?.getTracks().forEach((t) => t.stop());
  ctx?.close();
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }));
  $("stop").disabled = true;
  setStatus("Building report...");
}

function handle(m) {
  if (m.type === "transcript") caption(m);
  else if (m.type === "metrics") meters(m);
  else if (m.type === "nudge") nudge(m.message);
  else if (m.type === "report") report(m);
}

function caption(m) {
  const html = wordsHTML(m.words.map((w) => ({ ...w, filler: fillerSet.has(norm(w.text)) })));
  if (m.kind === "final") {
    finalHTML += (finalHTML ? " " : "") + html;
    $("final").innerHTML = finalHTML;
    $("partial").innerHTML = "";
  } else {
    $("partial").innerHTML = html;
  }
  $("captions").scrollTop = $("captions").scrollHeight;
}

function meters(m) {
  const frac = m.elapsed / targetS;
  $("m-time").textContent = clock(m.elapsed);
  $("m-time").className = frac > 1 ? "bad" : frac > 0.9 ? "warn" : "";
  $("m-time-sub").textContent = frac > 1 ? `over by ${clock(m.elapsed - targetS)}` : frac > 0.9 ? "wrap up" : `of ${clock(targetS)}`;

  $("m-wpm").textContent = m.wpm ?? "-";
  $("m-wpm").className = m.wpm == null ? "" : m.wpm > band[1] || m.wpm < band[0] ? "warn" : "good";
  $("m-sps").textContent = m.sps ?? "-";

  const f = m.fillers, p = m.pauses;
  $("m-fillers").textContent = f.total;
  $("m-fillers-sub").textContent = `${f.per_min}/min ` +
    Object.entries(f.by_word).slice(0, 3).map(([w, n]) => `${w} ×${n}`).join(", ");
  $("m-pauses").textContent = p.good + p.long + p.freeze;
  $("m-pauses-sub").textContent = `${p.good} good · ${p.long} long · ${p.freeze} freeze`;

  setStatus(m.speaking ? "Speaking" : "Listening");
  $("latency").textContent = `lag ${m.lag_ms} ms · RTF ${m.rtf ?? "-"}`;
}

function nudge(text) {
  if (document.body.dataset.level === "silent") return;
  $("nudge").textContent = text;
  $("nudge").hidden = false;
  clearTimeout(nudgeTimer);
  nudgeTimer = setTimeout(() => ($("nudge").hidden = true), 4000);
}

function report(r) {
  const f = r.fillers, p = r.pauses.counts;
  const longPauses = r.pauses.items.filter((x) => x.kind !== "good");
  $("report").innerHTML = `
    <h2>Rehearsal report</h2>
    <div class="meters">
      ${stat("Duration", clock(r.duration))}
      ${stat("Average pace", `${r.avg_wpm} wpm`, `${r.avg_sps} syl/s · target ${r.pace_band[0]}-${r.pace_band[1]}`)}
      ${stat("Fillers", f.total, `${f.per_min}/min`)}
      ${stat("Pauses", p.good + p.long + p.freeze, `${p.good} good · ${p.long} long · ${p.freeze} freeze`)}
      ${stat("Hesitations", r.hesitations, "filler, then pause")}
    </div>
    <h3>Pace over time</h3>${paceChart(r.pace_timeline, r.pace_band)}
    <h3>Fillers</h3>
    <p>${Object.entries(f.by_word).map(([w, n]) => `<mark>${esc(w)}</mark> ×${n}`).join(" · ") || "None"}</p>
    <h3>Pauses over 2 s</h3>
    <p>${longPauses.map((x) => `${clock(x.start)}-${clock(x.end)} (${x.kind})`).join(" · ") || "None"}</p>
    <h3>Transcript</h3>
    <p class="transcript">${wordsHTML(r.transcript)}</p>
    <div class="actions">
      <a class="button" href="/api/sessions/${r.id}.md">Download Markdown</a>
      <button onclick="print()">Print / save as PDF</button>
      <button onclick="location.reload()">New rehearsal</button>
    </div>`;
  setStatus("Done");
  $("latency").textContent = "";
  show("report");
}

function paceChart(pts, [lo, hi]) {
  if (pts.length < 2) return "<p>Too short for a chart.</p>";
  const W = 600, H = 160, maxT = pts.at(-1).t, maxY = Math.max(220, ...pts.map((p) => p.wpm));
  const x = (t) => ((t / maxT) * W).toFixed(1), y = (v) => (H - (v / maxY) * H).toFixed(1);
  return `<svg class="chart" viewBox="0 0 ${W} ${H + 20}" role="img" aria-label="Pace over time">
    <rect x="0" y="${y(hi)}" width="${W}" height="${y(lo) - y(hi)}" class="band"/>
    <polyline points="${pts.map((p) => `${x(p.t)},${y(p.wpm)}`).join(" ")}" class="line"/>
    <text x="4" y="${y(hi) - 4}">target ${lo}-${hi} wpm</text>
    <text x="0" y="${H + 16}">0:00</text>
    <text x="${W}" y="${H + 16}" text-anchor="end">${clock(maxT)}</text>
  </svg>`;
}

const wordsHTML = (ws) => ws.map((w) => (w.filler ? `<mark>${esc(w.text)}</mark>` : esc(w.text))).join(" ");
const stat = (label, value, sub = "") => `<div class="meter"><span class="label">${label}</span><b>${value}</b><small>${sub}</small></div>`;
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const norm = (t) => t.toLowerCase().replace(/[^\p{L}\p{N}\s'-]/gu, "").replace(/^[-' ]+|[-' ]+$/g, "");
const esc = (t) => t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const setStatus = (t) => ($("status").textContent = t);

function show(id) {
  for (const s of ["setup", "live", "report"]) $(s).hidden = s !== id;
}

function fail(msg) {
  $("error").textContent = msg;
  $("error").hidden = false;
  show("setup");
}
