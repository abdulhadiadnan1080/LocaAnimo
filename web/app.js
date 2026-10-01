"use strict";

/* ============================================================================
   AnimoLocal Studio. One script object (S) is the source of truth; the Story
   builder and the YAML editor are two views of it.
   ========================================================================= */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const STORE = "animolocal.script.v1";
const PALETTE = ["#2F5BEA", "#E5484D", "#12A594", "#F76B15", "#8E4EC6", "#0090FF"];

let S = null;                 // the episode script
let OPT = null;               // enum options from the server
let VOICES = [];              // Kokoro English voices
let VOICE_MISSING = null;     // why voices can't run, if anything
let lastCheck = null;         // latest validation result
let job = null;               // the render job shown in the dock
let yamlText = "";            // what's in the editor
let storyEdited = true;       // story changed since the editor last synced

// ---------- tiny DOM helpers ----------------------------------------------------

function h(sel, props, ...kids) {
  const [tag, ...cls] = sel.split(".");
  const el = document.createElement(tag || "div");
  if (cls.length) el.className = cls.join(" ");
  if (props && (typeof props !== "object" || props instanceof Node || Array.isArray(props))) {
    kids.unshift(props); props = null;
  }
  for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid instanceof Node ? kid : String(kid));
  for (const [k, v] of Object.entries(props || {})) {   // after children, so select.value sticks
    if (v == null || v === false) continue;
    if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "class") el.className = (el.className + " " + v).trim();
    else if (k === "style") el.style.cssText = v;
    else if (k in el) el[k] = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  return el;
}

const ICONS = {
  play: '<path d="M7 5v14l12-7z" fill="currentColor" stroke="none"/>',
  up: '<path d="M6 15l6-6 6 6"/>',
  down: '<path d="M6 9l6 6 6-6"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  dup: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
  chev: '<path d="M8 10l4 4 4-4"/>',
  min: '<path d="M6 12h12"/>',
};
function icon(name) {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  s.setAttribute("viewBox", "0 0 24 24");
  s.innerHTML = ICONS[name];
  return s;
}
const iconBtn = (name, title, onclick, cls = "") =>
  h("button.icon-btn", { type: "button", title, onclick, class: cls || null }, icon(name));

const pretty = (s) => String(s).replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

function select(options, value, onchange, blank) {
  const el = h("select", { onchange: (e) => onchange(e.target.value) },
    blank != null ? h("option", { value: "" }, blank) : null,
    options.map((o) => (typeof o === "string" ? h("option", { value: o }, pretty(o)) : h("option", { value: o.value }, o.label))));
  el.value = value ?? "";
  return el;
}

function seg(options, value, onchange) {
  const wrap = h("div.seg");
  for (const [v, label] of options) {
    const b = h("button", { type: "button", class: v === value ? "on" : null, onclick: () => {
      $$("button", wrap).forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      onchange(v);
    } }, label);
    wrap.append(b);
  }
  return wrap;
}

const toggle = (label, checked, onchange) =>
  h("label.switch", null, h("input", { type: "checkbox", checked, onchange: (e) => onchange(e.target.checked) }), h("i"), label);

const field = (label, control) => h("div", null, h("label.field-label", null, label), control);

function textInput(value, placeholder, oninput, extra = {}) {
  return h("input", { type: "text", value: value ?? "", placeholder, spellcheck: true,
    oninput: (e) => oninput(e.target.value), ...extra });
}

let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 2800);
}

const api = {
  get: (u) => fetch(u).then((r) => r.json()),
  post: (u, body) => fetch(u, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
};

// ---------- script model ----------------------------------------------------------

const starter = () => ({
  episode: { title: "", style: "anime_tv", resolution: "1080p", music: true, subtitles: true, narrator_voice: "auto" },
  characters: [{ id: "hero", name: "Hero", look: "", age: "teen", gender: "female", voice: "auto" }],
  locations: [{ id: "home", look: "" }],
  scenes: [{ id: "s1", location: "home", time: "day", mood: "calm", beats: [{ camera: "establishing" }] }],
});

function normalize(s) {
  s.episode = { title: "", style: "anime_tv", resolution: "1080p", music: true, subtitles: true, narrator_voice: "auto", ...(s.episode || {}) };
  s.characters = (s.characters || []).map((c) => ({ age: "adult", gender: "neutral", voice: "auto", look: "", ...c }));
  s.locations = (s.locations || []).map((l) => ({ look: "", ...l }));
  s.scenes = (s.scenes || []).map((sc) => ({ time: "day", mood: "calm", beats: [], ...sc }));
  return s;
}

const slug = (s) => String(s).toLowerCase().trim().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "item";
function uniqueId(base, taken) {
  let id = base, n = 2;
  while (taken.includes(id)) id = `${base}_${n++}`;
  return id;
}
const charColor = (id) => PALETTE[Math.max(0, S.characters.findIndex((c) => c.id === id)) % PALETTE.length];
const charName = (id) => S.characters.find((c) => c.id === id)?.name || id;

function beatKind(b) {
  if (b.line) return "line";
  if ("action" in b) return "action";
  if ("narration" in b) return "narration";
  if ("pause" in b) return "pause";
  return "camera";
}

function* allBeats() {
  for (const sc of S.scenes) for (const b of sc.beats) yield b;
}

function renameCharacter(oldId, newId) {
  for (const b of allBeats()) {
    if (b.line?.who === oldId) b.line.who = newId;
    if (Array.isArray(b.characters)) b.characters = b.characters.map((c) => (c === oldId ? newId : c));
  }
}

function renameLocation(oldId, newId) {
  for (const sc of S.scenes) if (sc.location === oldId) sc.location = newId;
}

function changed({ rerender } = {}) {
  storyEdited = true;
  try { localStorage.setItem(STORE, JSON.stringify(S)); } catch {}
  if (rerender) rerender();
  scheduleCheck();
}

// ---------- voices -----------------------------------------------------------------

const voiceById = (id) => VOICES.find((v) => v.id === id);
const voiceLabel = (id) => voiceById(id)?.name || id;
const voiceMeta = (id) => { const v = voiceById(id); return v ? `${v.accent} ${v.gender}` : ""; };

function sampleFor(charId) {
  const line = [...allBeats()].find((b) => b.line?.who === charId && b.line.text?.trim());
  if (line) return { text: line.line.text, emotion: line.line.emotion || "neutral" };
  const c = S.characters.find((x) => x.id === charId);
  return { text: `Hi, I'm ${c?.name || "here"}. Nice to meet you!`, emotion: "neutral" };
}
function narratorSample() {
  const n = [...allBeats()].find((b) => b.narration?.trim());
  return { text: n ? n.narration : "Every story begins with a single moment.", emotion: "neutral" };
}

// One audio element for the whole app; a newer preview always wins.
const audio = new Audio();
const audioCache = new Map();
let audioToken = 0, playingEl = null;

function markPlaying(el, state) {
  if (playingEl && playingEl !== el) playingEl.classList.remove("playing", "loading");
  playingEl = el;
  if (!el) return;
  el.classList.remove("playing", "loading");
  if (state) el.classList.add(state);
}

async function speak(voice, text, emotion = "neutral", el = null) {
  if (VOICE_MISSING) return toast(VOICE_MISSING);
  if (!voice || !text?.trim()) return;
  const token = ++audioToken;
  audio.pause();
  markPlaying(el, "loading");
  const key = `${voice}|${emotion}|${text}`;
  let url = audioCache.get(key);
  if (!url) {
    const r = await api.post("/api/voices/preview", { voice, text, emotion });
    if (!r.ok) { markPlaying(null); const e = await r.json().catch(() => ({})); return toast(e.error || "Couldn't play that voice"); }
    url = URL.createObjectURL(await r.blob());
    audioCache.set(key, url);
  }
  if (token !== audioToken) return;     // something newer started meanwhile
  audio.src = url;
  audio.play().catch(() => {});
  markPlaying(el, "playing");
  audio.onended = () => { if (token === audioToken) markPlaying(null); el?.classList.remove("playing"); };
}

function stopAudio() { audioToken++; audio.pause(); if (playingEl) playingEl.classList.remove("playing", "loading"); playingEl = null; }

// Voice picker: a button that opens a list; hovering a voice plays it.
const voicePainters = new Set();

function voicePicker({ get, set, autoPick, sample }) {
  const btn = h("button.voice-btn", { type: "button" });
  const play = h("span.play", { title: "Listen" }, icon("play"));
  const name = h("span.v-name");
  const sub = h("span.v-sub");
  btn.append(play, h("span", null, name, h("br"), sub), h("span.chev", null, icon("chev")));

  const current = () => (get() === "auto" ? autoPick() : get());
  function paint() {
    const v = get(), real = current();
    name.textContent = v === "auto" ? (real ? `Auto · ${voiceLabel(real)}` : "Auto-cast") : voiceLabel(v);
    sub.textContent = real ? voiceMeta(real) : "Picked from age and gender";
  }
  paint();
  voicePainters.add(paint);

  play.addEventListener("click", (e) => {
    e.stopPropagation();
    const real = current();
    if (!real) return toast("Fix the script issues first, so a voice can be auto-cast");
    const s = sample();
    speak(real, s.text, s.emotion, btn);
  });
  btn.addEventListener("click", () => openVoicePop(btn, get(), (v) => { set(v); paint(); }, sample, autoPick));
  btn._paint = paint;
  return btn;
}

function openVoicePop(anchor, currentValue, onPick, sample, autoPick) {
  const pop = $("#voice-pop");
  pop.innerHTML = "";
  pop.append(h("div.hint", null, "Hover to listen · click to choose"));
  const rows = [];
  let hoverTimer = null;

  const addRow = (value, label, star) => {
    const row = h("div.voice-row", { class: value === currentValue ? "sel" : null },
      star ? h("span.star", null, "★") : null, h("span", null, label),
      h("span.eq", null, h("b"), h("b"), h("b")));
    const voice = value === "auto" ? autoPick() : value;
    const listen = () => { if (voice) { const s = sample(); speak(voice, s.text, s.emotion, row); } };
    row.addEventListener("mouseenter", () => { clearTimeout(hoverTimer); hoverTimer = setTimeout(listen, 320); highlight(rows.indexOf(row), false); });
    row.addEventListener("mouseleave", () => clearTimeout(hoverTimer));
    row.addEventListener("click", () => { onPick(value); listen(); close(); changed(); });
    row._listen = listen; row._value = value;
    rows.push(row);
    pop.append(row);
  };

  addRow("auto", `Auto${autoPick() ? ` (${voiceLabel(autoPick())})` : ""}`);
  let group = "";
  for (const v of VOICES) {
    const g = `${v.accent} · ${pretty(v.gender)}`;
    if (g !== group) { pop.append(h("div.group", null, g)); group = g; }
    addRow(v.id, v.name, v.featured);
  }

  let hl = rows.findIndex((r) => r._value === currentValue);
  function highlight(i, play = true) {
    rows.forEach((r) => r.classList.remove("hl"));
    hl = Math.max(0, Math.min(rows.length - 1, i));
    rows[hl].classList.add("hl");
    rows[hl].scrollIntoView({ block: "nearest" });
    if (play) { clearTimeout(hoverTimer); hoverTimer = setTimeout(rows[hl]._listen, 250); }
  }
  function onKey(e) {
    if (e.key === "ArrowDown") { e.preventDefault(); highlight(hl + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); highlight(hl - 1); }
    else if (e.key === "Enter") { e.preventDefault(); rows[hl]?.click(); }
    else if (e.key === "Escape") close();
  }
  function onOutside(e) { if (!pop.contains(e.target) && !anchor.contains(e.target)) close(); }
  function close() {
    pop.hidden = true;
    clearTimeout(hoverTimer);
    document.removeEventListener("keydown", onKey, true);
    document.removeEventListener("mousedown", onOutside, true);
    $(".views").removeEventListener("scroll", close);
  }

  pop.hidden = false;
  const r = anchor.getBoundingClientRect();
  const below = window.innerHeight - r.bottom;
  pop.style.left = `${Math.min(r.left, window.innerWidth - 300)}px`;
  pop.style.top = below > 300 ? `${r.bottom + 6}px` : "";
  pop.style.bottom = below > 300 ? "" : `${window.innerHeight - r.top + 6}px`;
  if (hl >= 0) { rows[hl].classList.add("hl"); rows[hl].scrollIntoView({ block: "center" }); }
  document.addEventListener("keydown", onKey, true);
  document.addEventListener("mousedown", onOutside, true);
  $(".views").addEventListener("scroll", close, { once: true });
}

// ---------- story view --------------------------------------------------------------

function renderAll() {
  voicePainters.clear();
  $("#title").value = S.episode.title || "";
  renderFeel(); renderCast(); renderPlaces(); renderScenes();
}

function renderFeel() {
  const e = S.episode;
  $("#feel").replaceChildren(
    field("Style", seg([["anime_tv", "Anime TV"], ["chibi", "Chibi"], ["cartoon", "Cartoon"]], e.style, (v) => { e.style = v; changed(); })),
    field("Quality", seg([["720p", "720p"], ["1080p", "1080p"]], e.resolution, (v) => { e.resolution = v; changed(); })),
    field("Extras", h("div", { style: "display:flex;gap:18px;padding:6px 0" },
      toggle("Music", e.music, (v) => { e.music = v; changed(); }),
      toggle("Subtitles", e.subtitles, (v) => { e.subtitles = v; changed(); }))),
    h("div", { style: "min-width:250px" }, h("label.field-label", null, "Narrator voice"),
      voicePicker({ get: () => e.narrator_voice, set: (v) => (e.narrator_voice = v),
        autoPick: () => lastCheck?.summary?.narrator, sample: narratorSample })),
  );
}

function renderCast() {
  const wrap = $("#cast");
  wrap.replaceChildren();
  S.characters.forEach((c, i) => {
    const avatar = h("div.avatar", { style: `background:${PALETTE[i % PALETTE.length]}` }, (c.name || "?")[0].toUpperCase());
    const nameInput = textInput(c.name, "Name", (v) => { c.name = v; avatar.textContent = (v || "?")[0].toUpperCase(); changed(); }, {
      onchange: () => {
        const newId = uniqueId(slug(c.name), S.characters.filter((x) => x !== c).map((x) => x.id));
        if (newId !== c.id) { renameCharacter(c.id, newId); c.id = newId; }
        changed({ rerender: renderScenes });
      },
    });
    wrap.append(h("div.card.entity", null,
      h("div.entity-head", null, avatar, nameInput,
        iconBtn("x", "Remove character", () => removeCharacter(c), "danger")),
      field("Look", h("textarea.field", { value: c.look, rows: 2, placeholder: "teenage girl, silver ponytail, white surf jacket…",
        oninput: (e) => { c.look = e.target.value; changed(); } })),
      h("div.row", null,
        field("Age", select(OPT.ages, c.age, (v) => { c.age = v; changed(); })),
        field("Gender", select(OPT.genders, c.gender, (v) => { c.gender = v; changed(); }))),
      field("Voice", voicePicker({ get: () => c.voice, set: (v) => (c.voice = v),
        autoPick: () => lastCheck?.summary?.cast?.[c.id], sample: () => sampleFor(c.id) })),
    ));
  });
  wrap.append(h("button.add-card", { type: "button", onclick: addCharacter }, "+ Add character"));
}

function addCharacter() {
  const id = uniqueId("character", S.characters.map((c) => c.id));
  S.characters.push({ id, name: "", look: "", age: "teen", gender: "female", voice: "auto" });
  changed({ rerender: () => { renderCast(); renderScenes(); } });
  $$("#cast .entity-head input").at(-1)?.focus();
}

function removeCharacter(c) {
  const lines = [...allBeats()].filter((b) => b.line?.who === c.id).length;
  if (lines && !confirm(`${c.name || "This character"} has ${lines} line(s). Remove them too?`)) return;
  for (const sc of S.scenes) {
    sc.beats = sc.beats.filter((b) => b.line?.who !== c.id);
    for (const b of sc.beats) if (Array.isArray(b.characters)) b.characters = b.characters.filter((x) => x !== c.id);
  }
  S.characters = S.characters.filter((x) => x !== c);
  changed({ rerender: () => { renderCast(); renderScenes(); } });
}

function placeColor(id) {
  let hash = 0;
  for (const ch of id) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return `linear-gradient(135deg, hsl(${hash % 360} 60% 62%), hsl(${(hash + 30) % 360} 55% 42%))`;
}

function renderPlaces() {
  const wrap = $("#places");
  wrap.replaceChildren();
  for (const l of S.locations) {
    const sw = h("div.place-swatch", { style: `background:${placeColor(l.id)}` });
    wrap.append(h("div.card.entity", null,
      h("div.entity-head", null, sw,
        textInput(l.id, "Name, e.g. beach", () => {}, {
          onchange: (e) => {
            const newId = uniqueId(slug(e.target.value), S.locations.filter((x) => x !== l).map((x) => x.id));
            e.target.value = newId;
            if (newId !== l.id) { renameLocation(l.id, newId); l.id = newId; sw.style.background = placeColor(newId); }
            changed({ rerender: renderScenes });
          },
        }),
        iconBtn("x", "Remove place", () => removeLocation(l), "danger")),
      field("Look", h("textarea.field", { value: l.look, rows: 3, placeholder: "empty beach at sunset, big waves, wooden pier…",
        oninput: (e) => { l.look = e.target.value; changed(); } })),
    ));
  }
  wrap.append(h("button.add-card", { type: "button", onclick: () => {
    const id = uniqueId("place", S.locations.map((l) => l.id));
    S.locations.push({ id, look: "" });
    changed({ rerender: () => { renderPlaces(); renderScenes(); } });
  } }, "+ Add place"));
}

function removeLocation(l) {
  const used = S.scenes.findIndex((sc) => sc.location === l.id);
  if (used >= 0) return toast(`Scene ${used + 1} takes place here. Change that scene's place first.`);
  S.locations = S.locations.filter((x) => x !== l);
  changed({ rerender: renderPlaces });
}

const move = (arr, i, d) => { const j = i + d; if (j < 0 || j >= arr.length) return false; [arr[i], arr[j]] = [arr[j], arr[i]]; return true; };

function renderScenes() {
  const wrap = $("#scenes");
  wrap.replaceChildren();
  const locs = S.locations.map((l) => l.id);
  S.scenes.forEach((sc, i) => {
    const beats = h("div.beats");
    sc.beats.forEach((b, j) => beats.append(beatRow(sc, b, j)));
    const addBtn = (kind, label) => h("button", { type: "button", onclick: () => addBeat(sc, kind) }, label);
    wrap.append(h("div.card.scene", null,
      h("div.scene-head", null,
        h("span.scene-no", null, `Scene ${i + 1}`),
        select(locs, sc.location, (v) => { sc.location = v; changed(); }, locs.includes(sc.location) ? null : "Choose a place"),
        seg([["day", "Day"], ["sunset", "Sunset"], ["night", "Night"]], sc.time, (v) => { sc.time = v; changed(); }),
        select(OPT.moods.map((m) => ({ value: m, label: `${pretty(m)} music` })), sc.mood, (v) => { sc.mood = v; changed(); }),
        h("div.spacer"),
        iconBtn("up", "Move scene up", () => move(S.scenes, i, -1) && changed({ rerender: renderScenes })),
        iconBtn("down", "Move scene down", () => move(S.scenes, i, 1) && changed({ rerender: renderScenes })),
        iconBtn("dup", "Duplicate scene", () => {
          const copy = structuredClone(sc);
          copy.id = uniqueId(`s${S.scenes.length + 1}`, S.scenes.map((s) => s.id));
          S.scenes.splice(i + 1, 0, copy);
          changed({ rerender: renderScenes });
        }),
        iconBtn("x", "Delete scene", () => {
          if (sc.beats.length && !confirm(`Delete scene ${i + 1} and its ${sc.beats.length} beat(s)?`)) return;
          S.scenes.splice(i, 1);
          changed({ rerender: renderScenes });
        }, "danger")),
      beats,
      h("div.add-beats", null, addBtn("line", "+ Line"), addBtn("action", "+ Action"),
        addBtn("narration", "+ Narration"), addBtn("camera", "+ Camera"), addBtn("pause", "+ Pause")),
    ));
  });
}

function addBeat(sc, kind) {
  const first = S.characters[0]?.id || "";
  const fresh = {
    line: { line: { who: first, text: "", emotion: "neutral" } },
    action: { action: "", characters: [] },
    narration: { narration: "" },
    camera: { camera: "wide" },
    pause: { pause: 1 },
  }[kind];
  sc.beats.push(fresh);
  changed({ rerender: renderScenes });
  const scenes = $$("#scenes .scene");
  const last = $$(".beat", scenes[S.scenes.indexOf(sc)]).at(-1);
  $("input[type=text], input[type=number]", last)?.focus();
}

const BAR = { action: "#9ca3af", narration: "#8e4ec6", camera: "#0090ff", pause: "#d1d5db" };

function beatRow(sc, b, j) {
  const kind = beatKind(b);
  const bar = h("div.bar", { style: `background:${kind === "line" ? charColor(b.line.who) : kind === "action" && b.big ? "var(--big)" : BAR[kind]}` });
  const body = h("div.body");

  if (kind === "line") {
    const ln = b.line;
    const say = h("button.say", { type: "button", title: "Hear this line" }, icon("play"));
    say.addEventListener("click", () => {
      const voice = S.characters.find((c) => c.id === ln.who)?.voice;
      const real = voice && voice !== "auto" ? voice : lastCheck?.summary?.cast?.[ln.who];
      if (!real) return toast("Fix the script issues first, so this character gets a voice");
      speak(real, ln.text, ln.emotion || "neutral", say);
    });
    body.append(
      select(S.characters.map((c) => ({ value: c.id, label: c.name || c.id })), ln.who,
        (v) => { ln.who = v; bar.style.background = charColor(v); changed(); }, S.characters.some((c) => c.id === ln.who) ? null : "Who?"),
      textInput(ln.text, "What do they say?", (v) => { ln.text = v; changed(); }),
      select(OPT.emotions, ln.emotion || "neutral", (v) => { ln.emotion = v; changed(); }),
      select(OPT.gestures, ln.gesture || "", (v) => { if (v) ln.gesture = v; else delete ln.gesture; changed(); }, "No gesture"),
      say,
    );
  } else if (kind === "narration") {
    const say = h("button.say", { type: "button", title: "Hear the narrator" }, icon("play"));
    say.addEventListener("click", () => {
      const v = S.episode.narrator_voice !== "auto" ? S.episode.narrator_voice : lastCheck?.summary?.narrator;
      if (!v) return toast("Fix the script issues first, so the narrator gets a voice");
      speak(v, b.narration, "neutral", say);
    });
    body.append(textInput(b.narration, "Narrator says…", (v) => { b.narration = v; changed(); }), say);
  } else if (kind === "action") {
    b.characters ||= [];
    const chips = h("div.chips", null, S.characters.map((c) => {
      const on = b.characters.includes(c.id);
      const chip = h("button.chip", { type: "button", class: on ? "on" : null, style: on ? `background:${charColor(c.id)}` : null,
        onclick: () => {
          b.characters = b.characters.includes(c.id) ? b.characters.filter((x) => x !== c.id) : [...b.characters, c.id];
          const nowOn = b.characters.includes(c.id);
          chip.classList.toggle("on", nowOn);
          chip.style.background = nowOn ? charColor(c.id) : "";
          changed();
        } }, c.name || c.id);
      return chip;
    }));
    const big = h("button.big-toggle", { type: "button", class: b.big ? "on" : null,
      title: "Animate this moment with AI video. Keep it to 4–6 per 10 minutes.",
      onclick: () => { b.big = !b.big; if (!b.big) delete b.big; big.classList.toggle("on", !!b.big);
        bar.style.background = b.big ? "var(--big)" : BAR.action; changed(); } }, "⚡ AI motion");
    body.append(
      textInput(b.action, "What happens?", (v) => { b.action = v; changed(); }, { style: "flex:1 1 100%" }),
      chips, big,
      select(OPT.cameras, b.camera || "", (v) => { if (v) b.camera = v; else delete b.camera; changed(); }, "Auto camera"),
      textInput((b.sfx || []).join(", "), "Sound effects, e.g. whoosh, wave_roar", (v) => {
        const list = v.split(",").map((x) => x.trim()).filter(Boolean);
        if (list.length) b.sfx = list; else delete b.sfx;
        changed();
      }, { style: "flex:1 1 220px" }),
    );
  } else if (kind === "camera") {
    body.append(select(OPT.cameras, b.camera, (v) => { b.camera = v; changed(); }));
  } else if (kind === "pause") {
    body.append(h("input.num", { type: "number", min: 0.1, max: 10, step: 0.1, value: b.pause,
      oninput: (e) => { b.pause = parseFloat(e.target.value) || 0; changed(); } }), h("span.muted", { style: "padding-top:7px" }, "seconds of silence"));
  }

  return h("div.beat", null, bar, h("div.kind", null, kind), body,
    h("div.tools", null,
      iconBtn("up", "Move up", () => move(sc.beats, j, -1) && changed({ rerender: renderScenes })),
      iconBtn("down", "Move down", () => move(sc.beats, j, 1) && changed({ rerender: renderScenes })),
      iconBtn("x", "Delete", () => { sc.beats.splice(j, 1); changed({ rerender: renderScenes }); }, "danger")));
}

// ---------- validation & status pill ---------------------------------------------------

let checkTimer;
function scheduleCheck() {
  clearTimeout(checkTimer);
  checkTimer = setTimeout(check, 450);
}

async function check() {
  const r = await api.post("/api/validate", { script: S });
  lastCheck = await r.json();
  paintStatus();
  for (const paint of voicePainters) paint();
}

const fmtMin = (m) => (m < 1 ? `${Math.max(1, Math.round(m * 60))}s` : `${m.toFixed(1)} min`);

function paintStatus() {
  const pill = $("#status");
  const running = job && ["queued", "running"].includes(job.status);
  if (!lastCheck) { pill.className = "pill"; pill.textContent = "Checking…"; }
  else if (lastCheck.ok) {
    const s = lastCheck.summary;
    pill.className = "pill ok";
    pill.textContent = `Ready · ~${fmtMin(s.minutes)} · ${s.beats} beats${s.big ? ` · ${s.big} ⚡` : ""}`;
  } else {
    const n = lastCheck.errors.length;
    pill.className = "pill warn";
    pill.textContent = `${n} thing${n === 1 ? "" : "s"} to fix`;
  }
  const create = $("#create");
  create.disabled = !lastCheck?.ok || running || !!VOICE_MISSING;
  $("#storyboard").disabled = create.disabled;
  $("span", create).textContent = running ? `Creating… ${Math.round(job.progress * 100)}%` : "Create episode";
}

function friendlyWhere(where) {
  const m = where.match(/^scenes\.(\d+)(?:\.beats\.(\d+))?/);
  if (m) return `Scene ${+m[1] + 1}${m[2] != null ? `, beat ${+m[2] + 1}` : ""}`;
  const c = where.match(/^characters\.(\d+)/);
  if (c) return `Character ${+c[1] + 1}`;
  const l = where.match(/^locations\.(\d+)/);
  if (l) return `Place ${+l[1] + 1}`;
  return where;
}

function toggleIssues() {
  const pop = $("#issues");
  if (!pop.hidden || !lastCheck || lastCheck.ok) { pop.hidden = true; return; }
  pop.replaceChildren(h("h4", null, "Before you can create the episode"),
    h("ul", null, lastCheck.errors.map((e) => h("li", null, e.where ? h("span.where", null, friendlyWhere(e.where)) : null, e.message))));
  const r = $("#status").getBoundingClientRect();
  pop.style.top = `${r.bottom + 8}px`;
  pop.style.left = `${Math.max(12, r.right - 360)}px`;
  pop.hidden = false;
}

// ---------- render job dock --------------------------------------------------------------

let pollTimer;
async function createEpisode(mode = "full") {
  if ($("#create").disabled) return;
  const r = await api.post("/api/jobs", { script: S, mode });
  const data = await r.json();
  if (!r.ok) { lastCheck = { ok: false, errors: data.errors || [{ where: "", message: data.error }] }; paintStatus(); return toggleIssues(); }
  job = data.job;
  $("#dock").classList.remove("collapsed");
  renderDock(); paintStatus(); poll();
}

async function poll() {
  clearTimeout(pollTimer);
  if (!job) return;
  const data = await api.get(`/api/jobs/${job.id}`);
  const was = job.status;
  job = data.job;
  renderDock(); paintStatus();
  if (["queued", "running"].includes(job.status)) pollTimer = setTimeout(poll, 700);
  else if (was !== job.status && job.status === "done") { toast("Your episode is ready"); renderLibrary(); }
}

const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function renderDock() {
  const dock = $("#dock");
  if (!job) { dock.hidden = true; return; }
  dock.hidden = false;
  dock.classList.toggle("done", job.status === "done");
  dock.classList.toggle("failed", job.status === "failed");
  const running = ["queued", "running"].includes(job.status);
  const elapsed = ((job.finished || Date.now() / 1000) - (job.started || job.created));
  const eta = running && job.progress > 0.08 ? ` · ~${clock(elapsed * (1 - job.progress) / job.progress)} left` : "";
  const headline = { queued: "Waiting…", running: "Creating", done: "Ready", failed: "Failed", cancelled: "Cancelled" }[job.status];

  dock.replaceChildren(
    h("div.dock-head", null, h("strong", null, `${headline} · ${job.title || "Untitled"}`),
      iconBtn(dock.classList.contains("collapsed") ? "up" : "min", "Show/hide steps", () => { dock.classList.toggle("collapsed"); renderDock(); }),
      running ? null : iconBtn("x", "Dismiss", () => { job = null; renderDock(); paintStatus(); })),
    h("div.dock-msg", null, h("span", null, job.status === "failed" ? job.error : job.message), h("span", null, clock(elapsed) + eta)),
    h("div.progress", null, h("i", { style: `width:${Math.round(job.progress * 100)}%` })),
    h("ul.stages", null, job.stages.map((s) => h("li", { class: s.state }, h("span.st"),
      h("span", null, s.label, s.implemented ? null : h("span.soon", null, "SOON"), s.detail ? h("span.detail", null, s.detail) : null)))),
    h("div.dock-actions", null,
      running ? h("button.ghost", { type: "button", onclick: () => api.post(`/api/jobs/${job.id}/cancel`, {}) }, "Cancel") : null,
      job.status === "done" ? h("button.primary", { type: "button", onclick: () => openPlayer(job.video.replace(/\.mp4$/, "")) }, icon("play"), "Watch") : null,
      job.status === "done" ? h("button.ghost", { type: "button", onclick: () => reveal(job.video.replace(/\.mp4$/, "")) }, "Show in Finder") : null),
  );
}

// ---------- library & player ---------------------------------------------------------------

let LIB = [];
async function renderLibrary() {
  LIB = (await api.get("/api/library")).items;
  const wrap = $("#library");
  if (!LIB.length) {
    wrap.replaceChildren(h("div.empty", { style: "grid-column:1/-1" },
      h("strong", null, "No episodes yet"), "Build a story and press Create episode. Finished videos show up here."));
    return;
  }
  wrap.replaceChildren(...LIB.map((ep) => h("div.card.ep", { onclick: () => openPlayer(ep.id) },
    h("div.thumb", { style: ep.thumb ? `background-image:url('/media/${encodeURIComponent(ep.thumb)}')` : null },
      h("span.dur", null, clock(ep.duration)), ep.kind === "animatic" ? h("span.tag", null, "Animatic") : null),
    h("div.info", null, h("strong", null, ep.title || "Untitled"),
      h("span.muted", null, `${new Date(ep.created * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })} · ${ep.size_mb} MB`)))));
}

async function openPlayer(id) {
  if (!LIB.some((e) => e.id === id)) await renderLibrary();
  const ep = LIB.find((e) => e.id === id);
  if (!ep) return toast("Couldn't find that episode");
  stopAudio();
  const dlg = $("#player"), video = $("#player-video");
  video.src = `/media/${encodeURIComponent(ep.video)}`;
  $("#player-title").textContent = ep.title || "Untitled";
  $("#player-meta").textContent = `${clock(ep.duration)} · ${ep.scenes} scene${ep.scenes === 1 ? "" : "s"}`;
  $("#player-reveal").onclick = () => reveal(id);
  dlg.showModal();
  video.play().catch(() => {});
}

const reveal = (id) => api.post(`/api/library/${encodeURIComponent(id)}/reveal`, {});

// ---------- script (YAML) view -----------------------------------------------------------------

async function syncEditorFromStory() {
  if (!storyEdited && yamlText) return;
  const data = await (await api.post("/api/yaml/dump", { script: S })).json();
  yamlText = data.yaml;
  $("#yaml").value = yamlText;
  storyEdited = false;
  $("#yaml-state").textContent = "In sync with Story";
}

let parseTimer;
function onYamlInput() {
  yamlText = $("#yaml").value;
  clearTimeout(parseTimer);
  parseTimer = setTimeout(async () => {
    const r = await api.post("/api/yaml/parse", { yaml: yamlText });
    const data = await r.json();
    if (!r.ok) {
      $("#yaml-state").textContent = `⚠ ${data.error}`;
      lastCheck = { ok: false, errors: [{ where: "", message: `The script text has a formatting problem: ${data.error}` }] };
      return paintStatus();
    }
    S = normalize(data.script);
    try { localStorage.setItem(STORE, JSON.stringify(S)); } catch {}
    storyEdited = false;
    storyStale = true;
    $("#yaml-state").textContent = "In sync with Story";
    check();
  }, 400);
}

let storyStale = false;

function renderGuide() {
  const tags = (list) => h("div.tags", null, list.map((t) => h("code", { title: "Click to copy",
    onclick: () => { navigator.clipboard?.writeText(t); toast(`Copied “${t}”`); } }, t)));
  $("#guide").replaceChildren(
    h("h3", null, "How to write a script"),
    h("p", null, "A script has four parts. ", h("b", null, "episode"), " sets the title and look. ",
      h("b", null, "characters"), " lists who appears: an id you reuse, a name, a look (comma-separated visual tags work best), age, gender and a voice (or ", h("code", null, "auto"), "). ",
      h("b", null, "locations"), " lists places, each with an id and a look. ",
      h("b", null, "scenes"), " play in order. Each one picks a location, time and mood, then lists its ", h("b", null, "beats"), ":"),
    h("p", null, h("code", null, "line"), " (who, text, emotion, optional gesture), ",
      h("code", null, "action"), " (what happens; add ", h("code", null, "big: true"), " for an AI-animated moment, at most 4–6 per 10 minutes), ",
      h("code", null, "narration"), ", ", h("code", null, "camera"), " and ", h("code", null, "pause"), " (seconds)."),
    h("p.muted", null, "Edits here and in Story stay in sync. Paste a whole script, and the Story view picks it up."),
    h("h4", null, "Shape"),
    h("pre", null, `episode: { title: "…", style: anime_tv }
characters:
  - { id: kai, name: Kai, look: "…", age: teen, gender: male, voice: auto }
locations:
  - { id: beach, look: "…" }
scenes:
  - id: s1
    location: beach
    beats:
      - camera: establishing
      - line: { who: kai, text: "…", emotion: happy }
      - action: "Kai jumps off the pier"
        characters: [kai]
        big: true
      - pause: 1.0`),
    h("h4", null, "Emotions"), tags(OPT.emotions),
    h("h4", null, "Gestures"), tags(OPT.gestures),
    h("h4", null, "Camera"), tags(OPT.cameras),
    h("h4", null, "Times · moods · styles"), tags([...OPT.times, ...OPT.moods, ...OPT.styles]),
    h("h4", null, "Voices"), tags(VOICES.map((v) => v.id)),
  );
}

// ---------- plain-text import ------------------------------------------------------------------------

const TEXT_STORE = "animolocal.storytext.v1";
const EXAMPLE_TEXT = `Title: The Last Wave

Characters:
Kai is a teenage boy with messy black hair, blue eyes and an orange hoodie. Slim build.
Mira is a teenage girl with long silver hair in a ponytail, green eyes and a white surf jacket.

Scene 1: An empty beach at sunset, big waves rolling in, a wooden pier in the distance. Calm.
Narrator: Every summer, the biggest wave came on the last day.
Kai walks along the shore carrying his surfboard, footsteps in the sand.
KAI (determined): Today's the day. I can feel it.
Mira runs up behind him, waving.
MIRA (worried): Kai! You're not seriously going out there?
KAI (confident, pointing at the ocean): Watch me.

Scene 2: Same beach, epic.
A giant wave rises over the ocean with a roar.
BIG MOMENT: Kai paddles hard, then leaps up and rides the giant wave, spray flying everywhere.
MIRA (amazed): He's actually doing it!`;

let importTask = null, importTimer = null;

function setMode(mode) {
  $$("#script-mode button").forEach((b) => b.classList.toggle("on", b.dataset.mode === mode));
  $("#mode-text").hidden = mode !== "text";
  $("#mode-yaml").hidden = mode !== "yaml";
  $("#mode-hint").textContent = mode === "text"
    ? "Paste your story in any format, and the fields fill in automatically."
    : "The exact script the pipeline runs. Edits here and in Story stay in sync.";
  if (mode === "yaml") syncEditorFromStory();
}

function renderTextGuide(result) {
  const parts = [];
  if (result) parts.push(result);
  parts.push(
    h("h3", null, "What to include"),
    h("p", null, "Write it however you like, as a screenplay, a story or rough notes. The local story model reads it and fills in every field. The more you describe, the less it has to guess."),
    h("ul", null,
      h("li", null, h("b", null, "Title"), " of the episode."),
      h("li", null, h("b", null, "Characters"), ": name, what they look like (hair, eyes, clothes), age and gender."),
      h("li", null, h("b", null, "Places"), ": what each location looks like."),
      h("li", null, h("b", null, "Scenes"), ": where and when (day, sunset, night) and the mood."),
      h("li", null, h("b", null, "Dialogue"), " as ", h("code", null, "NAME (emotion): line"), "."),
      h("li", null, h("b", null, "Actions"), " as plain sentences. Mark the spectacular one with ", h("code", null, "BIG MOMENT:"), " to get AI motion."),
      h("li", null, h("b", null, "Narration"), " as ", h("code", null, "Narrator: …"), ".")),
    h("p.muted", null, "Dialogue is kept word for word. Anything the model had to invent, like a missing hairstyle, is listed after it finishes, so you can check it in Story."),
    h("p.muted", null, "A structured YAML script pasted here loads instantly, with no model needed."),
  );
  $("#text-guide").replaceChildren(...parts);
}

async function fillFromText() {
  const text = $("#story-text").value.trim();
  if (!text) return toast("Paste your story first");
  stopImport();
  const r = await api.post("/api/import", { text });
  const data = await r.json();
  if (!r.ok) return toast(data.error || "Couldn't start reading");
  importTask = data;
  $("#fill-progress").hidden = false;
  $("#fill-cancel").hidden = false;
  $("#fill").disabled = true;
  $("#fill span").textContent = "Reading…";
  pollImport();
}

async function pollImport() {
  clearTimeout(importTimer);
  if (!importTask) return;
  const t = importTask = await api.get(`/api/import/${importTask.id}`);
  const pct = t.status === "done" ? 100 : Math.min(95, Math.round(100 * t.progress / t.expected));
  $("#fill-bar").style.width = `${pct}%`;
  $("#fill-msg").textContent = t.progress ? `Writing the script… ${pct}%` : "Loading the story model…";
  if (t.status === "running") { importTimer = setTimeout(pollImport, 600); return; }
  finishImport(t);
}

function finishImport(t) {
  $("#fill-progress").hidden = true;
  $("#fill-cancel").hidden = true;
  $("#fill").disabled = false;
  $("#fill span").textContent = "✨ Fill in fields";
  importTask = null;
  if (t.status === "cancelled") return;
  if (t.status !== "done") {
    renderTextGuide(h("div.result.bad", null, h("strong", null, "Couldn't fill in the fields"), t.error || "Unknown problem"));
    return;
  }
  // Keep settings the text doesn't cover: chosen voices (matched by name) and output options.
  const oldVoices = Object.fromEntries(S.characters.filter((c) => c.voice !== "auto").map((c) => [c.name.toLowerCase(), c.voice]));
  const next = normalize(t.script);
  next.episode = { ...S.episode, title: next.episode.title, style: next.episode.style };
  for (const c of next.characters) if (oldVoices[c.name.toLowerCase()]) c.voice = oldVoices[c.name.toLowerCase()];
  S = next;
  storyEdited = true;
  changed();
  renderAll();
  const beats = S.scenes.reduce((n, s) => n + s.beats.length, 0);
  const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  renderTextGuide(h("div.result", null,
    h("strong", null, `Filled in ${plural(S.characters.length, "character")}, ${plural(S.locations.length, "place")}, ${plural(S.scenes.length, "scene")} and ${plural(beats, "beat")}`),
    t.via === "structured" ? h("span.muted", null, "Read as a structured script.") : null,
    t.notes?.length ? h("ul", null, t.notes.map((n) => h("li", null, n))) : null,
    h("button.primary", { type: "button", onclick: () => show("story"), style: "margin-top:6px" }, "Review in Story →")));
  toast("Fields filled in");
}

function stopImport() {
  if (importTask) api.post(`/api/import/${importTask.id}/cancel`, {});
  clearTimeout(importTimer);
}

// ---------- navigation & boot ---------------------------------------------------------------------

function show(view) {
  try { localStorage.setItem("locaanimo.view", view); } catch {}
  $$(".nav[data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  stopAudio();
  if (view === "script" && !$("#mode-yaml").hidden) syncEditorFromStory();
  if (view === "story" && storyStale) { storyStale = false; renderAll(); }
  if (view === "library") renderLibrary();
}

async function loadTemplate() {
  const t = await api.get("/api/template");
  S = normalize(t.script);
  storyEdited = true;
  changed();
  renderAll();
}

async function boot() {
  const [opt, voices, latest] = await Promise.all([api.get("/api/options"), api.get("/api/voices"), api.get("/api/jobs/latest")]);
  OPT = opt; VOICES = voices.voices; VOICE_MISSING = voices.missing;

  const engine = $("#engine");
  engine.classList.add(VOICE_MISSING ? "bad" : "ok");
  engine.title = VOICE_MISSING || `Voice engine ready · ${VOICES.length} English voices`;
  if (VOICE_MISSING) { const b = $("#banner"); b.hidden = false; b.textContent = `Voices aren't ready: ${VOICE_MISSING}`; }

  let saved = null;
  try { saved = JSON.parse(localStorage.getItem(STORE)); } catch {}
  if (saved?.episode) S = normalize(saved);
  else { S = normalize((await api.get("/api/template")).script); }

  renderAll(); renderGuide(); renderTextGuide(); check();

  try { $("#story-text").value = localStorage.getItem(TEXT_STORE) || ""; } catch {}
  $("#story-text").addEventListener("input", (e) => { try { localStorage.setItem(TEXT_STORE, e.target.value); } catch {} });
  $$("#script-mode button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $("#fill").addEventListener("click", fillFromText);
  $("#fill-cancel").addEventListener("click", () => { stopImport(); finishImport({ status: "cancelled" }); });
  $("#text-example").addEventListener("click", () => {
    $("#story-text").value = EXAMPLE_TEXT;
    try { localStorage.setItem(TEXT_STORE, EXAMPLE_TEXT); } catch {}
  });
  $("#text-clear").addEventListener("click", () => { $("#story-text").value = ""; try { localStorage.removeItem(TEXT_STORE); } catch {} });

  $$(".nav[data-view]").forEach((b) => b.addEventListener("click", () => show(b.dataset.view)));
  $("#title").addEventListener("input", (e) => { S.episode.title = e.target.value; changed(); });
  $("#status").addEventListener("click", toggleIssues);
  document.addEventListener("mousedown", (e) => { if (!$("#issues").contains(e.target) && e.target !== $("#status")) $("#issues").hidden = true; });
  $("#create").addEventListener("click", () => createEpisode("full"));
  $("#storyboard").addEventListener("click", () => createEpisode("storyboard"));
  $("#add-scene").addEventListener("click", () => {
    S.scenes.push({ id: uniqueId(`s${S.scenes.length + 1}`, S.scenes.map((s) => s.id)), location: S.locations[0]?.id || "", time: "day", mood: "calm", beats: [] });
    changed({ rerender: renderScenes });
  });
  $("#new-btn").addEventListener("click", () => {
    if (!confirm("Start a new, blank episode? The current one will be replaced (export it from Script first if you want to keep it).")) return;
    S = starter(); changed(); renderAll(); show("story");
  });
  $("#yaml").addEventListener("input", onYamlInput);
  $("#yaml").addEventListener("keydown", (e) => {          // two-space indent instead of focus jump
    if (e.key !== "Tab") return;
    e.preventDefault();
    const t = e.target, s = t.selectionStart;
    t.setRangeText("  ", s, t.selectionEnd, "end");
    onYamlInput();
  });
  $("#use-example").addEventListener("click", async () => {
    if (!confirm("Replace the current script with the example episode?")) return;
    await loadTemplate(); storyEdited = true; syncEditorFromStory();
  });
  $("#download").addEventListener("click", async () => {
    await syncEditorFromStory();
    const name = `${slug(S.episode.title || "episode")}.yaml`;
    const a = h("a", { href: URL.createObjectURL(new Blob([$("#yaml").value], { type: "text/yaml" })), download: name });
    a.click();
  });
  $("#load-file").addEventListener("click", () => $("#file-input").click());
  $("#file-input").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    $("#yaml").value = await file.text();
    onYamlInput();
    e.target.value = "";
  });
  $("#player-close").addEventListener("click", () => $("#player").close());
  $("#player").addEventListener("close", () => $("#player-video").pause());
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") { e.preventDefault(); createEpisode(); }
  });

  if (latest.job && ["queued", "running"].includes(latest.job.status)) { job = latest.job; renderDock(); poll(); }
  const v = new URLSearchParams(location.search).get("view");
  if (v) show(v);
}

boot();
