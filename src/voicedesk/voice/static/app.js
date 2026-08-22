import { createEnergyVAD } from "./vad.js";
import {
  IDLE, LISTENING, THINKING, SPEAKING, initialState, next,
} from "./turn-state.js";

const talk = document.getElementById("talk");
const transcriptEl = document.getElementById("transcript");
const replyEl = document.getElementById("reply");
const timingsEl = document.getElementById("timings");

// One session per page load, so the agent remembers this caller across turns.
const sessionId = crypto.randomUUID();

let lang = "en";

let state = initialState("hands-free");
let vad = null;
let audioCtx = null;
let analyser = null;
let micStream = null;
let recorder = null;
let chunks = [];
let rafId = null;
// Tracks the previous frame's vad.isPending() so tick() can detect the edges
// (nascent turn just began / just ended) rather than re-testing every frame.
let wasPending = false;

// Live diagnostics, enabled with ?debug=1 only. Echo rejection is a heuristic
// tuned against real hardware, and the numbers it acts on are invisible
// otherwise -- this shows why a barge-in fired, or why it did not. Never
// rendered for an ordinary visitor.
const debugEl = new URLSearchParams(location.search).get("debug") === "1"
  ? Object.assign(document.createElement("pre"), {
      style: "position:fixed;top:0;right:0;margin:0;padding:.5rem;"
           + "background:#111;color:#0f0;font:11px/1.4 ui-monospace,monospace;"
           + "z-index:9999;white-space:pre;text-align:left",
    })
  : null;
if (debugEl) document.body.appendChild(debugEl);
let postCount = 0;

function renderDebug(rms) {
  const d = vad.debug();
  const n = (v) => (v === null || v === undefined ? "  --  " : v.toFixed(4));
  debugEl.textContent = [
    `state      ${state.name}${state.capturing ? " +REC" : ""}`,
    `strict     ${state.name === SPEAKING}`,
    `rms        ${n(rms)}`,
    `floor      ${n(d.echoFloor)}`,
    `needs      ${n(d.requirement)}`,
    `over?      ${d.requirement !== null && rms >= d.requirement}`,
    `pending    ${d.pending}`,
    `turns sent ${postCount}`,
  ].join("\n");
}

// How far text-to-speech got through the last reply before it was cut off.
// null means "nothing was interrupted"; the server then leaves history alone.
let heardChars = null;
let spokenText = "";
let ttsSource = null;      // the AudioBufferSourceNode currently playing, or null
let ttsStartedAt = 0;      // performance.now() when the current reply started playing
let ttsDurationMs = 0;     // total duration (ms) of the current reply's audio
// Bumped at the top of every speak() call, before any await. SPEAKING is
// re-entered every turn, so `state.name === SPEAKING` alone cannot tell a
// stale reply's resumed continuation from the current one -- comparing the
// call's own seq against this can. See speak()'s post-await guard.
let speakSeq = 0;
// True from the start of a speak() round trip until ttsSource.start()
// actually begins playback. Lets cancelSpeech() recognize a barge-in that
// lands in the silent window between the reply text appearing (button
// already reads "Speaking…") and audio actually starting -- without this,
// heardChars would stay null for a reply the caller heard zero of.
let ttsAwaitingPlayback = false;

const LABELS = {
  en: { idle: "Start call", listening: "Listening…", thinking: "Thinking…",
        speaking: "Speaking… (interrupt any time)", ptt: "Hold to talk",
        recording: "Listening… release to send", blocked:
        "Microphone blocked — allow mic access and reload.",
        didnt: "(didn't catch that)", endCall: "End call" },
  zh: { idle: "开始通话", listening: "正在聆听…", thinking: "思考中…",
        speaking: "正在回答…（可随时打断）", ptt: "按住说话",
        recording: "正在聆听…松开发送", blocked: "麦克风被阻止，请允许后重新加载。",
        didnt: "（没有听清）", endCall: "结束通话" },
};

// Appended to the hands-free in-call labels so the button visibly doubles as
// a hang-up control (see Defect 2: hands-free calls used to be unstoppable).
const END_CALL_HINT = { en: " (tap to end)", zh: "（点击结束）" };
const withEndCallHint = (text) => text + END_CALL_HINT[lang];

function render() {
  const L = LABELS[lang];
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  talk.classList.remove("recording", "listening", "speaking");
  talk.title = "";
  if (state.mode === "ptt") {
    talk.textContent = state.capturing ? L.recording : L.ptt;
    if (state.capturing) talk.classList.add("recording");
    return;
  }
  if (state.name === IDLE) {
    talk.textContent = L.idle;
    return;
  }
  talk.title = L.endCall;
  if (state.name === THINKING) talk.textContent = withEndCallHint(L.thinking);
  else if (state.name === SPEAKING) {
    talk.textContent = withEndCallHint(L.speaking);
    talk.classList.add("speaking");
  } else {
    talk.textContent = withEndCallHint(L.listening);
    talk.classList.add(state.capturing ? "recording" : "listening");
  }
}

function dispatch(event) {
  const result = next(state, event);
  state = result.state;
  for (const action of result.actions) runAction(action);
  render();
}

function runAction(action) {
  if (action === "START_RECORDING") startRecording();
  else if (action === "STOP_AND_SEND") stopRecordingAndSend();
  else if (action === "CANCEL_TTS") cancelSpeech();
  else if (action === "SPEAK") speak(pendingReply, pendingLang);
  else if (action === "DISCARD_PENDING_REPLY") heardChars = 0;
}

// --- microphone + VAD ------------------------------------------------------

async function openMic() {
  if (micStream) return true;
  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    replyEl.textContent = LABELS[lang].blocked;
    return false;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  audioCtx.createMediaStreamSource(micStream).connect(analyser);
  return true;
}

// Stops the mic and tears down the audio graph so openMic() can genuinely
// re-open later (it short-circuits on a non-null micStream) and so the
// browser's mic-in-use indicator turns off between calls.
function releaseMic() {
  if (micStream) {
    for (const track of micStream.getTracks()) track.stop();
  }
  if (audioCtx && audioCtx.state !== "closed") {
    audioCtx.close();
  }
  micStream = null;
  audioCtx = null;
  analyser = null;
}

function startVadLoop() {
  const buf = new Float32Array(analyser.fftSize);
  vad = createEnergyVAD();
  const tick = () => {
    try {
      analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const rms = Math.sqrt(sum / buf.length);
      const event = vad.process(rms, performance.now(), state.name === SPEAKING);
      if (event === "speech-start") dispatch("SPEECH_START");
      else if (event === "speech-end") dispatch("SPEECH_END");

      // Capture speculatively from the first hint of sound: waiting for the
      // VAD to confirm speech costs ~250ms, which clipped the opening word
      // of every utterance. If the sound turns out not to be speech, the
      // recording is thrown away below — a discarded buffer is far cheaper
      // than a lost first word.
      const pending = vad.isPending();
      if (pending && !wasPending && !state.capturing) startRecording();
      else if (!pending && wasPending && !state.capturing) discardRecording();
      wasPending = pending;

      if (debugEl) renderDebug(rms);
    } finally {
      rafId = requestAnimationFrame(tick);
    }
  };
  rafId = requestAnimationFrame(tick);
}

function stopVadLoop() {
  if (rafId !== null) cancelAnimationFrame(rafId);
  rafId = null;
  vad = null;
  wasPending = false;
}

function startRecording() {
  // Idempotent: a speculative recording may already be running from
  // vad.isPending() going true. SPEECH_START still dispatches the
  // START_RECORDING action on confirmation, and without this guard that
  // would replace the in-progress speculative recorder — discarding the
  // exact pre-roll audio this whole mechanism exists to keep, and
  // reintroducing the clipping this fixes.
  if (recorder && recorder.state === "recording") return;
  if (!micStream) return;
  recorder = new MediaRecorder(micStream);
  chunks = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = () => send(new Blob(chunks, { type: "audio/webm" }));
  recorder.start();
}

// Abandons a speculative recording once the sound that triggered it turns
// out not to have been speech (vad.isPending() fell back to false without
// ever reaching speech-start). onstop is detached FIRST so stopping the
// recorder can never POST audio for a turn the caller never actually took.
function discardRecording() {
  if (!recorder) return;
  recorder.onstop = null;
  if (recorder.state === "recording") recorder.stop();
  recorder = null;
}

function stopRecordingAndSend() {
  if (recorder && recorder.state === "recording") recorder.stop();
  // Nothing to stop (e.g. the mic track ended and MediaRecorder self-stopped):
  // without this the machine would sit in THINKING forever and the call is dead.
  else dispatch("TURN_ABORTED");
}

// --- server round trip -----------------------------------------------------

let pendingReply = "";
let pendingLang = "en";

async function send(blob) {
  if (!blob || blob.size < 1000) {
    transcriptEl.textContent = LABELS[lang].didnt;
    dispatch("TURN_ABORTED");   // nothing usable; get out of THINKING
    return;
  }
  transcriptEl.textContent = "…";
  replyEl.textContent = "";
  timingsEl.textContent = "";

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("lang", lang);
  form.append("audio", blob, "turn.webm");
  postCount += 1;
  if (heardChars !== null) {
    form.append("heard_chars", String(heardChars));
    heardChars = null;
  }

  try {
    const res = await fetch("/turn", { method: "POST", body: form });
    const data = await res.json();
    transcriptEl.textContent = data.transcript || LABELS[lang].didnt;
    replyEl.textContent = data.reply;
    const t = data.timings;
    timingsEl.textContent =
      `stt ${t.stt_ms}ms · agent ${t.agent_ms}ms · total ${t.total_ms}ms`;
    pendingReply = data.reply;
    pendingLang = data.lang;
    dispatch("REPLY");
  } catch (err) {
    replyEl.textContent = "Something went wrong. Please try again.";
    dispatch("TURN_ABORTED");
  }
}

// --- text to speech --------------------------------------------------------

async function speak(text, replyLang) {
  const seq = ++speakSeq;
  ttsAwaitingPlayback = true;
  const ctx = audioCtx;   // captured now: a hang-up during the awaits below
                           // can null the global before this resumes
  let buffer;
  try {
    const res = await fetch("/tts", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ text, lang: replyLang }),
    });
    if (!res.ok) throw new Error(`tts failed: ${res.status}`);
    const bytes = await res.arrayBuffer();
    // The closed-context check sits before decode, not after: a hang-up
    // landing during decodeAudioData itself is still caught below because
    // every releaseMic() path dispatches DISARM first, which the
    // `state.name !== SPEAKING` guard past the try/catch catches --
    // load-bearing today, easy to break by moving this check.
    if (!ctx || ctx.state === "closed") {
      if (seq === speakSeq) ttsAwaitingPlayback = false;
      return;   // hung up mid-fetch
    }
    buffer = await ctx.decodeAudioData(bytes);
    // decodeAudioData's promise form assumes a modern browser; the
    // window.webkitAudioContext fallback in openMic() exists for one old
    // enough that it may not support it and resolve to undefined instead of
    // rejecting. Route that into the catch below rather than letting
    // `ttsSource.buffer = undefined` throw uncaught past this try, which
    // would skip TTS_END entirely and strand the call in SPEAKING forever.
    if (!buffer) throw new Error("decodeAudioData returned no buffer");
  } catch (err) {
    // Synthesis, network, or decode failed -- nothing to speak. Move on
    // rather than leaving the call stuck in SPEAKING forever. Guarded by
    // seq: a failed /tts for a reply the caller has already barged past
    // must not dispatch TTS_END and pull the *current* reply out of
    // SPEAKING out from under it.
    if (seq !== speakSeq) return;
    ttsAwaitingPlayback = false;
    spokenText = "";
    dispatch("TTS_END");
    return;
  }
  // seq catches what state.name alone cannot: SPEAKING is re-entered every
  // turn, so if this reply's /tts round trip outlasts a barge-in-then-
  // reply-B cycle (A dispatches SPEAK and awaits /tts; caller barges in
  // -> LISTENING; caller finishes an utterance -> THINKING; /turn returns
  // reply B -> SPEAKING, SPEAK for B), a bare `state.name === SPEAKING`
  // check would pass for A's stale resumed continuation, overwrite the
  // global ttsSource out from under B's still-playing node, and orphan it
  // -- its onended would still fire TTS_END while B's audio is audibly
  // still playing, dropping the state machine out of SPEAKING and flipping
  // vad.js's `strict` argument (in startVadLoop()'s `state.name ===
  // SPEAKING`) to false mid-speech, silently disabling the echo floor while
  // the agent is still talking out loud. That is the exact self-
  // interruption failure this whole plan exists to prevent.
  if (seq !== speakSeq) return;
  // The caller may also have simply hung up (state.name !== SPEAKING with
  // seq still current): do not start playback for a call the state machine
  // has already moved past, and stop advertising a pending reply that will
  // never arrive.
  if (state.name !== SPEAKING) {
    ttsAwaitingPlayback = false;
    return;
  }
  // Deferred until here (not set at the top of the function): a speak()
  // call that bails out above must not clobber a heardChars = 0 that
  // DISCARD_PENDING_REPLY may have just set for an unrelated, still-pending
  // /turn send. spokenText is only ever read while ttsSource is non-null,
  // so it is likewise safe to defer.
  spokenText = text;
  heardChars = null;
  ttsSource = ctx.createBufferSource();
  ttsSource.buffer = buffer;
  ttsSource.connect(ctx.destination);
  ttsDurationMs = buffer.duration * 1000;
  ttsStartedAt = performance.now();
  ttsAwaitingPlayback = false;
  ttsSource.onended = () => {
    ttsSource = null;
    heardChars = null;
    spokenText = "";
    dispatch("TTS_END");
  };
  ttsSource.start();
}

function cancelSpeech() {
  if (ttsSource) {
    // Deterministic from elapsed playback time -- unlike the speechSynthesis
    // onboundary event this replaces, which Chrome's network-backed zh-CN
    // voice never fired, silently disabling heard_chars tracking for that
    // language.
    const elapsedMs = performance.now() - ttsStartedAt;
    const fraction = ttsDurationMs > 0 ? Math.min(1, elapsedMs / ttsDurationMs) : 0;
    heardChars = Math.round(spokenText.length * fraction);
    ttsSource.onended = null;   // suppress TTS_END: this is a barge-in, not natural completion
    ttsSource.stop();
    ttsSource = null;
    spokenText = "";
  } else if (ttsAwaitingPlayback) {
    // Nothing is playing yet, but a /tts round trip is in flight and the
    // button already reads "Speaking…" -- the caller barged in during that
    // silent window and so heard exactly zero characters of this reply.
    // Record that deterministically rather than leaving heardChars null,
    // which would keep the whole (unheard) reply in history.
    heardChars = 0;
  }
}

// --- controls --------------------------------------------------------------

document.querySelectorAll(".lang").forEach((btn) => {
  btn.addEventListener("click", () => {
    lang = btn.dataset.lang;
    document.querySelectorAll(".lang").forEach((b) =>
      b.classList.toggle("active", b === btn));
    render();
  });
});

document.querySelectorAll(".mode").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const mode = btn.dataset.mode;
    document.querySelectorAll(".mode").forEach((b) =>
      b.classList.toggle("active", b === btn));
    cancelSpeech();
    if (recorder && recorder.state === "recording") {
      // Abandon this turn: detach onstop first so stopping the recorder does
      // NOT trigger send() for audio the caller no longer intends to submit.
      recorder.onstop = null;
      recorder.stop();
    }
    stopVadLoop();
    dispatch("DISARM");
    state = initialState(mode);
    if (mode === "hands-free" && await openMic()) {
      startVadLoop();
      dispatch("ARM");
    } else if (mode !== "hands-free") {
      // Push-to-talk opens the mic per press; it does not need it hot
      // continuously, and leaving it open would keep the mic-in-use
      // indicator lit for no reason.
      releaseMic();
    }
    render();
  });
});

talk.addEventListener("click", async () => {
  if (state.mode !== "hands-free") return;
  if (state.name === IDLE) {
    if (await openMic()) {
      startVadLoop();
      dispatch("ARM");
    }
    return;
  }
  // The call is active: the button doubles as a hang-up control.
  cancelSpeech();
  if (recorder && recorder.state === "recording") {
    // Abandon this turn: detach onstop first so stopping the recorder does
    // NOT trigger send() for audio the caller no longer intends to submit.
    recorder.onstop = null;
    recorder.stop();
  }
  stopVadLoop();
  dispatch("DISARM");
  releaseMic();
});

const pttDown = async (e) => {
  if (state.mode !== "ptt") return;
  e.preventDefault();
  if (await openMic()) dispatch("PTT_DOWN");
};
const pttUp = (e) => {
  if (state.mode !== "ptt") return;
  e.preventDefault();
  dispatch("PTT_UP");
};

talk.addEventListener("mousedown", pttDown);
talk.addEventListener("mouseup", pttUp);
talk.addEventListener("mouseleave", pttUp);
talk.addEventListener("touchstart", pttDown);
talk.addEventListener("touchend", pttUp);

render();
