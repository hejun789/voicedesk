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
const BCP47 = { en: "en-US", zh: "zh-CN" };

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

// How far text-to-speech got through the last reply before it was cut off.
// null means "nothing was interrupted"; the server then leaves history alone.
let heardChars = null;
let spokenText = "";

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

function speak(text, replyLang) {
  window.speechSynthesis.cancel();
  spokenText = text;
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = BCP47[replyLang] || BCP47.en;
  utterance.rate = 1.05;
  // onboundary reports how far speech has progressed. If the caller interrupts,
  // this is how we know what they actually heard. Not every browser or voice
  // fires it; when it does not, heardChars stays null and the server leaves
  // history untouched.
  utterance.onboundary = (e) => {
    if (typeof e.charIndex === "number") heardChars = e.charIndex;
  };
  // Natural completion (not a barge-in): the caller heard the whole reply,
  // so clear the interruption-tracking state before the next turn starts.
  utterance.onend = () => {
    heardChars = null;
    spokenText = "";
    dispatch("TTS_END");
  };
  utterance.onerror = (e) => {
    // cancel() fires 'error' with "interrupted"/"canceled" — that is our own
    // barge-in path, and cancelSpeech() has already recorded how much the
    // caller heard. Only a real synthesis failure should discard it.
    if (e.error !== "interrupted" && e.error !== "canceled") {
      heardChars = null;
      spokenText = "";
    }
    dispatch("TTS_END");
  };
  window.speechSynthesis.speak(utterance);
}

function cancelSpeech() {
  // cancel() fires 'error', not 'onend', so heardChars keeps whatever the last
  // boundary reported. If no boundary ever fired we genuinely do not know how
  // much the caller heard — Chrome's network-backed voices (the zh-CN default)
  // never fire them — so we send nothing and leave history untouched rather
  // than guessing zero, which would delete the whole reply.
  window.speechSynthesis.cancel();
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
