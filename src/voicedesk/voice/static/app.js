const talk = document.getElementById("talk");
const transcriptEl = document.getElementById("transcript");
const replyEl = document.getElementById("reply");
const timingsEl = document.getElementById("timings");

// One session per page load, so the agent remembers this caller across turns.
const sessionId = crypto.randomUUID();

let lang = "en";
const BCP47 = { en: "en-US", zh: "zh-CN" };

function setLabels() {
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  if (talk.classList.contains("recording")) {
    talk.textContent = lang === "zh" ? "正在聆听…松开发送" : "Listening… release to send";
  } else {
    talk.textContent = lang === "zh" ? "按住说话" : "Hold to talk";
  }
}

document.querySelectorAll(".lang").forEach((btn) => {
  btn.addEventListener("click", () => {
    lang = btn.dataset.lang;
    document.querySelectorAll(".lang").forEach((b) =>
      b.classList.toggle("active", b === btn));
    setLabels();
  });
});

let recorder = null;
let chunks = [];
let wantsToRecord = false;

async function startRecording() {
  wantsToRecord = true;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    replyEl.textContent = "Microphone blocked — allow mic access and reload.";
    talk.classList.remove("recording");
    talk.textContent = "Hold to talk";
    wantsToRecord = false;
    return;
  }
  if (!wantsToRecord) {           // released before permission resolved
    stream.getTracks().forEach((t) => t.stop());
    return;
  }
  recorder = new MediaRecorder(stream);
  chunks = [];
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    send(new Blob(chunks, { type: "audio/webm" }));
  };
  recorder.start();
  talk.classList.add("recording");
  setLabels();
}

function stopRecording() {
  wantsToRecord = false;
  if (recorder && recorder.state === "recording") recorder.stop();
  talk.classList.remove("recording");
  setLabels();
}

async function send(blob) {
  if (!blob || blob.size < 1000) {
    transcriptEl.textContent = lang === "zh" ? "（没有听清）" : "(didn't catch that)";
    replyEl.textContent = lang === "zh"
      ? "抱歉，我没有听清，可以再说一遍吗？"
      : "Sorry, I didn't catch that. Could you say that again?";
    return;
  }

  talk.disabled = true;
  transcriptEl.textContent = "…";
  replyEl.textContent = "";
  timingsEl.textContent = "";

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("lang", lang);
  form.append("audio", blob, "turn.webm");

  try {
    const res = await fetch("/turn", { method: "POST", body: form });
    const data = await res.json();
    transcriptEl.textContent = data.transcript || (lang === "zh" ? "（没有听清）" : "(didn't catch that)");
    replyEl.textContent = data.reply;
    const t = data.timings;
    timingsEl.textContent =
      `stt ${t.stt_ms}ms · agent ${t.agent_ms}ms · total ${t.total_ms}ms`;
    speak(data.reply, data.lang);
  } catch (err) {
    replyEl.textContent = "Something went wrong. Please try again.";
  } finally {
    talk.disabled = false;
  }
}

function speak(text, replyLang) {
  // Browser TTS: starts instantly, costs nothing, adds no network latency.
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = BCP47[replyLang] || BCP47.en;
  utterance.rate = 1.05;
  window.speechSynthesis.speak(utterance);
}

talk.addEventListener("mousedown", startRecording);
talk.addEventListener("mouseup", stopRecording);
talk.addEventListener("mouseleave", stopRecording);
talk.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
talk.addEventListener("touchend", (e) => { e.preventDefault(); stopRecording(); });
