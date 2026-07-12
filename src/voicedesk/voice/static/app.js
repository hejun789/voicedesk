const talk = document.getElementById("talk");
const transcriptEl = document.getElementById("transcript");
const replyEl = document.getElementById("reply");
const timingsEl = document.getElementById("timings");

// One session per page load, so the agent remembers this caller across turns.
const sessionId = crypto.randomUUID();

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
  talk.textContent = "Listening… release to send";
}

function stopRecording() {
  wantsToRecord = false;
  if (recorder && recorder.state === "recording") recorder.stop();
  talk.classList.remove("recording");
  talk.textContent = "Hold to talk";
}

async function send(blob) {
  if (!blob || blob.size < 1000) {
    transcriptEl.textContent = "(didn't catch that)";
    replyEl.textContent = "Sorry, I didn't catch that. Could you say that again?";
    return;
  }

  talk.disabled = true;
  transcriptEl.textContent = "…";
  replyEl.textContent = "";
  timingsEl.textContent = "";

  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("audio", blob, "turn.webm");

  try {
    const res = await fetch("/turn", { method: "POST", body: form });
    const data = await res.json();
    transcriptEl.textContent = data.transcript || "(didn't catch that)";
    replyEl.textContent = data.reply;
    const t = data.timings;
    timingsEl.textContent =
      `stt ${t.stt_ms}ms · agent ${t.agent_ms}ms · total ${t.total_ms}ms`;
    speak(data.reply);
  } catch (err) {
    replyEl.textContent = "Something went wrong. Please try again.";
  } finally {
    talk.disabled = false;
  }
}

function speak(text) {
  // Browser TTS: starts instantly, costs nothing, adds no network latency.
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1.05;
  window.speechSynthesis.speak(utterance);
}

talk.addEventListener("mousedown", startRecording);
talk.addEventListener("mouseup", stopRecording);
talk.addEventListener("mouseleave", stopRecording);
talk.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
talk.addEventListener("touchend", (e) => { e.preventDefault(); stopRecording(); });
