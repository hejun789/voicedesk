// Energy-based voice activity detection.
//
// Pure decision logic: it is handed a loudness value and a timestamp and
// answers whether a turn just started or just ended. It knows nothing about
// microphones or the DOM, which is what makes it unit-testable under Node.
//
// A better detector (Silero via onnxruntime-web is what production frameworks
// use) can replace this behind the same interface without touching app.js.

export function createEnergyVAD({
  threshold = 0.02,   // RMS above this counts as speech
  hangoverMs = 800,   // silence must persist this long to end a turn
  minSpeechMs = 200,  // sound must persist this long to start one
  bargeInThreshold = 0.08,   // strict-mode RMS floor while the agent is speaking
  bargeInMinSpeechMs = 500,  // strict-mode sustain time while the agent is speaking
} = {}) {
  let speaking = false;
  let loudSince = null;
  let quietSince = null;

  return {
    process(level, nowMs, strict = false) {
      const startThreshold = strict ? bargeInThreshold : threshold;
      const startMinSpeechMs = strict ? bargeInMinSpeechMs : minSpeechMs;

      if (!speaking) {
        const loud = level >= startThreshold;
        if (!loud) {
          loudSince = null;       // a dip cancels a nascent turn
          return null;
        }
        if (loudSince === null) loudSince = nowMs;
        if (nowMs - loudSince >= startMinSpeechMs) {
          speaking = true;
          loudSince = null;
          quietSince = null;
          return "speech-start";
        }
        return null;
      }

      // speaking: ending a turn always uses hangoverMs/threshold, regardless
      // of strict — strict affects only the speech-START decision above.
      const loud = level >= threshold;
      if (loud) {
        quietSince = null;        // a pause between words is not the end
        return null;
      }
      if (quietSince === null) quietSince = nowMs;
      if (nowMs - quietSince >= hangoverMs) {
        speaking = false;
        quietSince = null;
        loudSince = null;
        return "speech-end";
      }
      return null;
    },
  };
}
