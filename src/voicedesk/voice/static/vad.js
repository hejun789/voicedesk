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
  bargeInMinSpeechMs = 300,  // strict-mode sustain time while the agent is speaking
  dipToleranceMs = 150,      // a gap this short does not reset a nascent turn
  echoFloorAlpha = 0.02,     // smoothing factor for the strict-mode ambient estimate
  bargeInMargin = 2.0,       // a real interruption must clear the ambient estimate by this factor
  bargeInGraceMs = 600,      // no interruption may START this long after strict goes true
} = {}) {
  let speaking = false;
  let loudSince = null;
  let quietSince = null;

  // Running estimate of the ambient (echo) level while strict, so a loud
  // voice (e.g. Chrome's network-backed zh-CN TTS, which plays louder and
  // bypasses the pipeline's echo cancellation) doesn't get mistaken for a
  // caller barging in just because it clears the fixed bargeInThreshold.
  // null means "no estimate yet" -- distinct from 0, and the signal that a
  // fresh interruption window is starting.
  let echoFloor = null;
  let wasStrict = false;

  // Timestamp of the rising edge of strict (when the agent started
  // speaking), so the grace window below can be measured from it. There is
  // a real delay between the browser being told to speak and audio actually
  // reaching the speakers -- strict becomes true at the moment speech is
  // requested, not when sound arrives. Without a grace window the echo floor
  // initialises against the SILENCE of that gap and settles near zero,
  // collapsing the start requirement to bargeInThreshold; the agent's own
  // audio then arrives, clears that trivial requirement, starts a nascent
  // turn, and freezes the floor at its uselessly-low value -- the agent
  // interrupts itself. Found via live mic testing with the zh-CN voice.
  let strictSince = null;
  let lastRequirement = null;   // diagnostics only, see debug()

  return {
    process(level, nowMs, strict = false) {
      if (strict) {
        if (!wasStrict) strictSince = nowMs;
        const inGrace = nowMs - strictSince < bargeInGraceMs;
        // Only track ambient/echo while no turn is nascent. Once a candidate
        // turn starts accumulating (loudSince set), whatever is making the
        // noise is potentially speech, not ambient, so the estimate freezes
        // at whatever it was the instant the turn began. Without this, a
        // caller's own sustained voice would keep dragging the estimate (and
        // therefore the requirement) up toward itself while their
        // interruption was still accumulating toward bargeInMinSpeechMs,
        // raising the bar out from under them mid-word -- found via the
        // real animation-frame cadence (~16ms), where enough update calls
        // land inside the 300ms sustain window for the drift to matter.
        if (loudSince === null) {
          if (inGrace) {
            // Inside the grace window, track the PEAK level rather than an
            // EMA. An EMA initialised from the first frame -- silence, since
            // audio output lags the request to speak -- only reaches ~52% of
            // the true echo level after a 600ms window at ~60fps cadence
            // (1 - 0.98^36). With bargeInMargin 2.0 that lands the
            // requirement level with the echo itself, so a normal
            // fluctuation in a real (non-flat) echo signal clears it, starts
            // a nascent turn, and freezes the floor at that too-low value --
            // the agent then interrupts itself on its own echo. Peak-hold
            // means that by the time the window ends, the floor already
            // equals the loudest echo actually observed, so steady echo
            // (fluctuations included) cannot clear floor * bargeInMargin.
            // Found via live mic testing with the (louder) zh-CN voice.
            //
            // DELIBERATE TRADE-OFF (owner's call, made from live evidence,
            // not an accident of this implementation): this treats ANY loud
            // sound observed during the grace window as echo, including a
            // genuine caller who happens to start talking in the first
            // 600ms of a reply. There is no way to avoid this -- from the
            // detector's point of view, "the caller started talking late in
            // the window" and "the reply's own audio arrived late in the
            // window" are literally the same signal (loud sound landing
            // during the audio-output latency gap). We chose to always
            // resolve that ambiguity as echo because the self-talk loop
            // this window exists to prevent happens on EVERY reply, while a
            // caller interrupting within a reply's first 600ms is rare. The
            // cost is bounded and temporary: once the loud sound stops, the
            // EMA below decays the floor back down, and a later
            // interruption at a normal volume fires again (see
            // "grace window: the cost of ... is temporary" in
            // tests/js/vad.test.mjs).
            echoFloor = echoFloor === null ? level : Math.max(echoFloor, level);
          } else {
            // Past the grace window: resume the slow EMA. Slow adaptation is
            // deliberate and load-bearing here too -- steady echo pulls the
            // floor up to its own level within about a second, but a sudden
            // loud voice clears the margin well before the floor can catch
            // up.
            echoFloor = echoFloor === null
              ? level
              : echoFloor + (level - echoFloor) * echoFloorAlpha;
          }
        }
        if (inGrace) {
          // Still within the post-onset grace window: the floor above keeps
          // learning the real echo level (that's the whole point), but no
          // turn may start, and any turn that started accumulating this
          // frame must be cleared -- otherwise the freeze rule above would
          // lock the floor in at exactly the low value this window exists
          // to avoid.
          loudSince = null;
          wasStrict = strict;
          return null;
        }
      } else if (wasStrict) {
        // The interruption window just closed -- start the next one fresh
        // rather than inheriting a stale estimate.
        echoFloor = null;
      }
      wasStrict = strict;

      const startThreshold = strict
        ? Math.max(bargeInThreshold, echoFloor * bargeInMargin)
        : threshold;
      const startMinSpeechMs = strict ? bargeInMinSpeechMs : minSpeechMs;
      lastRequirement = startThreshold;   // diagnostics only, see debug()

      if (!speaking) {
        const loud = level >= startThreshold;
        if (!loud) {
          // A dip SHORTER than dipToleranceMs must not cancel a nascent turn.
          // Real speech drops below the RMS floor many times a second — between
          // syllables and on stop consonants — so demanding an unbroken run made
          // barge-in impossible to trigger by talking (found in live testing).
          // Only a sustained gap means the caller is not actually speaking.
          if (quietSince === null) quietSince = nowMs;
          if (nowMs - quietSince >= dipToleranceMs) {
            loudSince = null;
            quietSince = null;
          }
          return null;
        }
        quietSince = null;        // still going; the dip was momentary
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

    // Read-only: true exactly while a turn is nascent — sound has been
    // observed but the sustain requirement has not yet been met. False once
    // the turn is confirmed (speech-start resets loudSince) and once it is
    // abandoned (a gap beyond dipToleranceMs resets loudSince). Lets app.js
    // start capturing speculatively on the first hint of sound without
    // process() having to carry a second signal in its single return value.
    // Diagnostics only — never consulted by the agent. Exposes the internal
    // estimates so the page can render a live overlay while tuning echo
    // rejection against real hardware, which is the only way to see why a
    // barge-in fired or failed to.
    debug() {
      return {
        echoFloor,
        requirement: lastRequirement,
        pending: loudSince !== null,
        speaking,
      };
    },

    isPending() {
      return loudSince !== null;
    },
  };
}
