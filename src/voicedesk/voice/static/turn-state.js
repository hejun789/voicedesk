// The conversation state machine.
//
// A pure reducer: given the current state and an event, it returns the next
// state plus the side effects app.js should perform. It touches no browser API,
// so every transition — including the awkward ones like a reply arriving after
// the caller already interrupted — is unit-testable.
//
// Both input modes share this machine. Hands-free feeds it VAD events; push-to-
// talk feeds it button events. The transitions are otherwise identical.

export const IDLE = "IDLE";
export const LISTENING = "LISTENING";
export const THINKING = "THINKING";
export const SPEAKING = "SPEAKING";

export function initialState(mode = "hands-free") {
  return { name: IDLE, capturing: false, mode };
}

const stay = (state) => ({ state, actions: [] });

export function next(state, event) {
  const { name, capturing, mode } = state;

  if (event === "DISARM") {
    return { state: { ...state, name: IDLE, capturing: false }, actions: [] };
  }

  if (event === "ARM") {
    if (name !== IDLE) return stay(state);
    return { state: { ...state, name: LISTENING }, actions: [] };
  }

  if (event === "SPEECH_START" || event === "PTT_DOWN") {
    // Barge-in: the caller talks over the agent. Cut the audio immediately.
    if (name === SPEAKING) {
      return {
        state: { ...state, name: LISTENING, capturing: true },
        actions: ["CANCEL_TTS", "START_RECORDING"],
      };
    }
    // Deliberately NOT allowed while THINKING: the server cannot cancel an
    // in-flight agent call, so interrupting there would desync history.
    if (name === LISTENING || name === IDLE) {
      if (capturing) return stay(state);
      return {
        state: { ...state, name: LISTENING, capturing: true },
        actions: ["START_RECORDING"],
      };
    }
    return stay(state);
  }

  if (event === "SPEECH_END" || event === "PTT_UP") {
    if (!capturing) return stay(state);
    return {
      state: { ...state, name: THINKING, capturing: false },
      actions: ["STOP_AND_SEND"],
    };
  }

  if (event === "REPLY") {
    // A reply that lands after the caller already barged in must not speak
    // over them.
    if (name !== THINKING) return stay(state);
    return { state: { ...state, name: SPEAKING }, actions: ["SPEAK"] };
  }

  if (event === "TTS_END") {
    if (name !== SPEAKING) return stay(state);
    const back = mode === "hands-free" ? LISTENING : IDLE;
    return { state: { ...state, name: back }, actions: [] };
  }

  if (event === "TURN_ABORTED") {
    // The upload was unusable or the request failed. Without this the machine
    // would sit in THINKING forever and the call would be dead.
    if (name !== THINKING) return stay(state);
    const back = mode === "hands-free" ? LISTENING : IDLE;
    return { state: { ...state, name: back }, actions: [] };
  }

  return stay(state);
}
