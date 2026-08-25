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
    // Barge-in during THINKING: the server still cannot cancel the in-flight
    // agent call, so it runs to completion and its reply lands in history as
    // normal — but the caller does not have to sit through it. The REPLY
    // handler below already drops a reply that arrives after the caller has
    // moved off THINKING; DISCARD_PENDING_REPLY additionally tells the next
    // turn's request to report zero heard characters, so the server's
    // truncate_last_reply() removes it from the agent's own history too —
    // otherwise the agent would believe it said something the caller never
    // heard and could reference it later.
    if (name === THINKING) {
      return {
        state: { ...state, name: LISTENING, capturing: true },
        actions: ["DISCARD_PENDING_REPLY", "START_RECORDING"],
      };
    }
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
