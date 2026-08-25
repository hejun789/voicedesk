import { test } from "node:test";
import assert from "node:assert/strict";
import {
  IDLE, LISTENING, THINKING, SPEAKING, initialState, next,
} from "../../src/voicedesk/voice/static/turn-state.js";

test("a call starts idle and hands-free", () => {
  const s = initialState();
  assert.equal(s.name, IDLE);
  assert.equal(s.capturing, false);
  assert.equal(s.mode, "hands-free");
});

test("arming moves from idle to listening", () => {
  const { state, actions } = next(initialState(), "ARM");
  assert.equal(state.name, LISTENING);
  assert.deepEqual(actions, []);
});

test("speech while listening starts recording", () => {
  const armed = next(initialState(), "ARM").state;
  const { state, actions } = next(armed, "SPEECH_START");
  assert.equal(state.name, LISTENING);
  assert.equal(state.capturing, true);
  assert.deepEqual(actions, ["START_RECORDING"]);
});

test("end of speech sends the turn and moves to thinking", () => {
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  const { state, actions } = next(s, "SPEECH_END");
  assert.equal(state.name, THINKING);
  assert.equal(state.capturing, false);
  assert.deepEqual(actions, ["STOP_AND_SEND"]);
});

test("end of speech without a capture in progress is ignored", () => {
  const armed = next(initialState(), "ARM").state;
  const { state, actions } = next(armed, "SPEECH_END");
  assert.equal(state.name, LISTENING);
  assert.deepEqual(actions, []);
});

test("a reply moves to speaking and speaks it", () => {
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  const { state, actions } = next(s, "REPLY");
  assert.equal(state.name, SPEAKING);
  assert.deepEqual(actions, ["SPEAK"]);
});

test("barge-in cancels speech and starts recording", () => {
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  s = next(s, "REPLY").state;
  const { state, actions } = next(s, "SPEECH_START");
  assert.equal(state.name, LISTENING);
  assert.equal(state.capturing, true);
  assert.deepEqual(actions, ["CANCEL_TTS", "START_RECORDING"]);
});

test("speech during thinking interrupts it: the caller can talk over the LLM call", () => {
  // The server still cannot cancel the in-flight agent call (see /turn), so
  // it runs to completion regardless -- but the caller does not have to wait
  // for it. DISCARD_PENDING_REPLY tells app.js to report zero heard
  // characters on the next turn it sends, which the server's
  // truncate_last_reply() uses to drop the stale reply from history once it
  // lands (see the REPLY test below for the other half of this).
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  assert.equal(s.name, THINKING);
  const { state, actions } = next(s, "SPEECH_START");
  assert.equal(state.name, LISTENING);
  assert.equal(state.capturing, true);
  assert.deepEqual(actions, ["DISCARD_PENDING_REPLY", "START_RECORDING"]);
});

test("push-to-talk: pressing again during thinking interrupts it the same way", () => {
  let s = initialState("ptt");
  s = next(s, "PTT_DOWN").state;
  s = next(s, "PTT_UP").state;
  assert.equal(s.name, THINKING);
  const { state, actions } = next(s, "PTT_DOWN");
  assert.equal(state.name, LISTENING);
  assert.equal(state.capturing, true);
  assert.deepEqual(actions, ["DISCARD_PENDING_REPLY", "START_RECORDING"]);
});

test("a reply that lands after a thinking interrupt is not spoken", () => {
  // Mirrors "a late reply does not resurrect speaking after a barge-in"
  // above, but for a barge-in that happened during THINKING instead of
  // SPEAKING.
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  s = next(s, "SPEECH_START").state;   // interrupts THINKING
  assert.equal(s.name, LISTENING);
  const { state, actions } = next(s, "REPLY");
  assert.equal(state.name, LISTENING);
  assert.deepEqual(actions, []);
});

test("finishing speech returns to listening in hands-free mode", () => {
  let s = next(initialState("hands-free"), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  s = next(s, "REPLY").state;
  const { state } = next(s, "TTS_END");
  assert.equal(state.name, LISTENING);
});

test("finishing speech returns to idle in push-to-talk mode", () => {
  let s = initialState("ptt");
  s = next(s, "PTT_DOWN").state;
  s = next(s, "PTT_UP").state;
  s = next(s, "REPLY").state;
  const { state } = next(s, "TTS_END");
  assert.equal(state.name, IDLE);
});

test("a late reply does not resurrect speaking after a barge-in", () => {
  // The user interrupted and is already talking again; the in-flight reply
  // that lands afterwards must not start speaking over them.
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  s = next(s, "REPLY").state;
  s = next(s, "SPEECH_START").state;   // barge-in
  assert.equal(s.name, LISTENING);
  const { state, actions } = next(s, "REPLY");
  assert.equal(state.name, LISTENING);
  assert.deepEqual(actions, []);
});

test("push-to-talk drives the same machine with button events", () => {
  let s = initialState("ptt");
  let r = next(s, "PTT_DOWN");
  assert.equal(r.state.capturing, true);
  assert.deepEqual(r.actions, ["START_RECORDING"]);
  r = next(r.state, "PTT_UP");
  assert.equal(r.state.name, THINKING);
  assert.deepEqual(r.actions, ["STOP_AND_SEND"]);
});

test("an aborted turn returns to listening instead of hanging in thinking", () => {
  // The upload was too small to transcribe, or the request failed. Without this
  // the machine would sit in THINKING forever and the call would be dead.
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  s = next(s, "SPEECH_END").state;
  assert.equal(s.name, THINKING);
  const { state } = next(s, "TURN_ABORTED");
  assert.equal(state.name, LISTENING);
});

test("an aborted turn returns to idle in push-to-talk mode", () => {
  let s = initialState("ptt");
  s = next(s, "PTT_DOWN").state;
  s = next(s, "PTT_UP").state;
  const { state } = next(s, "TURN_ABORTED");
  assert.equal(state.name, IDLE);
});

test("disarming returns to idle and stops capturing", () => {
  let s = next(initialState(), "ARM").state;
  s = next(s, "SPEECH_START").state;
  const { state } = next(s, "DISARM");
  assert.equal(state.name, IDLE);
  assert.equal(state.capturing, false);
});

test("unknown events are ignored", () => {
  const s = next(initialState(), "ARM").state;
  const { state, actions } = next(s, "NONSENSE");
  assert.deepEqual(state, s);
  assert.deepEqual(actions, []);
});
