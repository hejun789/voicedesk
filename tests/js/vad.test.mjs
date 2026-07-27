import { test } from "node:test";
import assert from "node:assert/strict";
import { createEnergyVAD } from "../../src/voicedesk/voice/static/vad.js";

const OPTS = { threshold: 0.02, hangoverMs: 800, minSpeechMs: 200 };
const LOUD = 0.5;
const QUIET = 0.001;

test("silence never fires an event", () => {
  const vad = createEnergyVAD(OPTS);
  for (let t = 0; t < 5000; t += 50) {
    assert.equal(vad.process(QUIET, t), null);
  }
});

test("sustained speech fires speech-start exactly once", () => {
  const vad = createEnergyVAD(OPTS);
  const events = [];
  for (let t = 0; t < 2000; t += 50) {
    const e = vad.process(LOUD, t);
    if (e) events.push(e);
  }
  assert.deepEqual(events, ["speech-start"]);
});

test("speech-start waits for minSpeechMs before firing", () => {
  const vad = createEnergyVAD(OPTS);
  assert.equal(vad.process(LOUD, 0), null);
  assert.equal(vad.process(LOUD, 100), null);   // still under 200ms
  assert.equal(vad.process(LOUD, 200), "speech-start");
});

test("a short blip below minSpeechMs never fires", () => {
  // A click, a cough, a door closing.
  const vad = createEnergyVAD(OPTS);
  assert.equal(vad.process(LOUD, 0), null);
  assert.equal(vad.process(LOUD, 100), null);
  for (let t = 150; t < 3000; t += 50) {
    assert.equal(vad.process(QUIET, t), null);
  }
});

test("a brief pause mid-sentence does NOT end the turn", () => {
  // The single most common way energy VAD gets this wrong: natural gaps
  // between words must not be mistaken for the end of a turn.
  const vad = createEnergyVAD(OPTS);
  assert.equal(vad.process(LOUD, 0), null);
  assert.equal(vad.process(LOUD, 200), "speech-start");
  for (let t = 250; t < 900; t += 50) {         // 600ms gap, under hangover
    assert.equal(vad.process(QUIET, t), null);
  }
  assert.equal(vad.process(LOUD, 950), null);   // speaking again, no event
  assert.equal(vad.process(LOUD, 1000), null);
});

test("silence for hangoverMs ends the turn exactly once", () => {
  const vad = createEnergyVAD(OPTS);
  vad.process(LOUD, 0);
  assert.equal(vad.process(LOUD, 200), "speech-start");
  let events = [];
  for (let t = 250; t < 2000; t += 50) {
    const e = vad.process(QUIET, t);
    if (e) events.push(e);
  }
  assert.deepEqual(events, ["speech-end"]);
});

test("a second utterance fires start and end again", () => {
  const vad = createEnergyVAD(OPTS);
  const events = [];
  const feed = (level, from, to) => {
    for (let t = from; t < to; t += 50) {
      const e = vad.process(level, t);
      if (e) events.push(e);
    }
  };
  feed(LOUD, 0, 500);
  feed(QUIET, 500, 1500);
  feed(LOUD, 1500, 2000);
  feed(QUIET, 2000, 3000);
  assert.deepEqual(events, ["speech-start", "speech-end", "speech-start", "speech-end"]);
});

test("defaults are supplied when no options are passed", () => {
  const vad = createEnergyVAD();
  assert.equal(typeof vad.process, "function");
  assert.equal(vad.process(0.0, 0), null);
});

// --- strict mode (barge-in hardening) ---------------------------------------

test("strict mode: a level above normal threshold but below bargeInThreshold never fires, even sustained", () => {
  // 0.05 is above OPTS.threshold (0.02) but below the default bargeInThreshold (0.08).
  const vad = createEnergyVAD(OPTS);
  for (let t = 0; t < 5000; t += 50) {
    assert.equal(vad.process(0.05, t, true), null);
  }
});

test("strict mode: a level above bargeInThreshold sustained past bargeInMinSpeechMs fires speech-start exactly once", () => {
  // Strict mode only ever begins when the agent starts speaking, so a
  // constant loud level from the very first strict frame (no ambient/echo
  // period beforehand) cannot happen in real use -- the adaptive floor would
  // treat it as the ambient level itself. This prelude models the agent's
  // own voice leaking into the mic and settling the floor BEFORE the caller
  // interrupts, matching the real sequence. Do not collapse this back to a
  // bare constant 0.5 -- see the adaptive-floor tests above for why.
  const vad = createEnergyVAD(OPTS);
  for (let t = 0; t < 1000; t += 50) vad.process(0.05, t, true);
  const events = [];
  for (let t = 1000; t < 3000; t += 50) {
    const e = vad.process(0.5, t, true);
    if (e) events.push(e);
  }
  assert.deepEqual(events, ["speech-start"]);
});

test("strict mode: loud sustained past normal minSpeechMs but not bargeInMinSpeechMs does not fire", () => {
  // See the prelude comment above: strict mode always follows an
  // ambient/echo period in real use, so settle the floor first rather than
  // starting the loud level on the very first strict frame.
  // Loud (above bargeInThreshold) from t=1000, sustained to t=1250 -- past
  // the normal 200ms minSpeechMs, but short of the default 300ms
  // bargeInMinSpeechMs.
  const vad = createEnergyVAD(OPTS);
  for (let t = 0; t < 1000; t += 50) vad.process(0.05, t, true);
  for (let t = 1000; t <= 1250; t += 50) {
    assert.equal(vad.process(0.5, t, true), null);
  }
});

test("strict mode: once started, the turn still ends normally after hangoverMs of silence", () => {
  // See the prelude comment above: settle the floor on an ambient/echo
  // level before the loud level begins, matching the real sequence.
  const vad = createEnergyVAD(OPTS);
  for (let t = 0; t < 1000; t += 50) vad.process(0.05, t, true);

  // Reach the default bargeInMinSpeechMs (300ms) while loud and strict.
  let started = null;
  for (let t = 1000; t <= 1550; t += 50) {
    const e = vad.process(0.5, t, true);
    if (e) started = e;
  }
  assert.equal(started, "speech-start");

  // Ending uses hangoverMs/threshold regardless of strict.
  const events = [];
  for (let t = 1600; t < 3000; t += 50) {
    const e = vad.process(QUIET, t, true);
    if (e) events.push(e);
  }
  assert.deepEqual(events, ["speech-end"]);
});

test("strict = false explicitly behaves identically to the existing non-strict tests", () => {
  const vad = createEnergyVAD(OPTS);
  assert.equal(vad.process(LOUD, 0, false), null);
  assert.equal(vad.process(LOUD, 100, false), null);   // still under 200ms
  assert.equal(vad.process(LOUD, 200, false), "speech-start");
});

test("omitting strict entirely behaves identically to the existing non-strict tests", () => {
  const vad = createEnergyVAD(OPTS);
  assert.equal(vad.process(LOUD, 0), null);
  assert.equal(vad.process(LOUD, 100), null);   // still under 200ms
  assert.equal(vad.process(LOUD, 200), "speech-start");
});

test("a syllable-length dip does not reset a nascent turn", () => {
  // THE bug that made barge-in unusable in live testing: any single frame
  // below threshold used to reset the start timer to zero. Real speech dips
  // below the RMS floor many times a second — between syllables and on stop
  // consonants — so requiring an unbroken run was impossible to satisfy by
  // talking. Brief dips must be tolerated; only a sustained gap resets.
  const vad = createEnergyVAD(OPTS);
  assert.equal(vad.process(LOUD, 0), null);
  assert.equal(vad.process(QUIET, 50), null);    // 50ms dip mid-word
  assert.equal(vad.process(LOUD, 100), null);
  assert.equal(vad.process(QUIET, 150), null);   // another dip
  // 200ms of speech has now elapsed despite the dips, so the turn starts.
  assert.equal(vad.process(LOUD, 200), "speech-start");
});

test("strict barge-in survives the dips in natural speech", () => {
  // The live failure: a caller could not interrupt the agent by speaking
  // normally, because the strict sustain window never survived a syllable gap.
  //
  // See the prelude comment on the tests above: strict mode always follows
  // an ambient/echo period in real use (it only begins once the agent is
  // speaking), so settle the floor on a modest ambient level first rather
  // than starting the loud, dip-laced voice on the very first strict frame.
  // Do not collapse this back to a bare constant-from-t=0 form.
  const vad = createEnergyVAD(OPTS);
  for (let t = 0; t < 1000; t += 50) vad.process(0.05, t, true);

  const events = [];
  // Loud speech with a 50ms dip every 150ms — a normal speaking cadence.
  for (let t = 1000; t <= 1400; t += 50) {
    const level = ((t - 1000) % 150 === 100) ? QUIET : 0.5;
    const e = vad.process(level, t, true);
    if (e) events.push(e);
  }
  assert.deepEqual(events, ["speech-start"]);
});

test("a sustained gap still resets a nascent turn", () => {
  // The tolerance must not swallow a genuine pause: a click, then silence
  // well past the tolerance, then a later click must NOT accumulate into one
  // turn.
  const vad = createEnergyVAD(OPTS);
  assert.equal(vad.process(LOUD, 0), null);
  for (let t = 50; t <= 400; t += 50) {
    assert.equal(vad.process(QUIET, t), null);
  }
  // Timer was reset by the long gap, so this lone loud frame starts over.
  assert.equal(vad.process(LOUD, 450), null);
  assert.equal(vad.process(LOUD, 500), null);   // only 50ms accumulated
});

// --- speculative recording: isPending() query -------------------------------
//
// Recording that waits for CONFIRMED speech (speech-start) always clips the
// first ~200-300ms of an utterance, because that is exactly how long
// minSpeechMs makes the caller wait before confirmation. The fix: start
// capturing at the very first loud frame and throw the recording away if the
// sound never turns into real speech. process()'s single-value-per-call
// contract cannot carry that extra signal without breaking every existing
// test that asserts null/speech-start on those frames (tried and reverted —
// see the report), so instead of a new return value this exposes the
// detector's already-tracked nascent-turn state as a read-only query that
// app.js can poll independently of process()'s return value.

test("isPending() is false on a fresh detector", () => {
  const vad = createEnergyVAD(OPTS);
  assert.equal(vad.isPending(), false);
});

test("isPending() is true after one loud frame", () => {
  const vad = createEnergyVAD(OPTS);
  vad.process(LOUD, 0);
  assert.equal(vad.isPending(), true);
});

test("isPending() stays true across a dip shorter than dipToleranceMs", () => {
  const vad = createEnergyVAD(OPTS);
  vad.process(LOUD, 0);
  assert.equal(vad.isPending(), true);
  vad.process(QUIET, 50);          // 50ms dip, under the 150ms tolerance
  assert.equal(vad.isPending(), true);
  vad.process(LOUD, 100);
  assert.equal(vad.isPending(), true);
});

test("isPending() is false once process() returns speech-start", () => {
  const vad = createEnergyVAD(OPTS);
  vad.process(LOUD, 0);
  assert.equal(vad.isPending(), true);
  assert.equal(vad.process(LOUD, 200), "speech-start");
  assert.equal(vad.isPending(), false);
});

test("isPending() is false after a nascent turn is abandoned by a sustained gap", () => {
  const vad = createEnergyVAD(OPTS);
  vad.process(LOUD, 0);
  assert.equal(vad.isPending(), true);
  for (let t = 50; t <= 400; t += 50) {
    vad.process(QUIET, t);
  }
  // The gap exceeded dipToleranceMs well before minSpeechMs elapsed, so the
  // nascent turn was reset without ever confirming.
  assert.equal(vad.isPending(), false);
});

test("isPending() is never true during continuous silence from the start", () => {
  const vad = createEnergyVAD(OPTS);
  for (let t = 0; t < 5000; t += 50) {
    vad.process(QUIET, t);
    assert.equal(vad.isPending(), false);
  }
});

test("strict mode: isPending() only becomes true above bargeInThreshold, not merely above threshold", () => {
  const vad = createEnergyVAD(OPTS);
  // 0.05 is above OPTS.threshold (0.02) but below the default bargeInThreshold
  // (0.08): in strict mode this must not count as the start of a nascent turn.
  vad.process(0.05, 0, true);
  assert.equal(vad.isPending(), false);
  vad.process(0.5, 50, true);   // above bargeInThreshold
  assert.equal(vad.isPending(), true);
});

// --- adaptive echo floor (Chinese-voice barge-in loop) ----------------------
//
// The zh-CN browser voice is network-backed, plays louder than en-US, and its
// output does not pass through the pipeline where echo cancellation applies.
// Its echo exceeds the fixed bargeInThreshold (0.08) and re-triggers the
// barge-in loop: the agent hears itself, "barges in" on itself, and replies
// to itself. A fixed threshold cannot distinguish "loud echo" from "loud
// caller" for a voice this loud. The fix: while strict, track a slow running
// estimate of the ambient (echo) level and require a real interruption to
// clear that estimate by a margin, on top of the existing absolute floor.

test("adaptive floor: steady loud echo never triggers barge-in", () => {
  // 0.12 is above the fixed bargeInThreshold (0.08) -- this is the exact live
  // failure with the zh-CN voice's echo. The floor should track up to ~0.12,
  // making the requirement ~0.24, so this constant echo never reads as loud.
  const vad = createEnergyVAD(OPTS);
  for (let t = 0; t < 5000; t += 50) {
    assert.equal(vad.process(0.12, t, true), null);
  }
});

test("adaptive floor: a real voice over that same echo does trigger", () => {
  const vad = createEnergyVAD(OPTS);
  // Settle the floor at the echo level first.
  for (let t = 0; t < 2000; t += 50) {
    assert.equal(vad.process(0.12, t, true), null);
  }
  // Now the caller speaks over the echo, sustained past bargeInMinSpeechMs.
  const events = [];
  for (let t = 2000; t < 2500; t += 50) {
    const e = vad.process(0.6, t, true);
    if (e) events.push(e);
  }
  assert.deepEqual(events, ["speech-start"]);
});

test("adaptive floor: the absolute minimum still applies in a quiet room", () => {
  const vad = createEnergyVAD(OPTS);
  // Settle the floor near zero (a genuinely quiet room).
  for (let t = 0; t < 2000; t += 50) {
    assert.equal(vad.process(0.001, t, true), null);
  }
  // 0.05 is below bargeInThreshold (0.08) but far above floor * margin
  // (~0.002). The absolute floor must still block it, even sustained.
  for (let t = 2000; t < 4000; t += 50) {
    assert.equal(vad.process(0.05, t, true), null);
  }
});

test("adaptive floor: the estimate resets when the interruption window closes", () => {
  const fresh = createEnergyVAD(OPTS);
  const reopened = createEnergyVAD(OPTS);

  // `reopened` lives through a loud strict window first, driving its floor
  // estimate up toward 0.5...
  for (let t = 0; t < 1000; t += 50) {
    reopened.process(0.5, t, true);
  }
  // ...then the interruption window closes (strict -> false for a frame).
  reopened.process(QUIET, 1000, false);

  // From here, feed both detectors the identical modest-level strict
  // sequence (same relative timing) and compare outputs step for step. If
  // the reset happened, `reopened` behaves exactly like a brand-new
  // detector instead of carrying its old ~0.5 floor forward.
  const freshEvents = [];
  const reopenedEvents = [];
  for (let t = 0; t < 2000; t += 50) {
    const ef = fresh.process(0.12, t, true);
    const er = reopened.process(0.12, t + 1050, true);
    if (ef) freshEvents.push(ef);
    if (er) reopenedEvents.push(er);
  }
  assert.deepEqual(reopenedEvents, freshEvents);
  assert.deepEqual(freshEvents, []);   // sanity: steady 0.12 alone never fires
});

test("adaptive floor: non-strict mode is completely unaffected", () => {
  // 0.05 is above the normal threshold (0.02); in non-strict mode this must
  // still fire on the ordinary schedule regardless of any floor estimate --
  // the floor is strict-only and must not leak into the non-strict path.
  const vad = createEnergyVAD(OPTS);
  assert.equal(vad.process(0.05, 0), null);
  assert.equal(vad.process(0.05, 100), null);   // still under 200ms
  assert.equal(vad.process(0.05, 200), "speech-start");
});

test("adaptive floor: freezes once a turn is nascent, so the caller's own sustained voice cannot raise the bar against them", () => {
  // Regression test for a bug in the first cut of the adaptive floor: it
  // updated the estimate on every strict call, including while a candidate
  // turn was already accumulating. That meant a caller's own interruption
  // kept dragging the estimate -- and therefore the requirement -- up
  // toward their own voice while they were still mid-word, raising the bar
  // out from under them before bargeInMinSpeechMs elapsed. Real animation
  // frames land roughly every 16ms (~60fps), not the coarser 50ms step used
  // elsewhere in this file -- and that finer cadence is exactly what
  // exposes the bug: ~20 update calls land inside the 300ms sustain window
  // instead of ~6, giving the (buggy) unfrozen estimate enough steps to
  // climb meaningfully within a single interruption. Do not "simplify" this
  // back to a 50ms step; at that cadence the bug is invisible.
  const vad = createEnergyVAD(OPTS);
  // Settle the floor at a modest ambient/echo level first.
  for (let t = 0; t < 1000; t += 50) {
    assert.equal(vad.process(0.12, t, true), null);
  }
  // The caller interrupts at 0.4 -- only ~1.7x the frozen requirement
  // (0.12 * bargeInMargin = 0.24), but well under what an unfrozen estimate
  // would demand by the end of the window (it would climb toward ~0.47,
  // permanently exceeding 0.4 and preventing speech-start from ever firing).
  const events = [];
  for (let t = 1000; t <= 1400; t += 15) {
    const e = vad.process(0.4, t, true);
    if (e) events.push(e);
  }
  assert.deepEqual(events, ["speech-start"]);
});
