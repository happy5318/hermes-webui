"""Behavioral race tests for the OpenAI-compatible TTS chunk chain.

Reviewer feedback (nesquena-hermes on #7529): `_ttsSpeaking` is a single
global boolean and cannot distinguish playback A from playback B.  A
stop→start gap lets a late fetch or `onended` callback from A resume the
cancelled chain under B.  Round-3 review added four more objective blockers:

  1. The prefetch pipeline must pace requests to the server's per-client
     2 s TTS rate limit (api/routes.py _TtsRateLimiter) or a fast synthesis
     makes the prefetched chunk N+1 hit HTTP 429 and playback stops after
     chunk one.
  2. Automatic replacement (autoReadLastAssistant calls _playOpenaiTts()
     without stopping the current source) can overlap audio and lose the
     active handle.
  3. Synchronous Web Audio failures (createBufferSource/connect/start
     throwing) must not leave speaking state dangling.
  4. The behavioral driver must cover those schedules, not just
     stop→start transitions.

Round-4 review added three more objective blockers:

  1. The pacing clock was per-playback: a replacement playback reset it to
     zero and issued its first request inside the previous playback's
     server window (HTTP 429). Pacing must be shared across playback
     generations AND across engines (the server limiter is per-client, so
     an Edge/ElevenLabs request occupies the same window).
  2. Automatic replacement still bypassed the canonical stop boundary:
     autoReadLastAssistant() dispatched straight to each engine, so
     browser speech was not cancelled and manual buttons stayed stuck in
     the speaking state.
  3. Terminal failure could still launch a later network request: the
     prefetch timer was created before decode/construct/start succeeded,
     and `_fail` did not invalidate the generation, so a paced timer could
     fire a follow-on request after a terminal error. `AudioContext.resume()`
     rejection was also unobserved.

These tests drive the real extracted functions under node with controllable
fetch promises and fake audio sources and assert the generation/pacing/
ownership guards:

  1. Start A, leave its first fetch pending, stop A, start B, then resolve
     A; assert no A source starts. B's first request must wait for the
     pacing window instead of issuing immediately.
  2. Start A, prefetch A2 (paced), stop A, start B, fire A1 `onended`;
     assert A2 never issues a request and B remains the active source.
  3. Reject an abandoned prefetch and assert there is no
     `unhandledrejection` (prefetches settle into {ok}/{err} immediately).
  4. Automatic replacement: play A, call _playOpenaiTts() for B with no
     intervening stop; assert A is stopped, B's first request is paced,
     B owns the active handle, a stale A `onended` cannot erase B, and a
     later stopTTS() stops B.
  5. Synchronous AudioContext/source failures clear speaking state via the
     terminal handler, and no follow-on request fires past the pacing
     window (failure-matrix).
  6. A prior request from another TTS engine occupies the client window:
     the first OpenAI request waits instead of issuing immediately.
  7. decodeAudioData error: terminal handler clears state and zero
     follow-on requests fire past the pacing window.
  8. AudioContext.resume() rejection is observed and routed to the
     terminal handler, with zero follow-on requests.
  9. HTTP 429 is retried (bounded) after the pacing window and playback
     succeeds.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")

_DRIVER = r'''
const fs = require('fs');
const ui = fs.readFileSync(process.argv[2], 'utf8');

function extractFunction(name) {
  const re = new RegExp('function\\s+' + name + '\\s*\\(');
  const start = ui.search(re);
  if (start < 0) throw new Error(name + ' not found');
  let i = ui.indexOf('{', ui.indexOf(')', start));
  let depth = 1;
  i += 1;
  while (depth > 0 && i < ui.length) {
    if (ui[i] === '{') depth += 1;
    else if (ui[i] === '}') depth -= 1;
    i += 1;
  }
  return ui.slice(start, i);
}

// ---- module-level state the extracted functions touch ----
let _ttsSpeaking = false;
let _ttsGeneration = 0;
let _openaiTtsMinGapMs = 2000;
let _openaiTtsLastRequestTs = 0;
let _ttsCurrentUtterance = null;
let _ttsChunkQueue = [];
let _ttsChunkIndex = 0;
let _ttsActiveBtn = null;
let _playingEdgeAudio = null;
let _ttsAudioCtx = null;

// ---- controllable fakes ----
const toasts = [];
function showToast(msg) { toasts.push(msg); }

let fetchCalls = [];
function resetFetch() { fetchCalls = []; }
globalThis.fetch = function (url, opts) {
  return new Promise((resolve, reject) => {
    fetchCalls.push({ resolve, reject, url, opts });
  });
};

const startedSources = [];
class FakeAudioContext {
  constructor() {
    this.state = 'running';
    this.destination = {};
    this._decode = null;
    this.decodeCalls = 0;
    this.throwOnCreateSource = false;
    this.throwOnStart = false;
    this.rejectResume = false;
  }
  resume() {
    if (this.rejectResume) return Promise.reject(new Error('resume denied'));
    this.state = 'running';
    return Promise.resolve();
  }
  decodeAudioData(buf, ok, err) { this.decodeCalls += 1; this._decode = { ok, err }; }
  createBufferSource() {
    if (this.throwOnCreateSource) throw new Error('boom: createBufferSource');
    const ctx = this;
    const src = {
      buffer: null,
      connect() {},
      start() {
        if (ctx.throwOnStart) throw new Error('boom: start');
        this.started = true;
      },
      stop() { this.stopped = true; },
      disconnect() { this.disconnected = true; },
      onended: null,
    };
    startedSources.push(src);
    return src;
  }
}
const window = { AudioContext: FakeAudioContext };
const document = { baseURI: 'http://localhost/', querySelectorAll() { return []; } };
const location = { href: 'http://localhost/' };

eval(['_splitForTTS', '_playOpenaiTts', '_getTtsAudioCtx', '_playAudioBuf', 'stopTTS', '_noteTtsRequestSent']
  .map(extractFunction).join('\n'));

const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

function resetState() {
  _ttsSpeaking = false; _ttsGeneration = 0; _ttsCurrentUtterance = null;
  _ttsChunkQueue = []; _ttsChunkIndex = 0; _ttsActiveBtn = null;
  _playingEdgeAudio = null; _ttsAudioCtx = null;
  _openaiTtsLastRequestTs = 0;
  toasts.length = 0; fetchCalls.length = 0; startedSources.length = 0;
}

const longText = Array(60).fill('这是一段足够长的用于测试分块播放的中文文本段落。').join('');
const okResp = () => Promise.resolve({ ok: true, arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)) });
const rateLimitedResp = () => Promise.resolve({ ok: false, status: 429, json: () => Promise.resolve({}) });

// 1. Start A, leave its first fetch pending, stop A, start B, resolve A.
//    The generation guard must stop A's chain before it reaches the
//    AudioContext; resolving A's fetch must NOT create a context or issue
//    a decode. Only B may decode — and B's first request must wait for
//    the pacing window (shared clock), not issue immediately inside A's
//    server cooldown.
function scenario1() {
  resetState();
  _openaiTtsMinGapMs = 200;
  _playOpenaiTts('AAAA', { dataset: {} });
  if (fetchCalls.length !== 1) throw new Error('s1: A fetch not issued');
  stopTTS();
  _playOpenaiTts('BB', { dataset: {} });
  if (fetchCalls.length !== 1) {
    throw new Error('s1: B issued during A cooldown (calls=' + fetchCalls.length + ')');
  }
  fetchCalls[0].resolve(okResp());                       // late A1 response
  return sleep(30).then(() => {
    if (_ttsAudioCtx !== null) {
      throw new Error('s1: A resumed the chain after stop->start');
    }
    if (startedSources.length !== 0) {
      throw new Error('s1: A source started after stop->start');
    }
    return sleep(250);                                   // pacing window passes
  }).then(() => {
    if (fetchCalls.length !== 2) {
      throw new Error('s1: B fetch not issued after cooldown (calls=' + fetchCalls.length + ')');
    }
    fetchCalls[1].resolve(okResp());           // B decodes normally
    return sleep(30);
  }).then(() => {
    if (_ttsAudioCtx === null || _ttsAudioCtx.decodeCalls !== 1) {
      throw new Error('s1: B did not decode exactly once (calls=' +
        (_ttsAudioCtx && _ttsAudioCtx.decodeCalls) + ')');
    }
    if (startedSources.length !== 0) {
      throw new Error('s1: source started before decode ok');
    }
    return 'PASS';
  });
}

// 2. Start A, resolve A1, prefetch A2 (request must be paced to the
//    server rate window), stop A, start B, fire A1 onended, then let the
//    stale A2 prefetch lapse: assert A2 is never requested, never decoded,
//    and B remains the active source. B's own first request is paced too.
function scenario2() {
  resetState();
  _openaiTtsMinGapMs = 200;
  _playOpenaiTts(longText, { dataset: {} });           // fetch A1 (call 0)
  fetchCalls[0].resolve(okResp());
  return sleep(30).then(() => {
    if (fetchCalls.length !== 1) {
      throw new Error('s2: A2 prefetch was not paced (calls=' + fetchCalls.length + ')');
    }
    const ctxA = _ttsAudioCtx;
    if (!ctxA || !ctxA._decode) throw new Error('s2: A decode not requested');
    ctxA._decode.ok({});                              // A1 plays; src0 started
    if (startedSources.length !== 1) throw new Error('s2: A1 source missing');
    stopTTS();
    _playOpenaiTts('B', { dataset: {} });
    if (fetchCalls.length !== 1) {
      throw new Error('s2: B issued during A cooldown (calls=' + fetchCalls.length + ')');
    }
    startedSources[0].onended();                      // late A1 onended
    return sleep(250);                                // pacing window passes
  }).then(() => {
    // A2's paced timer must have fired but found the generation stale.
    // B's request goes out only after its own pacing wait.
    if (fetchCalls.length !== 2) {
      throw new Error('s2: unexpected fetch after stop->start (calls=' + fetchCalls.length + ')');
    }
    fetchCalls[1].resolve(okResp());                  // B plays: then → decodeAudioData
    return sleep(30);
  }).then(() => {
    if (_ttsAudioCtx.decodeCalls !== 2) {
      throw new Error('s2: abandoned A2 was decoded after stop->start (calls=' + _ttsAudioCtx.decodeCalls + ')');
    }
    _ttsAudioCtx._decode.ok({});                      // B plays; src1 started
    if (startedSources.length !== 2) {
      throw new Error('s2: unexpected extra source started: ' + startedSources.length);
    }
    if (_playingEdgeAudio !== startedSources[1]) {
      throw new Error('s2: B is not the active source');
    }
    return 'PASS';
  });
}

// 3. Reject an abandoned prefetch; no unhandled rejection, no toast.
function scenario3() {
  resetState();
  _openaiTtsMinGapMs = 200;
  const unhandled = [];
  const onUnhandled = (e) => { unhandled.push(e); };
  process.on('unhandledRejection', onUnhandled);
  _playOpenaiTts(longText, { dataset: {} });
  fetchCalls[0].resolve(okResp());
  return sleep(30).then(() => {
    _ttsAudioCtx._decode.ok({});                      // A1 plays; A2 prefetch paced
    if (fetchCalls.length !== 1) throw new Error('s3: A2 prefetch not paced');
    return sleep(250);                                // A2 request goes out
  }).then(() => {
    if (fetchCalls.length !== 2) throw new Error('s3: A2 prefetch missing');
    stopTTS();
    fetchCalls[1].reject(new Error('network down'));  // abandon the prefetch
    return sleep(30);
  }).then(() => {
    process.removeListener('unhandledRejection', onUnhandled);
    if (unhandled.length !== 0) throw new Error('s3: unhandled rejection: ' + unhandled[0]);
    if (toasts.length !== 0) throw new Error('s3: error surfaced for abandoned prefetch: ' + toasts[0]);
    return 'PASS';
  });
}

// 4. Automatic replacement without an intervening stop: A is playing,
//    _playOpenaiTts() starts B directly (the autoReadLastAssistant path).
//    A must be stopped/released, B's first request paced, B owns the
//    active handle, a stale A1 onended must not erase B, and a later
//    stopTTS() stops B.
function scenario4() {
  resetState();
  _openaiTtsMinGapMs = 200;
  _playOpenaiTts(longText, { dataset: {} });          // A: fetch A1 (call 0)
  fetchCalls[0].resolve(okResp());
  return sleep(30).then(() => {
    if (fetchCalls.length !== 1) throw new Error('s4: A2 not paced yet');
    _ttsAudioCtx._decode.ok({});                      // A1 plays; src0 started
    if (startedSources.length !== 1) throw new Error('s4: A1 source missing');
    _playOpenaiTts('BBBB', { dataset: {} });          // auto-replace, no stop
    if (fetchCalls.length !== 1) {
      throw new Error('s4: B issued during A cooldown (calls=' + fetchCalls.length + ')');
    }
    if (!startedSources[0].stopped) throw new Error('s4: A source not stopped on replacement');
    if (!startedSources[0].disconnected) throw new Error('s4: A source not disconnected on replacement');
    startedSources[0].onended();                      // stale A1 onended before B owns handle
    return sleep(250);                                // pacing window passes
  }).then(() => {
    if (fetchCalls.length !== 2) {
      throw new Error('s4: B fetch not issued after cooldown (calls=' + fetchCalls.length + ')');
    }
    fetchCalls[1].resolve(okResp());                  // B decodes
    return sleep(30);
  }).then(() => {
    if (_ttsAudioCtx.decodeCalls !== 2) {
      throw new Error('s4: decode calls != A1+B1: ' + _ttsAudioCtx.decodeCalls);
    }
    _ttsAudioCtx._decode.ok({});                      // B plays; src1 started
    startedSources[0].onended();                      // stale A1 onended after B owns handle
    if (_playingEdgeAudio !== startedSources[1]) {
      throw new Error('s4: B does not own the active handle');
    }
    if (startedSources.length !== 2) {
      throw new Error('s4: extra source started: ' + startedSources.length);
    }
    stopTTS();                                        // must reach B
    if (!startedSources[1].stopped) throw new Error('s4: stopTTS did not stop B');
    return 'PASS';
  });
}

// 5. Synchronous Web Audio failure: createBufferSource throws inside the
//    decode success callback. The terminal handler must clear speaking
//    state, toast the error, never leave an unhandled rejection, and no
//    follow-on request may fire past the pacing window (failure-matrix).
function scenario5() {
  resetState();
  _openaiTtsMinGapMs = 200;
  const unhandled = [];
  const onUnhandled = (e) => { unhandled.push(e); };
  process.on('unhandledRejection', onUnhandled);
  _playOpenaiTts(longText, { dataset: {} });
  fetchCalls[0].resolve(okResp());
  return sleep(30).then(() => {
    if (!_ttsAudioCtx || !_ttsAudioCtx._decode) throw new Error('s5: decode not requested');
    _ttsAudioCtx.throwOnCreateSource = true;
    _ttsAudioCtx._decode.ok({});                      // synchronous throw inside callback
    return sleep(30);
  }).then(() => {
    if (_ttsSpeaking !== false) throw new Error('s5: speaking state left dangling');
    if (_playingEdgeAudio !== null) throw new Error('s5: active handle not cleared');
    if (toasts.length === 0) throw new Error('s5: no error toast');
    if (startedSources.length !== 0) throw new Error('s5: a source started despite failure');
    return sleep(300);                                // advance past the pacing window
  }).then(() => {
    process.removeListener('unhandledRejection', onUnhandled);
    if (unhandled.length !== 0) throw new Error('s5: unhandled rejection: ' + unhandled[0]);
    if (fetchCalls.length !== 1) {
      throw new Error('s5: follow-on request after terminal failure (calls=' + fetchCalls.length + ')');
    }
    return 'PASS';
  });
}

// 6. A prior request from another TTS engine (Edge/ElevenLabs) occupies
//    the client-wide server window: the first OpenAI request must wait
//    for the pacing window instead of issuing immediately.
function scenario6() {
  resetState();
  _openaiTtsMinGapMs = 200;
  _openaiTtsLastRequestTs = Date.now();               // e.g. Edge engine request just went out
  _playOpenaiTts('AAAA', { dataset: {} });
  if (fetchCalls.length !== 0) {
    throw new Error('s6: first request issued during another engine cooldown');
  }
  return sleep(250).then(() => {
    if (fetchCalls.length !== 1) {
      throw new Error('s6: request not issued after cooldown (calls=' + fetchCalls.length + ')');
    }
    return 'PASS';
  });
}

// 7. decodeAudioData error: the terminal handler clears speaking state and
//    toasts, and zero follow-on requests fire past the pacing window.
function scenario7() {
  resetState();
  _openaiTtsMinGapMs = 200;
  const unhandled = [];
  const onUnhandled = (e) => { unhandled.push(e); };
  process.on('unhandledRejection', onUnhandled);
  _playOpenaiTts(longText, { dataset: {} });
  fetchCalls[0].resolve(okResp());
  return sleep(30).then(() => {
    if (!_ttsAudioCtx || !_ttsAudioCtx._decode) throw new Error('s7: decode not requested');
    _ttsAudioCtx._decode.err(new Error('decode boom')); // decode failure path
    return sleep(30);
  }).then(() => {
    if (_ttsSpeaking !== false) throw new Error('s7: speaking state left dangling');
    if (_playingEdgeAudio !== null) throw new Error('s7: active handle not cleared');
    if (toasts.length === 0) throw new Error('s7: no error toast');
    return sleep(300);                                // advance past the pacing window
  }).then(() => {
    process.removeListener('unhandledRejection', onUnhandled);
    if (unhandled.length !== 0) throw new Error('s7: unhandled rejection: ' + unhandled[0]);
    if (fetchCalls.length !== 1) {
      throw new Error('s7: follow-on request after decode failure (calls=' + fetchCalls.length + ')');
    }
    return 'PASS';
  });
}

// 8. AudioContext.resume() rejection (autoplay policy) is observed and
//    routed to the terminal handler; zero follow-on requests.
function scenario8() {
  resetState();
  _openaiTtsMinGapMs = 200;
  const unhandled = [];
  const onUnhandled = (e) => { unhandled.push(e); };
  process.on('unhandledRejection', onUnhandled);
  _playOpenaiTts(longText, { dataset: {} });
  fetchCalls[0].resolve(okResp());
  return sleep(30).then(() => {
    if (!_ttsAudioCtx || !_ttsAudioCtx._decode) throw new Error('s8: decode not requested');
    _ttsAudioCtx.state = 'suspended';
    _ttsAudioCtx.rejectResume = true;
    _ttsAudioCtx._decode.ok({});                      // doStart waits on resume()
    return sleep(30);
  }).then(() => {
    if (_ttsSpeaking !== false) throw new Error('s8: speaking state left dangling');
    if (_playingEdgeAudio !== null) throw new Error('s8: active handle not cleared');
    if (toasts.length === 0) throw new Error('s8: no error toast');
    if (startedSources.length !== 0) throw new Error('s8: source started despite resume rejection');
    return sleep(300);                                // advance past the pacing window
  }).then(() => {
    process.removeListener('unhandledRejection', onUnhandled);
    if (unhandled.length !== 0) throw new Error('s8: unhandled rejection: ' + unhandled[0]);
    if (fetchCalls.length !== 1) {
      throw new Error('s8: follow-on request after resume rejection (calls=' + fetchCalls.length + ')');
    }
    return 'PASS';
  });
}

// 9. HTTP 429 is retried (bounded) after the pacing window; playback
//    succeeds on the retry and no error toast is shown.
function scenario9() {
  resetState();
  _openaiTtsMinGapMs = 200;
  _playOpenaiTts(longText, { dataset: {} });
  if (fetchCalls.length !== 1) throw new Error('s9: first fetch not issued');
  fetchCalls[0].resolve(rateLimitedResp());           // server window still open
  return sleep(300).then(() => {                      // retry after pacing window
    if (fetchCalls.length !== 2) {
      throw new Error('s9: no retry after 429 (calls=' + fetchCalls.length + ')');
    }
    if (toasts.length !== 0) throw new Error('s9: error toast on retriable 429: ' + toasts[0]);
    fetchCalls[1].resolve(okResp());                  // retry succeeds
    return sleep(30);
  }).then(() => {
    if (!_ttsAudioCtx || _ttsAudioCtx.decodeCalls !== 1) {
      throw new Error('s9: retried chunk not decoded (calls=' +
        (_ttsAudioCtx && _ttsAudioCtx.decodeCalls) + ')');
    }
    _ttsAudioCtx._decode.ok({});                      // plays; src0 started
    if (startedSources.length !== 1) {
      throw new Error('s9: source not started after retry: ' + startedSources.length);
    }
    if (_ttsSpeaking !== true) throw new Error('s9: speaking state not active after retry');
    return 'PASS';
  });
}

// 10. start() throws synchronously after the source was created: the
//     terminal handler must stop/disconnect the partially constructed
//     source, clear speaking state, toast, and leave zero follow-on
//     requests past the pacing window.
function scenario10() {
  resetState();
  _openaiTtsMinGapMs = 200;
  const unhandled = [];
  const onUnhandled = (e) => { unhandled.push(e); };
  process.on('unhandledRejection', onUnhandled);
  _playOpenaiTts(longText, { dataset: {} });
  fetchCalls[0].resolve(okResp());
  return sleep(30).then(() => {
    if (!_ttsAudioCtx || !_ttsAudioCtx._decode) throw new Error('s10: decode not requested');
    _ttsAudioCtx.throwOnStart = true;
    _ttsAudioCtx._decode.ok({});                      // src created; start() throws
    return sleep(30);
  }).then(() => {
    if (startedSources.length !== 1) throw new Error('s10: source was not created');
    if (!startedSources[0].stopped) throw new Error('s10: partially created source not stopped');
    if (!startedSources[0].disconnected) throw new Error('s10: partially created source not disconnected');
    if (_ttsSpeaking !== false) throw new Error('s10: speaking state left dangling');
    if (_playingEdgeAudio !== null) throw new Error('s10: active handle not cleared');
    if (toasts.length === 0) throw new Error('s10: no error toast');
    return sleep(300);                                // advance past the pacing window
  }).then(() => {
    process.removeListener('unhandledRejection', onUnhandled);
    if (unhandled.length !== 0) throw new Error('s10: unhandled rejection: ' + unhandled[0]);
    if (fetchCalls.length !== 1) {
      throw new Error('s10: follow-on request after start failure (calls=' + fetchCalls.length + ')');
    }
    return 'PASS';
  });
}

const scenario = process.argv[3];
const runner = {
  scenario1: scenario1, scenario2: scenario2, scenario3: scenario3,
  scenario4: scenario4, scenario5: scenario5, scenario6: scenario6,
  scenario7: scenario7, scenario8: scenario8, scenario9: scenario9,
  scenario10: scenario10,
}[scenario];
if (!runner) throw new Error('unknown scenario: ' + scenario);
let outcome;
try {
  outcome = runner();
} catch (e) {
  process.stderr.write(String(e && e.stack || e));
  process.exit(1);
}
if (outcome && typeof outcome.then === 'function') {
  outcome.then(
    (verdict) => { process.stdout.write(JSON.stringify({ verdict: verdict })); process.exit(0); },
    (err) => { process.stderr.write(String(err && err.stack || err)); process.exit(1); }
  );
} else {
  process.stdout.write(JSON.stringify({ verdict: outcome }));
  process.exit(0);
}
'''


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
@pytest.mark.parametrize("scenario", [
    "scenario1", "scenario2", "scenario3", "scenario4", "scenario5",
    "scenario6", "scenario7", "scenario8", "scenario9", "scenario10",
])
def test_openai_tts_chunk_chain_race(tmp_path, scenario):
    """Behavioral race coverage: late callbacks from a stopped playback must
    never resume the chunk chain under a new playback; requests are paced to
    the server rate limit across generations and engines; automatic
    replacement stops the prior source; terminal failures (decode, source
    construction, resume rejection) leave no follow-on requests; HTTP 429 is
    retried within bounds."""
    driver = tmp_path / "tts_race_driver.js"
    driver.write_text(_DRIVER, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(driver), str(REPO / "static" / "ui.js"), scenario],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, f"node driver failed: {result.stderr}"
    assert json.loads(result.stdout) == {"verdict": "PASS"}
