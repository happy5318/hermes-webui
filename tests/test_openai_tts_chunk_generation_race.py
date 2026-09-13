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

These tests drive the real extracted functions under node with controllable
fetch promises and fake audio sources and assert the generation/pacing/
ownership guards:

  1. Start A, leave its first fetch pending, stop A, start B, then resolve
     A; assert no A source starts.
  2. Start A, prefetch A2 (paced), stop A, start B, fire A1 `onended`;
     assert A2 never issues a request and B remains the active source.
  3. Reject an abandoned prefetch and assert there is no
     `unhandledrejection` (prefetches settle into {ok}/{err} immediately).
  4. Automatic replacement: play A, call _playOpenaiTts() for B with no
     intervening stop; assert A is stopped, B owns the active handle, a
     stale A `onended` cannot erase B, and a later stopTTS() stops B.
  5. Synchronous AudioContext/source failures clear speaking state via the
     terminal handler (toast + no dangling _ttsSpeaking).
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
  }
  resume() {}
  decodeAudioData(buf, ok, err) { this.decodeCalls += 1; this._decode = { ok, err }; }
  createBufferSource() {
    if (this.throwOnCreateSource) throw new Error('boom: createBufferSource');
    const src = {
      buffer: null,
      connect() {},
      start() { startedSources.push(this); this.started = true; },
      stop() { this.stopped = true; },
      disconnect() { this.disconnected = true; },
      onended: null,
    };
    return src;
  }
}
const window = { AudioContext: FakeAudioContext };
const document = { baseURI: 'http://localhost/', querySelectorAll() { return []; } };
const location = { href: 'http://localhost/' };

eval(['_splitForTTS', '_playOpenaiTts', '_getTtsAudioCtx', '_playAudioBuf', 'stopTTS']
  .map(extractFunction).join('\n'));

const sleep = (ms) => new Promise((res) => setTimeout(res, ms));

function resetState() {
  _ttsSpeaking = false; _ttsGeneration = 0; _ttsCurrentUtterance = null;
  _ttsChunkQueue = []; _ttsChunkIndex = 0; _ttsActiveBtn = null;
  _playingEdgeAudio = null; _ttsAudioCtx = null;
  toasts.length = 0; fetchCalls.length = 0; startedSources.length = 0;
}

const longText = Array(60).fill('这是一段足够长的用于测试分块播放的中文文本段落。').join('');
const okResp = () => Promise.resolve({ ok: true, arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)) });

// 1. Start A, leave its first fetch pending, stop A, start B, resolve A.
//    The generation guard must stop A's chain before it reaches the
//    AudioContext; resolving A's fetch must NOT create a context or issue
//    a decode. Only B may decode.
function scenario1() {
  resetState();
  _playOpenaiTts('AAAA', { dataset: {} });
  if (fetchCalls.length !== 1) throw new Error('s1: A fetch not issued');
  stopTTS();
  _playOpenaiTts('BB', { dataset: {} });
  if (fetchCalls.length !== 2) throw new Error('s1: B fetch not issued');
  fetchCalls[0].resolve(okResp());                       // late A1 response
  return sleep(30).then(() => {
    if (_ttsAudioCtx !== null) {
      throw new Error('s1: A resumed the chain after stop->start');
    }
    if (startedSources.length !== 0) {
      throw new Error('s1: A source started after stop->start');
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
//    and B remains the active source.
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
    _playOpenaiTts('B', { dataset: {} });             // fetch B (call 1), immediate
    startedSources[0].onended();                      // late A1 onended
    return sleep(250);                                // pacing window passes
  }).then(() => {
    // A2's paced timer must have fired but found the generation stale.
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
//    A must be stopped/released, B owns the active handle, a stale A1
//    onended must not erase B, and a later stopTTS() stops B.
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
    if (fetchCalls.length !== 2) throw new Error('s4: B fetch not issued');
    if (!startedSources[0].stopped) throw new Error('s4: A source not stopped on replacement');
    if (!startedSources[0].disconnected) throw new Error('s4: A source not disconnected on replacement');
    startedSources[0].onended();                      // stale A1 onended before B owns handle
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
//    state, toast the error, and never leave an unhandled rejection.
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
    process.removeListener('unhandledRejection', onUnhandled);
    if (unhandled.length !== 0) throw new Error('s5: unhandled rejection: ' + unhandled[0]);
    if (_ttsSpeaking !== false) throw new Error('s5: speaking state left dangling');
    if (_playingEdgeAudio !== null) throw new Error('s5: active handle not cleared');
    if (toasts.length === 0) throw new Error('s5: no error toast');
    if (startedSources.length !== 0) throw new Error('s5: a source started despite failure');
    return 'PASS';
  });
}

const scenario = process.argv[3];
const runner = {
  scenario1: scenario1, scenario2: scenario2, scenario3: scenario3,
  scenario4: scenario4, scenario5: scenario5,
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
@pytest.mark.parametrize("scenario", ["scenario1", "scenario2", "scenario3", "scenario4", "scenario5"])
def test_openai_tts_chunk_chain_race(tmp_path, scenario):
    """Behavioral race coverage: late callbacks from a stopped playback must
    never resume the chunk chain under a new playback; requests are paced to
    the server rate limit; automatic replacement stops the prior source;
    synchronous Web Audio failures clear speaking state."""
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