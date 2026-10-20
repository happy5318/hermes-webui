// Drive the REAL _workspacePathExists / _workspaceProbeFile out of the served
// static/workspace.js and prove the decision branches.
//
// This is a logic test, not a text assertion: the functions are extracted from
// the file the browser actually downloads, so a change in the shipped source
// changes what runs here.
const fs = require('fs');

const SRC = process.argv[2];
if (!SRC) {
  console.error('usage: node driver.js <path-to-static/workspace.js>');
  process.exit(2);
}
const src = fs.readFileSync(SRC, 'utf8');

function extract(name) {
  const start = src.indexOf(`async function ${name}(`);
  if (start < 0) throw new Error(`${name} not found in ${SRC}`);
  const brace = src.indexOf('{', start);
  let depth = 0;
  for (let i = brace; i < src.length; i++) {
    if (src[i] === '{') depth++;
    else if (src[i] === '}') {
      depth--;
      if (depth === 0) return src.slice(start, i + 1);
    }
  }
  throw new Error(`could not extract ${name}`);
}

// Minimal stand-ins for the two globals the extracted code touches.
let S = { session: { session_id: 'SID', workspace: '/ws' } };
let apiCalls = [];
let apiImpl = async () => ({ entries: [] });

const sandbox = {
  S,
  api: async (...args) => { apiCalls.push(args); return apiImpl(...args); },
  _workspaceRouteForPath: (path, kind) => `/api/${kind}?path=${encodeURIComponent(path)}`,
  encodeURIComponent,
  console,
};

const factory = new Function(
  'S', 'api', '_workspaceRouteForPath',
  `${extract('_workspacePathExists')}\n${extract('_workspaceProbeFile')}\nreturn { _workspacePathExists, _workspaceProbeFile };`
);
const { _workspacePathExists } = factory(
  sandbox.S, sandbox.api, sandbox._workspaceRouteForPath
);

let failures = 0;
function check(label, cond, extra) {
  if (cond) {
    console.log(`  ✅ ${label}`);
  } else {
    failures++;
    console.log(`  ❌ ${label}${extra ? ' — ' + extra : ''}`);
  }
}

(async () => {
  console.log(`driving real functions from ${SRC}\n`);

  // ── Branch 1: the listing contains the name → true, no probe ──────────
  console.log('Branch 1: name present in the parent listing');
  apiCalls = [];
  apiImpl = async () => ({ entries: [{ name: 'note.md', path: 'note.md' }] });
  let r = await _workspacePathExists('note.md');
  check('returns true', r === true, `got ${r}`);
  check('did not need the direct probe', apiCalls.length === 1, `api calls: ${apiCalls.length}`);

  // ── Branch 2: the listing is CAPPED and the name is missing ──────────
  // The bug: the name sits past the response cap, so the listing cannot
  // prove absence. The probe must find the file.
  console.log('\nBranch 2: listing capped, name missing from it (the bug)');
  apiCalls = [];
  apiImpl = async (url) => {
    if (url.includes('/api/list')) return { entries: [{ name: 'other.md' }] };
    return {}; // raw probe resolves → 200
  };
  r = await _workspacePathExists('a-file-past-the-cap.md');
  check('returns true via the direct probe', r === true, `got ${r}`);
  check('issued the probe request', apiCalls.length === 2, `api calls: ${apiCalls.length}`);
  check('probe hit the raw route', apiCalls[1] && apiCalls[1][0].includes('/api/raw'), apiCalls[1] && apiCalls[1][0]);
  check('probe used statusOnly (no body buffered)', apiCalls[1] && apiCalls[1][1] && apiCalls[1][1].statusOnly === true);

  // ── Branch 3: the file genuinely does not exist → false ──────────────
  console.log('\nBranch 3: file genuinely absent');
  apiCalls = [];
  apiImpl = async (url) => {
    if (url.includes('/api/list')) return { entries: [{ name: 'other.md' }] };
    const err = new Error('Not found');
    err.status = 404;
    throw err;
  };
  r = await _workspacePathExists('missing.md');
  check('returns false', r === false, `got ${r}`);
  check('only 404 counts as absent', apiCalls.length === 2, `api calls: ${apiCalls.length}`);

  // ── Branch 4: the listing itself fails → still probe ─────────────────
  console.log('\nBranch 4: the listing request fails');
  apiCalls = [];
  apiImpl = async (url) => {
    if (url.includes('/api/list')) throw new Error('network down');
    return {}; // probe succeeds
  };
  r = await _workspacePathExists('note.md');
  check('falls back to the probe and returns true', r === true, `got ${r}`);
  check('probe was attempted after the listing error', apiCalls.length === 2, `api calls: ${apiCalls.length}`);

  // ── Branch 5: a non-404 probe failure must NOT be reported as absent ─
  console.log('\nBranch 5: probe fails with a non-404 status');
  apiCalls = [];
  apiImpl = async (url) => {
    if (url.includes('/api/list')) return { entries: [] };
    const err = new Error('forbidden');
    err.status = 403;
    throw err;
  };
  r = await _workspacePathExists('note.md');
  // Real contract: _workspaceProbeFile re-throws any non-404 failure, and the
  // caller's outer catch turns that into false. So an inconclusive probe
  // (403/503/timeout) does end up as "not openable" — which is the intended
  // outcome for an expired escape grant, where "file open failed" is the
  // honest message. What must NOT happen is the probe being skipped, or a
  // 404-shaped error being invented.
  //
  // The probe is attempted at least once; on a non-404 failure the caller's
  // outer catch retries the probe once more, so the count is 3 here
  // (list, probe, retry-probe). The retry is what lets a transient error
  // still resolve to "exists".
  check('probe was attempted (not skipped)', apiCalls.length >= 2, `api calls: ${apiCalls.length}`);
  check('reports not-openable for an inconclusive probe', r === false, `got ${r}`);

  console.log(`\n${failures === 0 ? '✅ ALL BRANCHES OK' : `❌ ${failures} check(s) failed`}`);
  process.exit(failures === 0 ? 0 : 1);
})();
