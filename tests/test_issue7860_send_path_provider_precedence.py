"""Model-picker → send-path hand-off: which provider actually answers.

Same lexical-extractor + node-driver pattern as ``test_issue7860`` (which
follows #6131's precedent): the production functions are lifted out of
``static/ui.js`` and driven in a real node process, so the assertions run
against shipped code rather than a re-implementation.

Scope: the state the picker leaves behind once a selection has been made. The
session then stores the *bare* model name while ``session.model_provider`` is
still the provider the session was created under, and the send path is called
with that bare name — which is what the UI puts on the wire. The equal bare
name must not make the stale provider authoritative (#7860 review round 2):
equality of one stored field does not prove the pair was updated atomically.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
UI_JS = REPO_ROOT / "static" / "ui.js"
EXTRACTOR_JS = Path(__file__).resolve().parent / "_ui_js_extractor.js"
NODE = (
    "node"
    if subprocess.run(["which", "node"], capture_output=True).returncode == 0
    else None
)

_PROVIDER = "claude-subscription-directsdk-experimental"
_STALE = "openai-codex"

_DRIVER = r"""
const fs = require('fs');
const uiSrc = fs.readFileSync(process.argv[1], 'utf8');
const extractorSrc = fs.readFileSync(process.argv[2], 'utf8');
eval(extractorSrc);
for (const name of [
  '_providerFromModelValue', '_getOptionProviderId', '_modelStateForSelect',
  '_modelProviderForSend', '_readPersistedModelState',
]) {
  eval(extractFunction(uiSrc, name));
}

const PROVIDER = process.argv[3];
const STALE = process.argv[4];
const QUALIFIED = '@' + PROVIDER + ':claude-sonnet-5[1m]';
const BARE = 'claude-sonnet-5[1m]';

function catalogSelect(value) {
  const opt = {
    value, textContent: value,
    parentElement: { tagName: 'OPTGROUP', dataset: { provider: PROVIDER } },
  };
  return { id: 'modelSelect', options: [opt], value, selectedOptions: [opt] };
}

const LS = {};
globalThis.localStorage = {
  getItem: k => (k in LS ? LS[k] : null),
  setItem: (k, v) => { LS[k] = String(v); },
  removeItem: k => { delete LS[k]; },
  clear: () => { for (const k of Object.keys(LS)) delete LS[k]; },
};

const out = {};

// (a) The state the picker actually leaves behind: the session stores the BARE
//     model name, model_provider is still the stale one from creation, and the
//     picker's current option is the provider's qualified entry.
const picker = catalogSelect(QUALIFIED);
globalThis.$ = function () { return picker; };
globalThis.S = { session: { model: BARE, model_provider: STALE } };
out.send_from_bare_session = _modelProviderForSend(BARE);

// (b) The same state with NO picker mounted (a fresh page before the catalog
//     renders) and no persisted state: the stored provider is the account
//     default and must still be the answer.
globalThis.$ = function () { return null; };
localStorage.clear();
globalThis.S = { session: { model: '', model_provider: STALE } };
out.send_from_fresh_session = _modelProviderForSend(BARE);

// (c) An explicit provider tag on the outgoing value outranks everything.
globalThis.S = { session: { model: BARE, model_provider: STALE } };
out.send_from_qualified_value = _modelProviderForSend(QUALIFIED);

process.stdout.write(JSON.stringify(out));
"""


def _run_driver() -> dict:
    proc = subprocess.run(
        [NODE, "-e", _DRIVER, str(UI_JS), str(EXTRACTOR_JS), _PROVIDER, _STALE],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_stale_provider_is_not_authoritative_when_the_model_is_already_bare():
    """#7860 round 2: the malformed pair the picker actually writes.

    ``session.model`` is already the bare name after a selection, so the send
    path is called with the bare name while ``session.model_provider`` is still
    the session's creation provider. The equal bare name must not promote that
    stale value — the dropdown option for the model decides, and the wire gets
    a matching model/provider pair instead of e.g. a Claude model pinned to
    Codex (the bogus "Codex quota exhausted (429)").
    """
    result = _run_driver()

    assert result["send_from_bare_session"] == _PROVIDER, (
        "the stale session provider still wins when the outgoing model is "
        "already bare — the malformed @provider:model pair survives on the wire"
    )


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_session_provider_stands_when_nothing_else_answers():
    """Control: nothing selected anywhere, so the account default stands."""
    result = _run_driver()

    assert result["send_from_fresh_session"] == _STALE


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_explicit_provider_tag_outranks_the_session_provider():
    """Control: a qualified outgoing value carries its own provider."""
    result = _run_driver()

    assert result["send_from_qualified_value"] == _PROVIDER
