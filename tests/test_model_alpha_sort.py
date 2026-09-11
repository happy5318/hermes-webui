"""新增测试：provider 组内模型按字母排序（2026-09-05 用户需求）。

覆盖静态 catalog 路径的组内排序行为——模型列表不该保持 config/live
probe 的插入顺序，而应按 id 大小写不敏感字母序排列。
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

import api.config as config


REPO = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


_FRONTEND_SORT_DRIVER = r'''
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

eval([
  '_modelPickerSortValue',
  '_compareModelPickerEntries',
  '_sortModelPickerEntries',
].map(extractFunction).join('\n'));

const entries = [
  {id: 'jd-deepseek-v4-flash-0731', providerId: 'custom:newapi'},
  {id: 'sn-kimi-k3', providerId: 'custom:newapi'},
  {id: 'hf-deepseek-v4-flash', providerId: 'custom:newapi'},
  {id: 'sub-glm-5.3', providerId: 'custom:newapi'},
  {id: 'ab-glm-5.3-flash', providerId: 'custom:newapi'},
  {id: '@custom:newapi:MiniMax-M3', providerId: 'custom:newapi'},
  {id: '@custom:newapi:model-a:free', providerId: 'custom:newapi'},
];
process.stdout.write(JSON.stringify(_sortModelPickerEntries(entries).map(entry => entry.id)));
'''


def _install_fake_hermes_cli(monkeypatch):
    fake_pkg = types.ModuleType("hermes_cli")
    fake_pkg.__path__ = []

    fake_models = types.ModuleType("hermes_cli.models")
    fake_models.list_available_providers = lambda: []
    fake_models.provider_model_ids = lambda pid: []

    fake_auth = types.ModuleType("hermes_cli.auth")
    fake_auth.get_auth_status = lambda _pid: {}

    monkeypatch.setitem(sys.modules, "hermes_cli", fake_pkg)
    monkeypatch.setitem(sys.modules, "hermes_cli.models", fake_models)
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", fake_auth)
    monkeypatch.delitem(sys.modules, "agent.credential_pool", raising=False)
    monkeypatch.delitem(sys.modules, "agent", raising=False)

    config.invalidate_models_cache()


@pytest.fixture(autouse=True)
def _isolate_cache():
    _saved_cfg = copy.deepcopy(config.cfg)
    _saved_paths = {
        name: getattr(config, name)
        for name in ("_cfg_path", "_cfg_mtime", "_cfg_fingerprint")
    }
    try:
        config.invalidate_models_cache()
    except Exception:
        pass
    yield
    try:
        config.invalidate_models_cache()
    except Exception:
        pass
    try:
        if isinstance(config.cfg, dict):
            config.cfg.clear()
            config.cfg.update(_saved_cfg)
        for name, value in _saved_paths.items():
            setattr(config, name, value)
    except Exception:
        pass


def _setup_config(tmp_path, monkeypatch, yaml_text):
    _install_fake_hermes_cli(monkeypatch)

    cfgfile = tmp_path / "config.yaml"
    cfgfile.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setattr(config, "_get_config_path", lambda: cfgfile)

    auth_path = tmp_path / "auth.json"
    monkeypatch.setattr(config, "_get_auth_store_path", lambda: auth_path)

    config.reload_config()


class TestInGroupModelAlphabetical:
    def test_static_catalog_sorts_models_within_group(self, tmp_path, monkeypatch):
        """组内模型按 id 忽略大小写字母排序，而非 config 插入顺序。"""
        _setup_config(
            tmp_path,
            monkeypatch,
            (
                "model:\n"
                "  provider: custom:mylocal\n"
                "  default: z-last\n"
                "custom_providers:\n"
                "  - name: MyLocal\n"
                "    base_url: http://localhost:8080/v1\n"
                "    api_key: local-key\n"
                "    models:\n"
                "      - z-last\n"
                "      - m-mid\n"
                "      - a-first\n"
            ),
        )

        result = config.get_available_models(force_refresh=True)
        groups = result.get("groups", [])
        custom = next((g for g in groups if g.get("provider_id") == "custom:mylocal"), None)
        assert custom is not None, f"custom:mylocal group missing: {[g.get('provider_id') for g in groups]}"
        ids = [m.get("id") for m in custom.get("models", [])]
        assert ids == ["a-first", "m-mid", "z-last"], (
            f"expected alphabetical model order, got {ids}"
        )


@pytest.mark.skipif(NODE is None, reason="node not on PATH")
def test_frontend_picker_sort_ignores_provider_routing_prefix(tmp_path):
    """自绘下拉框按模型 ID 排序，不按 @provider: 路由前缀或异步到达顺序排序。"""
    driver = tmp_path / "frontend_sort_driver.js"
    driver.write_text(_FRONTEND_SORT_DRIVER, encoding="utf-8")
    result = subprocess.run(
        [NODE, str(driver), str(REPO / "static" / "ui.js")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "ab-glm-5.3-flash",
        "hf-deepseek-v4-flash",
        "jd-deepseek-v4-flash-0731",
        "@custom:newapi:MiniMax-M3",
        "@custom:newapi:model-a:free",
        "sn-kimi-k3",
        "sub-glm-5.3",
    ]
