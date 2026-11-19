"""Both real parsers and host forwarding, without booting a robot or model."""
from itertools import combinations
import json
from pathlib import Path

import pytest

from pyrualean import cli
import pyrualean.play as play
from pyrualean.migrations import cli_migration, validate_migration


FLAGS = {
    "--openvl-visual-config": "openvl_visual",
    "--openvl-routed-config": "openvl_routed",
    "--openvl-capabilities-config": "openvl",
}


def arguments(tmp_path, runtime="openai"):
    return ["--robot", "libero", "--arm", "cells", "--task", "0", "--suite", "libero_spatial",
            "--out", str(tmp_path / "episode"), "--runtime", runtime, "--model", "unchanged-model",
            "--budget-s", "37", "--max-episode-steps", "123", "--max-turns", "7"]


def intercept_episode(monkeypatch):
    captured = []
    def run(cell, out, **kwargs):
        captured.append((cell, out, kwargs))
        out.mkdir(parents=True, exist_ok=True)
        result = {"status": "completed", "environment_success": False}
        (out / "result.json").write_text(json.dumps(result))
        return result
    monkeypatch.setattr(play, "play_episode", run)
    return captured


@pytest.mark.parametrize("flag,kind", FLAGS.items())
def test_top_level_forwards_to_real_child_parser_and_expected_adapter(tmp_path, monkeypatch, flag, kind):
    captured = intercept_episode(monkeypatch)
    commands = []
    def host(argv, **kwargs):
        commands.append(argv)
        assert argv[1:3] == ["-m", "pyrualean.play"]
        return play.main(argv[3:])
    monkeypatch.setattr(cli, "run_host", host)
    config = tmp_path / "native config.yml"
    assert cli.main(["play", *arguments(tmp_path), flag, str(config)]) == 0
    cell, _, options = captured[0]
    assert options["migration"] == {"kind": kind, "config": str(config.resolve())}
    validate_migration(options["migration"])
    assert commands[0][commands[0].index(flag) + 1] == str(config.resolve())
    assert options["runtime"] == "openai" and options["model"] == "unchanged-model"
    assert options["budget_s"] == 37 and options["max_turns"] == 7
    assert cell.max_episode_steps == 123
    assert options["openvl_evidence"] is None


@pytest.mark.parametrize("flag,kind", FLAGS.items())
def test_direct_submodule_selector_preserves_historical_kind(tmp_path, monkeypatch, flag, kind):
    captured = intercept_episode(monkeypatch)
    assert play.main([*arguments(tmp_path), flag, "relative-capabilities.yml"]) == 0
    assert captured[0][2]["migration"] == {"kind": kind, "config": str(Path("relative-capabilities.yml").resolve())}


@pytest.mark.parametrize("entry", ["top", "child"])
@pytest.mark.parametrize("runtime", ["codex", "claude", "openvlharness"])
@pytest.mark.parametrize("flag", list(FLAGS))
def test_configs_reject_wrong_runtime_before_host_or_episode(tmp_path, monkeypatch, capsys, entry, runtime, flag):
    monkeypatch.setattr(cli, "run_host", lambda *a, **k: pytest.fail("Host must not start"))
    monkeypatch.setattr(play, "play_episode", lambda *a, **k: pytest.fail("Episode must not start"))
    args = [*arguments(tmp_path, runtime), flag, "native.yml"]
    with pytest.raises(SystemExit) as exc:
        (cli.main(["play", *args]) if entry == "top" else play.main(args))
    assert exc.value.code == 2
    assert f"{flag} requires --runtime openai" in capsys.readouterr().err
    assert not (tmp_path / "episode").exists()


@pytest.mark.parametrize("entry", ["top", "child"])
@pytest.mark.parametrize("pair", list(combinations([*FLAGS, "--openvl-evidence"], 2)))
def test_all_selectors_and_evidence_are_mutually_exclusive(tmp_path, monkeypatch, capsys, entry, pair):
    monkeypatch.setattr(cli, "run_host", lambda *a, **k: pytest.fail("Host must not start"))
    monkeypatch.setattr(play, "play_episode", lambda *a, **k: pytest.fail("Episode must not start"))
    args = arguments(tmp_path)
    for flag in pair:
        args += [flag] if flag == "--openvl-evidence" else [flag, "native.yml"]
    with pytest.raises(SystemExit) as exc:
        (cli.main(["play", *args]) if entry == "top" else play.main(args))
    assert exc.value.code == 2 and "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize("entry", ["top", "child"])
@pytest.mark.parametrize("evidence", [False, True])
def test_omitting_config_preserves_baseline_or_explicit_legacy_evidence(tmp_path, monkeypatch, entry, evidence):
    captured = intercept_episode(monkeypatch)
    monkeypatch.setattr(cli, "run_host", lambda argv, **kwargs: play.main(argv[3:]))
    args = arguments(tmp_path) + (["--openvl-evidence"] if evidence else [])
    assert (cli.main(["play", *args]) if entry == "top" else play.main(args)) == 0
    assert captured[0][2]["migration"] is None
    assert captured[0][2]["openvl_evidence"] == ({} if evidence else None)


def test_programmatic_namespace_cannot_bypass_selector_conflict():
    from argparse import Namespace
    with pytest.raises(ValueError, match="only one"):
        cli_migration(Namespace(runtime="openai", openvl_evidence=True, openvl_visual_config=Path("native.yml")))
