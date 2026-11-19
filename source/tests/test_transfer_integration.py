"""Real migration classes through the CLI/runtime boundary with scripted inference."""

import io
import json
from argparse import Namespace
from pathlib import Path

import pytest
from PIL import Image

from pyrualean.arms import ArmToolkit
from pyrualean.migrations import cli_migration, configure_migration, migration_prompt
from pyrualean.openai_runtime import run_openai_agent
from tests.test_evo_harness import frozen_config
from tests.test_openai_runtime import ScriptedTransport, _reply
from tests.test_pos_harness import audit, belief

CONFIGS = Path(__file__).resolve().parents[1] / "integrations/migrations/general_harness"


def report(payload):
    return _reply([("state", "submit_harness_state", {"payload": payload})])


def finish():
    return _reply([("end", "finish", {"status": "stuck", "summary": "offline integration"})])


def run(robo, tmp_path, config, replies, *, initial_images=None, **options):
    kit = ArmToolkit(
        robo,
        arm="cells",
        workspace=tmp_path / "workspace",
        max_turns=4,
        feedback="pure",
        images="on-demand",
    )
    provenance = configure_migration(
        kit, config, task="Place the block", initial_images=initial_images or {}, runtime="openai"
    )
    transport = ScriptedTransport(replies)
    result = run_openai_agent(
        migration_prompt("Place the block", kit),
        [],
        kit,
        model="same-model",
        base_url="http://unused/v1",
        api_key=None,
        out_dir=tmp_path / "episode",
        budget_s=10,
        max_turns=4,
        transport=transport,
        sleep=lambda _: None,
        retries=0,
        **options,
    )
    assert not transport.replies, "The actual harness skipped part of its expected inference loop"
    return kit, provenance, transport, result


def new_config(kind, tmp_path):
    if kind == "evoharness":
        config, book = frozen_config(tmp_path)
        return config, book
    return json.loads((CONFIGS / f"{kind}.json").read_text()), None


def responses(kind, book=None):
    execute = _reply([("execute", "python", {"code": "answer = 40; print(answer + 2)"})])
    if kind == "pos":
        initial, after = belief("obs-0"), belief("obs-1")
        # No camera or primitive evidence is supplied in this arithmetic-only episode.
        initial["world"]["states"][0]["status"] = "uncertain"
        after["world"]["states"][0]["status"] = "uncertain"
        return [
            report(initial),
            report(audit()),
            execute,
            report(after),
            report(audit(False)),
            finish(),
        ]
    if kind == "longharness":
        return [
            report(
                {
                    "operations": [
                        {
                            "op": "upsert",
                            "id": "instruction",
                            "sources": ["s0001"],
                            "statement": "Place the block",
                            "scope": "instruction",
                            "slot": "task.goal",
                        }
                    ]
                }
            ),
            report(
                {
                    "established": [
                        {
                            "id": "task-goal",
                            "text": "Place the block",
                            "evidence": ["instruction"],
                            "scope": "instruction",
                        }
                    ],
                    "open": [
                        {
                            "id": "grasp-check",
                            "question": "Is the block held?",
                            "next_observation": "Inspect wrist view after lifting",
                        }
                    ],
                }
            ),
            execute,
            report(
                {
                    "operations": [
                        {
                            "op": "upsert",
                            "id": "cell-output",
                            "sources": ["s0002"],
                            "statement": "Python printed 42",
                            "scope": "history",
                            "slot": "cell.stdout",
                        }
                    ]
                }
            ),
            report(
                {
                    "hypotheses": [
                        {
                            "id": "python-worked",
                            "text": "Python printed 42",
                            "evidence": ["cell-output"],
                            "scope": "history",
                        }
                    ]
                }
            ),
            finish(),
        ]
    return [
        report({"selected_ids": [book["skills"][0]["id"]], "reason": "Applicable"}),
        execute,
        finish(),
    ]


@pytest.mark.parametrize("kind", ["pos", "longharness", "evoharness"])
def test_actual_harnesses_use_envelope_and_preserve_robot_interface(robo, tmp_path, kind):
    config, book = new_config(kind, tmp_path)
    # PoS admits complete two-call rounds, including the second observation.
    options = {"auxiliary_budget_s": 4} if kind == "pos" else {}
    kit, provenance, transport, result = run(
        robo, tmp_path, config, responses(kind, book), **options)
    assert provenance["kind"] == kind
    assert {tool["name"] for tool in kit.get_tools_spec()} == {"python", "finish"}
    assert kit.evidence_adapter is None and kit.stage_verifier is None
    assert kit.harnesspro_hook is None and kit.robo._action_feedback is None
    assert kit._globals["answer"] == 40 and kit._cell_index == 1
    assert not result["errors"] and not result["warnings"]
    expected_auxiliary = 1 if kind == "evoharness" else 4
    assert result["requests"] == expected_auxiliary + 2
    assert result["auxiliary_requests"] == expected_auxiliary
    assert result["input_tokens"] == 1000 * result["requests"]
    assert result["auxiliary_usage"]["prompt"] == 1000 * expected_auxiliary
    assert result["tool_calls"] == {"python": 1, "finish": 1}

    policies, auxiliaries = [], []
    for request in transport.requests:
        body = request["body"]
        assert body["model"] == "same-model"
        names = {tool["function"]["name"] for tool in body["tools"]}
        if names == {"submit_harness_state"}:
            auxiliaries.append(body)
            assert "payload field" in body["messages"][0]["content"]
        else:
            assert names == {"python", "finish"}
            policies.append(body)
    assert len(policies) == 2 and len(auxiliaries) == expected_auxiliary
    context = json.dumps(policies[-1]["messages"])
    if kind == "pos":
        assert kit.general_harness.revision == 2
        assert "POS_CONTEXT" in context and '"progress": false' not in context
        assert kit.general_harness.transitions[-1]["progress"] is False
    elif kind == "longharness":
        assert kit.general_harness.warning is None
        assert "Inspect wrist view after lifting" in context and "Python printed 42" in context
        assert kit.general_harness.evidence["cell-output"]["status"] == "active"
    else:
        assert book["skills"][0]["procedure"] in context
        assert kit.general_harness.context()["evaluation_updates"] is False
        assert json.loads(Path(config["book"]).read_text()) == book


@pytest.mark.parametrize("kind", ["pos", "longharness", "evoharness"])
def test_cli_resolves_new_kinds_and_relative_frozen_book(tmp_path, kind):
    config, _ = new_config(kind, tmp_path)
    path = tmp_path / f"{kind}.json"
    if kind == "evoharness":
        config["book"] = Path(config["book"]).name
    path.write_text(json.dumps(config))
    resolved = cli_migration(Namespace(general_harness_config=path, runtime="openai"))
    assert resolved["kind"] == kind
    if kind == "evoharness":
        assert Path(resolved["book"]).is_absolute()
        assert Path(resolved["book"]).exists()
    with pytest.raises(ValueError, match="requires runtime=openai"):
        cli_migration(Namespace(general_harness_config=path, runtime="codex"))


def test_actual_pos_rejects_missing_sentinel_budget_before_any_request(robo, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid PoS budget reached transport")
    monkeypatch.setattr(ScriptedTransport, "__call__", forbidden)
    with pytest.raises(ValueError, match="required maintenance stages and a planner"):
        run(robo, tmp_path, {"kind": "pos"}, [], max_requests=2)


def test_actual_longharness_rejects_invented_evidence_via_runtime(robo, tmp_path):
    replies = [
        report(
            {
                "operations": [
                    {
                        "op": "upsert",
                        "id": "invented",
                        "sources": ["s404"],
                        "statement": "Block is grasped",
                        "scope": "world",
                        "slot": "block.holding",
                    }
                ]
            }
        ),
        report({}),
        finish(),
    ]
    kit, _, _, result = run(robo, tmp_path, {"kind": "longharness"}, replies)
    assert not kit.general_harness.evidence
    assert kit.general_harness.receipts[0]["accepted"] is False
    assert not result["errors"] and result["python_cells"] == 0


def test_actual_evo_selector_rejects_fabricated_skill_without_changing_book(robo, tmp_path):
    config, book = new_config("evoharness", tmp_path)
    old = Path(config["book"]).read_bytes()
    kit, _, _, result = run(
        robo, tmp_path, config, [report({"selected_ids": ["imagined-skill"]}), finish()]
    )
    assert kit.general_harness.selection_status == "invalid_selection"
    assert not kit.general_harness.selected
    assert Path(config["book"]).read_bytes() == old
    assert kit.general_harness.book == book and result["auxiliary_requests"] == 1


def test_pos_initial_image_reaches_both_auxiliary_calls_with_provenance(robo, tmp_path):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, format="PNG")
    candidate = belief("obs-0-image-0")
    kit, _, transport, _ = run(
        robo,
        tmp_path,
        {"kind": "pos"},
        [report(candidate), report(audit()), finish()],
        initial_images={"front.png": buffer.getvalue()},
    )
    assert kit.general_harness.revision == 1
    for request in transport.requests[:2]:
        parts = request["body"]["messages"][1]["content"]
        assert any(part.get("type") == "image_url" for part in parts)
        assert "obs-0-image-0" in parts[0]["text"]
    logs = [
        json.loads(line)
        for line in (tmp_path / "episode/harness_model_calls.jsonl").read_text().splitlines()
    ]
    assert [line["role"] for line in logs] == ["pos_belief", "pos_sentinel"]
    assert all("image_ref" in json.dumps(line["messages"]) for line in logs)
