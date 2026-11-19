"""Ablations isolate prompts without adding execution or auxiliary calls."""
import copy
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import pytest

from pyrualean.arms import ArmToolkit
from pyrualean.ablation_v17 import validate_options, effective_config
from pyrualean.minimal_fusion import MinimalFusionHarness, NEXT_ACTION_PROMPT
from pyrualean.migrations import cli_migration, configure_migration
from pyrualean.progress_lite import ProgressLiteHarness
from pyrualean.robodojo import RobodojoRobot
from pyrualean.static_rule_reminders import STATIC_RULES
from pyrualean.openai_runtime import run_openai_agent
from tests.test_openai_runtime import ScriptedTransport, _reply
from tests.test_openvl_routed import Candidates
from tests.test_robodojo import FakeRobodojoBackend

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT/'integrations/migrations/general_harness'
PROFILES = ['pos', 'fusion', 'next_action']


@pytest.fixture
def configure(tmp_path, monkeypatch):
    import pyrualean.openvl_routed as routed
    original = routed.OpenVLRoutedAdapter
    monkeypatch.setattr(routed, 'OpenVLRoutedAdapter', lambda *a, **kw:
                        original(*a, **kw, segmenter=Candidates()))
    counter = 0
    def build(profile='fusion', rules=False, label=None):
        nonlocal counter
        counter += 1
        label = label or profile + ('_rules' if rules else '')
        config = cli_migration(Namespace(general_harness_config=CONFIGS/f'ablation_{label}.json', runtime='openai'))
        backend = FakeRobodojoBackend()
        robo = RobodojoRobot(backend)
        kit = ArmToolkit(robo, arm='cells', workspace=tmp_path/str(counter)/'workspace',
                         images='on-demand', feedback='pure')
        record = configure_migration(kit, config, task='Insert all three tubes', initial_images={})
        return config, record, backend, robo, kit
    return build


@pytest.mark.parametrize('profile', PROFILES)
def test_rules_toggle_changes_only_executor_prompt(configure, profile):
    c0, r0, _, _, k0 = configure(profile)
    c1, r1, _, _, k1 = configure(profile, rules=True)
    assert {k:v for k,v in c0.items() if k != 'static_rules'} == {
        k:v for k,v in c1.items() if k != 'static_rules'}
    assert k1.general_harness.prompt() == k0.general_harness.prompt() + '\n' + STATIC_RULES
    assert k0.get_tools_spec() == k1.get_tools_spec()
    assert '# Failure dependencies' not in k1.general_harness.prompt()
    assert effective_config(c0) == effective_config(c1)
    assert r1['adaptation']['v17_ablation']['static_rules_sha256'] == hashlib.sha256(STATIC_RULES.encode()).hexdigest()


def test_pos_is_original_progresslite_without_openvl_or_fusion_context(configure, tmp_path):
    config, record, _, _, kit = configure('pos')
    assert isinstance(kit.general_harness, ProgressLiteHarness)
    assert not isinstance(kit.general_harness, MinimalFusionHarness)
    assert kit.evidence_adapter is None and 'native' not in record
    direct_kit = ArmToolkit(RobodojoRobot(FakeRobodojoBackend()), arm='cells', workspace=tmp_path/'direct')
    direct_kit.general_harness = ProgressLiteHarness(direct_kit, {'kind':'pos_progress_lite'}, 'Insert all three tubes')
    assert kit.general_harness.prompt() == direct_kit.general_harness.prompt()
    assert kit.get_tools_spec() == direct_kit.get_tools_spec()
    assert [t['name'] for t in kit.get_tools_spec()] == ['python','finish']


def test_next_action_diff_is_one_executor_prompt_suffix(configure):
    c0, _, _, _, k0 = configure('fusion')
    c1, _, _, _, k1 = configure('next_action')
    assert k1.general_harness.prompt() == k0.general_harness.prompt() + NEXT_ACTION_PROMPT
    assert k0.get_tools_spec() == k1.get_tools_spec()
    assert k0.general_harness.action_focus is k1.general_harness.action_focus is None
    assert k0.general_harness.call_reliability is k1.general_harness.call_reliability is None
    assert k0.general_harness.dependency_mode == k1.general_harness.dependency_mode == 'off'
    assert k0.general_harness.reflection is k1.general_harness.reflection is None


@pytest.mark.parametrize('profile', PROFILES)
@pytest.mark.parametrize('rules', [False, True])
def test_real_runtime_executes_one_action_and_no_new_aux_calls(configure, tmp_path, profile, rules):
    _, _, _, robo, kit = configure(profile, rules)
    kit._globals['executions'] = 0
    transport = ScriptedTransport([
        _reply([('grip','python',{'code':"executions+=1\nrobo.set_gripper('right',robo.CLOSE)"})]),
        _reply([('stop','finish',{'status':'stuck','summary':'offline scripted check'})])])
    result = run_openai_agent(kit.general_harness.prompt(), [], kit, model='offline',
        base_url='http://unused/v1', api_key=None, transport=transport, budget_s=60,
        max_turns=3, max_requests=3, auxiliary_budget_s=0, out_dir=tmp_path/'runtime')
    assert result['requests'] == 2 and result['auxiliary_requests'] == 0
    assert result['python_cells'] == 1 and not result['errors']
    assert kit._globals['executions'] == 1
    assert len([r for r in robo._ledger.records if r.stateful]) == 1
    for request in transport.requests:
        # PyRUA sends the initial task/instruction prompt as a user message.
        texts = []
        for message in request['body']['messages']:
            content = message.get('content', '')
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                texts.extend(p.get('text', '') for p in content if p.get('type') == 'text')
        assert sum(text.count(STATIC_RULES) for text in texts) == int(rules)


@pytest.mark.parametrize('profile', PROFILES)
def test_original_success_audit_cannot_be_bypassed(configure, profile):
    _, _, _, _, kit = configure(profile, True)
    pending = kit.execute_tool('finish', {'status':'success', 'summary':'unsupported completion'})
    assert pending.result['completion_review_pending'] and kit.finish is None


def test_v15_switches_reconnect_existing_components_separately(configure):
    _, _, _, _, focused = configure(label='v15_focus')
    _, _, _, _, reliable = configure(label='v15_reliability')
    assert focused.general_harness.action_focus is not None
    assert focused.general_harness.call_reliability is None
    assert reliable.general_harness.action_focus is None
    assert reliable.general_harness.call_reliability is not None
    props = next(t for t in focused.get_tools_spec() if t['name']=='python')['input_schema']['properties']['step']['properties']
    assert 'focus' in props and 'dependencies' not in props
    assert '# Observation for the next action' not in focused.general_harness.prompt()


@pytest.mark.parametrize('patch', [
    {'dependency_mode':'advisory'}, {'profile':'typo'}, {'static_rules':1},
    {'profile':'pos','reliability':True}, {'profile':'pos','openvl':{}},
    {'profile':'next_action','action_focus':True}, {'pos':{'kind':'pos'}},
    {'openvl':{'enabled':False}},
])
def test_invalid_or_confounding_profiles_rejected(patch):
    config = json.loads((CONFIGS/'ablation_fusion.json').read_text())
    config.update(patch)
    with pytest.raises(ValueError):
        validate_options(config)


def test_reminders_match_frozen_v15_text():
    assert STATIC_RULES == (ROOT/'FROZEN_RULES.txt').read_text()
    assert hashlib.sha256(STATIC_RULES.encode()).hexdigest() == 'c7fad68f7f8afe068e1e5838cefa4b6b557d00d2c7d7681239094033140c75d0'
