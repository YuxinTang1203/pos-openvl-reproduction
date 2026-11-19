import copy
from pathlib import Path
import pytest
from pyrualean.arms import ArmToolkit
from pyrualean.migrations import configure_migration
from pyrualean.robodojo import RobodojoRobot
from pyrualean.openai_runtime import run_openai_agent
from pyrualean.static_rule_reminders import STATIC_RULES
from tests.test_openai_runtime import ScriptedTransport, _reply
from tests.test_openvl_routed import Candidates, CONFIG
from tests.test_robodojo import FakeRobodojoBackend


@pytest.fixture
def build(tmp_path, monkeypatch):
    import pyrualean.openvl_routed as routed
    original = routed.OpenVLRoutedAdapter
    monkeypatch.setattr(routed, 'OpenVLRoutedAdapter', lambda *a, **kw:
                        original(*a, **kw, segmenter=Candidates()))
    n = 0
    def make(condition, original_kind=False):
        nonlocal n
        n += 1
        kit = ArmToolkit(RobodojoRobot(FakeRobodojoBackend()), arm='cells',
                         workspace=tmp_path/str(n), images='on-demand', feedback='pure')
        cfg = {'kind':'v28_reminder_presentation', 'condition':condition,
               'pos':{}, 'openvl':{'enabled':True,'config':str(CONFIG)}}
        if original_kind:
            cfg.pop('condition'); cfg.update(kind='v17_ablation', profile='fusion',
                static_rules=condition=='B', action_focus=False, reliability=False)
        configure_migration(kit, cfg, task='Insert all tubes', initial_images={})
        return kit
    return make


def text(message):
    content = message.get('content') or ''
    return content if isinstance(content, str) else '\n'.join(
        p.get('text','') for p in content if p.get('type')=='text')


def test_original_behavior_and_tools_preserved(build):
    a, b, c = [build(x) for x in 'ABC']
    direct_a, direct_b = build('A',True), build('B',True)
    assert a.general_harness.prompt() == direct_a.general_harness.prompt()
    assert b.general_harness.prompt() == direct_b.general_harness.prompt()
    assert b.general_harness.prompt() == c.general_harness.prompt()
    assert b.general_harness.prompt() == a.general_harness.prompt()+'\n'+STATIC_RULES
    assert all(k.get_tools_spec() == a.get_tools_spec() for k in [b,c,direct_a,direct_b])
    for k in [a,b,c]:
        assert k.robo._action_feedback is None
        assert not hasattr(k.general_harness,'feedback_guard')
        assert not hasattr(k.general_harness,'max_executed_calls_per_response')
        assert k.general_harness.action_focus is None
        assert k.general_harness.call_reliability is None
        assert k.general_harness.dependency_mode == 'off'
        assert not any(x in k._globals for x in ['try_motion_candidates'])


@pytest.mark.parametrize('condition,counts', [('A',[0,0,0]),('B',[1,1,1]),('C',[1,2,3])])
def test_real_runtime_new_reminder_delivery_and_no_new_actions(build,tmp_path,condition,counts):
    kit = build(condition)
    transport = ScriptedTransport([
        _reply([('one','python',{'code':"robo.set_gripper('right',robo.CLOSE)"})]),
        _reply([('two','python',{'code':"observed = robo.state()"})]),
        _reply([('done','finish',{'status':'stuck','summary':'scripted test'})])])
    result = run_openai_agent(kit.general_harness.prompt(), [], kit, model='offline',
        base_url='http://unused/v1', api_key=None, transport=transport, budget_s=60,
        max_turns=4, max_requests=4, auxiliary_budget_s=0, out_dir=tmp_path/'runtime')
    assert result['requests']==3 and result['auxiliary_requests']==0 and not result['errors']
    assert [sum(text(m).count(STATIC_RULES.strip()) for m in r['body']['messages'])
            for r in transport.requests] == counts
    assert len([r for r in kit.robo._ledger.records if r.stateful])==1
    # No guard: empty CLOSE was not automatically turned into a stop.
    assert result['python_cells']==2
    assert all(r['body']['tools']==transport.requests[0]['body']['tools'] for r in transport.requests)
