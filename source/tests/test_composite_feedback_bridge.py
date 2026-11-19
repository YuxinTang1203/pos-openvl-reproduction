"""Regression checks for lost reviewer issues and stale action/directive handoff."""
import copy
import json

from pyrualean.composite_harness import CompositeHarness
from pyrualean.arms import ArmResult
from tests.test_composite_harness import setup, args
from tests.test_pos_harness import Script, audit


def test_rejected_audit_reaches_planner_and_next_original_pos_input(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False);h=kit.general_harness
    h.options['repair_attempts']=0
    issue='Image widths are pixels; their values are not measured millimetres.'
    reviewer={**audit(None),'accepted':False,'issues':[issue], 'directive':{'goal_id':'a-root','mode':'observe','next_step':'Inspect the relevant original pixels.','decision_critical_missing':'Metric geometry is unknown.','why_change':'Units are inconsistent.'}}
    before=copy.deepcopy(h.belief)
    h.before_request(Script(copy.deepcopy(h.belief),reviewer))
    assert h.last_status=='audit_rejected' and h.revision==0 and h.belief==before
    assert issue in h.runtime_context()
    result=kit.execute_tool('python',args('print(robo.state())'))
    assert result.result['component_review_feedback']['issues']==[issue]
    assert 'component_review_feedback' in h.evidence['obs-1']['text']
    next_model=Script(copy.deepcopy(h.belief),{**audit(None),'directive':{**reviewer['directive'],'why_change':'New evidence available.'}})
    h.before_request(next_model)
    assert len(next_model.calls)==2
    assert all(issue in json.dumps(c['payload'])for c in next_model.calls)
    assert h.decision_context()['maintenance_feedback']['status']=='committed'
    assert not h.decision_context()['maintenance_feedback']['issues']


def test_unavailable_review_is_visible_without_fabricating_issue_or_world(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False);h=kit.general_harness
    h.before_request(Script(copy.deepcopy(h.belief),None))
    context=h.decision_context()
    assert context['maintenance_feedback']['status']=='audit_unavailable'
    assert context['maintenance_feedback']['issues']==[]
    assert h.revision==0 and h.committed_sequence is None


def test_stale_directive_is_retained_but_versioned_and_outcome_is_exposed(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False);h=kit.general_harness
    directive={'goal_id':'a-root','mode':'act','next_step':'Attempt the supported placement.',
               'decision_critical_missing':'No additional fact requested.','why_change':'Proceed from delivered evidence.'}
    h.before_request(Script(copy.deepcopy(h.belief),{**audit(None),'directive':directive}))
    assert h.decision_context()['directive_source']['same_motion_epoch']
    out=kit.execute_tool('python',args('robo.move_to([0,0,.6])'))
    ctx=out.result['composite_decision']
    assert ctx['directive'] is None
    assert ctx['historical_directive']==directive
    assert not ctx['directive_source']['same_motion_epoch']
    assert not ctx['directive_source']['same_observation_sequence']
    assert ctx['latest_feedback']['action_outcomes']==out.result['task_progress']['action_outcomes']
    assert ctx['recent_action_feedback'][-1]['action_outcomes']
    kit.execute_tool('python',args('print(robo.state())'))
    ctx=h.decision_context()
    assert ctx['recent_action_feedback'][-1]['sequence']<ctx['latest_feedback']['sequence']
    assert not ctx['latest_feedback']['action_outcomes']


def test_empty_grasp_feedback_remains_historical_without_overriding_new_feedback(robo,tmp_path):
    from types import SimpleNamespace
    kit=setup(robo,tmp_path,visual=False);h=kit.general_harness
    def rec(index,tool,kwargs,summary):
        return SimpleNamespace(index=index,tool=tool,kwargs=kwargs,summary=summary,error=None,stateful=True)
    fail=rec(100,'grip',{'gripper':robo.CLOSE},{'holding':False,'steps_used':7})
    h.after_tool('python',ArmResult({'stdout':''}),[fail])
    first=h.decision_context();assert first['last_failure']['type']=='empty_grasp'
    success=rec(101,'grip',{'gripper':robo.CLOSE},{'holding':True,'steps_used':7})
    h.after_tool('python',ArmResult({'stdout':''}),[success])
    ctx=h.decision_context()
    assert ctx['latest_feedback']['action_outcomes'][0]['feedback']['holding']is True
    assert ctx['latest_feedback']['last_failure']is None
    assert ctx['last_failure_source_sequence']<ctx['latest_feedback']['sequence']
    assert ctx['last_failure']==first['last_failure']  # original PoS bookkeeping unchanged
    assert not h.belief['achievement']==[]  # holding is not task completion


def test_no_new_tools_helpers_robot_queries_or_mutable_aliases(robo,backend,tmp_path):
    kit=setup(robo,tmp_path);h=kit.general_harness
    calls=copy.deepcopy(backend.calls)
    names={t['name']for t in kit.get_tools_spec()}
    assert names=={'python','finish','Visual_Grounding_Tool','Zoom_In_Tool','Recall_Images'}
    for name in ['visual_point','visual_axis','visual_contact_geometry','visual_pose_check']:
        assert not hasattr(robo,name) and name not in kit._globals
    for _ in range(3):h.runtime_context()
    assert backend.calls==calls
    ctx=h.decision_context();ctx['unfinished_goals'].clear()
    assert h.belief['achievement']


def test_handoff_does_not_repeat_full_world_and_retains_task_obligations(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False);h=kit.general_harness
    text='distinctive world detail '*80
    h.belief['world']['states']=[{'id':'s-long','value':text}]
    ctx=h.decision_context();serialized=h.runtime_context()
    assert text not in serialized
    assert ctx['belief_reference']['world_record_counts']['states']==1
    assert ctx['unfinished_goals']==h.belief['achievement']
    assert ctx['decision_critical_missing']==h.belief['epistemic']
    assert h.belief['world']['states'][0]['value']==text


def test_handoff_action_window_and_review_issues_are_bounded(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False);h=kit.general_harness
    for _ in range(7):kit.execute_tool('python',args('robo.move_to([0,0,.6])'))
    assert len(h.decision_context()['recent_action_feedback'])==4
    h._log('maintenance_degraded',status='candidate_rejected',issues=['x'*1500]*11)
    report=h.decision_context()['maintenance_feedback']
    assert len(report['issues'])==8 and all(len(x)==800 for x in report['issues'])
    assert report['omitted_issues']==3


def test_new_belief_commit_does_not_refresh_rejected_directive(robo,tmp_path):
    kit=setup(robo,tmp_path,visual=False);h=kit.general_harness
    first={'goal_id':'a-root','mode':'observe','next_step':'Inspect the original outcome.',
           'decision_critical_missing':'Outcome unverified.','why_change':'Need evidence.'}
    h.before_request(Script(copy.deepcopy(h.belief),{**audit(None),'directive':first}))
    origin=copy.deepcopy(h.decision_context()['directive_source'])
    kit.execute_tool('python',args('robo.move_to([0,0,.6])'))
    rejected={**first,'goal_id':'not-an-audited-goal'}
    h.before_request(Script(copy.deepcopy(h.belief),{**audit(None),'directive':rejected}))
    ctx=h.decision_context()
    assert h.last_status=='committed' and h.revision==2
    assert ctx['directive'] is None
    assert ctx['historical_directive']==first
    assert ctx['directive_source']['sequence']==origin['sequence']
    assert ctx['directive_source']['revision']==origin['revision']
    assert not ctx['directive_source']['same_motion_epoch']
    assert ctx['committed_sequence']>ctx['directive_source']['sequence']
