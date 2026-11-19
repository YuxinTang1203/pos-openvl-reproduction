import copy
import pytest
from pyrualean.composite_harness import CompositeHarness,validate_options
from pyrualean.call_reliability import CallReliability
from tests.test_composite_harness import setup,events
from tests.test_pos_harness import Script,audit
from tests.test_composite_v10_feedback import directive


def test_flag_is_explicit_and_default_keeps_previous_behavior(robo,tmp_path):
 kit=setup(robo,tmp_path,visual=False,reflection=False);h=kit.general_harness
 assert h.call_reliability is None and 'state_sync' not in h.decision_context()
 cfg={'kind':'openvl_pos_reflact','openvl':{'enabled':False},'reflact':{'enabled':False},'reliability':{'enabled':True}}
 h=CompositeHarness(kit,cfg,'Place the block')
 assert isinstance(h.call_reliability,CallReliability) and h.reflection is None
 assert h.provenance['call_reliability']['total_budget_increased'] is False
 assert 'state_sync' in h.prompt()


@pytest.mark.parametrize('value',[True,{'enabled':1},{'enabled':True,'unknown':1}])
def test_bad_reliability_options_rejected(value):
 with pytest.raises(ValueError):validate_options({'kind':'openvl_pos_reflact','openvl':{'enabled':False},'reliability':value})


def test_unreviewed_feedback_marked_without_changing_audited_world(robo,tmp_path):
 kit=setup(robo,tmp_path,visual=False,reflection=False);h=kit.general_harness;h.call_reliability=CallReliability()
 h.before_request(Script(copy.deepcopy(h.belief),{**audit(None),'directive':directive()}))
 before=copy.deepcopy(h.belief['world']);old_revision=h.revision
 assert h.decision_context()['state_sync']['belief_current_for_latest_feedback']
 out=kit.execute_tool('python',{'code':'robo.move_to([0,0,.6])'})
 sync=out.result['composite_decision']['state_sync']
 assert not sync['belief_current_for_latest_feedback'] and sync['committed_revision']==old_revision
 assert sync['unreviewed_action_feedback_tail'][-1]['evidence_id']=='obs-'+str(h.sequence)
 assert h.belief['world']==before and h.revision==old_revision
 assert sync['pending_evidence_ids'] and not sync['tail_is_complete_history']
 assert 'latest_feedback' in out.result['composite_decision']


def test_read_only_new_observation_also_remains_unreviewed_until_commit(robo,tmp_path):
 kit=setup(robo,tmp_path,visual=False,reflection=False);h=kit.general_harness;h.call_reliability=CallReliability()
 h.before_request(Script(copy.deepcopy(h.belief),{**audit(None),'directive':directive()}))
 kit.execute_tool('python',{'code':'print(robo.state())'})
 sync=h.decision_context()['state_sync']
 assert not sync['belief_current_for_latest_feedback'] and sync['unreviewed_action_feedback_tail']==[]
 assert sync['committed_motion_epoch']==sync['latest_motion_epoch']
