import copy
import json
from tests.test_composite_harness import setup, events
from tests.test_pos_harness import Script, audit
from tests.test_composite_v10_feedback import directive
from tests.test_openvl_locate_candidate import configured as configured, CONFIG
from pyrualean.composite_harness import CompositeHarness


def test_failed_visual_update_keeps_original_sources_for_next_review(robo,tmp_path):
 kit=setup(robo,tmp_path,reflection=False)
 kit.execute_tool('python',{'code':"robo.segment('blocks')"})
 h=kit.general_harness
 original={eid:copy.deepcopy(h.evidence[eid])for eid in h.pending}
 failed=Script(None)
 h.before_request(failed)
 assert h.revision==0 and h.last_status=='update_unavailable'
 assert all(eid in h.pending for eid in original)
 assert len(events(kit,'fusion_uncommitted_evidence_retained'))==1
 for eid,entry in original.items():
  assert h.evidence[eid]['motion_epoch']==entry['motion_epoch']
  assert h.evidence[eid]['sequence']==entry['sequence']
  assert h.evidence[eid].get('openvl_source')==entry.get('openvl_source')
 candidate=copy.deepcopy(h.belief)
 reviewed=Script(candidate,{**audit(None),'directive':directive()})
 h.before_request(reviewed)
 assert h.revision==1 and not h.pending
 for call in reviewed.calls:
  payload=call['payload'];handoff=payload['openvl_pos_handoff']
  assert set(original).issubset({e['id']for e in payload['new_evidence']})
  assert handoff['recent_packets'][-1]['evidence_id']=='obs-1'
  assert 'candidate_or_routing_feedback' in handoff['recent_packets'][-1]['evidence_kinds']
 assert h.belief==candidate


def test_failed_audit_preserves_evidence_without_committing_candidate(robo,tmp_path):
 kit=setup(robo,tmp_path,reflection=False);h=kit.general_harness
 kit.execute_tool('python',{'code':'robo.show()'})
 original=copy.deepcopy(h.belief);ids=list(h.pending)
 h.before_request(Script(copy.deepcopy(original),None))
 assert h.last_status=='audit_unavailable' and h.revision==0
 assert h.belief==original and h.pending==ids


def test_motion_does_not_refresh_requeued_image_or_enable_current_claim(robo,tmp_path):
 kit=setup(robo,tmp_path,reflection=False);h=kit.general_harness
 kit.execute_tool('python',{'code':'robo.show()'})
 ids=[eid for eid in h.pending if h.evidence[eid]['type']=='image']
 h.before_request(Script(None))
 kit.execute_tool('python',{'code':'robo.move_to([0,0,.6])'})
 packet=h.decision_context()['openvl_pos_handoff']['recent_packets'][0]
 assert packet['same_motion_epoch'] is False and not packet['current_image_evidence_ids']
 import pytest
 with pytest.raises(ValueError,match='temporally eligible'):
  h._refs(ids,current=True)
 assert all(h.evidence[eid]['motion_epoch']<h.motion_epoch for eid in ids)


def test_calibration_capture_not_claimed_as_delivered_pixels(configured):
 _,_,_,robo,kit,_,_=configured
 kit.general_harness=CompositeHarness(kit,{'kind':'openvl_pos_reflact','openvl':{'config':str(CONFIG)},'reflact':{'enabled':False}},'Place the object')
 h=kit.general_harness
 out=kit.execute_tool('python',{'code':"print(robo.camera_meta('cam_head'))"})
 packet=h.decision_context()['openvl_pos_handoff']['recent_packets'][-1]
 assert packet['calibration_image_ids']
 assert packet['calibration_ids_are_not_automatically_delivered_pixels']
 assert packet['current_image_evidence_ids']==[] and not out.images
 assert 'calibration_with_conditional_projection' in packet['evidence_kinds']
 assert 'conditional on that height' in h.visual_handoff()['semantics']['projection']


def test_same_delivered_sources_reach_belief_repair_and_independent_audit(robo,tmp_path):
 kit=setup(robo,tmp_path,reflection=False);h=kit.general_harness
 kit.execute_tool('python',{'code':"robo.segment('blocks')"})
 bad=copy.deepcopy(h.belief);bad['undeclared']=1
 model=Script(bad,copy.deepcopy(h.belief),{**audit(None),'directive':directive()})
 h.before_request(model)
 assert h.last_status=='committed'
 assert [c['role']for c in model.calls]==['pos_belief','pos_repair','pos_sentinel']
 supplied=[c['payload']['openvl_pos_handoff']for c in model.calls]
 assert supplied[0]==supplied[1]==supplied[2]
 assert supplied[0]['recent_packets']==h.decision_context()['openvl_pos_handoff']['recent_packets']
 assert kit._cell_index==1 and len(events(kit,'decision_bound'))==1
