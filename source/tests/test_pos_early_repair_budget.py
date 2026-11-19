"""Recorded early schema/evidence rejections must not discard unused audit time."""
import copy
from types import SimpleNamespace

import pytest

from pyrualean.pos_harness import PoSHarness
from tests.test_auxiliary_scheduling import Clock, is_aux, run, finish
from tests.test_openai_runtime import _reply
from tests.test_pos_harness import belief, audit


def candidates(kind):
    repaired=belief('obs-0');repaired['world']['states'][0]['status']='uncertain'
    repaired['achievement'][0]['id']='a-root'
    bad=copy.deepcopy(repaired)
    if kind=='overflow':
        bad['world']['states']=[dict(bad['world']['states'][0],id=f's-{i}',property=f'p-{i}')for i in range(24)]
    else:bad['world']['states'][0]['status']='current'  # no temporally eligible physical observation
    return bad,repaired


@pytest.mark.parametrize('kind,belief_latency,round_budget',[
    ('overflow',23.835,240.0039182342594),
    ('stale',44.508,251.2488982293948),
])
@pytest.mark.parametrize('accepted',[True,False])
def test_actual_pos_repairs_recorded_failure_within_original_round(
    robo,tmp_path,monkeypatch,kind,belief_latency,round_budget,accepted):
    h=PoSHarness(SimpleNamespace(arm='cells',workspace=tmp_path/'workspace'),{'kind':'pos'},'Place the block')
    bad,repaired=candidates(kind);clock=Clock();sent=[];roles=[]
    replies=iter([(belief_latency,bad),(10,repaired),(20,audit(accepted=accepted))])
    def transport(url,body,timeout):
        if is_aux(body):
            seconds,payload=next(replies);assert seconds<=timeout
            sent.append((copy.deepcopy(body),timeout));clock.advance(seconds)
            assert h.revision==0  # repairs never commit before independent audit
            return 200,_reply([('state','submit_harness_state',{'payload':payload})])
        return 200,finish()
    result,events=run(robo,tmp_path,monkeypatch,transport,clock,h,budget_s=5400,
        auxiliary_budget_s=1080,auxiliary_stage_budget_s=120,
        auxiliary_round_budget_s=round_budget,request_timeout_s=900)
    roles=[e['role']for e in events if e['type']=='harness_stage_start']
    assert roles==['pos_belief','pos_repair','pos_sentinel']
    assert len(sent)==3 and result['auxiliary_requests']==3
    assert sent[1][1]==pytest.approx((round_budget-belief_latency)/2)
    assert result['auxiliary_wall_s']==pytest.approx(belief_latency+30,abs=.001)
    assert result['auxiliary_wall_s']<=round_budget and result['options']['auxiliary_budget_s']==1080
    assert result['options']['planner_reserve_s']==540 and not result['errors']
    assert h.revision==(1 if accepted else 0)
    if accepted:
        assert h.belief==repaired
    assert h.options['max_records']==24 and h.options['repair_attempts']==1
    assert any(e['type']=='harness_repair_budget_reused'for e in events)
    assert not any(e['type']=='harness_stage_chain_skipped'for e in events)
    repair_input=sent[1][0]['messages']
    assert 'repair_issues' in str(repair_input) and 'rejected_candidate' in str(repair_input)


def test_early_repair_still_reserves_audit_and_planner_request_slots(robo,tmp_path,monkeypatch):
    h=PoSHarness(SimpleNamespace(arm='cells',workspace=tmp_path/'workspace'),{'kind':'pos'},'Place block')
    bad,_=candidates('overflow');clock=Clock();aux=[]
    def transport(url,body,timeout):
        if is_aux(body):aux.append(body);clock.advance(24);return 200,_reply([('state','submit_harness_state',{'payload':bad})])
        return 200,finish()
    result,events=run(robo,tmp_path,monkeypatch,transport,clock,h,budget_s=5400,
        auxiliary_budget_s=1080,auxiliary_stage_budget_s=120,auxiliary_round_budget_s=240,max_requests=3,request_timeout_s=900)
    assert len(aux)==1 and h.revision==0 and result['requests']==2
    assert any(e['type']=='harness_stage_chain_skipped'and e['role']=='pos_repair'for e in events)


def test_repair_timeout_cannot_exceed_round_or_commit_without_auditor(robo,tmp_path,monkeypatch):
    h=PoSHarness(SimpleNamespace(arm='cells',workspace=tmp_path/'workspace'),{'kind':'pos'},'Place block')
    bad,_=candidates('overflow');clock=Clock();aux=0
    def transport(url,body,timeout):
        nonlocal aux
        if is_aux(body):
            aux+=1
            if aux==1:clock.advance(24);return 200,_reply([('state','submit_harness_state',{'payload':bad})])
            assert timeout==pytest.approx(108)
            clock.advance(timeout);return 0,'repair timeout'
        return 200,finish()
    result,events=run(robo,tmp_path,monkeypatch,transport,clock,h,budget_s=5400,
        auxiliary_budget_s=1080,auxiliary_stage_budget_s=120,auxiliary_round_budget_s=240,retries=0,request_timeout_s=900)
    assert aux==2 and h.revision==0 and result['auxiliary_wall_s']==pytest.approx(132)
    assert not any(e['type']=='harness_stage_start'and e['role']=='pos_sentinel'for e in events)
