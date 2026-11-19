import copy
import pytest
from pyrualean.call_reliability import CallReliability
from tests.test_auxiliary_scheduling import Clock,run,is_aux,finish,Maintenance
from tests.test_staged_harness_runtime import TwoStage,report
from tests.test_openai_runtime import _reply


MM={'error':{'message':'Download multimodal file timed out','code':'invalid_parameter_error'}}


@pytest.mark.parametrize('code,payload,expected',[
 (400,MM,True),(400,{'error':{'message':'invalid image URL'}},False),
 (400,{'error':{'message':'context length exceeded'}},False),
 (400,'Download multimodal file timed out',False),(401,'unauthorized',False),
 (403,'forbidden',False),(422,'schema',False),(0,'ReadTimeout: timed out',True),
 (0,'ConnectError: connection reset',True),(0,'AssertionError: coding bug',False),
 (429,'rate limit',True),(500,'temporary',True),(503,'busy',True),(200,'bad response',False)])
def test_classify_only_known_transient_failures(code,payload,expected):
 assert CallReliability.retryable(code,payload) is expected


@pytest.mark.parametrize('left,followups',[(120,1),(240,1),(300,1),(480,0),(120,0),(239,1),(363,1)])
def test_allocation_reserves_followup_within_original_round(left,followups):
 p=CallReliability();p.observe('pos_belief',120,False,True);p.observe('pos_sentinel',80,True)
 a=p.allocate('pos_belief',left,followups,120)
 if left<120*(1+followups):assert a is None
 else:
  assert 120<=a['stage_cap_s']<=243
  assert a['stage_cap_s']+a['minimum_followup_s']<=left


def test_short_history_never_shrinks_or_invents_tail_latency_estimate():
 p=CallReliability()
 for role in ('pos_belief','pos_sentinel','pos_repair'):
  for _ in range(30):p.observe(role,20,True)
  assert p.target(role,120)==120
  p.observe(role,120,False,True)
  assert p.target(role,120)==120
  a=p.allocate(role,154,0,120)
  assert a['stage_cap_s']==120 and not a['retry_capacity_reserved']


def reliable(h):
 h.call_reliability=CallReliability();return h


def test_planner_transient_400_retries_same_body_with_absolute_deadline(robo,tmp_path,monkeypatch):
 clock=Clock();sent=[];h=reliable(Maintenance())
 def transport(url,body,timeout):
  sent.append((copy.deepcopy(body),timeout))
  clock.advance(70 if len(sent)==1 else 1)
  return (400,MM) if len(sent)==1 else (200,finish())
 result,events=run(robo,tmp_path,monkeypatch,transport,clock,h,budget_s=600,request_timeout_s=100,retries=0)
 assert len(sent)==2 and sent[0][0]==sent[1][0]
 assert sent[0][1]==100 and 27<=sent[1][1]<=28
 assert not result['errors'] and result['requests']==2 and result['python_cells']==0
 assert any(x['type']=='request_retry_classified' and x['robot_replay'] is False for x in events)


def test_retry_bound_does_not_inherit_large_global_retry_count(robo,tmp_path,monkeypatch):
 clock=Clock();sent=[]
 def transport(url,body,timeout):
  sent.append(body);clock.advance(1);return 400,MM
 result,_=run(robo,tmp_path,monkeypatch,transport,clock,reliable(Maintenance()),budget_s=600,retries=10)
 assert len(sent)==2 and result['errors']


def test_invalid_request_never_retries(robo,tmp_path,monkeypatch):
 sent=[]
 def transport(url,body,timeout):sent.append(body);return 400,{'error':{'message':'invalid image URL'}}
 r,_=run(robo,tmp_path,monkeypatch,transport,Clock(),reliable(Maintenance()),budget_s=600)
 assert len(sent)==1 and r['errors']


def test_sentinel_retry_does_not_repeat_belief_or_commit_without_audit(robo,tmp_path,monkeypatch):
 clock=Clock();h=reliable(TwoStage());aux=0;sent=[]
 def transport(url,body,timeout):
  nonlocal aux
  sent.append(copy.deepcopy(body))
  if is_aux(body):
   aux+=1;clock.advance(10);assert h.commits==0
   return (400,MM) if aux==2 else (200,report())
  return 200,finish()
 r,events=run(robo,tmp_path,monkeypatch,transport,clock,h,budget_s=5400,
  auxiliary_budget_s=1080,auxiliary_stage_budget_s=120,auxiliary_round_budget_s=480,request_timeout_s=900)
 assert h.commits==1 and r['auxiliary_requests']==3 and not r['errors']
 assert sent[1]==sent[2] and sent[0]!=sent[1]
 roles=[e['role']for e in events if e['type']=='request_start']
 assert roles==['pos_belief','pos_sentinel','pos_sentinel','planner']
 assert 32<=r['auxiliary_wall_s']<=33


def test_retry_respects_reserved_auditor_and_planner_request_slots(robo,tmp_path,monkeypatch):
 sent=[]
 def transport(url,body,timeout):
  sent.append(is_aux(body));return (400,MM) if is_aux(body) else (200,finish())
 h=reliable(TwoStage())
 r,_=run(robo,tmp_path,monkeypatch,transport,Clock(),h,max_requests=3,budget_s=5400,
  auxiliary_budget_s=1080,auxiliary_stage_budget_s=120,auxiliary_round_budget_s=480)
 assert sent==[True,False] and h.commits==0 and r['requests']==2


def test_exhausted_stage_timeout_does_not_receive_fresh_retry_allowance(robo,tmp_path,monkeypatch):
 clock=Clock();h=reliable(TwoStage());aux=0
 def transport(url,body,timeout):
  nonlocal aux
  if is_aux(body):aux+=1;clock.advance(timeout);return 0,'ReadTimeout: timed out'
  return 200,finish()
 r,_=run(robo,tmp_path,monkeypatch,transport,clock,h,budget_s=5400,
  auxiliary_budget_s=1080,auxiliary_stage_budget_s=120,auxiliary_round_budget_s=240,request_timeout_s=900)
 assert aux==1 and h.commits==0 and r['auxiliary_wall_s']==pytest.approx(120)
 assert r['status']=='completed'


def test_full_read_timeout_retries_then_audits_without_redoing_belief(robo,tmp_path,monkeypatch):
 clock=Clock();h=reliable(TwoStage());sent=[]
 def transport(url,body,timeout):
  sent.append((copy.deepcopy(body),timeout))
  if is_aux(body):
   assert timeout==pytest.approx(120)
   if len(sent)==1:
    clock.advance(timeout);return 0,'ReadTimeout: timed out'
   clock.advance(80);return 200,report()
  return 200,finish()
 r,events=run(robo,tmp_path,monkeypatch,transport,clock,h,budget_s=5400,
  auxiliary_budget_s=1080,auxiliary_stage_budget_s=120,auxiliary_round_budget_s=480,request_timeout_s=900)
 assert h.commits==1 and r['auxiliary_requests']==3 and not r['errors']
 assert sent[0][0]==sent[1][0] and sent[1][0]!=sent[2][0]
 assert 282<=r['auxiliary_wall_s']<=283
 assert [e['attempt'] for e in events if e['type']=='request_start']==[1,2,1,1]
 assert r['options']['auxiliary_budget_s']==1080 and r['options']['planner_reserve_s']==540
 assert any(e['type']=='request_retry_scheduled' for e in events)


def test_optional_repair_is_skipped_when_full_followup_floor_is_unfunded(robo,tmp_path,monkeypatch):
 clock=Clock();h=reliable(TwoStage(repair=True));latencies=iter([90,100]);aux=0
 def transport(url,body,timeout):
  nonlocal aux
  if is_aux(body):aux+=1;clock.advance(next(latencies));return 200,report()
  return 200,finish()
 r,events=run(robo,tmp_path,monkeypatch,transport,clock,h,budget_s=5400,
  auxiliary_budget_s=1080,auxiliary_stage_budget_s=120,auxiliary_round_budget_s=240)
 assert aux==2 and h.commits==0 and r['auxiliary_wall_s']==190
 assert any(e.get('reason')=='insufficient_dynamic_followup_reservation' for e in events)


def test_retry_after_executed_python_never_replays_robot_action(robo,backend,tmp_path,monkeypatch):
 clock=Clock();h=reliable(Maintenance());h.before_request=lambda call:''
 sent=[]
 def transport(url,body,timeout):
  sent.append(copy.deepcopy(body));clock.advance(1)
  if len(sent)==1:return 200,_reply([('act','python',{'code':'robo.move_to([0,0,.6])'})])
  if len(sent)==2:return 400,MM
  return 200,finish()
 r,_=run(robo,tmp_path,monkeypatch,transport,clock,h,budget_s=600)
 assert len(sent)==3 and sent[1]==sent[2]
 assert sum(k=='move_to' for k,_ in backend.calls)==1
 assert r['python_cells']==1 and not r['errors']


def test_capacity_preserves_scheduler_final_reserve_and_round_limit():
 p=CallReliability()
 assert p.recovery_allowance(240,840,480,120,2)==364
 assert p.recovery_allowance(240,300,480,120,2)==300
 assert p.recovery_allowance(240,840,240,120,2)==240
 # A funded audit is never squeezed to 60s to create a retry opportunity.
 assert p.allocate('pos_belief',239,1,120) is None
 a=p.allocate('pos_belief',300,1,120)
 assert a['stage_cap_s']==120 and a['minimum_followup_s']==120
 assert not a['retry_capacity_reserved']


def test_sentinel_full_timeout_retry_keeps_completed_belief(robo,tmp_path,monkeypatch):
 clock=Clock();h=reliable(TwoStage());roles=[];aux=0;sent=[]
 def transport(url,body,timeout):
  nonlocal aux
  sent.append(copy.deepcopy(body))
  if is_aux(body):
   aux+=1
   if aux==2:clock.advance(timeout);return 0,'ReadTimeout: timed out'
   clock.advance(20);return 200,report()
  return 200,finish()
 r,events=run(robo,tmp_path,monkeypatch,transport,clock,h,budget_s=5400,
  auxiliary_budget_s=1080,auxiliary_stage_budget_s=120,auxiliary_round_budget_s=480,request_timeout_s=900)
 assert h.commits==1 and r['auxiliary_requests']==3
 assert sent[1]==sent[2] and sent[0]!=sent[1]
 assert [e['role']for e in events if e['type']=='request_start']==['pos_belief','pos_sentinel','pos_sentinel','planner']


def test_two_full_timeouts_stop_at_one_retry_without_partial_commit(robo,tmp_path,monkeypatch):
 clock=Clock();h=reliable(TwoStage());timeouts=[]
 def transport(url,body,timeout):
  if is_aux(body):
   timeouts.append(timeout);clock.advance(timeout);return 0,'ReadTimeout: timed out'
  return 200,finish()
 r,events=run(robo,tmp_path,monkeypatch,transport,clock,h,budget_s=5400,
  auxiliary_budget_s=1080,auxiliary_stage_budget_s=120,auxiliary_round_budget_s=480,request_timeout_s=900)
 assert timeouts==[120,120] and h.commits==0
 assert 242<=r['auxiliary_wall_s']<=243 and r['status']=='completed'
 assert any(e['type']=='request_retry_skipped' and e['reason']=='attempt_limit'for e in events)


def test_no_full_retry_capacity_is_explicit_not_a_fake_attempt(robo,tmp_path,monkeypatch):
 clock=Clock();h=reliable(TwoStage());n=0
 def transport(url,body,timeout):
  nonlocal n
  if is_aux(body):n+=1;clock.advance(timeout);return 0,'ReadTimeout: timed out'
  return 200,finish()
 r,events=run(robo,tmp_path,monkeypatch,transport,clock,h,budget_s=5400,
  auxiliary_budget_s=240,auxiliary_stage_budget_s=120,auxiliary_round_budget_s=480,request_timeout_s=900)
 assert n==1 and h.commits==0
 assert any(e['type']=='request_retry_skipped'and e['reason']=='insufficient_full_retry_time'for e in events)
 assert not any(e['type']=='request_retry_scheduled'for e in events)


def test_initial_paced_round_reserves_retry_despite_setup_time(robo,tmp_path,monkeypatch):
 clock=Clock();h=reliable(TwoStage());aux=0
 h.maintenance_plan=lambda **kw:{'due':True,'allowance_s':240,'reserve_s':240,'reason':'event_due'}
 original=h.call_reliability.allocate
 def allocate(*a):
  # Real payload/setup delay must not silently disable a funded retry.
  clock.advance(.01)
  role,left,followups,nominal=a
  return original(role,left-.01,followups,nominal)
 h.call_reliability.allocate=allocate
 def transport(url,body,timeout):
  nonlocal aux
  if is_aux(body):
   aux+=1
   if aux==1:clock.advance(timeout);return 0,'ReadTimeout: timed out'
   clock.advance(10);return 200,report()
  return 200,finish()
 r,events=run(robo,tmp_path,monkeypatch,transport,clock,h,budget_s=5400,
  auxiliary_budget_s=1080,auxiliary_stage_budget_s=120,auxiliary_round_budget_s=480,request_timeout_s=900)
 assert aux==3 and h.commits==1 and not r['errors']
 reservation=next(e for e in events if e['type']=='harness_recovery_capacity')
 assert reservation['reserved_allowance_s']==364 and reservation['final_chain_reserve_s']==240
 assert any(e['type']=='request_start'and e['attempt']==2 for e in events)
