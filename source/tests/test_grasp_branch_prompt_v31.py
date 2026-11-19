import pytest
import importlib.util,json,os,sys
from pathlib import Path
from pyrualean.arms import ArmToolkit
from pyrualean.migrations import configure_migration
from pyrualean.robodojo import RobodojoRobot
from pyrualean.openai_runtime import run_openai_agent
from pyrualean.static_rule_reminders import STATIC_RULES
from pyrualean.grasp_branch_prompt_v31 import BRANCH_RULES,reminder_text
from tests.test_openai_runtime import ScriptedTransport,_reply
from tests.test_openvl_routed import Candidates,CONFIG
from tests.test_robodojo import FakeRobodojoBackend

@pytest.fixture
def build(tmp_path,monkeypatch):
 import pyrualean.openvl_routed as routed
 original=routed.OpenVLRoutedAdapter
 monkeypatch.setattr(routed,'OpenVLRoutedAdapter',lambda *a,**kw:original(*a,**kw,segmenter=Candidates()))
 n=0
 def make(condition='A',direct=False):
  nonlocal n
  n+=1
  kit=ArmToolkit(RobodojoRobot(FakeRobodojoBackend()),arm='cells',workspace=tmp_path/str(n),images='on-demand',feedback='pure')
  cfg={'kind':'v31_grasp_branch_prompt','condition':condition,'pos':{},'openvl':{'enabled':True,'config':str(CONFIG)}}
  if direct:
   cfg.pop('condition');cfg.update(kind='v17_ablation',profile='fusion',static_rules=False,action_focus=False,reliability=False)
  configure_migration(kit,cfg,task='Insert all tubes',initial_images={})
  return kit
 return make

def text(m):
 c=m.get('content') or ''
 return c if isinstance(c,str) else '\n'.join(p.get('text','') for p in c if p.get('type')=='text')

def test_only_grasp_paragraph_changes():
 assert BRANCH_RULES.split('\n\n',1)[1]==STATIC_RULES.split('\n\n',1)[1]
 assert 'Python if/else' in BRANCH_RULES and 'printing is not a check' in BRANCH_RULES
 assert len(STATIC_RULES)==732 and len(BRANCH_RULES)==752
 assert 'checked(' not in BRANCH_RULES

def test_original_api_initial_prompt_and_tools_identical(build):
 kits=[build(c) for c in 'ARP'];direct=build(direct=True)
 assert all(k.general_harness.prompt()==direct.general_harness.prompt() for k in kits)
 assert all(k.get_tools_spec()==direct.get_tools_spec() for k in kits)
 for k in kits:
  assert k.robo._action_feedback is None
  assert k.general_harness.dependency_mode=='off'
  assert k.general_harness.action_focus is None and k.general_harness.call_reliability is None
  assert not hasattr(k.general_harness,'max_executed_calls_per_response')
  assert not hasattr(k.general_harness,'feedback_guard')
  assert 'try_motion_candidates' not in k._globals

@pytest.mark.parametrize('condition','ARP')
def test_real_runtime_all_rounds_prompt_only_and_no_action_interception(build,tmp_path,condition):
 kit=build(condition)
 code="g=robo.set_gripper('left',robo.CLOSE)\nrobo.move_to([-.3,-.2,1.05],arm='left',gripper=robo.CLOSE)\ncontinued=True"
 t=ScriptedTransport([
  _reply([('first','python',{'code':code})]),
  _reply([('second','python',{'code':'state=robo.state()'})]),
  _reply([('done','finish',{'status':'stuck','summary':'scripted software contract'})])])
 result=run_openai_agent(kit.general_harness.prompt(),[],kit,model='offline',base_url='http://unused/v1',api_key=None,
  transport=t,budget_s=60,max_turns=4,max_requests=4,auxiliary_budget_s=0,out_dir=tmp_path/'runtime')
 assert result['requests']==3 and result['auxiliary_requests']==0 and not result['errors']
 payload=reminder_text(condition)
 assert all(text(r['body']['messages'][0]).count('For object pickup, make close feedback')==0 for r in t.requests)
 counts=[sum(text(m).count(payload.strip()) for m in r['body']['messages']) if payload else 0 for r in t.requests]
 assert counts==([0,0,0] if condition=='A' else [1,2,3])
 assert all(r['body']['tools']==t.requests[0]['body']['tools'] for r in t.requests)
 acts=[r for r in kit.robo._ledger.records if r.stateful]
 assert len(acts)==2 and acts[0].summary['holding'] is False
 assert acts[1].summary['steps_used']>0 and kit._globals['continued'] is True

 # Exercise the real saved runtime transcript against the official experiment
 # delivery auditor, rather than just counting an in-memory prompt string.
 evaluation_root=Path(os.environ.get('V31_EVALUATION_ROOT',Path(__file__).resolve().parents[2]/'evaluation'))
 sys.path.insert(0,str(evaluation_root))
 spec=importlib.util.spec_from_file_location('_v31_delivery_contract',evaluation_root/'collect.py')
 collector=importlib.util.module_from_spec(spec);spec.loader.exec_module(collector)
 saved=json.loads((tmp_path/'runtime/messages.json').read_text())
 (tmp_path/'runtime/prompt.txt').write_text(text(saved[0]))
 audited=collector.delivery(tmp_path/'runtime',condition)
 assert audited['complete'] and audited['errors']==[]
 assert audited['new_repeat_injections']==(0 if condition=='A' else 3)
