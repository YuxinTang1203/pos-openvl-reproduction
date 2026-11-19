"""Nonblocking metadata adapter for PoS task-progress control.

Corrections affect planner bookkeeping only, never robot code or world facts.
"""
import copy
from .progress_harness import ProgressHarness, PHASES, OUTCOMES, STEP_SCHEMA
from .pos_harness import validate_options as validate_pos

KIND='pos_progress_lite'
def validate_options(config):
 if config.get('kind')!=KIND:raise ValueError('unknown progress-lite kind')
 validate_pos({**config,'kind':'pos'})

class ProgressLiteHarness(ProgressHarness):
 def __init__(self,kit,config,task,initial_images=()):
  validate_options(config)
  super().__init__(kit,{**config,'kind':'pos_progress'},task,initial_images)
  self.kind=KIND;self.metadata_warnings=[];self.delivered_image_calls=set()
  self.provenance.update(kind=KIND,implementation='pyrualean-pos-progress-im-lite-v1',required_step_contract=False,
    metadata_corrections='nonblocking,logged; never rewrite code or fabricate physical success')

 def prepare_tools(self,specs):
  specs=super().prepare_tools(specs)
  for t in specs:
   if t['name']=='python':
    schema=copy.deepcopy(STEP_SCHEMA);schema['required']=['goal','phase','expected']
    schema['properties']['evidence_calls']['description']='Optional. Host associates delivered current images automatically; never invent indices.'
    t['input_schema']['properties']['step']=schema;t['input_schema']['required']=['code']
    t['description']='Execute code in the original persistent Python sandbox. Optional step={goal,phase,expected} updates intent; omitted intent inherits the active audited task. missing/change/previous_outcome/evidence_calls are optional. Metadata problems produce warnings, never resubmission of robot actions.'
  return specs

 def _validate_step(self,raw):
  warnings=[]
  def note(kind,**details):warnings.append({'kind':kind,**details})
  if raw is None:raw={}
  elif not isinstance(raw,dict):note('invalid_metadata_object');raw={}
  eligible=self._eligible_goals()
  fallback=self.active if self.active in eligible else (self.directive or {}).get('goal_id')
  if fallback not in eligible:fallback='task-root'
  goal=raw.get('goal',fallback)
  if not isinstance(goal,str)or goal not in eligible:
   note('goal_rebound_to_audited_target',requested=str(goal)[:160],selected=fallback);goal=fallback
  # The bounded table is bookkeeping; archive an inactive record rather than
  # wasting a robot call if a later audited goal uses a new identifier.
  if goal not in self.goals and len(self.goals)>=24:
   retired=next((g for g in self.goals if g!=self.active),next(iter(self.goals)))
   self._log('progress_goal_archived',goal=retired,record=self.goals[retired]);del self.goals[retired]
  phase=raw.get('phase')
  if phase not in PHASES:
   phase='observe' if (self.directive or {}).get('mode')=='observe' else 'recover' if (self.directive or {}).get('mode')=='recover' else self.goals.get(goal,{}).get('last_phase','other_action')
   if 'phase'in raw:note('invalid_phase_replaced',selected=phase)
  def text(key,default=None):
   value=raw.get(key,default)
   if value is None:return None
   if not isinstance(value,str):note('invalid_text_ignored',field=key);return default
   value=value.strip()
   if len(value)>240:note('annotation_truncated',field=key,original_length=len(value));value=value[:240]
   return value or default
  expected=text('expected',(self.directive or {}).get('next_step','Read actual tool feedback before judging this attempt.'))
  if expected is not None:expected=expected[:240]
  missing=text('missing');change=text('change')
  if phase=='observe' and not missing:note('observation_question_not_declared',instruction='State the one decision-critical uncertainty if another observation is needed.')
  if (self.failure_streak or self.no_progress_cells>=6)and not change:note('strategy_change_not_declared',instruction='Use failure/stagnation feedback to change the approach or justify a retry.')
  outcome=raw.get('previous_outcome','unknown' if self.active is None else 'attempted')
  if outcome not in OUTCOMES:note('unknown_outcome',requested=str(outcome)[:80]);outcome='uncertain'
  if self.active is None and outcome!='unknown':note('no_prior_subgoal_claim_ignored');outcome='unknown'
  indices=raw.get('evidence_calls',[])
  if not isinstance(indices,list):note('invalid_evidence_list_ignored');indices=[]
  rows={r.index:r for r in self.kit.robo._ledger.records};accepted=[]
  for index in indices[:8]:
   if type(index)is int and index in rows and rows[index].ok:accepted.append(index)
   else:note('invalid_evidence_index_ignored',index=str(index)[:80])
  fresh=sorted(i for i in self.delivered_image_calls if i>self.last_action_call and i in rows and rows[i].ok)
  if outcome=='observed_complete':
   if fresh:
    accepted=list(dict.fromkeys(accepted+fresh))[-8:]
    note('current_delivered_image_associated',calls=fresh)
   else:
    outcome='uncertain';note('completion_not_verified',instruction='No delivered current outcome image after the latest action; retain unfinished/uncertain status. The requested code will still run.')
  self.metadata_warnings=warnings
  for warning in warnings:self._log('progress_metadata_warning',warning=warning)
  return {'goal':goal,'phase':phase,'expected':expected,'missing':missing,'change':change,'previous_outcome':outcome,'evidence_calls':accepted}

 def _feedback(self,result,records):
  feedback=super()._feedback(result,records)
  # Only camera artifacts delivered in the tool response can support a later
  # completion claim. A private/unshown array or a historical display cannot.
  self.delivered_image_calls={i for i in self.delivered_image_calls if i>self.last_action_call}
  labels=result.result.get('images',[])
  cameras=[c for c in self.kit.robo.CAMERAS if any(str(n)==c or str(n).startswith(c+'.') for n in labels)]
  artifact_names={name for (cam,res),name in self.kit.robo.IMAGE_ARTIFACTS.items()if cam in cameras}
  if result.images and cameras:
   for record in records:
    if record.tool=='artifact'and record.ok and record.kwargs.get('step',-1)==-1 and record.kwargs.get('name')in artifact_names and record.index>self.last_action_call:
     self.delivered_image_calls.add(record.index)
  feedback['metadata_warnings']=copy.deepcopy(self.metadata_warnings)
  feedback['current_outcome_image_calls']=sorted(self.delivered_image_calls)
  feedback['metadata_policy']='Intent annotations may be inherited/corrected; unverified completion remains uncertain. Robot code executes once under the original budget.'
  return feedback

 def prompt(self):
  # Keep PoS + audited execution direction, replacing the rigid cell contract.
  base=super().prompt().split('# Task-progress feedback (Inner Monologue adaptation)')[0]
  return base+'''\n# Task progress with automatic bookkeeping
Call python(code, step=...) using the original robot API. step is optional: when intent changes, provide {goal,phase,expected}; otherwise the current audited target is inherited. goal is an achievement ID from POS_CONTEXT or task-root. Do not turn observation/measurement into a new task goal.
Read the automatically injected action outcomes, controller directive and unfinished targets. When needed, add missing (a decision-critical question), change (what differs after failure), or previous_outcome. Evidence indices are optional: current delivered camera images are associated automatically. Metadata warnings do not require resubmitting robot actions.
Moving is not grasping, holding does not identify the intended object, and releasing is not verified placement. Unsupported completion assertions remain uncertain while subsequent observations/actions continue. Show current outcome images after actions. Whole-task success still requires the existing independent PoS/Sentinel audit within the same1080s auxiliary budget. No automatic robot retry or new primitive is introduced.\n'''
