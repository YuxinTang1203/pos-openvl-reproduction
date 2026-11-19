"""PoS + public task-progress/action feedback inspired by Inner Monologue.

Domain adaptation, not a reproduction: existing analytic robot API, no learned
pick/place primitive, no oracle scene predicates, no extra model allowance.
"""
import copy
import json
from .pos_v4 import ScheduledPoSHarness
from .pos_harness import validate_options as validate_pos

KIND='pos_progress'
PHASES=['observe','grasp','transport','place','verify','recover','other_action']
OUTCOMES=['unknown','attempted','observed_complete','failed','uncertain']
STEP_SCHEMA={'type':'object','additionalProperties':False,'properties':{
 'goal':{'type':'string','minLength':1,'maxLength':160,'description':'ID of an audited PoS achievement gap, or task-root for the original task, from eligible_task_goals. Never invent an inspect/measure meta-goal.'},
 'phase':{'type':'string','enum':PHASES},
 'expected':{'type':'string','minLength':1,'maxLength':240,'description':'Observable result expected from this code.'},
 'missing':{'type':['string','null'],'maxLength':240,'description':'For observation: the specific decision-critical uncertainty and how its answer changes the next action.'},
 'change':{'type':['string','null'],'maxLength':240,'description':'After failure/stagnation: the changed hypothesis, arm, pose, path, observation, or justified stochastic retry.'},
 'previous_outcome':{'type':'string','enum':OUTCOMES,'description':'Outcome of the PREVIOUS active subgoal; observed_complete requires fresh outcome image evidence.'},
 'evidence_calls':{'type':'array','maxItems':8,'items':{'type':'integer'}}},
 'required':['goal','phase','expected','missing','change','previous_outcome','evidence_calls']}

def validate_options(config):
 if config.get('kind')!=KIND:raise ValueError('unknown progress harness')
 validate_pos({**config,'kind':'pos'})

class ProgressHarness(ScheduledPoSHarness):
 def __init__(self,kit,config,task,initial_images=()):
  validate_options(config)
  super().__init__(kit,{**config,'kind':'pos_schedule'},task,initial_images)
  self.kind=KIND;self.active=None;self.goals={};self.current_step=None
  self.no_progress_cells=0;self.no_execution_cells=0;self.failure_streak=0
  self.last_failure=None;self.last_image_sequence=-1;self.last_action_call=-1
  self.finish_requested=False;self.completion_approved=None;self.contract_denials=0
  self.directive=None;self.pending_directive=None
  self.provenance.update(kind=KIND,implementation='pyrualean-pos-progress-im-v2',
   reference_paper='https://proceedings.mlr.press/v205/huang23c/huang23c.pdf',
   reference_method='Inner Monologue: action outcome + task-progress scene feedback + replanning',
   reproduction=False,oracle_feedback=False,learned_skills_added=False,
   automatic_action_execution=False,required_step_contract=True,
   extra_auxiliary_budget=False,execution_checks='feedback-driven recovery; no blanket failed-motion gripper gate')

 def response_schema(self,role):
  schema=super().response_schema(role)
  if role=='pos_sentinel':
   schema=copy.deepcopy(schema)
   schema['properties']['directive']={'type':'object','additionalProperties':False,'properties':{
     'goal_id':{'type':['string','null'],'maxLength':160},
     'mode':{'type':'string','enum':['act','observe','recover','complete','stuck']},
     'next_step':{'type':'string','minLength':1,'maxLength':400},
     'decision_critical_missing':{'type':['string','null'],'maxLength':240},
     'why_change':{'type':'string','minLength':1,'maxLength':240}},
     'required':['goal_id','mode','next_step','decision_critical_missing','why_change']}
   schema['required'].append('directive')
  return schema

 def _eligible_goals(self):
  return {'task-root':self.task,**{g['id']:g['target'] for g in self.belief['achievement']}}

 def prepare_tools(self,specs):
  specs=copy.deepcopy(specs)
  for t in specs:
   if t['name']=='python':
    t['input_schema']['properties']['step']=STEP_SCHEMA
    t['input_schema']['required']=['code','step']
    t['description']+=' Include step metadata in the SAME call as code. Feedback refers to the previous subgoal; it is not a hidden task verdict.'
   if t['name']=='finish':
    t['description']='End with success only after fresh evidence and the PoS whole-task audit; failure/stuck is always available.'
  return specs

 def _validate_step(self,step):
  if not isinstance(step,dict) or set(step)!=set(STEP_SCHEMA['properties']):raise ValueError('step requires exactly '+', '.join(STEP_SCHEMA['required']))
  for k,n in [('goal',160),('expected',240)]:
   if not isinstance(step[k],str) or not step[k].strip() or len(step[k])>n:raise ValueError(k+' must be concise nonempty text')
  for k in ['missing','change']:
   if step[k] is not None and (not isinstance(step[k],str) or len(step[k])>240):raise ValueError(k+' must be null or concise text')
  if step['phase']not in PHASES or step['previous_outcome']not in OUTCOMES:raise ValueError('unknown phase/outcome')
  if not isinstance(step['evidence_calls'],list) or len(step['evidence_calls'])>8 or any(type(x)is not int for x in step['evidence_calls']):raise ValueError('evidence_calls must list real public call indices')
  if step['phase']=='observe' and not (step['missing']or'').strip():raise ValueError('observation must identify a decision-critical missing fact')
  if (self.failure_streak or self.no_progress_cells>=6) and not (step['change']or'').strip():raise ValueError('state what changes after failure or stagnation; a justified stochastic retry is allowed')
  if self.active is None and step['previous_outcome']!='unknown':raise ValueError('no previous subgoal to mark complete')
  if step['goal']not in self._eligible_goals():raise ValueError('goal must be an audited achievement ID or the original task-root from eligible_task_goals; observations belong in phase/missing, not new task goals')
  rows={r.index:r for r in self.kit.robo._ledger.records}
  if any(i not in rows or not rows[i].ok for i in step['evidence_calls']):raise ValueError('unknown or unsuccessful evidence call')
  if step['previous_outcome']=='observed_complete':
   images=[rows[i] for i in step['evidence_calls'] if rows[i].tool=='artifact' and str(rows[i].kwargs.get('name','')).endswith('.png') and rows[i].kwargs.get('step',-1)==-1 and i>self.last_action_call]
   if not images or self.last_image_sequence!=self.sequence:raise ValueError('completion claim needs a delivered CURRENT outcome image after the latest action; first show it, then cite its public call index')
  return copy.deepcopy(step)

 def execute_python(self,args,execute):
  from .arms import ArmResult
  self.current_step=None
  try:step=self._validate_step(args.get('step'))
  except (ValueError,TypeError) as exc:
   self.contract_denials+=1
   result=execute('pass')  # Rejected Python calls still consume the original cell allowance.
   result.result.update(error=str(exc),progress_contract_denied=True,instruction='Correct step metadata and resubmit; no intended robot action executed. This rejected call consumes one cell.')
   self._log('progress_contract_denied',error=str(exc))
   return result
  if self.active:
   row=self.goals[self.active]
   row['status']=step['previous_outcome'];row['evidence_calls']=step['evidence_calls']
   if step['previous_outcome']=='observed_complete':
    row['observed_at_sequence']=self.sequence;row['current_evidence']=True
   row['verification']='executor model assertion with evidence-delivery validation; not ground truth'
   if step['previous_outcome']=='observed_complete':self.no_progress_cells=0
  goal=step['goal'].strip()
  if goal not in self.goals and len(self.goals)>=24:
   result=execute('pass');result.result['error']='24 subgoals already tracked; reuse an existing goal identifier instead of renaming failures.';return result
  self.goals.setdefault(goal,{'status':'unknown','attempts':0,'evidence_calls':[]})
  self.goals[goal]['target']=self._eligible_goals()[goal]
  self.active=goal;self.goals[goal]['attempts']+=1
  self.current_step=step
  self._log('progress_step',step=step)
  # Every code execution still goes through the existing sandbox and cell budget.
  return execute(str(args.get('code','')))

 def _call_index(self,limit=10):
  return [{'call':r.index,'tool':r.tool,'kwargs':r.kwargs,'summary':r.summary,'error':r.error}
          for r in self.kit.robo._ledger.records[-limit:]]

 def _feedback(self,result,records):
  public=[];failure=None;executed=False
  for r in records:
   if r.stateful:
    self.last_action_call=max(self.last_action_call,r.index)
    summary=copy.deepcopy(r.summary)
    public.append({'call':r.index,'tool':r.tool,'parameters':r.kwargs,'feedback':summary,'error':r.error})
    executed|=bool(summary.get('steps_used',0)>0)
    if r.error:failure={'type':'tool_error','call':r.index,'detail':r.error}
    elif summary.get('ik_error'):failure={'type':'unreachable_or_path','call':r.index,'detail':summary['ik_error']}
    elif summary.get('blocked') or summary.get('reached')is False:
     failure={'type':'contact_or_motion_shortfall' if summary.get('blocked')else 'motion_shortfall','call':r.index,'detail':summary}
    elif r.tool=='grip' and r.kwargs.get('gripper')==self.kit.robo.CLOSE and summary.get('holding')is False:
     failure={'type':'empty_grasp','call':r.index,'detail':summary}
  if any(r.stateful for r in records):
   for row in self.goals.values():
    if row.get('status')=='observed_complete':
     row['current_evidence']=False
     row['recheck']='Previously observed completion; later motion may disturb it. Reobserve if relevant.'
  if result.result.get('exception')or result.result.get('error'):
   failure=failure or {'type':'code_or_protocol','detail':str(result.result.get('exception')or result.result.get('error'))[-800:]}
  self.failure_streak=self.failure_streak+1 if failure else 0
  if failure:self.last_failure=failure;self.error_event=True
  if self.current_step:
   self.no_progress_cells+=1
   self.no_execution_cells=0 if executed else self.no_execution_cells+1
   self.goals[self.active]['status']='failed' if failure else 'attempted'
   self.goals[self.active]['last_phase']=self.current_step['phase']
  names=result.result.get('images',[])
  if result.images and any(str(n).startswith('cam_') for n in names):self.last_image_sequence=self.sequence
  suggestions={
   'unreachable_or_path':'Use the reported reachable alternatives: change pose/orientation/path or arm; a fresh observation may be needed. Do not repeat an infeasible pose unchanged.',
   'contact_or_motion_shortfall':'Determine from fresh images whether contact is intended support or obstruction before choosing release, realignment, or retreat. reached=false alone does not decide that.',
   'motion_shortfall':'Compare final pose to target; inspect the relevant tolerance/geometry and change the approach if necessary.',
   'empty_grasp':'The gripper reports holding=false. Reobserve the target and adjust alignment/height/orientation before another grasp; do not count transport or placement.',
   'code_or_protocol':'Repair the specific code/schema error while keeping the existing target and unresolved obligations.',
   'tool_error':'Use the original tool error to choose a feasible next action.'}
  review_needed=self.no_progress_cells>=6 or self.failure_streak>=2 or self.no_execution_cells>=4
  return {'active_subgoal':self.active,'eligible_task_goals':self._eligible_goals(),'controller_directive':self.directive,'step':self.current_step,'action_outcomes':public[-6:],
   'last_failure':failure,'recovery_guidance':suggestions.get((failure or {}).get('type')),
   'no_observed_subgoal_progress_cells':self.no_progress_cells,'no_executed_motion_cells':self.no_execution_cells,
   'strategy_review_required':review_needed,
   'next_decision':('Choose one targeted observation that changes a decision, a feasible task action, a revised approach, or report stuck. State the concrete change in the next step; repeated measurements alone do not complete the task.' if review_needed else 'Use actual feedback to select the next phase; preserve all unfinished goals.'),
   'subgoals':copy.deepcopy(self.goals),'public_calls':self._call_index(),
   'verification':'Motion arrival is not grasp success. holding is not intended-object identity. Release is not placement completion. Subgoal claims are fallible; no task evaluator is consulted.'}

 def after_tool(self,name,result,records):
  if name=='finish' and self.kit.finish is None:name='completion_request'
  out=super().after_tool(name,result,records)
  if name!='finish':
   out.result['task_progress']=self._feedback(out,records)
   # Supply automatically collected feedback to the existing memory/auditor too.
   # One bounded public summary replaces only this observation's text.
   eid='obs-'+str(self.sequence)
   if eid in self.evidence:
    obj=json.loads(self.evidence[eid]['text']);obj['task_progress']={k:out.result['task_progress'][k] for k in ['active_subgoal','last_failure','strategy_review_required','subgoals']}
    self.evidence[eid]['text']=json.dumps(obj,ensure_ascii=False,default=str)
   self._log('progress_feedback',feedback=out.result['task_progress'])
   self.current_step=None
  return out

 def maintenance_plan(self,**kw):
  gap=json.dumps(self.belief.get('active_gap'),sort_keys=True)
  result=self.schedule.decide(sequence=self.sequence,epoch=self.motion_epoch,gap=gap,
    important=self.important_event,error=self.error_event,finish=self.finish_requested,
    **kw)
  self._log('schedule_decision',sequence=self.sequence,motion_epoch=self.motion_epoch,
            auxiliary_spent_s=kw['spent'],**result)
  return result

 def before_request(self,model_call):
  final=self.finish_requested;self.pending_directive=None
  def feedback_call(**kw):
   kw['payload']={**kw['payload'],'task_progress':{'active':self.active,'subgoals':self.goals,'last_failure':self.last_failure,'claims_are_ground_truth':False}}
   kw['prompt']+='\nKeep fields within the declared schema limits (gap target <=240 characters). Task-progress claims are executor reports; independently check original task obligations against public outcome evidence.'
   if final:kw['prompt']+='\nFINAL AUDIT: verify the ENTIRE ORIGINAL TASK, not just the active subgoal. Preserve incomplete or unobserved obligations. Partial placement or executor claims do not establish task completion.'
   if kw['role']in ('pos_belief','pos_repair'):
    kw['prompt']+='\nJSON structure reminder: {"world":{"entities":[],"states":[],"relations":[]},"epistemic":[],"achievement":[],"active_gap":null}. Close world before epistemic. Populate the arrays using the real evidence and preserve unfinished requirements.\nAchievement gaps must describe original-task outcomes in the external world, not observing/measuring/estimating. When useful, expose one concrete next physical subgoal plus the remaining original obligations; epistemic gaps hold missing observations.'
   if kw['role']=='pos_sentinel':
    kw['prompt']=kw['prompt'].replace('progress_reason: string.', 'progress_reason: string, directive: {goal_id, mode, next_step, decision_critical_missing, why_change}. Return all FIVE top-level fields, including directive, as required by the supplied schema.')
    kw['prompt']+='\nAlso return directive per schema, a bounded actionable recommendation for the NEXT code cell. Ground it in delivered evidence and the candidate task gaps. Choose act when enough information exists for a feasible task action; observe only for a specific decision-critical missing fact; recover after failure. Repeated global remapping is not physical task progress. Do not invent poses or force action with missing essential geometry. goal_id must name a candidate achievement gap, or null only for complete/stuck. Give a concrete next_step, not a generic reminder. This is public control guidance, not hidden reasoning. Continue independently auditing the candidate; producing a directive never justifies accepting an unsupported state.'
   report=model_call(**kw)
   if kw['role']=='pos_sentinel' and isinstance(report,dict):self.pending_directive=report.get('directive')
   return report
  context=super().before_request(feedback_call)
  if self.last_status=='committed' and isinstance(self.pending_directive,dict):
   target=self.pending_directive.get('goal_id')
   if target in self._eligible_goals() or target is None and self.pending_directive.get('mode')in ('complete','stuck'):
    self.directive=copy.deepcopy(self.pending_directive)
    self._log('progress_directive',directive=self.directive)
    context+='\nTASK_PROGRESS_DIRECTIVE\n'+json.dumps(self.directive,ensure_ascii=False)
   else:self._log('progress_directive_rejected',reason='unknown achievement gap')
  if final:
   fresh=any(e.get('type')=='image' and e.get('motion_epoch')==self.motion_epoch and e.get('current_eligible') and e.get('inspected') for e in self.evidence.values())
   self.completion_approved=self.sequence if self.last_status=='committed' and not self.belief['achievement'] and not self.belief['epistemic'] and fresh else None
   self.finish_requested=False
   self._log('progress_completion_audit',approved=self.completion_approved==self.sequence)
  return context

 def execute_finish(self,args,finish):
  if args.get('status')!='success' or self.completion_approved==self.sequence:return finish(args)
  from .arms import ArmResult
  self.finish_requested=True
  return ArmResult({'completion_review_pending':True,'error':'Show current outcome images and request whole-task completion; existing PoS belief/Sentinel must support every original requirement within the same1080s auxiliary budget. Continue unfinished work or report failure/stuck if support is unavailable.'})

 def prompt(self):
  return super().prompt()+'''\n# Task-progress feedback (Inner Monologue adaptation)
Every python call includes code AND a compact step object. goal MUST be an audited achievement ID from POS_CONTEXT.belief.achievement or eligible_task_goals; task-root always denotes the authoritative original task, including when only epistemic gaps remain. For example use a-root if it exists; never invent 'inspect/measure/count' meta-goals. The target is an external task outcome. phase describes how you advance it, including observation when needed. State an observable expected result. previous_outcome describes the PREVIOUS active subgoal. For observe, missing must identify a decision-critical fact and how its answer changes the next action. After failure/stagnation, change states the revised hypothesis/pose/path/arm or a justified retry. This metadata does not consume another cell or model call.
Read TASK_PROGRESS_DIRECTIVE when present: follow its concrete next action/observation/recovery recommendation, or explain in missing/change the new public evidence that requires a different step. Read task_progress after EVERY cell: action outcomes, specific failure, remaining subgoals, and public evidence indices are injected automatically. You do not need to remember to call a progress helper. Report observed_complete only after showing a current outcome camera image and citing its public call index. Claims remain fallible; moving, closing, and releasing are separate attempted events, not semantic success.
Use a short observe->act->verify loop. Once the decision-critical geometry is adequate, attempt a feasible action and inspect its outcome. Avoid repeated global remapping when one local observation would decide the next action. After a failed grasp/IK/placement, change the failed part while preserving verified work; all existing robot primitives remain available. Contact may be intentional: assess it rather than banning all reached=false actions. No automatic robot retry is performed. Do not abandon unfinished original-task obligations when changing subgoals. finish(failure/stuck) remains available.\n'''
