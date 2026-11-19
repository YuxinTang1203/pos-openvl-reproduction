"""Budget-distributed PoS and an independently selectable execution-check layer.

Only registered public tool results/images are consumed. No evaluator access.
"""
import functools
import inspect
import json

from .pos_harness import PoSHarness, validate_options as validate_v3

KINDS={'pos_schedule','pos_schedule_guard'}

def validate_options(config):
    if config.get('kind') not in KINDS:raise ValueError('unknown scheduled PoS variant')
    validate_v3(config)

class ExecutionReviewRequired(BaseException):
    """Abort dependent code without swallowing the original motion feedback."""

class EventSchedule:
    """Paced spending with stale-state borrowing and one final chain reserved.

    A reserve is an admission bound, not a second auxiliary budget. The runtime
    retains the original stage deadlines, total cap and main-policy reserve.
    """
    def __init__(self):
        self.attempted=False;self.last_sequence=-1;self.last_epoch=-1
        self.last_gap=None;self.manipulation_started=False
        self.last_elapsed=0.0;self.decision_elapsed=0.0

    def decide(self, *, sequence, epoch, gap, important, error, finish,
               remaining, total, spent, auxiliary, round_limit, minimum):
        elapsed=max(0.0,total-remaining);self.decision_elapsed=elapsed
        age=max(0.0,elapsed-self.last_elapsed);cells=sequence-self.last_sequence
        late=remaining<=min(1200.0,total*2/9) or finish
        # Protect one complete belief+Sentinel chain, not two static thirds.
        reserve=0.0 if late else min(minimum,auxiliary/3)
        hard_available=max(0.0,auxiliary-spent-reserve)
        release_window=max(1.0,total-min(1200.0,total*2/9))
        initial=min(minimum,auxiliary/3)
        ceiling=auxiliary if late else min(auxiliary-reserve,
            initial+max(0.0,auxiliary-reserve-initial)*min(1.0,elapsed/release_window))
        credit=max(0.0,ceiling-spent)
        stale=self.attempted and (age>=600.0 or cells>=8)
        urgent=bool(error or important or gap!=self.last_gap)
        # Coalesce adjacent errors; tool feedback remains immediately visible.
        cooled=age>=180.0 and cells>=2
        trigger=('initial' if not self.attempted else
                 'completion_request' if finish else
                 'stale_state_checkpoint' if stale else
                 'execution_error' if error and cooled else
                 'grasp_or_release' if important and cooled else
                 'subgoal_change' if gap!=self.last_gap and cooled else
                 'motion_checkpoint' if epoch!=self.last_epoch and cells>=4 and age>=300 else
                 'observation_checkpoint' if cells>=4 and age>=300 else None)
        debt_needed=max(0.0,minimum-credit)
        borrowing_limit=minimum if stale else minimum/4
        borrow=bool(trigger and (not self.attempted or finish or
            ((stale or urgent and cooled) and spent<=ceiling and debt_needed<=borrowing_limit)))
        authorized_credit=max(credit,minimum) if borrow else credit
        allowance=max(0.0,min(hard_available,round_limit,remaining,
                               max(minimum, min(1.5*minimum,authorized_credit))))
        due=bool(trigger and allowance>=minimum and (credit>=minimum or borrow))
        reason=('event_due' if due else 'coalesced_observation' if not trigger else
                'auxiliary_or_final_chain_reserve' if hard_available<minimum else 'paced_spending')
        return {'due':due,'trigger':trigger,'reserve_s':reserve,'allowance_s':allowance,
                'phase':'late' if late else 'operation' if self.manipulation_started else 'initial_observation',
                'reason':reason,'age_s':age,'cells_since_attempt':cells,
                'spending_ceiling_s':ceiling,'borrowing_s':max(0.0,minimum-credit) if due and borrow else 0.0,
                'auxiliary_remaining_s':max(0.0,auxiliary-spent)}

    def attempted_at(self,sequence,epoch,gap):
        self.attempted=True;self.last_sequence=sequence;self.last_epoch=epoch
        self.last_gap=gap;self.last_elapsed=self.decision_elapsed

class ExecutionChecks:
    def __init__(self,harness):
        self.h=harness;self.robo=harness.kit.robo
        self.pending={};self.holding={};self.progress={};self.finish_requested=False
        self.approved_sequence=None;self.no_motion_rounds=0;self.error_rounds=0
        self._attach()

    def log(self,event,**data):self.h._log('execution_'+event,**data)

    def _records(self):return self.robo._ledger.records

    def _last_motion(self):
        return max((r.index for r in self._records() if r.stateful),default=-1)

    def _evidence(self,indices,after):
        if not isinstance(indices,list) or not indices or any(type(i)is not int for i in indices):
            raise ValueError('evidence_calls must list actual public ledger indices')
        rows={r.index:r for r in self._records()}
        selected=[]
        for i in indices:
            r=rows.get(i)
            if r is None or i<=after or not r.ok or r.stateful:
                raise ValueError('evidence must be a successful observation after the relevant action')
            if r.tool not in {'artifact','view_env_state','locate','back_project','segment'}:
                raise ValueError('not a public observation')
            # Historical artifacts do not refresh evidence.
            if r.tool=='artifact' and r.kwargs.get('step',-1)!=-1:
                raise ValueError('historical images cannot verify the current action')
            selected.append(r)
        if not any(r.tool=='artifact' and str(r.kwargs.get('name','')).endswith('.png') for r in selected):
            raise ValueError('include an actual current camera image after the action')
        return selected

    def _gate(self,arm,command):
        pending=self.pending.get(arm)
        dependent=command==1 or (command==-1 and self.holding.get(arm) is not False)
        if pending and dependent:
            self.log('dependency_blocked',arm=arm,failure=pending,command=command)
            raise ExecutionReviewRequired('Motion did not reach its target. Inspect current public evidence and call robo.review_motion before this dependent gripper command; or execute a successful recovery motion. Failure details: '+json.dumps(pending))

    def _attach(self):
        for name in ('move_to','set_gripper'):
            original=getattr(self.robo,name);sig=inspect.signature(original)
            def wrap(original=original,sig=sig,name=name):
                @functools.wraps(original)
                def call(*args,**kwargs):
                    bound=sig.bind(*args,**kwargs);bound.apply_defaults();a=bound.arguments
                    arm=a.get('arm','right');command=a.get('gripper',0)
                    self._gate(arm,command)
                    result=original(*args,**kwargs)
                    if name=='move_to':
                        if getattr(result,'reached',None) is False or getattr(result,'ik_error',None):
                            self.pending[arm]={'call':self._records()[-1].index,'reached':getattr(result,'reached',None),'ik_error':getattr(result,'ik_error',None),'blocked':getattr(result,'blocked',None),'final_xyz':getattr(result,'final_xyz',None)}
                        elif getattr(result,'reached',None) is True:
                            self.pending.pop(arm,None)
                        # Moving with OPEN releases; CLOSE alone does not certify holding.
                        if command==-1:self.holding[arm]=False
                    else:
                        self.holding[arm]=getattr(result,'holding',None)
                    return result
                return call
            setattr(self.robo,name,wrap())
        self.robo.review_motion=self.review_motion
        self.robo.record_progress=self.record_progress
        self.robo.execution_evidence=self.evidence_index

    def evidence_index(self,limit=8):
        if type(limit)is not int or not 1<=limit<=20:raise ValueError('limit must be1..20')
        return [{'call':r.index,'tool':r.tool,'kwargs':r.kwargs,'summary':r.summary,'ok':r.ok}
                for r in self._records()[-limit:]]

    def review_motion(self,arm,*,decision,evidence_calls,reason):
        if arm not in ('left','right'):raise ValueError('unknown arm')
        if decision not in ('release_on_support','retry_grasp','recover'):raise ValueError('unknown review decision')
        if not isinstance(reason,str) or len(reason.strip())<12 or len(reason)>400:
            raise ValueError('provide a concise evidence-based explanation')
        failure=self.pending.get(arm)
        if not failure:raise ValueError('no pending failed motion for this arm')
        self._evidence(evidence_calls,max(failure['call'],self._last_motion()))
        # This is explicit model review of real observations, not a physics oracle.
        self.pending.pop(arm)
        self.log('dependency_reviewed',arm=arm,decision=decision,evidence_calls=evidence_calls,reason=reason,failure=failure)
        return {'review_recorded':True,'physical_support_verified_by_host':False,'decision':decision}

    def record_progress(self,subgoal,*,status,evidence_calls=None,note=''):
        if not isinstance(subgoal,str) or not subgoal.strip() or len(subgoal)>160:raise ValueError('invalid subgoal')
        if status not in ('attempted','observed_complete','uncertain'):raise ValueError('invalid progress status')
        if status=='observed_complete':self._evidence(evidence_calls,self._last_motion())
        if subgoal not in self.progress and len(self.progress)>=24:raise ValueError('progress record limit reached')
        self.progress[subgoal]={'status':status,'evidence_calls':evidence_calls or [],'note':str(note)[:400],'sequence':self.h.sequence,'verification':'model assertion with source delivery checked; not official success'}
        self.log('progress_recorded',subgoal=subgoal,record=self.progress[subgoal])
        self.h.important_event=True
        return self.progress[subgoal]

    def after_tool(self,result,records):
        moved=any(r.stateful for r in records)
        self.no_motion_rounds=0 if moved else self.no_motion_rounds+1
        self.error_rounds=self.error_rounds+1 if ('exception'in result.result or 'error'in result.result) else 0
        if self.no_motion_rounds>=4 or self.error_rounds>=3:
            result.result['stagnation_check']={'no_motion_rounds':self.no_motion_rounds,'consecutive_error_rounds':self.error_rounds,'instruction':'Repeated analysis/errors have not produced a world change. Choose a discriminating observation, a different feasible action/arm, or report stuck; preserve unresolved task obligations.'}
            self.h.error_event=True
        result.result['execution_checks']={'pending_motion_reviews':self.pending,'persistent_progress':self.progress,'model_assertions_not_ground_truth':True}

    def completion_ready(self):
        return self.approved_sequence==self.h.sequence and not self.pending

    def after_review(self):
        h=self.h
        fresh=any(e.get('type')=='image' and e.get('motion_epoch')==h.motion_epoch and e.get('current_eligible') and e.get('inspected') for e in h.evidence.values())
        okay=(h.last_status=='committed' and not h.belief['achievement'] and not h.belief['epistemic'] and fresh and not self.pending)
        self.approved_sequence=h.sequence if self.finish_requested and okay else None
        self.log('completion_audit',requested=self.finish_requested,approved=self.approved_sequence is not None,fresh_image_delivered=fresh,remaining_achievement=h.belief['achievement'],remaining_epistemic=h.belief['epistemic'])
        self.finish_requested=False

class ScheduledPoSHarness(PoSHarness):
    def __init__(self,kit,config,task,initial_images=()):
        validate_options(config);self.kind=config['kind']
        super().__init__(kit,config,task,initial_images)
        self.schedule=EventSchedule();self.important_event=False;self.error_event=False
        self.checks=ExecutionChecks(self) if self.kind=='pos_schedule_guard' else None
        self.provenance.update(implementation='pyrualean-pos-v4.1-balanced-schedule',execution_checks=bool(self.checks),schedule={'event_driven':True,'paced_spending':True,'final_reserved_chains':1,'late_window_s':1200,'stale_seconds':600,'stale_cells':8,'urgent_cooldown_seconds':180,'urgent_cooldown_cells':2,'new_per_cell_cpu_limit':None},independent_completion_audit='existing PoS Sentinel over the entire original task' if self.checks else False)

    def maintenance_plan(self,*,spent,remaining,total,auxiliary,round_limit,minimum):
        gap=json.dumps(self.belief.get('active_gap'),sort_keys=True)
        result=self.schedule.decide(sequence=self.sequence,epoch=self.motion_epoch,gap=gap,important=self.important_event,error=self.error_event,finish=bool(self.checks and self.checks.finish_requested),remaining=remaining,total=total,spent=spent,auxiliary=auxiliary,round_limit=round_limit,minimum=minimum)
        self._log('schedule_decision',sequence=self.sequence,motion_epoch=self.motion_epoch,auxiliary_spent_s=spent,**result)
        return result

    def before_request(self,model_call):
        self.schedule.attempted_at(self.sequence,self.motion_epoch,json.dumps(self.belief.get('active_gap'),sort_keys=True))
        self.important_event=False;self.error_event=False
        def audited_call(**kw):
            if self.checks and self.checks.finish_requested:
                kw['prompt']+='\nFINAL COMPLETION AUDIT: audit the ENTIRE ORIGINAL TASK, including every remaining item. Reject removal of any unresolved requirement without fresh public outcome evidence. An attempted placement, a motion arrival, or an executor claim is insufficient. Preserve unfinished/uncertain obligations as gaps; do not accept a partial task as completion.'
                kw['payload']={**kw['payload'],'execution_progress':self.checks.progress,'pending_motion_reviews':self.checks.pending}
            return model_call(**kw)
        context=super().before_request(audited_call)
        if self.checks:self.checks.after_review()
        return context

    def after_tool(self,name,result,records):
        if name=='finish' and self.checks and self.kit.finish is None:
            name='completion_request'
        out=super().after_tool(name,result,records)
        if name!='finish':
            self.important_event|=any(r.tool=='grip' for r in records)
            self.error_event|=('exception'in result.result or 'error'in result.result or any(r.error or r.summary.get('reached')is False for r in records))
            self.schedule.manipulation_started|=any(r.tool=='grip' and r.kwargs.get('gripper')==1 or r.stateful and r.kwargs.get('gripper')==1 for r in records)
            if self.checks:self.checks.after_tool(out,records)
        return out

    def execute_finish(self,args,finish):
        if not self.checks or args.get('status')!='success':return finish(args)
        if self.checks.completion_ready():return finish(args)
        self.checks.finish_requested=True;self.important_event=True
        self._log('execution_finish_deferred',sequence=self.sequence,reason='whole-task fresh-evidence Sentinel audit required')
        return self._finish_denial()

    def _finish_denial(self):
        from .arms import ArmResult
        return ArmResult({'error':'Completion not independently supported yet. Inspect fresh current images, preserve ALL unfinished original-task obligations, resolve pending motion reviews, then retry finish(success). Existing PoS belief+Sentinel maintenance will audit completion within the SAME1080s auxiliary budget. If unsupported or budget unavailable, continue the task or finish(stuck/failure).','completion_review_pending':True})

    def prompt(self):
        base=super().prompt()
        if not self.checks:return base
        return base+'''\nEXECUTION CHECKS: failed approach/transport requires explicit public-evidence review before a dependent close or release. Use robo.execution_evidence() to list real public call indices; fetch a CURRENT camera image after the failed action, then call robo.review_motion(arm, decision="release_on_support"|"retry_grasp"|"recover", evidence_calls=[indices], reason="observed support/actual pose and why this next action is justified"). Review is a fallible model judgment. A successful recovery motion may clear a pending motion failure. Do not use stale/cropped coordinates as current world facts.\nRecord subgoal attempts and observed outcomes with robo.record_progress(subgoal, status="attempted"|"observed_complete"|"uncertain", evidence_calls=[indices], note="..."). observed_complete requires an actual current camera image after the most recent action. These host records persist across Python variable redefinitions; they do not establish official success.\nBefore finish(success), show current outcome images and request a whole-task audit; the existing PoS belief/Sentinel chain must support every original obligation. Failure/stuck remains available. Partial completion is not full success.\n'''
