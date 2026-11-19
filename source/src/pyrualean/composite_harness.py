"""Explicit OpenVL -> PoS Progress Lite -> same-decision ReflAct composition.

Only delivered public observations enter belief maintenance. Executor summaries
remain reports, never authoritative physical facts. No extra model or robot loop.
"""
from __future__ import annotations

import copy
import json
from collections import deque
from collections.abc import Mapping

from .progress_lite import ProgressLiteHarness, validate_options as validate_pos
from .reflact_harness import ReflActHarness, validate_options as validate_reflact
from .openvl_routed import validate_ambiguity_thresholds
from .composite_schedule import FreshnessSchedule
from .call_reliability import CallReliability
from .action_focus import ActionFocus, FOCUS_SCHEMA, FOCUS_PROMPT, AUX_PROMPT, FIELD_RECIPE, GRIP_RECIPE

KIND = "openvl_pos_reflact"
VISUAL_KEYS = ("routed_perception", "source", "image_contexts", "visual_images",
               "visual_geometry_context", "binding_ids", "candidates", "error", "automatic_multiview_geometry")


def validate_options(config):
    if not isinstance(config, Mapping) or config.get("kind") != KIND:
        raise ValueError("composition requires kind=openvl_pos_reflact")
    if set(config) - {"kind", "pos", "openvl", "reflact", "reliability", "action_focus"}:
        raise ValueError("unknown composition option")
    for key in ("pos", "openvl", "reflact"):
        if not isinstance(config.get(key, {}), dict):
            raise ValueError(key + " must be an object")
    pos = config.get("pos", {})
    if "kind" in pos:
        raise ValueError("composition fixes PoS to pos_progress_lite")
    validate_pos({"kind": "pos_progress_lite", **pos})
    visual = config.get("openvl", {})
    if set(visual) - {"enabled", "config", "ambiguity_score_margin", "duplicate_mask_iou"}:
        raise ValueError("unknown OpenVL composition option")
    reliability = config.get("reliability", {})
    if (not isinstance(reliability, dict) or set(reliability) - {"enabled"}
            or type(reliability.get("enabled", False)) is not bool):
        raise ValueError("invalid reliability options")
    action_focus = config.get("action_focus", {})
    if (not isinstance(action_focus, dict) or set(action_focus) - {"enabled"}
            or type(action_focus.get("enabled", False)) is not bool):
        raise ValueError("invalid action_focus options")
    reflection = config.get("reflact", {})
    for block in (visual, reflection):
        if type(block.get("enabled", True)) is not bool:
            raise ValueError("enabled must be boolean")
    if visual.get("enabled", True) and not isinstance(visual.get("config"), str):
        raise ValueError("enabled OpenVL requires a native config path")
    if visual.get("enabled", True) and not visual["config"].strip():
        raise ValueError("enabled OpenVL requires a nonempty config path")
    validate_ambiguity_thresholds(visual.get("ambiguity_score_margin", .10),
                                  visual.get("duplicate_mask_iou", .90))
    if "kind" in reflection:
        raise ValueError("composition fixes reflection to ReflAct v3")
    validate_reflact({"kind": "reflact", **{k: v for k, v in reflection.items()
                                                if k != "enabled"}})


class CompositeHarness(ProgressLiteHarness):
    """A data bridge around the existing three components, not a new controller."""

    def __init__(self, kit, config, task, initial_images=()):
        validate_options(config)
        self.call_reliability = CallReliability() if config.get("reliability", {}).get("enabled", False) else None
        self.visual_enabled = config.get("openvl", {}).get("enabled", True)
        if self.visual_enabled and kit.evidence_adapter is None:
            raise ValueError("install the OpenVL adapter before the composition")
        self.action_focus = ActionFocus() if config.get("action_focus", {}).get("enabled", False) else None
        self.delivered_visual_packets = deque(maxlen=4)
        self.directive_sequence = None
        self.directive_epoch = None
        self.directive_revision = None
        self.review_feedback = None
        self.committed_review_feedback = None
        self.latest_feedback = None
        self.last_audited_directive = None
        self.recent_actions = deque(maxlen=4)
        self.last_failure_sequence = None
        self.committed_sequence = None
        self.committed_epoch = None
        super().__init__(kit, {"kind": "pos_progress_lite", **config.get("pos", {})},
                         task, initial_images)
        self.kind = KIND
        self.schedule = FreshnessSchedule()
        reflection = config.get("reflact", {})
        self.reflection = (ReflActHarness(kit, {"kind": "reflact", **{
            k: v for k, v in reflection.items() if k != "enabled"}}, task)
            if reflection.get("enabled", True) else None)
        self.provenance = {
            "kind": KIND, "implementation": "pyrualean-openvl-pos-progress-lite-reflact-v11-interface",
            "pos": copy.deepcopy(self.provenance),
            "openvl_enabled": self.visual_enabled,
            "reflact": copy.deepcopy(self.reflection.provenance) if self.reflection else None,
            "shared_auxiliary_budget": True,
            "additional_reflection_requests": 0,
            "automatic_robot_replay": False,
            "pos_schema_and_validation": "inherited unchanged; no candidate rewrite",
            "pos_model_calls": "original payload fields/prompt prefixes plus schema contract and delivered-source handoff; no report rewriting",
            "pos_step_and_completion": "original step validation and completion audit; obsolete directives archived after motion",
            "robot_primitive_wrappers_added": False,
            "planner_handoff": "revision-stamped belief/gap, public feedback, compact delivered OpenVL source links and semantic limits; full original evidence retained in POS_CONTEXT",
            "fusion_adapter": "v14-delivered-source-gap-bridge",
            "auxiliary_input_change": "openvl_pos_handoff added to belief, repair and Sentinel payloads; original payload fields retained",
            "pos_schedule": "existing trigger/final reserve; funded retry capacity may be reserved inside original round/total limits",
            "additional_tools": [], "additional_host_queries": 0,
            "visual_evidence_policy": "delivered results only; original source time and eligibility retained",
        }
        if self.action_focus:
            self.provenance['action_focus'] = {
                'implementation': 'v15-advisory-action-scoped-gap-index',
                'optional_metadata': True, 'extra_model_requests': 0,
                'learned_affordances': False, 'action_gate': False,
                'core_inner_monologue_feedback_reused': True,
                'recipes': 'original pure-data examples; no injected robot API or automatic skills',
                'references': ['https://say-can.github.io/', 'https://code-as-policies.github.io/',
                               'https://innermonologue.github.io/']}
        if self.call_reliability:
            self.provenance['call_reliability'] = {
                'max_transport_retries':1, 'same_absolute_request_deadline':True,
                'latency_window':4, 'allocation_policy':'fixed_attempt_full_audit_reserved_retry',
                'latency_samples_for_diagnostics_only':True, 'retry_backoff_jitter_s':[2,3],
                'total_budget_increased':False, 'robot_replay':False,
                'state_sync_handoff':True}
        self._log("composition_initialized", provenance=self.provenance)

    def before_request(self, model_call):
        revision = self.revision
        # The original PoS owns scheduling, update, repair and Sentinel audit.
        # Do not alter its call roles, prompts, payloads or failure bookkeeping.
        def contract_call(**kwargs):
            kwargs["prompt"] += self.response_contract(kwargs["role"])
            if self.visual_enabled:
                # Same delivered evidence for belief, repair and independent
                # review. This is an input bridge, never a world-state rewrite.
                kwargs["payload"] = {**kwargs["payload"],
                    "openvl_pos_handoff": self.visual_handoff()}
            if self.action_focus:
                kwargs['payload'] = {**kwargs['payload'], 'action_focus': self.focus_context()}
                kwargs['prompt'] += AUX_PROMPT
            return model_call(**kwargs)
        attempted_evidence = list(self.pending)
        try:
            context = super().before_request(contract_call)
        finally:
            if self.revision == revision:
                restored = [eid for eid in attempted_evidence
                    if eid in self.evidence and eid not in self.pending]
                self.pending = restored + self.pending
                if restored:
                    self._log('fusion_uncommitted_evidence_retained', evidence_ids=restored,
                              status=self.last_status, sequence=self.sequence,
                              motion_epoch=self.motion_epoch,
                              original_ids_and_source_times_unchanged=True)
        if self.revision > revision:
            self.committed_sequence = self.sequence
            self.committed_epoch = self.motion_epoch
        return context

    def response_contract(self, role):
        """Render existing structural limits; do not normalize the returned report."""
        schema = self.response_schema(role)
        constraints = {}

        def visit(node, path):
            row = {key: copy.deepcopy(node[key]) for key in
                   ("required", "additionalProperties", "maxLength", "maxItems")
                   if key in node}
            if row:
                constraints[path] = row
            for name, child in node.get("properties", {}).items():
                visit(child, path + "." + name)
            if "items" in node:
                visit(node["items"], path + "[]")

        visit(schema, "payload")
        return "\nCOMPOSITE_RESPONSE_CONTRACT\n" + json.dumps({
            "role": role, "constraints": constraints,
            "combined_world_record_limit": self.options["max_records"],
            "instruction": "Submit exactly one submit_harness_state call with one payload object. "
                "Use every required field and no undeclared field. Keep bounded prose well below "
                "maxLength by summarizing before submission; do not emit a long narrative in target "
                "or directive. Preserve unresolved obligations, physical uncertainty and evidence IDs. "
                "These are the existing schema limits, not permission to omit facts, invent evidence "
                "or downgrade an unsupported physical claim silently. The original validator and "
                "independent audit still decide acceptance; no host-side truncation or coercion.",
        }, ensure_ascii=False, separators=(",", ":"))

    def _log(self, event, **data):
        # These are existing component diagnostics, not new observations or facts.
        super()._log(event, **data)
        if event == "progress_directive":
            # A belief commit does not necessarily install a new directive.
            self.directive_sequence = self.sequence
            self.directive_epoch = self.motion_epoch
            self.directive_revision = self.revision
            self.last_audited_directive = copy.deepcopy(data.get("directive"))
        if event in {"committed", "maintenance_degraded"}:
            audit = data.get("audit") or {}
            issues = (audit.get("issues") if event == "committed" else data.get("issues")) or []
            self.review_feedback = {
                "sequence": self.sequence, "motion_epoch": self.motion_epoch,
                "revision": self.revision,
                "status": "committed" if event == "committed" else data.get("status"),
                "issues": [str(x)[:800] for x in issues[:8]],
                "omitted_issues": max(0, len(issues)-8),
                "verification": "Reviewer/validator diagnostics; not physical facts or a replacement belief.",
            }

            if event == "committed":
                self.committed_review_feedback = copy.deepcopy(self.review_feedback)

    def _feedback(self, result, records):
        # Motion is already recorded by the original PoS before this method.
        # Retain the auditor's output, but stop presenting it as the NEXT action.
        if self.directive is not None and self.directive_epoch != self.motion_epoch:
            self.last_audited_directive = copy.deepcopy(self.directive)
            self.directive = None
            self._log("directive_archived", source_epoch=self.directive_epoch,
                      current_epoch=self.motion_epoch, reason="new_motion_feedback")
        feedback = super()._feedback(result, records)
        feedback["historical_controller_directive"] = (
            copy.deepcopy(self.last_audited_directive)
            if self.directive_epoch != self.motion_epoch else None)
        feedback["controller_directive_source"] = {
            "sequence": self.directive_sequence, "motion_epoch": self.directive_epoch,
            "revision": self.directive_revision,
            "applicable_to_current_epoch": self.directive_epoch == self.motion_epoch}
        return feedback

    def decision_context(self):
        gap = self.belief.get("active_gap")
        target = next((g["target"] for g in self.belief.get(gap["type"], [])
                       if g["id"] == gap["id"]), None) if gap else None
        context = {
            "belief_revision": self.revision,
            "observation_sequence": self.sequence,
            "committed_sequence": self.committed_sequence,
            "belief_updated_for_latest_feedback": self.committed_sequence == self.sequence
                and self.last_status == "committed",
            "belief_reference": {"context": "POS_CONTEXT", "revision": self.revision,
                "world_record_counts": {k: len(v) for k,v in self.belief["world"].items()},
                "note": "Full belief remains in original PoS context; this handoff does not repeat it or re-audit it."},
            "maintenance_feedback": copy.deepcopy(self.review_feedback),
            "last_committed_review": copy.deepcopy(self.committed_review_feedback),
            "active_gap": copy.deepcopy(gap), "gap_target": target,
            "unfinished_goals": copy.deepcopy(self.belief["achievement"]),
            "decision_critical_missing": copy.deepcopy(self.belief["epistemic"]),
            "directive": copy.deepcopy(self.directive) if self.directive_epoch == self.motion_epoch else None,
            "historical_directive": copy.deepcopy(self.last_audited_directive) if self.directive_epoch != self.motion_epoch else None,
            "directive_source": {"sequence": self.directive_sequence,
                "motion_epoch": self.directive_epoch, "revision": self.directive_revision,
                "same_observation_sequence": self.directive_sequence == self.sequence,
                "same_motion_epoch": self.directive_epoch == self.motion_epoch,
                "note": "Historical advisory output when versions differ; compare newer feedback before using it."},
            "directive_motion_epoch": self.directive_epoch,
            "current_motion_epoch": self.motion_epoch,
            "latest_feedback": copy.deepcopy(self.latest_feedback),
            "recent_action_feedback": copy.deepcopy(list(self.recent_actions)),
            "task_progress": {"active": self.active,
                "subgoals": {k: {f: copy.deepcopy(v[f]) for f in
                    ("status", "attempts", "last_phase", "current_evidence") if f in v}
                    for k,v in self.goals.items()}},
            "last_failure": copy.deepcopy(self.last_failure),
            "last_failure_source_sequence": self.last_failure_sequence,
            "verification": "Public feedback and fallible component outputs; no new evidence, physical inference or action gate. Historical failures are not proof of current failure.",
        }

        if self.visual_enabled:
            context['openvl_pos_handoff'] = self.visual_handoff()
        if self.call_reliability:
            synchronized = (self.committed_sequence == self.sequence
                and self.committed_epoch == self.motion_epoch
                and self.last_status == 'committed')
            unreviewed = [copy.deepcopy(x) for x in self.recent_actions
                if self.committed_sequence is None or x['sequence'] > self.committed_sequence]
            context['state_sync'] = {
                'belief_current_for_latest_feedback':synchronized,
                'committed_revision':self.revision,
                'committed_sequence':self.committed_sequence,
                'committed_motion_epoch':self.committed_epoch,
                'latest_sequence':self.sequence, 'latest_motion_epoch':self.motion_epoch,
                'maintenance_status':self.last_status,
                'unreviewed_action_feedback_tail':unreviewed,
                'tail_is_complete_history':False,
                'pending_evidence_ids':list(self.pending),
                'current_gap_is_last_audited_gap':True,
                'instruction':('Stored belief and gap are last audited outputs. Compare latest_feedback, '
                    'unreviewed action feedback and pending evidence before acting. An unavailable update '
                    'neither incorporates newer feedback nor disproves it. Do not turn an unreviewed '
                    'outcome into an audited world fact. Full evidence remains in POS_CONTEXT.')}
        if self.action_focus:
            context['action_focus'] = self.focus_context()
        return context

    def focus_context(self):
        return self.action_focus.context(self.belief, self.sequence, self.motion_epoch, self.active)

    def visual_handoff(self):
        packets = []
        for source in self.delivered_visual_packets:
            row = copy.deepcopy(source)
            row['registered_observation'] = row['evidence_id'] in self.evidence
            row['same_motion_epoch'] = row['motion_epoch'] == self.motion_epoch
            # Delivery freshness is historical metadata. A new robot action
            # invalidates reuse; re-reading this handoff never refreshes it.
            row['current_image_evidence_ids'] = [
                link['pos_evidence_id'] for link in row['image_links']
                if row['same_motion_epoch'] and link['eligible_at_delivery']
                and link['pos_evidence_id'] in self.evidence
                and self.evidence[link['pos_evidence_id']].get('current_eligible', False)]
            packets.append(row)
        return {
            'scope': 'Delivered OpenVL metadata only; not an alternative belief or new observation.',
            'latest_motion_epoch': self.motion_epoch,
            'recent_packets': packets,
            'retained_packet_limit': 4, 'complete_history': False,
            'full_sources': 'Registered obs/image IDs in POS_CONTEXT; omitted metadata is not absent evidence.',
            'semantics': {
                'candidate': 'Region proposals and score gaps are hypotheses, not object identity or 3D targets. Retain alternatives until the model resolves the relevant ambiguity.',
                'projection': 'visual_project/back_project on a supplied plane is conditional on that height and camera/source frame. It does not measure object height, contact, or seating. Preserve the supplied assumptions and source IDs when reporting geometry.',
                'calibration': 'Camera metadata/calibration is not proof that image pixels were delivered or that an object was observed. Only registered delivered image IDs identify attached pixels.',
                'freshness': 'same_motion_epoch is necessary, not sufficient. Eligibility at delivery never overrides source history, image inspection, original PoS validation, or independent review.',
                'action_result': 'Arm reachability, holding and release describe different outcomes. Keep task obligations and epistemic gaps until relevant public evidence supports the claim.'},
            'coordination': 'Start from the active audited gap and its evidence IDs. Use an existing OpenVL observation only for a decision-critical missing fact. If the available evidence supports an action, act and inspect its returned feedback; do not restart broad perception merely because a maintenance call failed. A conditional estimate remains conditional after audit.'}

    def runtime_context(self):
        return "COMPOSITE_HANDOFF\n" + json.dumps(self.decision_context(), ensure_ascii=False)

    def prepare_tools(self, specs):
        specs = super().prepare_tools(specs)
        if self.action_focus:
            python = next(t for t in specs if t['name'] == 'python')
            python['input_schema']['properties']['step']['properties']['focus'] = copy.deepcopy(FOCUS_SCHEMA)
        if not self.reflection:
            return specs
        python = next(t for t in specs if t["name"] == "python")
        step = copy.deepcopy(python["input_schema"]["properties"]["step"])
        specs = self.reflection.prepare_tools(specs)
        python = next(t for t in specs if t["name"] == "python")
        python["input_schema"]["properties"]["step"] = step
        python["description"] += " PoS step metadata remains optional; do not resubmit executed actions to fix metadata."
        return specs

    def execute_python(self, args, execute):
        if self.action_focus:
            original_execute = execute
            raw_step = args.get('step') if isinstance(args, Mapping) else None
            raw_focus = copy.deepcopy(raw_step.get('focus')) if isinstance(raw_step, dict) else None
            def focus_execute(code, **execution_options):
                # The inherited validator has already produced current_step.
                # Reflection rejection never reaches this execution callback.
                self.action_focus.begin(raw_focus, self.current_step, self.belief,
                                        self.sequence, self.motion_epoch)
                self._log('action_focus_proposal', context=self.focus_context(),
                          action_intercepted=False, extra_model_requests=0)
                return original_execute(code, **execution_options)
            execute = focus_execute
        if isinstance(args, Mapping):
            self._log("decision_bound", decision=self.decision_context(),
                      observed_state=args.get("observed_state"), goal_gap=args.get("goal_gap"),
                      verification="reported decision linked to actual PoS output; not proof of causal use")
        if not self.reflection:
            return super().execute_python(args, execute)
        if not isinstance(args, Mapping):
            return self.reflection.execute_python(args, execute)
        reflection_args = {k: v for k, v in args.items() if k != "step"}

        def run_once(code, *, before_execute):
            return super(CompositeHarness, self).execute_python(
                {"code": code, "step": args.get("step")},
                lambda body: execute(body, before_execute=before_execute))

        return self.reflection.execute_python(reflection_args, run_once)

    def after_tool(self, name, result, records):
        if self.reflection:
            result = self.reflection.after_tool(name, result, records)
        result = super().after_tool(name, result, records)
        if not self.closed and self.visual_enabled:
            self._link_visual_evidence(result)
        progress = result.result.get("task_progress", {})
        # Copy only outcomes already published by the original ProgressLite.
        # Do not query the backend, inspect simulator state or manufacture gaps.
        if isinstance(progress, dict) and progress:
            feedback = {k: copy.deepcopy(progress[k]) for k in
                ("active_subgoal", "action_outcomes", "last_failure", "recovery_guidance",
                 "strategy_review_required", "no_observed_subgoal_progress_cells") if k in progress}
            feedback.update(sequence=self.sequence, motion_epoch=self.motion_epoch,
                            evidence_id="obs-" + str(self.sequence))
            self.latest_feedback = feedback
            if feedback.get("action_outcomes"):
                self.recent_actions.append({k: copy.deepcopy(feedback[k]) for k in
                    ("sequence", "motion_epoch", "evidence_id", "action_outcomes", "last_failure")})
            if feedback.get("last_failure"):
                self.last_failure_sequence = self.sequence
        if not self.closed and self.review_feedback:
            # Feed the existing review diagnostic through the next public result,
            # without creating an observation, changing eligibility or spending a call.
            result.result["component_review_feedback"] = copy.deepcopy(self.review_feedback)
            result.result["last_committed_review"] = copy.deepcopy(self.committed_review_feedback)
            entry = self.evidence.get("obs-" + str(self.sequence))
            if entry:
                payload = json.loads(entry["text"])
                payload["component_review_feedback"] = copy.deepcopy(self.review_feedback)
                payload["last_committed_review"] = copy.deepcopy(self.committed_review_feedback)
                entry["text"] = json.dumps(payload, ensure_ascii=False, default=str)
        if self.action_focus and isinstance(progress, dict) and progress:
            self.action_focus.outcome(progress, self.sequence, self.motion_epoch,
                                      self.kit.robo.CLOSE, self.kit.robo.OPEN)
            outcome = copy.deepcopy(self.action_focus.last_outcome)
            self._log('action_focus_outcome', **outcome)
            result.result['action_focus_outcome'] = outcome
            entry = self.evidence.get('obs-' + str(self.sequence))
            if entry:
                payload = json.loads(entry['text'])
                payload['action_focus_outcome'] = outcome
                entry['text'] = json.dumps(payload, ensure_ascii=False, default=str)
        # Preserve the original task_progress/controller_directive verbatim.
        result.result["composite_decision"] = self.decision_context()
        return result

    def _link_visual_evidence(self, result):
        adapter = self.kit.evidence_adapter
        # Copy only results already delivered to the executor. Do not scan the
        # OpenVL archive or add unshown original camera captures to PoS.
        packet = {k: copy.deepcopy(result.result[k]) for k in VISUAL_KEYS
                  if k in result.result}
        contexts = []
        for ref in self.pending:
            evidence = self.evidence[ref]
            if evidence.get("sequence") != self.sequence or evidence.get("type") != "image":
                continue
            label = evidence.get("label")
            source = None
            if label in adapter.frames:
                source = adapter.frame_context(label)
            else:
                sources = result.result.get("visual_images", [])
                labels = result.result.get("images", [])
                if label in labels:
                    index = labels.index(label)
                    if index < len(sources):
                        source = sources[index]
            if source:
                evidence["openvl_source"] = copy.deepcopy(source)
                evidence["current_eligible"] = bool(source.get("current_camera_pixels"))
                evidence["timing"] = "OpenVL original capture step/action revision; crops, annotations and old frames are not fresh original camera observations"
                contexts.append({"pos_evidence_id": ref, "openvl_source": source})
            elif not any(label == cam or str(label).startswith(cam + ".")
                         for cam in self.kit.robo.CAMERAS):
                evidence["current_eligible"] = False
                evidence["timing"] = "unknown OpenVL image source; cannot certify current physical state"
        if packet or contexts:
            eid = "obs-" + str(self.sequence)
            # Keep small semantic links ahead of verbose metadata. Previously
            # only a potentially truncated observation text reached PoS; the
            # root handoff did not carry these visual/source links at all.
            kinds = []
            if packet.get('routed_perception'): kinds.append('candidate_or_routing_feedback')
            if packet.get('visual_geometry_context'): kinds.append('calibration_with_conditional_projection')
            if contexts: kinds.append('delivered_images')
            if packet.get('automatic_multiview_geometry'): kinds.append('automatic_calibrated_surface')
            compact = {'evidence_id': eid, 'sequence': self.sequence,
                'motion_epoch': self.motion_epoch, 'delivered_metadata_keys': list(packet),
                'evidence_kinds': kinds, 'image_links': [
                    {'pos_evidence_id': c['pos_evidence_id'],
                     'source': copy.deepcopy(c['openvl_source']),
                     'eligible_at_delivery': self.evidence[c['pos_evidence_id']].get('current_eligible', False)}
                    for c in contexts],
                'calibration_image_ids': [c.get('image_id') for c in packet.get('visual_geometry_context', [])
                    if isinstance(c, dict) and c.get('image_id')],
                'calibration_ids_are_not_automatically_delivered_pixels': True,
                'automatic_geometry': [{k: copy.deepcopy(g[k]) for k in
                    ('measurement_id','status','reason','source_step','views','accepted_matches','matcher_inference_calls','current_eligible','geometry','median_conditioning_sigma_m') if k in g}
                    for g in packet.get('automatic_multiview_geometry', [])[-8:]]}
            self.delivered_visual_packets.append(compact)
            self._log('fusion_evidence_linked', **compact)
            entry = self.evidence.get(eid)
            bridge = {"delivered_visual_results": packet, "image_links": contexts,
                      "interpretation": "Candidate regions are hypotheses. Crops retain source time and local pixels; they are not world targets. Executor reflections do not update world facts."}
            # Bound the public metadata without losing the registered evidence ID.
            text = json.dumps(bridge, ensure_ascii=False, default=str)
            if len(text) > 10000:
                bridge = {"image_links": contexts,
                          "visual_metadata_truncated": True,
                          "details": text[:8000]}
            if entry:
                payload = json.loads(entry["text"])
                payload["openvl_evidence"] = bridge
                entry["text"] = json.dumps(payload, ensure_ascii=False, default=str)
            self._log("visual_evidence_linked", evidence_id=eid, **bridge)

    def prompt(self):
        base = self._component_prompt()
        if self.action_focus:
            base += FOCUS_PROMPT + '\n```python\n' + FIELD_RECIPE + '\n\n' + GRIP_RECIPE + '\n```\n'
        return base

    def _component_prompt(self):
        text = super().prompt()
        if self.reflection:
            text += self.reflection.prompt()
        if self.reflection:
            direction = "Use the current PoS gap and actual feedback in ReflAct's observed_state/goal_gap before generating code in the same decision."
            title = "OpenVL -> modified PoS -> ReflAct -> execution feedback"
        else:
            direction = "Use the current PoS gap and actual feedback to choose code. Python accepts code and optional step metadata; ReflAct summaries are disabled."
            title = "OpenVL -> modified PoS -> execution feedback"
        if self.call_reliability:
            direction += (" Read state_sync: when belief_current_for_latest_feedback is false, "
                "the stored belief/gap has not absorbed all newer public evidence. "
                "Use the explicitly marked latest feedback; do not treat an unavailable review "
                "as validation or erase the unresolved gap.")
        return text + "\n# " + title + "\n" + direction + """
OpenVL delivers observation evidence with source time and uncertainty. Modified
PoS updates its belief, task progress and active gap through its original audit.
COMPOSITE_HANDOFF exposes those actual outputs without adding another policy.
Missing evidence calls for the relevant observation; an actionable achievement
gap calls for a task action.
The handoff's maintenance_feedback preserves the latest original reviewer or
validator diagnostics. Read last_committed_review too: an accepted belief may
still have reviewer caveats about conditional geometry, identity or verification.
Acceptance does not clear those caveats or certify the directive's coordinates.
A later unavailable review does not erase the last completed review. Its sequence,
motion_epoch and revision still mark historical advice, not fresh observations;
resolve relevant caveats against current tool feedback before dependent actions.
A rejected candidate was NOT committed: keep its specific
issues as questions to resolve, never adopt its unsupported values as facts.
After motion, the prior directive moves to historical_directive and is no longer
the current next action. Read directive_source and compare historical advice with
latest_feedback and recent_action_feedback before choosing code. An old empty-grasp report is not proof of current failure after new evidence;
a successful motion or unrelated read also does not prove that grasp succeeded.
Read openvl_pos_handoff together with the active gap: candidate identity,
conditional plane geometry, calibration and delivered pixels are distinct evidence.
Keep the source IDs and assumptions attached to a geometric estimate through
belief update, independent review and action choice; approval does not measure
an unknown height or resolve an unobserved contact. When observation is needed,
choose the existing tool relevant to that gap instead of starting a new broad
perception pass. When evidence is sufficient for the next action, proceed and
let its actual returned outcome update the next decision.
Use existing OpenVL tools to resolve the particular missing evidence identified
by PoS, and use the actual outcome of execution to select the next decision.
A no-progress review should identify what evidence or approach changes; do not
repeat a failed attempt just because the old directive still names that step.
Full world belief remains in the original POS_CONTEXT; this compact handoff adds
no competing world model, new tool, host query, automatic retry or action gate.
Actual execution results and delivered images return to PoS as public evidence.
Before numeric geometry calculations, check found/status and every field the
calculation needs. A None field remains unmeasured: do not round it, use it in
arithmetic or replace it with zero or a fixed guessed pose to enable an action.
Read the original tool's uncertainty/notes and use the existing observation tools
when the missing value is decision-critical. A code exception is feedback to
revise the computation; it does not justify replaying an already executed action.
Preserve the feedback boundary inside generated code: when a later action depends
on a result that does not exist yet, do not precommit that later action. Return the
result for the next decision, or explicitly test the returned result in code.
Printing a result is not a test. If a contact/placement decision needs a new image,
return and inspect that image before deciding; capturing it and then executing the
dependent action in the same unseen cell does not provide visual verification.
An IK result, blocked contact, or arrival tolerance alone cannot establish seating
or support. Use public results and delivered observations to decide whether the
next action is justified; these instructions do not intercept robot execution.
Keep all original task obligations. Beliefs and executor reports remain fallible;
read their revision, source time and the latest feedback. Crops and recalled
images retain original capture time. This bridge does not execute, retry, gate,
or select robot actions, and does not add a reflection-model request.
"""
