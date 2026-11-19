"""V16: minimal OpenVL/PoS fusion with explicit failure-dependency feedback."""
from __future__ import annotations

import copy
import functools
import inspect
import json

from .composite_harness import CompositeHarness, validate_options as validate_composite
from .failure_dependencies import DEPENDENCY_SCHEMA, FailureDependencies

KIND = "minimal_fusion"
MODES = {"off", "advisory", "pause_cell"}
OBSERVATION_SCOPES = {"baseline", "next_action"}
NEXT_ACTION_PROMPT = """
# Observation for the next action
Keep every task obligation in PoS, but choose visual work for the next action.
Name the next action and the specific unresolved question that could change it.
Reuse adequate current evidence; defer unrelated objects/targets until needed.
Do not reconstruct the entire scene or all target geometry before the first
useful action unless that information is a prerequisite for that action.
After motion, reobserve affected geometry when the next decision requires it;
historical estimates remain historical. Read API status, values and uncertainty
before deciding that a helper failed. An uncertain estimate is not automatically
an instruction for another observation: choose whether another view can resolve
the current question, or a justified reversible action can provide useful feedback.
This changes observation choice only; retain source freshness, public-feedback
checks and the original task completion audit.
"""
PROMPT = """
# Failure dependencies
COMPOSITE_HANDOFF.failure_dependencies indexes public API reports, never task-success labels.
Optional step.dependencies=[{"fact":"holding:right","belief_ids":[]}] links
the current intent to a typed prerequisite and existing physical belief IDs.
Only declare dependencies you actually use; declarations do not prove readiness.
After CLOSE reports holding=False, carrying that intended object lacks its
grasp prerequisite: end this action sequence, reobserve/relocalize and retry.
OPEN holding=False describes release. holding=True does not identify the object.
An explicit IK error WITH steps_used=0 means that requested motion did not run.
Do not assume arrival or retry unchanged conditions; use the returned reason to
change relevant pose/orientation/path/arm, or acquire discriminating evidence.
blocked/reached=False after execution or with unknown step count is different:
state may have changed. Check retention and the local object/target relation;
do not label the cause as collision, insertion or empty grasp without evidence.
Ask one observation question whose possible answers change the next action.
Use existing OpenVL views/Zoom/candidates as needed; keep their original source
time and assumptions. Recalling/cropping does not refresh a frame. Finish every
original task obligation under the existing PoS/Sentinel audit and budgets.
In advisory mode use short cells and explicit branches: the host does not stop
later actions. pause_cell is a separate intervention: the next instrumented
action in that cell is refused after a failure, even if it is a recovery action.
Already executed actions/variables remain; never replay the whole cell.
"""


def _base(config):
    return {"kind": "openvl_pos_reflact", "pos": config.get("pos", {}),
            "openvl": config.get("openvl", {}), "reflact": {"enabled": False},
            "action_focus": config.get("action_focus", {}),
            "reliability": config.get("reliability", {})}


def validate_options(config):
    if config.get("kind") != KIND or set(config) - {"kind", "pos", "openvl", "dependency_mode", "observation_scope", "action_focus", "reliability"}:
        raise ValueError("minimal_fusion accepts kind,pos,openvl,dependency_mode,observation_scope,action_focus,reliability only")
    if config.get("dependency_mode", "advisory") not in MODES:
        raise ValueError("dependency_mode must be off, advisory or pause_cell")
    if config.get("observation_scope", "baseline") not in OBSERVATION_SCOPES:
        raise ValueError("observation_scope must be baseline or next_action")
    validate_composite(_base(config))


class DependencyReviewRequired(RuntimeError):
    """A cell boundary intervention; not a failed robot motion or an episode stop."""


class MinimalFusionHarness(CompositeHarness):
    def __init__(self, kit, config, task, initial_images=()):
        validate_options(config)
        self.dependency_mode = config.get("dependency_mode", "advisory")
        self.observation_scope = config.get("observation_scope", "baseline")
        self.dependencies = FailureDependencies()
        self._dependency_cell_events = []
        self._seen_dependency_calls = set()
        self._inside_dependency_cell = False
        super().__init__(kit, _base(config), task, initial_images)
        self.kind = KIND
        self.provenance.update(kind=KIND, implementation="minimal-fusion-v17",
            observation_scope=self.observation_scope,
            dependency_mode=self.dependency_mode, reflact=None,
            robot_primitive_wrappers_added=self.dependency_mode != "off",
            additional_model_requests=0, additional_sensing_calls=0,
            dependency_links="optional executor declarations; invalidations only, no promotion",
            same_cell_intervention=self.dependency_mode == "pause_cell",
            original_temporal_validation_preserved=True,
            full_world_selective_freshness_implemented=False)
        if self.dependency_mode != "off":
            self._instrument()
        self._log("minimal_fusion_initialized", provenance=self.provenance)

    def _instrument(self):
        robo = self.kit.robo
        # Public RoboDojo atomic actions; do not wrap perception or internal backend handles.
        for name in ("move_to", "set_gripper", "home", "move_both", "move_waist", "move_base"):
            original = getattr(robo, name, None)
            if not callable(original):
                continue
            signature = inspect.signature(original)

            def make_wrapper(original=original, name=name, signature=signature):
                @functools.wraps(original)
                def call(*args, **kwargs):
                    if not self._inside_dependency_cell:
                        return original(*args, **kwargs)
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    params = dict(bound.arguments)
                    check = self.dependencies.before_action(name, params)
                    self._log("dependency_request", method=name, parameters=params, **check)
                    if check["same_cell_after_failure"]:
                        self._log("dependency_followup", method=name, parameters=params, **check)
                        if self.dependency_mode == "pause_cell":
                            raise DependencyReviewRequired(
                                "Previous public failure requires a new cell before another atomic action. "
                                "Inspect failure_dependencies; prior actions were executed exactly once. "
                                f"Source: {check['failure_event_id']}")
                    start = len(robo._ledger.records)
                    value = original(*args, **kwargs)
                    for record in robo._ledger.records[start:]:
                        self._consume(record, self.dependencies.signature(params))
                    return value  # Original dataclass/return identity is preserved.
                return call
            setattr(robo, name, make_wrapper())

    def _consume(self, record, signature=None):
        if record.index in self._seen_dependency_calls or not record.stateful:
            return
        self._seen_dependency_calls.add(record.index)
        event = self.dependencies.observe(record, close=self.kit.robo.CLOSE,
            open_=self.kit.robo.OPEN, cell=self.dependencies.cell,
            request_signature=signature)
        if event:
            self._dependency_cell_events.append(event)
            # Never remove obligations or turn model-selected links into physical truth.
            invalid_ids = {i for a in event["affected_declared_dependencies"] for i in a["belief_ids"]}
            for kind in ("states", "relations"):
                for row in self.belief["world"][kind]:
                    if row["id"] in invalid_ids:
                        row["status"] = "uncertain"
            for affected in event["affected_declared_dependencies"]:
                goal = self.goals.get(affected["goal"])
                if goal:
                    goal["status"] = "uncertain"
                    goal["current_evidence"] = False
            self._log("dependency_update", **event)

    def prepare_tools(self, specs):
        specs = super().prepare_tools(specs)
        if self.dependency_mode != "off":
            python = next(t for t in specs if t["name"] == "python")
            python["input_schema"]["properties"]["step"]["properties"]["dependencies"] = copy.deepcopy(DEPENDENCY_SCHEMA)
        return specs

    def execute_python(self, args, execute):
        if self.dependency_mode == "off":
            return super().execute_python(args, execute)
        raw_step = args.get("step") if isinstance(args, dict) else None
        step = copy.deepcopy(raw_step) if isinstance(raw_step, dict) else {}
        if not isinstance(step.get("goal"), str) or step["goal"] not in self._eligible_goals():
            step["goal"] = self.active if self.active in self._eligible_goals() else "task-root"
        self.dependencies.begin(self.kit._cell_index + 1, step, self.belief)
        self._dependency_cell_events = []
        self._inside_dependency_cell = True
        try:
            return super().execute_python(args, execute)
        finally:
            self._inside_dependency_cell = False

    def after_tool(self, name, result, records):
        if self.dependency_mode != "off":
            for record in records:
                self._consume(record)  # Covers native calls not instrumented above.
            if self._dependency_cell_events:
                result.result["failure_dependencies"] = {
                    "mode": self.dependency_mode, "events": copy.deepcopy(self._dependency_cell_events),
                    "facts": copy.deepcopy(self.dependencies.facts),
                    "metadata_warnings": list(self.dependencies.warnings)}
                self._dependency_cell_events = []
        # The original PoS registers this public result and OpenVL image lineage.
        result = super().after_tool(name, result, records)
        if self.dependency_mode != "off" and name != "finish":
            entry = self.evidence.get("obs-" + str(self.sequence))
            if entry and "failure_dependencies" in result.result:
                payload = json.loads(entry["text"])
                payload["failure_dependencies"] = copy.deepcopy(result.result["failure_dependencies"])
                entry["text"] = json.dumps(payload, ensure_ascii=False, default=str)
        return result

    def before_request(self, model_call):
        if self.dependency_mode == "off":
            return super().before_request(model_call)
        def contextual_call(**kw):
            kw["payload"] = {**kw["payload"], "failure_dependencies": self.dependencies.context()}
            kw["prompt"] += ("\nUse failure_dependencies as source-stamped public report metadata. "
                "Independently check evidence and all original task obligations; declarations and "
                "API reports do not prove object identity, insertion, or task success. Do not clear "
                "a failed prerequisite merely because another image was delivered.")
            return model_call(**kw)
        return super().before_request(contextual_call)

    def decision_context(self):
        context = super().decision_context()
        if self.dependency_mode != "off":
            context["failure_dependencies"] = self.dependencies.context()
        return context

    def prompt(self):
        base = super().prompt()
        if self.observation_scope == "next_action":
            base += NEXT_ACTION_PROMPT
        return base if self.dependency_mode == "off" else base + PROMPT + "\nMode: " + self.dependency_mode

