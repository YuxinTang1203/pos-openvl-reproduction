"""Same-decision ReflAct summaries required at the Python tool boundary.

The host records agent-authored summaries, not verified state or reasoning
traces. It neither maintains a plan nor requests a separate model decision.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping

from .arms import ArmResult

DEFAULTS = {"max_summary_chars": 320, "summary_length_policy": "strict"}
MAX_REFLECTIONS_PER_CELL = 1


def validate_options(config):
    if not isinstance(config, Mapping):
        raise ValueError("reflact config must be an object")
    if config.get("kind") != "reflact":
        raise ValueError("reflact requires kind=reflact")
    if set(config) - {"kind", *DEFAULTS}:
        raise ValueError("unknown reflact option")
    limit = config.get("max_summary_chars", DEFAULTS["max_summary_chars"])
    if type(limit) is not int or not 80 <= limit <= 640:
        raise ValueError("max_summary_chars must be an integer from 80 to 640")


    if config.get("summary_length_policy", "strict") not in ("strict", "advisory"):
        raise ValueError("summary_length_policy must be strict or advisory")


class ReflActHarness:
    kind = "reflact"

    def __init__(self, kit, config, task, initial_images=()):
        validate_options(config)
        if kit.arm != "cells":
            raise ValueError("reflact requires arm=cells")
        self.kit = kit
        self.options = {key: config.get(key, value) for key, value in DEFAULTS.items()}
        self.directory = kit.workspace.parent / "general_harness"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pending = []
        self.reflection_count = 0
        self.rejected_count = 0
        self.closed_cell = 0
        # task and initial_images already occur in the normal model context.
        # This adaptation does not parse, inspect, or duplicate them.
        self.provenance = {
            "kind": self.kind,
            "implementation": "pyrualean-reflact-v3-interface-v11",
            "paper": "https://aclanthology.org/2025.emnlp-main.1697/",
            "method_reference": "https://arxiv.org/html/2505.15182v2#S4",
            "prompt_reference": "https://arxiv.org/html/2505.15182v2#A11",
            "adaptation": "same-decision goal-state summaries required in Python tool arguments",
            "benchmark_reproduction": False,
            "additional_model_requests": 0,
            "additional_robot_actions": 0,
            "history_trimming": False,
            "independent_verifier": False,
            "completion_guard": False,
            "automatic_retry": False,
            "evaluation_updates": False,
            "audit_helper": None,
            "audit_interface": "python tool arguments: observed_state, goal_gap, code",
            "reflection_required_before_python": True,
            "missing_reflection_executes_code": False,
            "audit_is_ground_truth": False,
            "audit_interface_is_pyrualean_adaptation": True,
            "grounding_guidance": "actual feedback versus attempted actions; original goal constraints",
            "dependent_action_guidance": "check public results before dependent actions within a cell",
            "requested_argument_order": ["observed_state", "goal_gap", "code"],
            "serialization_order_proves_causality": False,
            "max_reflections_per_cell": MAX_REFLECTIONS_PER_CELL,
            **self.options,
        }
        self._event("initialized", provenance=self.provenance)

    def prompt(self):
        limit = self.options["max_summary_chars"]
        length_instruction = (
            f"Use one short sentence per summary field, at most {limit} characters each."
            if self.options["summary_length_policy"] == "strict" else
            f"Aim for one concise sentence, about {limit} characters per summary. This is a brevity target, not an execution validity limit; retain essential uncertainty instead of truncating it.")
        return f"""
# ReflAct: relate current state to the final task goal
For each Python decision, use the public observations and interaction history to
briefly describe your current situation in relation to the full task goal. State
what is established or uncertain and what still separates this situation from
that goal. Let this comparison guide the actions in the same decision.

In observed_state, compare the latest actual feedback with the expected result
of the previous action. Separate an attempted command from its observed outcome.
If feedback contradicts an earlier belief, correct that belief; if the outcome
has not been observed, say it is unverified instead of advancing a progress count.
A motion reaching a pose does not by itself establish the intended interaction,
and a blocked motion does not identify what caused contact. Use the documented
public feedback and observations, keeping unsupported interpretations uncertain.

In goal_gap, relate this evidence to the ORIGINAL full task instruction. Retain
unresolved identity, ordering and intermediate requirements even if the final
scene looks plausible. Describe the remaining mismatch rather than just listing
the next commands. Do not add cleanup or restoration objectives absent from the
instruction. If the original requirements are supported as complete, finish.

Every python TOOL CALL requires three sibling arguments: observed_state,
goal_gap, and code. Put the summaries in the tool arguments, not inside code.
Generate the arguments in that order: observed_state first, goal_gap second,
then code conditioned on those summaries. Do not write code first and append
a retrospective explanation of the commands.
Example for a hypothetical item-retrieval task:
{{"observed_state": "The latest feedback says the drawer is still closed.",
 "goal_gap": "The goal is to retrieve the item; it remains inaccessible.",
 "code": "print(robo.state())"}}
{length_instruction}
Before sending, check that code is a nonempty executable string. Never submit
only observed_state and goal_gap, and never put code inside either summary.
These are concise summaries, not a detailed reasoning trace. There is no
reflact Python helper to define, call, or replace. The host checks and records
the two fields before executing code. Missing or invalid fields reject the
whole call without executing code; fix the tool arguments in the next decision.
Use the normal Python API in code, with multiple related actions in one cell
when appropriate. Check returned public results immediately before a dependent
action in that cell. For example, do not release a carried object merely because
the transport command was issued: first check whether the required transport
and placement conditions actually hold. If a required condition fails or is
unknown, keep the dependent action from running, print the relevant feedback,
and return control to reassess. Printing a failure while unconditionally
executing the remaining commands is not a check. Use documented API fields;
do not invent success flags or repeatedly request the same inconclusive view.
Summary and actions belong to this same model decision;
no separate reflection request is needed. Continue using the existing finish tool.

The host only records your report for auditing. It does not verify the report,
execute an action, grant task success, or provide new evidence. Base the summary
on what the observations actually support, preserving uncertainty where needed.
"""

    def prepare_tools(self, standard):
        """Publish the same contract that execute_python checks, without loop ownership."""
        specs = copy.deepcopy(standard)
        python = next(spec for spec in specs if spec["name"] == "python")
        python["description"] += (
            " ReflAct: include observed_state and goal_gap alongside code in every call. "
            "The host records them before Python executes; do not define a reflact helper."
        )
        schema = python["input_schema"]
        schema["properties"]["code"]["minLength"] = 1
        for field, description in (
            ("observed_state", "Latest observed outcome versus the expected result; flag contradictions and unverified claims."),
            ("goal_gap", "Remaining mismatch to the original full goal, retaining identity and ordering requirements without adding goals."),
        ):
            schema["properties"][field] = {
                "type": "string", "minLength": 1,
                "description": description,
                **({"maxLength": self.options["max_summary_chars"]}
                   if self.options["summary_length_policy"] == "strict" else {}),
            }
        schema["properties"]["code"]["description"] = (
            "Required nonempty executable Python string. Include it alongside both summaries; "
            "summaries without code do not execute. Correct invalid arguments in a new decision; "
            "the host never fills code or replays actions.")
        schema["required"] = ["observed_state", "goal_gap", "code"]
        # Present reflection before action in the serialized schema as well as
        # in the prompt. JSON key order is not a validity constraint, and the
        # observed response order is only an audit signal, not causal proof.
        schema["properties"] = {
            field: schema["properties"][field]
            for field in ("observed_state", "goal_gap", "code")
        }
        schema["additionalProperties"] = False
        return specs

    def execute_python(self, args, execute):
        """Reject incomplete decisions before the cell counter or robot can change."""
        issues = []
        if not isinstance(args, Mapping):
            issues.append("python arguments must be an object")
        else:
            unexpected = sorted(set(args) - {"observed_state", "goal_gap", "code"})
            if unexpected:
                issues.append("unexpected arguments: " + ", ".join(unexpected))
            for field in ("observed_state", "goal_gap", "code"):
                value = args.get(field)
                if not isinstance(value, str) or not value.strip():
                    issues.append(f"{field}: required nonempty string")
                elif (field != "code" and self.options["summary_length_policy"] == "strict"
                      and len(value) > self.options["max_summary_chars"]):
                    limit = self.options["max_summary_chars"]
                    issues.append(f"{field}: exceeds {limit} characters")
        if issues:
            self.rejected_count += 1
            self._event("reflection_rejected", turn=len(self.kit.turns), cell=None,
                        interface="tool_arguments", issues=issues, code_executed=False)
            return ArmResult({
                "error": "ReflAct python arguments rejected: " + "; ".join(issues)
                + ". No code was executed. Supply observed_state, goal_gap and code as sibling "
                "tool arguments; do not define or call a reflact Python helper.",
                "reflact_protocol": {"accepted": False, "code_executed": False, "issues": issues,
                    "required_arguments": ["observed_state", "goal_gap", "code"],
                    "repair": "Generate a new complete call; no code was executed. The host will not synthesize code or automatically retry."},
            })

        def record():
            order = list(args)
            self._event(
                "decision_serialization", turn=len(self.kit.turns), cell=self.kit._cell_index,
                argument_order=order,
                reflection_fields_precede_code=all(
                    order.index(field) < order.index("code")
                    for field in ("observed_state", "goal_gap")
                ),
                verification="serialized field order only, not private reasoning or causality",
            )
            report = self.reflect(args["observed_state"], args["goal_gap"])
            if not report["recorded"]:
                raise RuntimeError("ReflAct host recording failed: " + report["error"])

        return execute(args["code"], before_execute=record)

    def _event(self, event, **fields):
        row = {"kind": self.kind, "event": event, **fields}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def reflect(self, observed_state=None, goal_gap=None, **extra):
        """Host-side recording callback; never installed in the Python namespace."""
        turn, cell = len(self.kit.turns), self.kit._cell_index
        error = None
        if extra:
            error = "expected only observed_state and goal_gap"
        elif cell <= self.closed_cell:
            error = "reflection must be recorded within a running Python cell"
        else:
            for key, value in (("observed_state", observed_state), ("goal_gap", goal_gap)):
                if not isinstance(value, str) or not value.strip():
                    error = f"{key} must be a nonempty string"
                    break
                if len(value) > self.options["max_summary_chars"]:
                    if self.options["summary_length_policy"] == "strict":
                        error = f"{key} exceeds max_summary_chars"
                        break
                    self._event("reflection_length_warning", field=key,
                                actual_chars=len(value), target_chars=self.options["max_summary_chars"],
                                policy="advisory", text_truncated=False, code_blocked=False)
        if error is None and len(self.pending) >= MAX_REFLECTIONS_PER_CELL:
            error = "maximum reflections per cell reached"
        if error is not None:
            self.rejected_count += 1
            self._event("reflection_rejected", turn=turn, cell=cell, error=error)
            return {"recorded": False, "error": error}

        # Only read the public-call counter. Never inspect the simulator, reward,
        # physical state, or the contents of earlier ledger records here.
        call_count = len(self.kit.robo._ledger)
        self.reflection_count += 1
        row = {
            "id": f"reflection-{self.reflection_count}",
            "turn": turn,
            "cell": cell,
            "observed_state": observed_state.strip(),
            "goal_gap": goal_gap.strip(),
            "source": "agent_report",
            "interface": "tool_arguments",
            "ledger_count_at_record": call_count,
        }
        self._event("reflection_recorded", **row)
        self.pending.append(row)
        return {"recorded": True, "id": row["id"], "source": "agent_report"}

    def after_tool(self, name, result, records):
        """Audit actual helper use without changing tool feedback or outcomes."""
        turn = len(self.kit.turns)
        cell = result.result.get("cell") if name == "python" else None
        reflections = [
            row
            for row in self.pending
            if name == "python" and row["turn"] == turn and row["cell"] == cell
        ]
        calls = [(row.index, bool(row.stateful)) for row in records]
        first_call = min((index for index, _ in calls), default=None)
        first_stateful = min((index for index, stateful in calls if stateful), default=None)
        usage = []
        for row in reflections:
            count = row["ledger_count_at_record"]
            usage.append(
                {
                    "id": row["id"],
                    "before_first_primitive": None if first_call is None else count <= first_call,
                    "before_first_stateful": (
                        None if first_stateful is None else count <= first_stateful
                    ),
                    "subsequent_primitive_count": sum(index >= count for index, _ in calls),
                    "subsequent_stateful_count": sum(
                        index >= count and stateful for index, stateful in calls
                    ),
                }
            )
        self._event(
            "after_tool",
            tool=name,
            turn=turn,
            cell=cell,
            adoption=("recorded" if reflections else "rejected" if self.rejected_count
                      else "not_executed")
            if name == "python"
            else "not_applicable",
            reflection_ids=[row["id"] for row in reflections],
            reflection_count=len(reflections),
            reflection_usage=usage,
            rejected_count=self.rejected_count,
            orphaned_reflection_count=len(self.pending) - len(reflections),
            tool_failed="error" in result.result or "exception" in result.result,
            primitive_error_count=sum(bool(row.error) for row in records),
            primitive_count=len(calls),
            stateful_count=sum(stateful for _, stateful in calls),
            verification="agent summaries; adoption does not certify factuality or causality",
        )
        self.pending.clear()
        self.rejected_count = 0
        if name == "python" and type(cell) is int:
            self.closed_cell = max(self.closed_cell, cell)
        return result
