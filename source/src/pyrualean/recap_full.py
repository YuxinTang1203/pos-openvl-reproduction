"""ReCAP's recursive controller and shared-context protocol for PyRUA.

Algorithmic port of ReCAP (Zhang et al., arXiv:2510.23822, Algorithm 1),
https://github.com/ReCAP-Stanford/ReCAP. This implementation uses the existing
Python cell as a leaf action and OpenAI function calls as the structured plan
format. It is not a reproduction of the upstream benchmark experiments.

The host advances the head, returns to the parent, and requests a revised plan.
It never treats successful Python execution as proof of a physical subgoal.
Completion remains an agent claim; the benchmark owns the official verdict.
"""

from __future__ import annotations

import copy
import json

from .arms import ArmResult

DEFAULTS = {"window_rounds": 64, "max_depth": 8, "max_pending": 48}


def validate_options(config):
    for key, default in DEFAULTS.items():
        value = config.get(key, default)
        if type(value) is not int or not 1 <= value <= 256:
            raise ValueError(f"recap_full {key} must be an integer in 1..256")


def _text(value, name, limit):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be nonempty text of at most {limit} characters")
    return value.strip()


class ReCAPFull:
    kind = "recap_full"
    controls_loop = True

    def __init__(self, kit, config, task):
        validate_options(config)
        if kit.arm != "cells":
            raise ValueError("recap_full requires arm=cells")
        self.kit = kit
        self.options = {k: config.get(k, v) for k, v in DEFAULTS.items()}
        self.frames = [self._frame(str(task), 0)]
        self.phase = "initial_plan"
        self.revision = 0
        self.next_id = 1
        self.pending_action = None
        self.closed = False
        self.last_outcome = None
        self.directory = kit.workspace.parent / "general_harness"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.provenance = {
            "kind": self.kind,
            "implementation": "pyrualean-recap-controller-v1",
            "paper": "https://arxiv.org/abs/2510.23822",
            "source": "https://github.com/ReCAP-Stanford/ReCAP",
            "adaptation": "algorithmic port; Python cell leaves; function-call plans",
            "automatic_parent_refinement": True,
            "history_trimming": True,
            "shared_context": True,
            "independent_verifier": False,
            "automatic_retry": False,
            "benchmark_reproduction": False,
            **self.options,
        }

    @staticmethod
    def _frame(goal, node_id):
        return {"id": node_id, "goal": goal, "thought": "", "remaining": [],
                "observation": None}

    def prompt(self):
        return """
# ReCAP recursive execution controller
Use exactly one recap_decide call per model reply. Return a brief planning
summary and a complete ordered subtask list for the CURRENT node, using the
latest observation. Keep the remaining plan when it is still appropriate;
revise it when execution changes the situation. The host executes only the
head and automatically asks you to refine the parent plan afterwards.
A subtask is either a goal needing decomposition or executable Python code.
EVERY kind=python item, including later items, must contain nonempty code.
Use kind=goal with no code for future work whose implementation is not ready.
The whole list is validated before the head runs; an error such as
subtasks[1].code refers to the second item, not necessarily the head. Correct that
item (or make it a goal), keeping a valid head unchanged when appropriate.
A Python cell is a leaf: it can observe or act with the normal robo API and
persistent Python session. Put useful executable work at the head whenever
possible. Keep decomposition within the current goal. After a leaf, inspect
its result and images before claiming the physical subgoal is complete.
Parent siblings are stored by the host; do not copy them into a child's plan.
Return status=complete with evidence only when the CURRENT goal is achieved;
return status=failed with evidence when it cannot be achieved. Both use an
empty subtask list. The host then restores the parent and requests a new plan.
Completion reporting is a recap_decide status, not another subtask or a Python
print operation. Once the current goal is achieved, report complete directly.
Root completion is an agent report, never an official success verdict.
The shared conversation uses a sliding window. The current ancestry, latest
planning summaries, remaining subtasks and latest outcome are reintroduced
on each request. Python variables persist even if their creation leaves the
window; don't assume you remember old images or measurements.
The legacy recap_plan/refine/done/backtrack helpers are not available here.
Use finish(status=failure or stuck) to stop if the budget or evidence is
insufficient. Planning, decomposition and execution all share the episode's
request and wall-clock budgets.
"""

    def tools(self, standard):
        spec = {
            "name": "recap_decide",
            "description": "Revise the CURRENT ReCAP goal's plan; the host executes only its head.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "maxLength": 800,
                                "description": "Brief planning summary, based on the latest observation."},
                    "status": {"type": "string", "enum": ["continue", "complete", "failed"]},
                    "subtasks": {
                        "type": "array", "maxItems": 12,
                        "items": {"anyOf": [
                            {"type": "object", "properties": {
                                "goal": {"type": "string", "minLength": 1, "maxLength": 240},
                                "kind": {"type": "string", "enum": ["python"]},
                                "code": {"type": "string", "minLength": 1, "maxLength": 16000,
                                         "description": (
                                             "Required for EVERY Python item, including future leaves.")},
                            }, "required": ["goal", "kind", "code"], "additionalProperties": False},
                            {"type": "object", "properties": {
                                "goal": {"type": "string", "minLength": 1, "maxLength": 240},
                                "kind": {"type": "string", "enum": ["goal"]},
                            }, "required": ["goal", "kind"], "additionalProperties": False},
                        ]},
                    },
                    "evidence": {"type": "string", "maxLength": 800,
                                 "description": "Required for complete/failed; identify the observed outcome."},
                },
                "required": ["thought", "status", "subtasks"], "additionalProperties": False,
            },
        }
        finish = copy.deepcopy(next(s for s in standard if s["name"] == "finish"))
        finish["description"] = "Abort the episode when stuck or failed. Report successful completion through recap_decide."
        finish["input_schema"]["properties"]["status"]["enum"] = ["failure", "stuck"]
        return [spec, finish]

    def _validate_decision(self, args):
        if set(args) - {"thought", "status", "subtasks", "evidence"}:
            raise ValueError("unknown ReCAP decision fields")
        thought = _text(args.get("thought"), "thought", 800)
        status = args.get("status")
        if status not in {"continue", "complete", "failed"}:
            raise ValueError("status must be continue, complete or failed")
        items = args.get("subtasks")
        if not isinstance(items, list) or len(items) > 12:
            raise ValueError("subtasks must be a list of at most 12 items")
        evidence = args.get("evidence", "")
        if evidence:
            evidence = _text(evidence, "evidence", 800)
        if status != "continue":
            if items:
                raise ValueError("complete/failed requires an empty subtask list")
            evidence = _text(evidence, "evidence", 800)
            return thought, status, [], evidence
        if not items:
            raise ValueError("continue requires at least one subtask; check the goal before returning complete")
        steps = []
        for index, item in enumerate(items):
            path = f"subtasks[{index}]"
            if not isinstance(item, dict) or set(item) - {"goal", "kind", "code"}:
                raise ValueError(f"{path}: expected goal, kind, and code only for a Python leaf")
            goal = _text(item.get("goal"), f"{path}.goal", 240)
            kind = item.get("kind")
            if kind not in {"goal", "python"}:
                raise ValueError(f"{path}.kind must be goal or python")
            step = {"goal": goal, "kind": kind}
            if kind == "python":
                try:
                    step["code"] = _text(item.get("code"), f"{path}.code", 16000)
                except ValueError as exc:
                    raise ValueError(
                        f"{exc}; every Python item needs code, including future items. "
                        f"If {path} is future work not yet implemented, "
                        "use kind='goal' and omit code. "
                        "No head was executed."
                    ) from None
            elif "code" in item:
                raise ValueError(f"{path}.code: omit code entirely for kind='goal'")
            steps.append(step)
        if sum(len(f["remaining"]) for f in self.frames[:-1]) + len(steps) > self.options["max_pending"]:
            raise ValueError("ReCAP pending-subtask limit reached; simplify the plan")
        if steps[0]["kind"] == "goal" and len(self.frames) >= self.options["max_depth"]:
            raise ValueError("ReCAP maximum depth reached; provide an executable Python head or report failure")
        return thought, status, steps, evidence

    def execute(self, name, args, run_python, finish):
        if self.closed:
            raise ValueError("ReCAP episode already closed")
        if name == "finish":
            if args.get("status") not in {"failure", "stuck"}:
                raise ValueError("successful completion requires recap_decide status=complete with evidence")
            result = finish(args)
            self.closed = self.kit.finish is not None
            if self.closed:
                self.phase = "aborted"
            return result
        if name != "recap_decide":
            raise ValueError("recap_full requires recap_decide; put executable Python in its head subtask")
        # Validate the complete update before mutating any frame or moving the robot.
        thought, status, steps, evidence = self._validate_decision(args)
        frame = self.frames[-1]
        frame["thought"] = thought
        self.revision += 1
        if status != "continue":
            outcome = {"goal": frame["goal"], "status": "agent_reported_" + status,
                       "evidence": evidence, "node": frame["id"]}
            self.last_outcome = outcome
            if len(self.frames) == 1:
                result = finish({"status": "success" if status == "complete" else "failure",
                                 "summary": evidence})
                self.closed = self.kit.finish is not None
                self.phase = "finished" if self.closed else "verify_goal"
                self._log("root_report", outcome=outcome)
                return result
            child = self.frames.pop()
            parent = self.frames[-1]
            # A failed child remains pending until the parent repairs the plan.
            if status == "complete":
                parent["remaining"].pop(0)
            parent["observation"] = outcome
            self.phase = "refine_parent" if parent["remaining"] else "verify_goal"
            self._log("return_to_parent", child=child["id"], outcome=outcome)
            return ArmResult({"recap_transition": "return_to_parent"})
        frame["remaining"] = steps
        head = steps[0]
        if head["kind"] == "goal":
            child = self._frame(head["goal"], self.next_id)
            self.next_id += 1
            self.frames.append(child)
            self.phase = "decompose"
            self._log("descend", parent=frame["id"], child=child["id"])
            return ArmResult({"recap_transition": "descend"})
        self.pending_action = {"node": frame["id"], "goal": head["goal"]}
        self.phase = "execute"
        self._log("execute_head", head={"goal": head["goal"], "kind": head["kind"]})
        result = run_python(head["code"])
        result.result["recap_executed_tool"] = "python"
        return result

    def after_tool(self, name, result, records):
        if self.pending_action is not None:
            action, self.pending_action = self.pending_action, None
            from .general_harness import GeneralHarness

            diagnostics = []
            for r in records:
                diagnostic = GeneralHarness._diagnostic(r, GeneralHarness._public_summary(r))
                if diagnostic:
                    diagnostics.append({"call": r.index, "tool": r.tool,
                                        "category": diagnostic[0], "error": diagnostic[1][:240]})
            result.result["recap_executed_tool"] = "python"
            failed = bool(result.result.get("error") or result.result.get("exception") or diagnostics)
            outcome = {
                **action, "status": "execution_failed" if failed else "executed_unverified",
                "cell": result.result.get("cell"), "diagnostics": diagnostics[-4:],
                "error": str(result.result.get("error") or result.result.get("exception") or "")[:500],
                "stdout": str(result.result.get("stdout") or "")[-1000:],
                "images": list(result.result.get("images") or [])[:8],
            }
            self.last_outcome = outcome
            frame = self.frames[-1]
            frame["observation"] = outcome
            if not failed:
                frame["remaining"].pop(0)
            self.phase = "repair_plan" if failed else ("refine_parent" if frame["remaining"] else "verify_goal")
            self.revision += 1
            self._log("leaf_return", outcome=outcome)
        context = self.context()
        # Prefer the control state when stdout is large; never hide the original error.
        from .arms import MAX_TEXT_CHARS

        original = result.content_blocks[0]["text"]
        result.result = {"recap_full": context, **result.result}
        # The full path is injected by request_prompt, outside the tool text cap.
        # A compact feedback header keeps large plans from hiding execution errors.
        compact = {"revision": self.revision, "phase": self.phase,
                   "current_goal": self.frames[-1]["goal"], "last_outcome": self.last_outcome}
        prefix = json.dumps({"recap_full": compact}, ensure_ascii=False, default=str)
        remaining = max(0, MAX_TEXT_CHARS - len(prefix) - 100)
        if len(original) > remaining:
            half = remaining // 2
            original = original[:half] + "\n[tool text truncated; head and tail retained]\n" + original[-half:]
        result.rendered_text = prefix + "\n" + original
        self._log("tool_feedback", tool=name, error=result.result.get("error"))
        return result

    def context(self):
        # Reinject descriptions, not potentially large future code blocks. Code remains
        # in the shared recent conversation and the archived controller events.
        frames = []
        for frame in self.frames:
            frames.append({"id": frame["id"], "goal": frame["goal"], "thought": frame["thought"],
                           "remaining": [{"goal": s["goal"], "kind": s["kind"]} for s in frame["remaining"]]})
        return {"revision": self.revision, "phase": self.phase, "path": frames,
                "last_outcome": copy.deepcopy(self.last_outcome),
                "verification": "agent claims; official verdict belongs to the evaluator"}

    def request_prompt(self):
        directive = {
            "initial_plan": "Generate an ordered plan for the root goal.",
            "decompose": "Decompose the current goal; execute a Python head when possible.",
            "refine_parent": "You are back at the parent. Revise its remaining plan using the latest outcome.",
            "repair_plan": "The leaf reported an error. Repair the current plan using its diagnostics; no action was automatically retried.",
            "verify_goal": "No subtasks remain. Check the current goal against observations. Report complete with evidence, or add missing work.",
        }.get(self.phase, "Inspect the current controller state.")
        return "ReCAP controller state:\n" + json.dumps(self.context(), ensure_ascii=False) + "\n" + directive

    def _log(self, event, **fields):
        with (self.directory / "events.jsonl").open("a") as f:
            f.write(json.dumps({"event": event, "revision": self.revision,
                                "context": self.context(), **fields}, ensure_ascii=False, default=str) + "\n")
        target = self.directory / "state.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps({"context": self.context(), "frames": self.frames,
                                    "closed": self.closed}, ensure_ascii=False, indent=2))
        temp.replace(target)
