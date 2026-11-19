"""Small, independent adaptations of ReCAP, ACE and failure-aware feedback.

These are mechanism ablations, not reproductions of the papers' full systems.
They use the existing planner and public tool records; no extra model calls,
simulator reads, retries, evaluator access, or changes to the episode budget.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import OrderedDict, deque
from pathlib import Path

KINDS = {"recap", "ace", "failure_memory"}
SOURCES = {
    "recap": "https://github.com/ReCAP-Stanford/ReCAP",
    "ace": "https://github.com/ace-agent/ace",
    "failure_memory": "https://arxiv.org/html/2609.32423v2",
}


def _text(value, limit=400):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"expected nonempty text of at most {limit} characters")
    return value.strip()


def _steps(value):
    if not isinstance(value, list) or not 1 <= len(value) <= 12:
        raise ValueError("a plan needs 1..12 ordered subgoals")
    return [_text(item, 160) for item in value]


def curate_playbook(base, deltas, *, sources, excluded_tasks):
    """Apply offline, auditable ACE-style bullet deltas; never call an LLM.

    An offline reviewer or a separately run reflector supplies deltas. Counters represent
    supplied annotations, not a claim of measured treatment effectiveness.
    Evaluation tasks must not occur anywhere in the source registry.
    """
    if any(s["task"] in excluded_tasks for s in sources.values()):
        raise ValueError("evaluation task leaked into development sources")
    bullets = {b["id"]: copy.deepcopy(b) for b in base}
    for delta in deltas:
        op, bid = delta["op"], _text(delta["id"], 64)
        if op == "add":
            text = _text(delta["text"])
            refs = delta["sources"]
            if not refs or any(ref not in sources for ref in refs):
                raise ValueError("every lesson needs registered development evidence")
            if bid in bullets or any(
                b["text"].casefold() == text.casefold() for b in bullets.values()
            ):
                raise ValueError("duplicate bullet; tag it instead")
            bullets[bid] = {
                "id": bid,
                "text": text,
                "sources": list(refs),
                "helpful": 0,
                "harmful": 0,
                "neutral": 0,
            }
        elif op == "tag":
            tag = delta["tag"]
            if tag not in {"helpful", "harmful", "neutral"}:
                raise ValueError("unknown annotation")
            bullets[bid][tag] += 1
        elif op == "remove":
            del bullets[bid]
        else:
            raise ValueError("unknown playbook delta")
    if len(bullets) > 24:
        raise ValueError("curate/prune to at most 24 bullets before freezing")
    for bullet in bullets.values():
        _text(bullet["text"])
        if not bullet["sources"] or any(ref not in sources for ref in bullet["sources"]):
            raise ValueError("unregistered evidence in base playbook")
    return {
        "schema": "pyrualean-ace-frozen-v1",
        "frozen": True,
        "reflector": "offline-agent-curated-deltas; no separate reflector calls or training",
        "excluded_tasks": list(excluded_tasks),
        "sources": sources,
        "bullets": list(bullets.values()),
    }


def load_playbook(config):
    raw = Path(config["playbook"]).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != config["sha256"]:
        raise ValueError("frozen ACE playbook hash mismatch")
    book = json.loads(raw)
    if book.get("schema") != "pyrualean-ace-frozen-v1" or book.get("frozen") is not True:
        raise ValueError("ACE evaluation requires a frozen playbook")
    if not book.get("excluded_tasks"):
        raise ValueError("declare held-out tasks before freezing ACE")
    ids = [b["id"] for b in book["bullets"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate bullet id")
    curate_playbook(
        book["bullets"], [], sources=book["sources"], excluded_tasks=book["excluded_tasks"]
    )
    return book


class GeneralHarness:
    def __init__(self, kit, config):
        if kit.arm != "cells":
            raise ValueError("general harness pilots currently require arm=cells")
        self.kit = kit
        self.kind = config["kind"]
        self.frames = []
        self.completed = deque(maxlen=4)
        self.revision = 0
        self.failures = OrderedDict()
        self.book = load_playbook(config) if self.kind == "ace" else None
        self.directory = kit.workspace.parent / "general_harness"
        self.directory.mkdir(parents=True, exist_ok=True)
        if self.book:
            (self.directory / "frozen_playbook.json").write_bytes(
                Path(config["playbook"]).read_bytes()
            )
        if self.kind == "recap":
            kit._globals.update(
                recap_plan=self.plan,
                recap_refine=self.refine,
                recap_done=self.done,
                recap_backtrack=self.backtrack,
            )
        self.provenance = {
            "kind": self.kind,
            "source": SOURCES[self.kind],
            "implementation": "pyrualean-general-harness-v1",
            "adaptation": True,
            "additional_model_requests": 0,
            "history_trimming": False,
            "automatic_retry": False,
            "evaluation_updates": False,
        }
        if self.book:
            self.provenance["playbook_sha256"] = config["sha256"]

    def plan(self, steps):
        """Initialize once. Execute the head; refine or backtrack thereafter."""
        if self.frames or self.revision:
            raise ValueError("plan exists; refine the head or backtrack")
        self.frames = [{"goal": "episode", "remaining": _steps(steps)}]
        self.revision += 1
        return self.plan_context()

    def refine(self, steps):
        if not self.frames or not self.frames[-1]["remaining"]:
            raise ValueError("no active subgoal")
        if len(self.frames) >= 4:
            raise ValueError("maximum recursive plan depth is 4")
        children = _steps(steps)
        if sum(len(f["remaining"]) for f in self.frames) + len(children) > 24:
            raise ValueError("keep at most 24 pending subgoals")
        self.frames.append({"goal": self.frames[-1]["remaining"][0], "remaining": children})
        self.revision += 1
        return self.plan_context()

    def done(self, evidence):
        evidence = _text(evidence, 240)
        if not self.frames or not self.frames[-1]["remaining"]:
            raise ValueError("no active subgoal")
        goal = self.frames[-1]["remaining"].pop(0)
        self.completed.append({"goal": goal, "agent_reported_evidence": evidence})
        # Completing children restores the parent, which requires its own check.
        if not self.frames[-1]["remaining"] and len(self.frames) > 1:
            self.frames.pop()
        self.revision += 1
        return self.plan_context()

    def backtrack(self, reason, steps):
        reason, steps = _text(reason, 240), _steps(steps)
        if not self.frames:
            raise ValueError("initialize a plan first")
        frames = copy.deepcopy(self.frames)
        if len(frames) > 1:
            frames.pop()
        # Replace the current head with recovery work while preserving siblings.
        pending = steps + frames[-1]["remaining"][1:]
        if sum(len(f["remaining"]) for f in frames[:-1]) + len(pending) > 24:
            raise ValueError("keep at most 24 pending subgoals")
        frames[-1]["remaining"] = pending
        self.frames = frames
        self.completed.append({"backtrack_reason": reason})
        self.revision += 1
        return self.plan_context()

    def plan_context(self):
        return {
            "revision": self.revision,
            "frames": copy.deepcopy(self.frames),
            "recent": list(self.completed),
            "verification": "agent reports, not task verdicts",
        }

    def prompt(self):
        if self.kind == "recap":
            return """
# Recursive subgoal memory (ReCAP adaptation)
In the existing python tool, initialize recap_plan([ordered subgoals]) before
acting. Execute only the head. Use recap_refine([smaller subgoals]) for a hard
head; the parent and its pending siblings persist. After observing its outcome,
call recap_done('cell/view reference and what changed'). Completing children
restores their parent for an explicit outcome check. Motion success alone does
not prove a subgoal. Use recap_backtrack('observed failure', [recovery subgoals])
when the approach fails. Keep plans short and batch these helpers with normal
Python work; they neither move the robot nor add a model request. Current parent
plans are reinjected in tool feedback. Completion records are your claims,
not ground truth. Stop honestly if verification or recovery is unaffordable.
"""
        if self.kind == "ace":
            entries = "\n".join(f"- [{b['id']}] {b['text']}" for b in self.book["bullets"])
            return "\n# Frozen development playbook (ACE adaptation)\n" + (
                "Use relevant lessons as hypotheses, checked against current observations. "
                "The playbook is frozen; do not learn test-layout solutions into it.\n"
                + entries
                + "\n"
            )
        return """
# Failure-aware tool feedback (PluginRSI component adaptation)
Tool feedback includes public primitive failures even when Python omitted their
return values. Repeated identical command/diagnostic pairs persist across cells.
On a repeat, inspect the earliest failure and change a relevant precondition,
measurement, target or approach before retrying. Contact flags are uncertain;
do not infer task failure or success from them alone. This feedback is advisory
and arrives after the cell: use ordinary Python conditions to stop dependent
actions inside a cell. No automatic retries or extra model calls are performed.
"""

    @staticmethod
    def _public_summary(record):
        s = dict(record.summary)
        # LIBERO's typed Move derives reached from these public fields; RoboDojo
        # already returns reached. Do not guess missing tolerances or distances.
        if (
            "reached" not in s
            and record.tool in {"move_to", "move_pose"}
            and "final_dist_m" in s
            and "tol" in record.kwargs
        ):
            s["reached"] = float(s["final_dist_m"]) <= float(record.kwargs["tol"])
        return s

    @staticmethod
    def _diagnostic(record, s):
        if record.error:
            return "tool_error", str(record.error)[:240]
        if s.get("ik_error"):
            return "ik_error", str(s["ik_error"])[:240]
        if s.get("reached") is False and s.get("blocked") is not True:
            return "target_not_reached", "Inspect actual pose before dependent actions."
        if s.get("found") is False:
            return "not_found", "No object evidence returned."
        return None

    def observe(self, records):
        recent = []
        for r in records:
            summary = self._public_summary(r)
            command = json.dumps([r.tool, r.kwargs], sort_keys=True, default=str)
            command_id = hashlib.sha256(command.encode()).hexdigest()[:16]
            diagnostic = self._diagnostic(r, summary)
            if diagnostic:
                category, detail = diagnostic
                normalized = re.sub(r"[-+]?\d+(?:\.\d+)?", "#", detail)
                key = command_id + ":" + category + ":" + normalized
                previous = self.failures.pop(key, None)
                if previous and previous["resolved"]:
                    previous = None
                row = {
                    "command_id": command_id,
                    "tool": r.tool,
                    "category": category,
                    "first_call": previous["first_call"] if previous else r.index,
                    "last_call": r.index,
                    "count": previous["count"] + 1 if previous else 1,
                    "first_diagnostic": previous["first_diagnostic"] if previous else detail,
                    "arguments": command[:500],
                    "resolved": False,
                }
                self.failures[key] = row
                recent.append(dict(row))
            elif summary.get("blocked") is True or (
                summary.get("holding") is False and r.kwargs.get("gripper") == 1
            ):
                recent.append(
                    {
                        "call": r.index,
                        "category": "contact_or_grasp_uncertain",
                        "advice": "Interpret against intent and visible outcome, not as failure.",
                    }
                )
            elif (
                summary.get("reached") is True
                or summary.get("found") is True
                or (summary.get("holding") is True and r.kwargs.get("gripper") == 1)
            ):
                # An unrelated successful action cannot clear an earlier failure.
                for row in self.failures.values():
                    if row["command_id"] == command_id:
                        row["resolved"] = True
            while len(self.failures) > 32:
                self.failures.popitem(last=False)
        active = [dict(r) for r in self.failures.values() if not r["resolved"]]
        return {
            "recent": recent[-4:],
            "unresolved": active[-4:],
            "replan_recommended": any(r["count"] >= 2 for r in active),
            "scope": "primitive feedback; physical outcome may remain unknown",
        }

    def after_tool(self, name, result, records):
        if self.kind == "recap":
            context = self.plan_context()
            if not self.frames:
                context["reminder"] = "Initialize recap_plan in your next Python cell."
        elif self.kind == "failure_memory":
            context = self.observe(records)
        else:
            # ACE is prompt-only at evaluation time; never consume test rewards.
            context = {"frozen": True, "bullet_count": len(self.book["bullets"])}
        event = {"turn": len(self.kit.turns), "tool": name, "kind": self.kind, "context": context}
        with (self.directory / "events.jsonl").open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        if self.kind == "ace":
            return result
        # Put bounded feedback first so a verbose stdout cannot hide it.
        from .arms import MAX_TEXT_CHARS

        result.result = {"general_harness": context, **result.result}
        prefix = json.dumps({"general_harness": context}, ensure_ascii=False, default=str)
        body = json.dumps(
            {k: v for k, v in result.result.items() if k != "general_harness"},
            ensure_ascii=False,
            default=str,
        )
        remaining = MAX_TEXT_CHARS - len(prefix) - 100
        if len(body) > remaining:
            half = remaining // 2
            body = body[:half] + "\n[body truncated; head and tail retained]\n" + body[-half:]
            result.rendered_text = prefix + "\n" + body
        return result
