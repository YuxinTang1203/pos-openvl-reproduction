"""LongHarness evidence/conclusion lifecycle adapted to Python robot cells.

The maintainer proposes edits; registered sources and lifecycle constraints are
checked here. Source fidelity and physical truth remain model judgements. Only
the external evaluator decides task success. No simulator state is read here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import OrderedDict

PAPER = "https://arxiv.org/html/2610.13075v1"
DEFAULTS = {
    "max_sources": 48, "max_evidence": 24, "max_claims": 12,
    "max_images": 2, "update_every": 1, "conclusion_repair_attempts": 1,
}
LIMITS = {
    "max_sources": (4, 128), "max_evidence": (2, 64), "max_claims": (1, 24),
    "max_images": (0, 4), "update_every": (1, 8), "conclusion_repair_attempts": (0, 1),
}
MAX_IMAGE_CANDIDATES = 8
VERDICT_KEYS = {
    "success", "score", "reward", "terminated", "truncated", "done",
    "eval_success", "environment_success", "task_success", "episode_done",
}

EVIDENCE_PROMPT = """Maintain a source-linked EvidenceBank for a robot task.
Return JSON {"operations": [...]} and no conclusion update in this phase.
Allowed operations:
{"op":"upsert","id":"e1","statement":"short observed fact",
 "sources":["s0001"],"scope":"world","slot":"object.location"}
{"op":"revise","id":"e1","replacement":{same entry fields, new id}}
{"op":"revise","id":"e1","replacement":null}
IDs are immutable. Cite only registered source IDs in the payload, never invented
IDs or memory alone. scope=world is current physical state and needs current
sources; scope=history only describes a past event; scope=instruction is a task
constraint. A stable slot names one proposition/attribute; revise conflicting
evidence in that slot instead of admitting both. Do not infer grasp/task success
from a completed motion, printed intention, or absence of an exception. Images
can support uncertain visual interpretations, not ground truth. Old sources do
not become fresh when cited again. Use at most 8 operations; [] is valid.
Keep records relevant to the task, retaining useful constraints and contradictions.
Only images marked attached_this_request have pixels in this request. Registration
and a previous inspected flag do not mean the pixels were supplied again. The
image_delivery manifest lists selected and omitted IDs; missing views are review
limitations, not observed world facts. Keep such questions in open conclusions.
"""

CONCLUSION_PROMPT = """Update conclusions AFTER observing the evidence receipts.
Return JSON with any of: established, hypotheses, open. Omitted lists persist;
an empty list clears that field. established/hypotheses items:
{"id":"c1","text":"short conclusion","evidence":["e1"],"scope":"world"}
scope is world, history, or instruction. Cite accepted ACTIVE evidence only.
For BOTH established AND hypotheses, scope=world requires EVERY cited evidence
entry to have scope=world. An instruction describes the requested task, not the
existence, pose, reachability, or success of a physical object. Do not cite
instruction/history entries for world hypotheses, even if the claim is tentative.
The payload lists allowed_evidence_by_scope. If its world list is empty, include
NO world-scoped established claims or hypotheses; hypotheses=[] is valid. Record
unsupported possibilities as questions in open instead. For example, an
instruction 'move object to target' may support an instruction-scope task goal;
'the object is reachable' must remain an open question until observed.
Established still means an agent inference, not evaluator truth. Preserve
unresolved task requirements. open items:
{"id":"g1","question":"what is still unknown or incomplete?",
 "next_observation":"specific discriminating observation/action to obtain"}
Prioritize gaps that affect the next decision. Correct claims with stale,
superseded, retracted, or contradicted evidence. Do not call tools or move the
robot here; the executor chooses the next Python cell. Check physical results
against current observations; movement completion does not prove task completion.
Only request public observations available through the documented robot API;
never propose inspecting hidden simulator state, rewards, or grading conditions.
"""


def validate_options(config):
    if not isinstance(config, dict) or config.get("kind") != "longharness":
        raise ValueError("expected kind=longharness")
    if set(config) - {"kind", *DEFAULTS}:
        raise ValueError("unknown LongHarness option")
    values = {**DEFAULTS, **{k: v for k, v in config.items() if k != "kind"}}
    for key, (low, high) in LIMITS.items():
        if type(values[key]) is not int or not low <= values[key] <= high:
            raise ValueError(f"LongHarness {key} must be an integer in {low}..{high}")
    return values


def _text(value, limit=500):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"expected nonempty text of at most {limit} characters")
    return value.strip()


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}", value):
        raise ValueError("invalid identifier")
    return value


def _public(value):
    if isinstance(value, dict):
        return {str(k): _public(v) for k, v in value.items() if k not in VERDICT_KEYS}
    if isinstance(value, (list, tuple)):
        return [_public(v) for v in value[:40]]
    return value


def _excerpt(value, limit):
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text[:limit]


class LongHarness:
    kind = "longharness"

    def __init__(self, kit, config, task, initial_images=()):
        if kit.arm != "cells":
            raise ValueError("LongHarness requires arm=cells")
        self.kit, self.task = kit, task
        self.options = validate_options(config)
        self.directory = kit.workspace.parent / "general_harness"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.sources = OrderedDict()
        self.evidence = OrderedDict()
        self.conclusions = {"established": [], "hypotheses": [], "open": []}
        self.images = OrderedDict()
        self._last_action_arms = set()
        self.epoch = self.revision = self.cell_count = self.source_counter = 0
        self._used_evidence_ids = set()
        self._dirty = True
        self._last_update_cell = -self.options["update_every"]
        self.receipts = []
        self.warning = None
        self.provenance = {
            "kind": self.kind, "paper": PAPER,
            "implementation": "pyrualean-longharness-v1",
            "adaptation": "source-linked robot evidence and model-driven conclusion lifecycle",
            "benchmark_reproduction": False, "history_trimming": False,
            "model_maintenance": "two phases plus bounded conclusion repair; shared runtime budget",
            "max_maintenance_requests": 2 + self.options["conclusion_repair_attempts"],
            "motion_invalidates_world_evidence": True, "automatic_retry": False,
            "official_success_access": False, "options": self.options,
        }
        self._source("instruction", str(task), current=False)
        pairs = initial_images.items() if isinstance(initial_images, dict) else initial_images
        for name, data in pairs:
            if isinstance(data, (bytes, bytearray)):
                self._image_source(str(name), bytes(data), current=True)
        self._event("initialized")

    def prompt(self):
        return """
# LongHarness evidence and conclusion state
Before decisions, a bounded maintainer supplies source-linked evidence,
conclusions, and unresolved gaps. Use the gaps to choose observations and actions
that can establish or refute the next condition. Inactive evidence and unsupported
claims describe history, not the current world. The records are model inferences;
they never certify physical success. Execute ordinary Python cells with multiple
actions as appropriate, checking returned failures before dependent actions.
"""

    def _event(self, event, **fields):
        data = {"kind": self.kind, "event": event, "cell": self.cell_count,
                "epoch": self.epoch, "revision": self.revision, **fields}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")

    def _source(self, kind, text, *, current, **extra):
        self.source_counter += 1
        sid = f"s{self.source_counter:04d}"
        source = {"id": sid, "kind": kind, "epoch": self.epoch,
                  "current_eligible": current, "text": str(text)[:1600], **extra}
        self.sources[sid] = source
        with (self.directory / "sources.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(source, ensure_ascii=False, default=str) + "\n")
        while len(self.sources) > self.options["max_sources"]:
            old_id = next(k for k, s in self.sources.items() if s["kind"] != "instruction")
            self.sources.pop(old_id)
            self.images.pop(old_id, None)
        self._mark_stale()
        return sid

    def _image_source(self, name, data, *, current):
        digest = hashlib.sha256(data).hexdigest()
        sid = self._source("image", name, current=current, sha256=digest, inspected=False)
        # Keep one candidate per view, rather than letting the last two initial
        # wrists evict the head before the first maintenance decision.
        for old_id in list(self.images):
            if self.sources.get(old_id, {}).get("text") == name:
                self.images.pop(old_id)
        self.images[sid] = (f"{sid}:{name}", data)
        while len(self.images) > min(MAX_IMAGE_CANDIDATES, self.options["max_sources"]):
            self.images.popitem(last=False)
        return sid

    def _select_image_ids(self):
        """Select existing views using public questions/actions; never take new images."""
        limit = self.options["max_images"]
        if not limit:
            return []
        question = " ".join(str(g.get(k, "")) for g in self.conclusions["open"]
                            for k in ("question", "next_observation")).casefold()
        candidates = [sid for sid in self.images if sid in self.sources]

        def view(sid):
            return self.sources[sid]["text"].casefold()

        def score(sid):
            source, label = self.sources[sid], view(sid)
            head = "head" in label or "front" in label
            wrist = "wrist" in label
            direct = int(label.removesuffix(".png") in question)
            if head and any(word in question for word in ("head", "overview", "whole scene")):
                direct += 2
            if wrist and any(word in question for word in ("wrist", "grasp", "gripper", "held")):
                direct += 1
            for arm in ("left", "right"):
                if arm in label and arm in question:
                    direct += 2
            active_arm = int(wrist and any(arm in label for arm in self._last_action_arms))
            fresh = int(source["current_eligible"] and source["epoch"] == self.epoch)
            return fresh, direct, active_arm, int(head), candidates.index(sid)

        ranked = sorted(candidates, key=score, reverse=True)
        if self.cell_count == 0:
            heads = [sid for sid in ranked if "head" in view(sid)]
            if heads:
                ranked.remove(heads[0])
                ranked.insert(0, heads[0])
        return ranked[:limit]

    def _mark_stale(self):
        for entry in self.evidence.values():
            if entry["status"] != "active":
                continue
            if any(s not in self.sources for s in entry["sources"]):
                entry.update(status="stale", inactive_reason="source evicted from active registry")
            elif entry["scope"] == "world" and any(
                self.sources[s]["epoch"] != self.epoch for s in entry["sources"]
            ):
                entry.update(status="stale",
                             inactive_reason="world changed after source observation")

    def after_tool(self, name, result, records):
        if name != "python":
            return result
        self.cell_count += 1
        action_arms = {
            str(record.kwargs.get("arm")) for record in records
            if getattr(record, "stateful", False) and record.kwargs.get("arm") in {"left", "right"}
        }
        if action_arms:
            self._last_action_arms = action_arms
        for record in records:
            if getattr(record, "stateful", False):
                self.epoch += 1
                self._mark_stale()
            if record.tool == "artifact":
                continue
            summary = _public(record.summary)
            # Keep failures ahead of large arguments or stdout. Source text has a
            # fixed limit, so a long command must not erase its actual outcome.
            diagnostics = {
                k: _excerpt(v, 180) if isinstance(v, (dict, list, str)) else v
                for k, v in summary.items()
                if k in {"ik_error", "reached", "found", "blocked", "holding"}
            }
            content = {"tool": str(record.tool)[:100],
                       "error": _excerpt(record.error, 400) if record.error else None,
                       "diagnostics": diagnostics, "summary": _excerpt(summary, 600),
                       "arguments": _excerpt(_public(record.kwargs), 300)}
            self._source("primitive", json.dumps(content, default=str), current=True,
                         call=getattr(record, "index", None))
        public = {
            k: _excerpt(result.result[k], limit)
            for k, limit in (("error", 400), ("exception", 600), ("stderr", 400),
                             ("stdout", 800), ("value", 400))
            if result.result.get(k)
        }
        if public:
            # stdout may include an earlier observation or a printed intention;
            # chronology cannot be recovered reliably from arbitrary Python text.
            self._source("cell_text", json.dumps(_public(public), default=str), current=False)
        names = result.result.get("images") or []
        for i, data in enumerate(result.images):
            label = str(names[i]) if i < len(names) else f"unbound_image_{i}"
            # Camera artifacts are fetched at the end of the cell by ArmToolkit.
            # display() images can be old arrays; never refresh them implicitly.
            current = i < len(names) and not label.startswith(("display_", "unbound_"))
            self._image_source(label, data, current=current)
        self._dirty = True
        self._event("observed", tool=name, context=self.context())
        result.result["longharness"] = {
            "epoch": self.epoch, "maintenance_pending": True,
            "unsupported_claims": self.context()["unsupported_claims"],
        }
        return result

    def _entry(self, value):
        if not isinstance(value, dict):
            raise ValueError("evidence entry must be an object")
        if set(value) - {"id", "statement", "sources", "scope", "slot"}:
            raise ValueError("unknown evidence field")
        eid = _identifier(value.get("id"))
        if eid in self._used_evidence_ids:
            raise ValueError("evidence IDs are immutable; revise using a new ID")
        statement = _text(value.get("statement"))
        slot = _text(value.get("slot"), 100)
        scope = value.get("scope", "world")
        if scope not in {"world", "history", "instruction"}:
            raise ValueError("invalid evidence scope")
        refs = value.get("sources")
        if not isinstance(refs, list) or not 1 <= len(refs) <= 8:
            raise ValueError("evidence needs 1..8 registered sources")
        if any(not isinstance(s, str) or s not in self.sources for s in refs):
            raise ValueError("unregistered source")
        if any(self.sources[s]["kind"] == "image" and not self.sources[s].get("inspected")
               for s in refs):
            raise ValueError("image was not supplied to a successful maintainer request")
        if scope == "world" and any(
            not self.sources[s]["current_eligible"] or self.sources[s]["epoch"] != self.epoch
            for s in refs
        ):
            raise ValueError("world evidence needs current observations, "
                             "not old sources or printed claims")
        if scope == "instruction" and any(self.sources[s]["kind"] != "instruction" for s in refs):
            raise ValueError("instruction evidence must cite the task instruction")
        return {"id": eid, "statement": statement, "sources": list(dict.fromkeys(refs)),
                "scope": scope, "slot": slot, "status": "active", "epoch": self.epoch,
                "authority": "agent interpretation of registered observations"}

    def apply_operations(self, operations):
        """Independently validate operations, returning receipts before conclusion revision."""
        if not isinstance(operations, list) or len(operations) > 8:
            raise ValueError("operations must be a list of at most 8 entries")
        receipts = []
        for operation in operations:
            try:
                if not isinstance(operation, dict):
                    raise ValueError("operation must be an object")
                op = operation.get("op")
                old = None
                if op == "upsert":
                    entry = self._entry({k: v for k, v in operation.items() if k != "op"})
                elif op == "revise":
                    if set(operation) != {"op", "id", "replacement"}:
                        raise ValueError("revise needs id and replacement (null to retract)")
                    old = self.evidence.get(_identifier(operation.get("id")))
                    if not old or old["status"] not in {"active", "stale"}:
                        raise ValueError("revise target must be active or stale")
                    entry = (None if operation["replacement"] is None
                             else self._entry(operation["replacement"]))
                else:
                    raise ValueError("unknown evidence operation")
                if entry and any(
                    other["status"] == "active" and other["slot"] == entry["slot"]
                    and other["scope"] == entry["scope"] and other is not old
                    for other in self.evidence.values()
                ):
                    raise ValueError("slot already has active evidence; explicitly revise it")
                if old:
                    old.update(status="superseded" if entry else "retracted")
                    if entry:
                        old["replacement"] = entry["id"]
                if entry:
                    self.evidence[entry["id"]] = entry
                    self._used_evidence_ids.add(entry["id"])
                while len(self.evidence) > self.options["max_evidence"]:
                    inactive = next(
                        (k for k, e in self.evidence.items() if e["status"] != "active"),
                        next(iter(self.evidence)))
                    self.evidence.pop(inactive)
                self.revision += 1
                receipt = {"accepted": True, "id": entry["id"] if entry else old["id"]}
            except (ValueError, TypeError, KeyError) as exc:
                receipt = {"accepted": False, "reason": str(exc)}
            receipts.append(receipt)
            self._event("evidence_update", operation=operation, receipt=receipt)
        self.receipts = receipts
        return receipts

    def update_conclusions(self, patch):
        if not isinstance(patch, dict) or set(patch) - set(self.conclusions):
            raise ValueError("conclusion patch permits established, hypotheses, open only")
        state = copy.deepcopy(self.conclusions)
        for key, items in patch.items():
            if not isinstance(items, list) or len(items) > self.options["max_claims"]:
                raise ValueError("conclusion list exceeds budget")
            parsed, ids = [], set()
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("conclusion entry must be an object")
                cid = _identifier(item.get("id"))
                if cid in ids:
                    raise ValueError("duplicate conclusion identifier")
                ids.add(cid)
                if key == "open":
                    if set(item) != {"id", "question", "next_observation"}:
                        raise ValueError("gap needs id, question and next_observation")
                    parsed.append({"id": cid, "question": _text(item["question"]),
                                   "next_observation": _text(item["next_observation"])})
                    continue
                if set(item) - {"id", "text", "evidence", "scope"}:
                    raise ValueError("unknown conclusion field")
                scope = item.get("scope", "world")
                if scope not in {"world", "history", "instruction"}:
                    raise ValueError("invalid conclusion scope")
                refs = item.get("evidence")
                if not isinstance(refs, list) or not 1 <= len(refs) <= 8:
                    raise ValueError("claim needs 1..8 active evidence references")
                for ref in refs:
                    if not isinstance(ref, str) or ref not in self.evidence:
                        raise ValueError("unknown evidence reference")
                    e = self.evidence[ref]
                    if e["status"] != "active":
                        raise ValueError("inactive evidence reference")
                    if scope != "history" and e["scope"] != scope:
                        raise ValueError(
                            f"{key} {cid}: historical/instruction evidence cannot establish "
                            f"current world state; {ref} has scope={e['scope']}, "
                            f"claim scope={scope}")
                parsed.append({"id": cid, "text": _text(item.get("text")),
                               "evidence": list(dict.fromkeys(refs)), "scope": scope})
            state[key] = parsed
        self.conclusions = state
        self.revision += 1
        self._event("conclusions_updated", context=self.context())

    def context(self):
        unsupported = []
        active_conclusions = copy.deepcopy(self.conclusions)
        for group in ("established", "hypotheses"):
            active_conclusions[group] = []
            for claim in self.conclusions[group]:
                invalid = [r for r in claim["evidence"]
                           if r not in self.evidence or self.evidence[r]["status"] != "active"]
                if invalid:
                    unsupported.append({**copy.deepcopy(claim), "group": group,
                                        "references": invalid, "support_valid": False,
                                        "support_status": "inactive_or_missing_evidence"})
                else:
                    active_conclusions[group].append(copy.deepcopy(claim))
        return {"kind": self.kind, "epoch": self.epoch, "revision": self.revision,
                "evidence": copy.deepcopy(list(self.evidence.values())),
                "conclusions": active_conclusions,
                "unsupported_claims": unsupported, "validation_receipts": self.receipts,
                "warning": self.warning, "verification": "agent inferences, not official verdicts"}

    def _response_payload(self, reply, *, phase):
        if not isinstance(reply, dict):
            raise ValueError("maintenance response must be an object")
        metadata = {k: reply[k] for k in ("reason", "explanation") if k in reply}
        if any(not isinstance(v, str) or len(v) > 2000 for v in metadata.values()):
            raise ValueError("reason/explanation metadata must be text of at most 2000 characters")
        if metadata:
            self._event("ignored_metadata", phase=phase, metadata=metadata)
        return {k: v for k, v in reply.items() if k not in metadata}

    def _conclusion_input(self):
        active = [e for e in self.evidence.values() if e["status"] == "active"]
        return {"task": self.task, "state": self.context(),
                "max_claims_per_list": self.options["max_claims"],
                "allowed_evidence_by_scope": {
                    scope: [e["id"] for e in active if scope == "history" or e["scope"] == scope]
                    for scope in ("world", "instruction", "history")
                }}

    def before_request(self, model_call):
        due = self.cell_count - self._last_update_cell >= self.options["update_every"]
        if self._dirty and due:
            self._dirty = False
            self._last_update_cell = self.cell_count
            self.warning = None
            try:
                selected = self._select_image_ids()
                sources = [{**source, "attached_this_request": sid in selected}
                           for sid, source in self.sources.items()]
                delivery = {
                    "selected_image_ids": selected,
                    "omitted_image_ids": [sid for sid, source in self.sources.items()
                                          if source["kind"] == "image" and sid not in selected],
                    "selection": (
                        "current evidence, open-question view, latest action arm; initial head first"),
                    "status": "selected for callback; runtime records actual dispatch",
                    "verification": "previous inspection is not current delivery or physical truth",
                }
                payload = {"task": self.task, "state": self.context(),
                           "registered_sources": sources, "image_delivery": delivery,
                           "max_claims_per_list": self.options["max_claims"]}
                self._event("image_selection", **delivery)
                reply = model_call("longharness_evidence", EVIDENCE_PROMPT,
                                   payload, [self.images[sid] for sid in selected])
                self._event("image_delivery", selected_image_ids=selected,
                            actually_sent_image_ids=selected if isinstance(reply, dict) else None,
                            response_received=isinstance(reply, dict))
                if reply is None:
                    raise ValueError("evidence maintenance unavailable or "
                                     "shared request budget exhausted")
                if not isinstance(reply, dict):
                    raise ValueError("maintenance response must be an object")
                for sid in selected:
                    if sid in self.sources:
                        self.sources[sid]["inspected"] = True
                reply = self._response_payload(reply, phase="evidence")
                if set(reply) != {"operations"}:
                    raise ValueError("evidence response must contain operations only")
                self.apply_operations(reply["operations"])
                reply = model_call("longharness_conclusions", CONCLUSION_PROMPT,
                                   self._conclusion_input(), [])
                if reply is None:
                    raise ValueError("conclusion maintenance unavailable or "
                                     "shared request budget exhausted")
                try:
                    self.update_conclusions(self._response_payload(reply, phase="conclusions"))
                except (ValueError, TypeError, KeyError) as exc:
                    # A rejected patch leaves the previous state intact. Show the
                    # precise failure once; never silently coerce physical claims.
                    error = str(exc)[:1000]
                    self._event("conclusion_rejected", reason=error, proposal=reply)
                    if not self.options["conclusion_repair_attempts"]:
                        raise
                    reply = model_call(
                        "longharness_conclusion_repair", CONCLUSION_PROMPT,
                        {**self._conclusion_input(), "rejected_proposal": reply,
                         "validation_error": error,
                         "repair_instruction": "Return a corrected complete patch. Preserve useful "
                         "supported claims and open questions. Remove unsupported world hypotheses "
                         "or express them as questions in open; do not relabel evidence scopes."},
                        [])
                    if reply is None:
                        raise ValueError("conclusion repair unavailable or shared request "
                                         "budget exhausted") from None
                    self.update_conclusions(
                        self._response_payload(reply, phase="conclusion_repair"))
                self._event("maintenance_complete", context=self.context())
            except Exception as exc:  # maintenance never hides original robot observations
                self.warning = f"{type(exc).__name__}: {exc}"[:500]
                self._event("maintenance_failed", reason=self.warning, context=self.context())
        return ("LongHarness current evidence and conclusion state:\n"
                + json.dumps(self.context(), ensure_ascii=False))
