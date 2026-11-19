"""Passive public history for the visual workspace, with no sensing or policy.

Only values already returned to the robot adapter are copied. Native MemoryStore
owns the bindings; links express provenance or temporal context, never a chosen
action target, object identity, causal effect, or successful task outcome.
"""
import copy
from dataclasses import asdict, is_dataclass
import json
import time


def copied(value):
    if is_dataclass(value):
        return copied(asdict(value))
    if isinstance(value, dict):
        return {str(k): copied(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [copied(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return copy.deepcopy(value)


class PassiveVisualHistory:
    """Append-only, explicitly historical records; errors never affect execution."""

    def __init__(self, adapter):
        self.adapter, self.kit = adapter, adapter.kit
        self.rows, self.aliases = {}, {}
        self.errors, self.error_count = [], 0
        self.cell_ref = None
        self.cell = None
        self.ledger_start = len(self.kit.robo._ledger.records)
        self.seen_calls = set()
        self.evidence_refs = []
        self.previous_sink = None

    def attach(self):
        robo = self.kit.robo
        previous = robo._evidence_adapter
        if previous is not None and not isinstance(previous, PassiveVisualHistory):
            raise ValueError("Visual history cannot replace another robot evidence adapter")
        sink = robo._perception_sink
        owner = getattr(sink, "__self__", None)
        self.previous_sink = owner.previous_sink if isinstance(owner, PassiveVisualHistory) else sink
        robo._evidence_adapter = self
        robo._perception_sink = self.perception

    def trace_error(self, exc):
        # RobotBase calls this from its own audit-error handler. Never raise here.
        self.error_count += 1
        try:
            message = f"{type(exc).__name__}: {exc}"
        except Exception:
            message = "History recording failed (error text unavailable)"
        self.errors.append(message[:1000])
        del self.errors[:-20]

    def safely(self, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            self.trace_error(exc)
            return None

    def _store(self, kind, *, links=(), **data):
        row = dict(kind=kind, timestamp=time.time(), cell=self.cell,
                   cell_ref=self.cell_ref, historical=True, current_target=False,
                   association="provenance_or_temporal_context_not_target_or_causality",
                   links=list(dict.fromkeys(x for x in (self.cell_ref, *links) if x)),
                   **copied(data))
        ref = self.adapter.memory.store_binding(
            prefix="visual_history", query=kind, value=copied(row),
            source_tool="pyrualean.passive_visual_history",
            source_args={"kind": kind, "cell": row["cell"]},
            descriptions={"links": "Recorded provenance/temporal relations, not action targets",
                          "current_target": "Always false: reading history never refreshes geometry"},
        )
        self.rows[ref] = row
        # Retain the in-memory record even when its optional disk archive fails.
        self.safely((self.adapter.root / f"{ref}.json").write_text,
                    json.dumps(row, default=str) + "\n")
        return ref

    def store(self, *args, **kwargs):
        return self.safely(self._store, *args, **kwargs)

    def begin(self, name, arguments):
        self.cell_ref = None
        self.cell = len(self.kit.turns)
        self.ledger_start = len(self.kit.robo._ledger.records)
        self.cell_ref = self.store("cell_start", tool=name, arguments=arguments or {},
                                   ledger_start=self.ledger_start)

    def before_action(self, **_kwargs):
        pass  # Recording is exclusively the existing post-action audit hook.

    def observe(self, *_args):
        pass  # No failure classification, action checks, sensing, or recovery.

    def record_action(self, record, result, state_before, state_observed, step_before, step_after):
        self.safely(self._action, record, result, state_before, state_observed, step_before, step_after)

    def _action(self, record, result, state_before, state_observed, step_before, step_after):
        before = self.store("state", call_index=record.index, environment_steps=step_before,
                            relation="cached_before_action_not_refreshed",
                            availability="available" if state_before is not None else "unknown",
                            value=state_before)
        after_value = self.kit.robo._last_state if state_observed else None
        after = self.store("state", call_index=record.index, environment_steps=step_after,
                           relation="returned_action_observation" if state_observed else "unknown_no_returned_observation",
                           availability="available" if after_value is not None else "unknown",
                           value=after_value)
        contextual = self.evidence_refs[-8:]
        ref = self.store("action", links=[before, after, *contextual],
                         tool=record.tool, call_index=record.index, record=record.to_dict(),
                         result=result, state_before=before, state_after=after,
                         environment_steps_before=step_before, environment_steps=step_after,
                         historical_evidence_context=contextual,
                         target_association="unknown", result_use="not_inferred_from_temporal_proximity")
        self.seen_calls.add(record.index)
        if record.error:
            self.store("exception", links=[ref], tool=record.tool, call_index=record.index,
                       scope="primitive", error=record.error)

    def perception(self, tool, arguments, payload):
        if self.previous_sink:
            self.previous_sink(tool, arguments, payload)  # Preserve an existing sink's behavior.
        self.safely(self._perception, tool, arguments, payload)

    def _perception(self, tool, arguments, payload):
        record = self.kit.robo._ledger.records[-1]
        ref = self.store("perception", tool=tool, call_index=record.index,
                         arguments=arguments, result=payload,
                         source_image=None, source_image_relation="not_captured_by_passive_history")
        if ref:
            self.evidence_refs.append(ref)

    def frame(self, image_id, frame):
        ref = self.store("frame", links=[frame.get("parent_image_id")], image_id=image_id,
                         source=copied(frame), relation="existing_capture_no_new_image_read")
        if ref:
            self.aliases[image_id] = ref
            self.evidence_refs.append(ref)

    def grounding(self, binding_id, source_image_id):
        ref = self.store("grounding", links=[binding_id, source_image_id],
                         binding_id=binding_id, source_image_id=source_image_id,
                         relation="native_grounding_binding_on_recorded_source_image")
        if ref:
            self.aliases[binding_id] = ref
            self.evidence_refs.append(ref)

    def event(self, kind, values):
        if kind not in {"visual_data_read", "visual_candidate_read", "selected_mask_geometry",
                        "visual_plane_projection", "perception_choice_required", "image_recall"}:
            return
        links = [values.get("binding_id"), values.get("source_image_id")]
        links.extend(values.get("image_ids") or [])
        source = values.get("source") or {}
        links.append(source.get("image_id"))
        ref = self.store(kind, links=links, call_index=values.get("primitive_ledger_index"),
                         result=values, relation="explicit_helper_or_capability_evidence")
        if ref:
            self.evidence_refs.append(ref)

    def finish(self, result):
        self.safely(self._finish, result)
        self.cell_ref, self.cell = None, None
        self.safely(self.adapter.persist)

    def _finish(self, result):
        for record in self.kit.robo._ledger.records[self.ledger_start:]:
            if record.index in self.seen_calls:
                continue
            ref = self.store("call", tool=record.tool, call_index=record.index,
                             record=record.to_dict(), relation="existing_ledger_no_new_observation")
            self.seen_calls.add(record.index)
            if record.error:
                self.store("exception", links=[ref], tool=record.tool, call_index=record.index,
                           scope="primitive", error=record.error)
        error = result.result.get("exception") or result.result.get("error")
        if error:
            self.store("exception", scope="cell", error=error)
        self.store("cell_end", links=[c["image_id"] for c in result.result.get("visual_images", [])],
                   output={k: result.result[k] for k in ("stdout", "stderr", "exception", "error")
                           if k in result.result},
                   delivered_images=result.result.get("images", []),
                   relation="delivered_at_cell_end_source_may_be_historical")

    @staticmethod
    def _limit(limit):
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("limit must be an integer from 1 to 50")

    @staticmethod
    def _summary(ref, row):
        return dict(reference=ref, **{k: row[k] for k in
                    ("kind", "cell", "tool", "call_index", "timestamp", "links") if k in row})

    def find(self, *, kind=None, tool=None, cell=None, call_index=None, limit=10):
        self._limit(limit)
        out = []
        for ref, row in reversed(self.rows.items()):
            if any(value is not None and row.get(key) != value for key, value in
                   (("kind", kind), ("tool", tool), ("cell", cell), ("call_index", call_index))):
                continue
            out.append(self._summary(ref, row))
            if len(out) >= limit:
                break
        return copied(out)

    def context(self, reference, *, limit=10):
        self._limit(limit)
        if not isinstance(reference, str):
            raise TypeError("reference must be a returned history, frame or Grounding binding ID")
        ref = self.aliases.get(reference, reference)
        if ref not in self.rows:
            raise ValueError("Unknown visual history reference")
        row = self.rows[ref]
        related = []
        names = {ref, reference, *(k for k, v in self.aliases.items() if v == ref)}
        linked = {self.aliases.get(x, x) for x in row["links"]}
        for other, value in reversed(self.rows.items()):
            if other != ref and (other in linked or names.intersection(value["links"])):
                related.append(self._summary(other, value))
                if len(related) >= limit:
                    break
        observations = []
        for other, value in reversed(self.rows.items()):
            if row["cell"] is not None and other != ref and value["cell"] == row["cell"] and value["kind"] in {
                "frame", "grounding", "perception", "selected_mask_geometry", "visual_plane_projection"
            }:
                observations.append({**self._summary(other, value), "relation": "same_cell_temporal_only"})
                if len(observations) >= limit:
                    break
        return copied({"reference": ref, "record": row, "related": related,
                       "same_cell_observations": observations,
                       "historical": True, "current_target": False,
                       "interpretation": "Stored public evidence only; nested current/status fields describe their recorded instant, never current validity. Reading history does not refresh geometry or establish object identity, result use, causality or success."})

    def summary(self):
        return {"trace_records": len(self.rows), "trace_error_count": self.error_count,
                "trace_errors": list(self.errors)}
