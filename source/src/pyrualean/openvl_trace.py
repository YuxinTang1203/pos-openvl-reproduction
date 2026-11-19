"""Append-only robot observation links in native OpenVL memory, never control flow."""

import copy
import time
from dataclasses import asdict, is_dataclass

from .arms import _to_png
from .openai_runtime import image_part


def plain(value):
    return copy.deepcopy(asdict(value) if is_dataclass(value) else value)


class RobotTrace:
    def __init__(self, adapter):
        self.adapter = adapter
        self.kit = adapter.kit
        self.rows = []
        self.errors = []
        self.context = None
        self.seen_calls = set()
        self.seen_perception = set()
        self.derived_images = {}

    def store(self, kind, *, links=(), **data):
        row = dict(
            kind=kind,
            timestamp=time.time(),
            cell=len(self.kit.turns),
            links=[ref for ref in links if ref is not None],
            **plain(data),
        )
        ref = self.adapter.memory.store_binding(
            prefix="trace",
            query=kind,
            value=row,
            source_tool="pyrualean.robot_trace",
            source_args={"kind": kind, "cell": row["cell"]},
            descriptions={
                "kind": "Historical observation type; never a current control target",
                "links": "Explicit evidence relations; temporal association is not causality",
            },
        )
        self.rows.append((ref, row))
        return ref

    def begin(self, name, args):
        self.context = None
        self.start = len(self.rows)
        self.ledger_start = len(self.kit.robo._ledger)
        self.context = self.store(
            "cell_start", tool=name, arguments=args or {}, step=self.kit.robo._backend.env_steps()
        )

    def state(self, value, *, relation, step, call_index):
        return self.store(
            "state",
            links=[self.context],
            value=plain(value),
            relation=relation,
            step=step,
            call_index=call_index,
            availability="available" if value is not None else "unavailable",
        )

    def action(self, record, result, state_before, state_observed, step_before, step_after):
        before = self.state(
            state_before,
            relation="cached_before_action_not_refreshed",
            step=step_before,
            call_index=record.index,
        )
        after = self.state(
            self.kit.robo._last_state if state_observed else None,
            relation="returned_action_observation" if state_observed else "unavailable",
            step=step_after,
            call_index=record.index,
        )
        # Old perception bindings are explicitly contextual, never asserted targets.
        candidates = list(self.adapter.bridge.refs[-3:])
        ref = self.store(
            "action",
            links=[self.context, before, after],
            tool=record.tool,
            arguments=record.kwargs,
            result=plain(result),
            step_before=step_before,
            step=step_after,
            elapsed_s=record.elapsed_s,
            call_index=record.index,
            ok=record.ok,
            error=record.error,
            state_before=before,
            state_after=after,
            historical_context=candidates,
            association="temporal_only_target_unknown",
        )
        self.seen_calls.add(record.index)
        if record.error:
            self.store(
                "exception",
                links=[self.context, ref],
                scope="primitive",
                tool=record.tool,
                call_index=record.index,
                step=step_after,
                error=record.error,
            )
        # Capture before the next primitive, including a failure caught by user code.
        # Only public image access; no SAM or state query is introduced.
        for camera in self.kit.robo.CAMERAS:
            if self.kit.images == "none":
                break
            try:
                start = self.kit.robo._backend.env_steps()
                data = _to_png(self.kit.robo.image(camera))
                end = self.kit.robo._backend.env_steps()
                image = self.adapter.memory.register_image(
                    image_part(data)["image_url"]["url"],
                    provenance=f"action:{record.index}:{camera}",
                )
                self.store(
                    "visual",
                    links=[ref, after, image],
                    image_ref=image,
                    camera=camera,
                    call_index=record.index,
                    step=end,
                    capture_step_before=start,
                    relation="after_action_same_step"
                    if start == end == step_after and end is not None
                    else "after_action_time_uncertain",
                )
            except Exception as exc:
                self.store(
                    "visual",
                    links=[ref],
                    camera=camera,
                    step=step_after,
                    availability="unavailable",
                    error=f"{type(exc).__name__}: {exc}",
                )

    def finish(self, result, image_refs):
        # Include readonly calls and errors that happened before the action hook.
        for record in self.kit.robo._ledger.records[self.ledger_start :]:
            if record.index in self.seen_calls:
                continue
            ref = self.store(
                "call",
                links=[self.context],
                tool=record.tool,
                call_index=record.index,
                record=record.to_dict(),
                relation="cell_record_no_per_primitive_image",
            )
            self.seen_calls.add(record.index)
            if record.error:
                self.store(
                    "exception",
                    links=[self.context, ref],
                    scope="primitive",
                    tool=record.tool,
                    call_index=record.index,
                    error=record.error,
                )
        for ref in self.adapter.bridge.refs:
            if ref not in self.seen_perception:
                value = self.adapter.bridge.raw(ref)
                source = value.get("source", {})
                self.store(
                    "perception",
                    links=[self.context, ref],
                    binding=ref,
                    image_ref=source.get("image_ref"),
                    source=source,
                    step=source.get("step"),
                    relation="perception_source",
                )
                self.seen_perception.add(ref)
        self.state(
            self.kit.robo._last_state,
            relation="cached_at_cell_end_not_refreshed",
            step=self.kit.robo._backend.env_steps(),
            call_index=None,
        )
        children = [ref for ref, _ in self.rows[self.start :]]
        if result.result.get("exception") or result.result.get("error"):
            children.append(
                self.store(
                    "exception",
                    links=[self.context, *children[1:]],
                    scope="cell",
                    error=result.result.get("exception") or result.result.get("error"),
                )
            )
        end = self.store(
            "cell_end",
            links=[
                *children,
                *image_refs,
                *(item["id"] for item in result.result.get("artifact_refs", [])),
            ],
            step=self.kit.robo._backend.env_steps(),
            output={
                k: result.result[k]
                for k in ("stdout", "stderr", "exception", "error")
                if k in result.result
            },
            delivered_images=image_refs,
            image_relation="delivered_at_cell_end_source_may_be_historical",
        )
        return {
            "cell": len(self.kit.turns),
            "start": self.context,
            "end": end,
            "records": len(self.rows) - self.start,
            "audit_errors": list(self.errors),
        }

    def find(
        self,
        *,
        kind=None,
        tool=None,
        cell=None,
        step_from=None,
        step_to=None,
        time_from=None,
        time_to=None,
        limit=10,
    ):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be 1..100")
        out = []
        for ref, row in reversed(self.rows):
            step = row.get("step")
            if time_from is not None and row["timestamp"] < time_from:
                continue
            if time_to is not None and row["timestamp"] > time_to:
                continue
            if kind is not None and row["kind"] != kind:
                continue
            if tool is not None and row.get("tool") != tool:
                continue
            if cell is not None and row["cell"] != cell:
                continue
            if step_from is not None and (step is None or step < step_from):
                continue
            if step_to is not None and (step is None or step > step_to):
                continue
            out.append(
                dict(
                    reference=ref,
                    **{
                        k: row[k]
                        for k in (
                            "kind",
                            "timestamp",
                            "cell",
                            "tool",
                            "step",
                            "call_index",
                            "links",
                        )
                        if k in row
                    },
                )
            )
            if len(out) == limit:
                break
        return copy.deepcopy(out)

    def read(self, reference):
        for ref, row in self.rows:
            if ref == reference:
                value = copy.deepcopy(row)
                value['temporal_context'] = {
                    'source_cell': row['cell'],
                    'source_step': row.get('step'),
                    'source_call': row.get('call_index'),
                    'phase': row.get('relation', row['kind']),
                    'assessment_step': self.kit.robo._backend.env_steps(),
                    'current_target': False,
                    'guidance': 'Historical record, including coordinates in arguments/output. '
                                'Check evidence_current(ref) or re-localize before targeting; '
                                'this record does not verify later actions.',
                }
                return value
        # One reader also accepts perception/native bindings and image IDs.
        if reference in self.adapter.bridge.refs:
            return {
                "kind": "perception",
                "historical": True,
                "value": self.adapter.bridge.read(reference, allow_historical=True),
            }
        if reference in self.adapter.memory.state.images.registry:
            return {
                "kind": "image",
                "image_ref": reference,
                "relation": "stored_image_not_current_geometry",
                "temporal_context": self.image_context(reference),
            }
        return self.adapter.read_data(reference)

    def image_context(self, reference):
        """Bounded capture summaries and parent IDs, never nested context copies.

        The full associations remain in trace/perception records. In particular,
        cropping repeatedly must not copy all ancestor metadata into each reply.
        Images deduplicated by bytes may have several capture phases; preserve
        those phases rather than treating delivery time as a new observation.
        """
        reference = self._image_reference(reference)
        parents = self._image_parents(reference)
        visited, roots, stack = set(), set(), [(reference, frozenset())]
        cycle = False
        complete = True
        while stack:
            ref, ancestors = stack.pop()
            if ref in ancestors:
                cycle = True
                continue
            if ref in visited:
                continue
            if len(visited) >= 128:
                complete = False
                break
            visited.add(ref)
            upstream = self._image_parents(ref)
            if upstream:
                stack.extend((parent, ancestors | {ref}) for parent in upstream)
            else:
                roots.add(ref)

        groups = {}

        def add(ref, source, *, validity=None):
            key = (ref, source.get('camera'), source.get('source_step'), source.get('phase'))
            group = groups.setdefault(key, dict(
                image_ref=ref, camera=source.get('camera'),
                source_step=source.get('source_step'), phase=source.get('phase'),
                associations=0, calls=[], cells=[], validity_statuses=set()))
            group['associations'] += 1
            if source.get('source_call') is not None:
                group['calls'].append(source['source_call'])
            if source.get('source_cell') is not None:
                group['cells'].append(source['source_cell'])
            if validity is not None:
                group['validity_statuses'].add(str(validity.get('status', 'unknown')))

        for _, row in self.rows:
            ref = row.get('image_ref')
            if ref not in visited:
                continue
            # The bridge has the original source fields and current validity;
            # do not repeat its perception again as a trace wrapper.
            if row.get('kind') == 'perception' and row.get('binding') in self.adapter.bridge.values:
                continue
            source = row.get('source') or {}
            add(ref, dict(camera=row.get('camera', source.get('camera')),
                          source_step=row.get('step', source.get('step')),
                          source_call=row.get('call_index', source.get('call_index')),
                          source_cell=row.get('cell'), phase=row.get('relation')))
            roots.add(ref)  # This image also has a direct recorded capture.
        for binding in self.adapter.bridge.refs:
            source = self.adapter.bridge.values[binding]['source']
            ref = source.get('image_ref')
            if ref in visited:
                add(ref, dict(camera=source.get('camera'), source_step=source.get('step'),
                              source_call=source.get('call_index'), phase='perception_source'),
                    validity=self.adapter.bridge.validity(binding))
                roots.add(ref)
        sources = []
        for group in groups.values():
            calls, cells = group.pop('calls'), group.pop('cells')
            group.update(source_call_range=[min(calls), max(calls)] if calls else None,
                         source_cell_range=[min(cells), max(cells)] if cells else None,
                         validity_statuses=sorted(group['validity_statuses']))
            sources.append(group)
        omitted_sources = max(0, len(sources) - 8)
        if omitted_sources:
            sources = sources[:4] + sources[-4:]
        root_refs = sorted(roots)
        return dict(image_ref=reference, sources=sources,
                    derived_from=parents[:16], root_image_refs=root_refs[:16],
                    omitted_sources=omitted_sources,
                    omitted_parent_refs=max(0, len(parents)-16),
                    omitted_root_refs=max(0, len(root_refs)-16),
                    lineage_complete=complete and not cycle and len(parents) <= 16 and len(root_refs) <= 16,
                    cycle_detected=cycle,
                    assessment_step=self.kit.robo._backend.env_steps(), current_target=False,
                    status='stored_image_not_current_geometry',
                    constraint='Recorded capture phases only; omitted/unknown capture phases remain unverified. '
                               'Parent/root image IDs can be inspected separately. Zoom does not refresh '
                               'the source or establish later action outcomes.')

    def _image_reference(self, reference):
        """Resolve registered aliases locally, with no fallback or network call."""
        images = self.adapter.memory.state.images
        if reference in images.registry:
            return reference
        return images.alias_to_ref.get(str(reference).strip().lower(), reference)

    def _image_parents(self, reference):
        # Accept older in-memory snapshots while emitting only canonical IDs.
        parents = self.derived_images.get(reference, [])
        refs = [parent.get('image_ref') if isinstance(parent, dict) else parent for parent in parents]
        return list(dict.fromkeys(self._image_reference(ref) for ref in refs if isinstance(ref, str)))

    def record_derivation(self, derived_ref, source_refs):
        """Store lineage edges only; native memory continues to own the pixels."""
        derived_ref = self._image_reference(derived_ref)
        parents = self._image_parents(derived_ref)
        parents.extend(self._image_reference(ref) for ref in source_refs if isinstance(ref, str))
        self.derived_images[derived_ref] = list(dict.fromkeys(ref for ref in parents if ref != derived_ref))

    def related(self, reference):
        self.read(reference)  # Reject unknown IDs instead of returning an ambiguous empty result.
        return [
            dict(
                reference=ref,
                **{
                    key: copy.deepcopy(row[key])
                    for key in (
                        "kind",
                        "cell",
                        "image_ref",
                        "camera",
                        "step",
                        "relation",
                        "availability",
                    )
                    if key in row
                },
            )
            for ref, row in self.rows
            if reference in row["links"] or reference in row.get("historical_context", [])
        ]

    def context_for_action(self, reference, limit=8):
        """Read stored action evidence without refreshing it or inferring object effects."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('limit must be an integer in 1..100')
        rows = dict(self.rows)
        if not isinstance(reference, str) or reference not in rows or rows[reference]['kind'] != 'action':
            raise ValueError('Expected an action reference from trace_find(kind="action")')
        action = rows[reference]
        index = action['call_index']
        next_index = min((r['call_index'] for _, r in self.rows
                          if r['kind'] == 'action' and r['call_index'] > index), default=None)
        before = []
        after = []
        for ref in action.get('historical_context', []):
            if ref in self.adapter.bridge.refs:
                before.append(self.adapter.bridge.history(ref))
        for _, row in self.rows:
            if row['kind'] != 'perception':
                continue
            call = row.get('source', {}).get('call_index')
            if call is not None and call > index and (next_index is None or call < next_index):
                after.append(self.adapter.bridge.history(row['binding']))
        # Nearest stored image per camera before the action, with known ordering.
        prior = {}
        visuals = []
        errors = []
        for ref, row in self.rows:
            if row['kind'] == 'visual':
                call, step = row.get('call_index'), row.get('step')
                if (row.get('image_ref') and call is not None and call < index
                        and step is not None and action.get('step_before') is not None
                        and step <= action['step_before']):
                    prior[row.get('camera')] = {'reference': ref, **self.read(ref)}
                if reference in row['links']:
                    visuals.append({'reference': ref, **self.read(ref)})
            if row['kind'] == 'exception' and reference in row['links']:
                errors.append({'reference': ref, **self.read(ref)})
        groups = {'perception_before': before, 'perception_after': after,
                  'images_before': list(prior.values()), 'images_after': visuals,
                  'exceptions': errors}
        omitted = {key: max(0, len(values)-limit) for key, values in groups.items()}
        missing = [key for key in ('perception_before', 'perception_after', 'images_before', 'images_after')
                   if not groups[key] or (key.startswith('images') and not any(v.get('image_ref') for v in groups[key]))]
        states = {}
        for key in ('state_before', 'state_after'):
            ref = action.get(key)
            states[key] = {'reference': ref, **self.read(ref)} if ref in rows else None
            if states[key] is None or states[key].get('availability') != 'available':
                missing.append(key)
        return copy.deepcopy({
            'reference': reference, 'action': self.read(reference), 'states': states,
            **{key: values[-limit:] for key, values in groups.items()},
            'missing': missing, 'omitted': omitted,
            'scope': {
                'purpose': 'historical_action_review', 'current_target': False,
                'association': 'temporal_not_causal', 'object_identity': 'unverified',
                'task_effect': 'not_verified',
                'perception_before': 'Recently stored context; not guaranteed to depict the action target',
                'perception_after': 'Stored measurements after this action and before the next action',
                'images_before': 'Latest previously stored action capture per camera, possibly old',
                'image_pixels_returned': False,
                'constraint': 'Missing evidence stays missing. No sensing, model request, action, or success check is performed.'},
        })
