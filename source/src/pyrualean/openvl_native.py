"""OpenVL native capabilities and shared memory inside the PyRUA executor.

No MainAgent, upstream orchestrator prompt or one-tool-per-turn policy is used.
The native config also configures the external router; no services are launched.
"""

import base64
import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from openvlharness.bootstrap import build_capabilities, load_agent_config, memory_factory
from openvlharness.contracts import VirtualCall

from .arms import ArmResult
from .openai_runtime import image_part
from .openvl_adapter import OpenVLEvidenceAdapter


class LocalZoomClient:
    """Run the upstream CPU crop backend without a router round trip."""

    def execute(self, tool_name, args, *, tool_call_id):
        from openvlharness.services.envelope import _build_success_payload
        from openvlharness.services.tools.zoom_in_image_by_bbox.tool import Zoom_In_Tool

        return _build_success_payload(tool_name, tool_call_id, Zoom_In_Tool().execute(**args))


class OpenVLNativeAdapter(OpenVLEvidenceAdapter):
    def __init__(
        self,
        toolkit,
        *,
        config,
        initial_images,
        model=None,
        local_zoom=True,
        local_coding=False,
        base_url=None,
        api_key=None,
        validate_geometry=True,
        max_failure_images=2,
        feedback_mode="full",
        expose_robot_tools=True,
        directory_mode="all",
    ):
        if feedback_mode not in {"full", "ondemand", "auto"}:
            raise ValueError("feedback_mode must be full, ondemand, or auto")
        self.feedback_mode = feedback_mode
        if type(expose_robot_tools) is not bool or directory_mode not in {"all", "evidence"}:
            raise ValueError("expose_robot_tools must be boolean; directory_mode must be all or evidence")
        self.expose_robot_tools = expose_robot_tools
        self.directory_mode = directory_mode
        self.automatic_feedback = feedback_mode != "ondemand"
        settings = load_agent_config(config)
        if not {"image", "tool_context", "url"} <= {
            item["name"] for item in settings["memory"]["modules"]
        }:
            raise ValueError("OpenVL migration requires image, tool_context and url memory modules")
        if local_coding and ("coding" not in settings["capabilities"]["tools"] or not all((base_url, api_key, model))):
            raise ValueError("local_coding requires a coding capability and planner gateway credentials")
        if not local_coding and "coding" in settings["capabilities"]["tools"]:
            from openvlharness.services.models import ModelAdapter

            profile = settings["services"]["models"][settings["agent"]["model_service"]]
            served_model = ModelAdapter(profile).model
            if model not in {served_model, profile["alias"]}:
                raise ValueError(
                    f"OpenVL coding backbone {served_model!r} must match PyRUA model {model!r}; "
                    "configure a matching native model profile or disable coding"
                )
        native = build_capabilities(settings)
        if set(native) & {
            "python",
            "run_program",
            "finish",
            "Robot_Perception_Tool",
            "Robot_Evidence_Tool",
        }:
            raise ValueError("OpenVL capability collides with a PyRUA tool")
        if local_zoom and "Zoom_In_Tool" in native:
            native["Zoom_In_Tool"].client = LocalZoomClient()
        super().__init__(
            toolkit,
            validate_geometry=validate_geometry,
            max_failure_images=max_failure_images,
            execution_recovery=False,
            stage_verification=False,
            temporal_feedback=False,
        )
        self.memory = memory_factory(settings["memory"])()
        self.memory.start([])
        self.bridge.bind(self.memory)
        self.native_names = set(native)
        self.tools.update(native)
        self.artifact_root = self.root / "native"
        self.artifact_root.mkdir(exist_ok=True)
        self.saved_images = {}
        self.saved_bindings = {}
        self.native_calls = 0
        self.remaining = None
        self.stopped = None
        if local_coding:
            from .openvl_coding import LocalCodingClient
            native["Python_Coding_Agent_Tool"].client = LocalCodingClient(self, base_url, api_key, model)
        from .openvl_trace import RobotTrace

        self.trace = RobotTrace(self)
        for name, data in initial_images.items():
            self.memory.register_image(
                image_part(data)["image_url"]["url"], aliases=(name,), provenance=f"initial:{name}"
            )
        self.initial_inventory = self.memory.build_environment_update()
        self._directory_visible_images = set(self.memory.state.images.registry)
        self._inventory_lines = set()
        for name, method in {
            "openvl_data": self.read_data,
            "openvl_bindings": self.memory.describe_bindings,
            "evidence_compare": self.compare_evidence,
            "trace_find": self.trace.find,
            "trace_read": self.trace.read,
            "trace_related": self.trace.related,
            "evidence_context": self.trace.context_for_action,
        }.items():
            toolkit._globals[name] = method
            setattr(toolkit.robo, name, method)
        self.provenance.update(
            mode="native_capabilities",
            feedback_mode=feedback_mode,
            expose_robot_tools=expose_robot_tools,
            directory_mode=directory_mode,
            agent_loop="pyrualean.openai_runtime",
            capabilities={
                name: {
                    "backend": cap.backend,
                    "sha256": hashlib.sha256(
                        Path(inspect.getfile(type(cap))).read_bytes()
                    ).hexdigest(),
                }
                for name, cap in native.items()
            },
            local_zoom=local_zoom,
            local_coding=local_coding,
            coding_model=model if local_coding else None,
            # Only a digest is persisted: expanded service config may contain keys.
            config_sha256=hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
            memory=copy.deepcopy(settings["memory"]),
            robot_trace={
                "enabled": True,
                "capture": "public_cameras_after_each_action",
                "association": "temporal_not_causal",
                "controls_execution": False,
            },
        )
        (self.root / "provenance.json").write_text(json.dumps(self.provenance, indent=2) + "\n")
        self.archive_native()

    def specs(self):
        specs = copy.deepcopy(super().specs())
        if not self.expose_robot_tools:
            specs = [s for s in specs if s['name'] not in {'Robot_Perception_Tool', 'Robot_Evidence_Tool'}]
        for spec in specs:
            if spec['name'] == 'Python_Coding_Agent_Tool' and self.provenance.get('local_coding'):
                spec['description'] += (' Delegate a multi-record analysis question with evidence_refs: '
                    'the adapter retrieves only those structured records and returns computation, citations '
                    'and limitations. Simple calculations and robot control can stay in Python.')
                spec['input_schema']['properties']['evidence_refs'] = {
                    'type':'array', 'items':{'type':'string'}, 'minItems':1, 'maxItems':16,
                    'description':'Optional explicit stored binding IDs from openvl_bindings/trace_find. '
                                  'Historical analysis only; linked image pixels are not inspected.'}
            if spec['name'] == 'Zoom_In_Tool':
                props = spec['input_schema']['properties']
                props['padding'].update(minimum=50, default=50)
                props['coords'].update(minItems=4, maxItems=4,
                                       items={'type':'number', 'minimum':0, 'maximum':1000})
        return specs

    def compare_evidence(self, reference, previous_ref=None):
        self.bridge.require_reference(reference)
        if previous_ref is None:
            previous_ref = self.bridge.entries[reference].get('previous_ref')
        else:
            self.bridge.require_reference(previous_ref)
        before = self.bridge.history(previous_ref) if previous_ref else None
        current_entry = self.bridge.entries[reference]
        previous_entry = self.bridge.entries[previous_ref] if previous_ref else None
        association = {
            'same_query': (current_entry['query'].casefold() == previous_entry['query'].casefold())
                if previous_entry else None,
            'same_camera': (current_entry['camera'] == previous_entry['camera'])
                if previous_entry else None,
            'object_identity': 'unverified',
            'causality': 'unverified',
        }
        self.event('evidence_compare', reference=reference, previous_ref=previous_ref,
                   association=association)
        return {'reference':reference, 'previous_ref':previous_ref,
                'before':before, 'after':self.bridge.current(reference),
                'association':association,
                'task_effect':'not established by these measurements alone'}

    def before_tool(self, *, name=None, args=None):
        super().before_tool(name=name, args=args)
        try:
            self.trace.begin(name, args)
        except Exception as exc:
            self.trace_error(exc)

    def record_action(self, *args):
        self.trace.action(*args)

    def trace_error(self, exc):
        self.trace.errors.append(f"{type(exc).__name__}: {exc}")

    def read_data(self, reference):
        """Copy a native binding; robot bindings retain their freshness checks."""
        if reference in self.bridge.refs:
            return self.bridge.read(reference)
        bindings = self.memory.state.bindings.bindings
        if reference not in bindings:
            raise ValueError(f"Unknown OpenVL binding: {reference!r}")
        return copy.deepcopy(bindings[reference])

    def prompt(self):
        text = """
# Native OpenVL tools
Native tool image_id and binding IDs share the robot evidence memory.
openvl_bindings() lists descriptors; openvl_data(id) reads copied data (also on robo).
Environment updates list new/changed entries; earlier IDs remain stored. Full
historical trace data stays accessible through the helpers below.
Zoom uses 0-1000 coordinates in its source image. Convert crop pixels back to the
original camera before back_project. Cropping does not refresh capture time.
Native depth/pose estimates have no automatic robot-world mapping.
Python_Coding_Agent_Tool, if configured, has a separate calculation context with
no robo or persistent robot variables. Use PyRUA Python for robot actions.
For multi-record analysis, pass objective plus evidence_refs (stored binding IDs)
when the schema supports them. Only selected structured records are retrieved;
linked records/images are not automatically inspected. The returned report and
citations are generated analysis, not verified task success. Simple computation
can remain in Python; no delegation is mandatory.
Inspect its printed_output, generation_error and execution_error before using a
calculation. Its copied historical bindings do not certify current coordinates
or task completion; use fresh observations for action targets.
trace_find(kind=None, tool=None, cell=None, step_from=None, step_to=None,
time_from=None, time_to=None, limit=10) returns summary DICTIONARIES with reference
STRINGS (Unix time filters). trace_read(id) reads historical records;
trace_related(id) lists incoming links, including visual image_ref/camera/step.
Follow action state references explicitly:
 a = trace_read(trace_find(kind='action')[0]['reference'])
 s = trace_read(a['state_after']); print(s['availability'], s['value'])
Cached states may be old/missing; temporal association does not prove causality.
evidence_context(action_id, limit=8) retrieves the action, linked states,
historical perception and saved before/after image references in one call.
It reports missing/truncated evidence and never captures images or moves the robot.
Use evidence_inspect(perception_id) or Zoom(image_id) to view stored pixels;
reading a reference alone is not visual verification or proof of task completion.
Trace records and images are not current geometry or proof of later action outcomes.
The latest evidence_temporal_context supersedes earlier freshness labels, including
coordinates printed in stdout or retained in Python variables. Missing/omitted
references are not certified current. Use evidence_current(ref), branch on usable,
or re-localize if no reference exists. Historical review uses evidence_history(ref).
Feedback evidence_refs contains dictionaries; use entry['reference'] as the STRING
argument. evidence_compare(ref, previous_ref=None) returns before/after observations;
by default it uses the preceding same-query/same-camera record. Check after.usable
before using coordinates. Matching query/view does not prove object identity.
Zoom padding defaults to50; nonnegative values below50 are explicitly raised to50.
Before claiming task completion, check the task requirements against currently
available observations. Clearly mark anything that cannot be confirmed as unconfirmed.
No one-tool-per-turn rule, automatic retry policy or stage gate is added.
"""
        reviewer = getattr(self, 'completion_reviewer', None)
        if reviewer:
            text += ("\nAn independent native Coding Agent reviews finish(success) using public evidence. "
                     "Its verdict may be satisfied, not_satisfied or unknown. "
                     "Mode: " + reviewer.mode + "; at most two reviews per episode. "
                     "A reviewer verdict is not an environment success signal.\n")
        if self.initial_inventory:
            text += "\n" + self.compact_inventory(self.initial_inventory["content"], initial=True)
        return text

    def invoke(self, name, args):
        if name not in self.native_names:
            return super().invoke(name, args)
        if self.stopped and self.stopped() or self.remaining and self.remaining() <= 0:
            return ArmResult({"error": "episode ended before OpenVL tool execution"})
        invocation_memory = self.memory
        delegation = None
        if name == 'Python_Coding_Agent_Tool' and 'evidence_refs' in args:
            if not self.provenance.get('local_coding'):
                return ArmResult({'error': 'Explicit evidence delegation requires local_coding'})
            from .openvl_delegation import prepare
            try:
                invocation_memory, args, delegation = prepare(self, args)
            except ValueError as exc:
                return ArmResult({'error': {'code':'invalid_delegation', 'message':str(exc)}})
        adjustment = None
        if name == 'Zoom_In_Tool':
            from .openvl_feedback import zoom_arguments
            try:
                args, adjustment = zoom_arguments(args)
            except ValueError as exc:
                return ArmResult({'error': {'code':'invalid_zoom_arguments', 'message':str(exc)}})
        cap = self.tools[name]
        # Upstream can silently substitute the sole stored image for a misspelled
        # ID. Robot episodes require explicit source association, even with one view.
        image_refs = [args["image_id"]] if isinstance(args.get("image_id"), str) else []
        if isinstance(args.get("image_ids"), list):
            image_refs.extend(args["image_ids"])
        for ref in image_refs:
            url, _, fallback = self.memory.resolve_image(ref)
            if not url or fallback == "single_image_registry_default":
                return ArmResult(
                    {
                        "error": {
                            "code": "unresolved_image_reference",
                            "message": f"Unknown image reference: {ref!r}",
                        }
                    }
                )
        if self.remaining:
            from openvlharness.services.client import ToolClient

            if isinstance(cap.client, ToolClient):
                cap.client.timeout_seconds = max(
                    0.001, min(cap.client.timeout_seconds, self.remaining())
                )
                cap.client.retry = 0
            if name == "Python_Coding_Agent_Tool":
                cap.service_config["execution_timeout_seconds"] = max(
                    0.001,
                    min(cap.service_config.get("execution_timeout_seconds", 30), self.remaining()),
                )
        self.serial += 1
        self.native_calls += 1
        result = cap.invoke(VirtualCall(f"openvl_{self.serial}", name, args), invocation_memory)
        self.event("native_capability", name=name, ok=result.ok, processor=result.processor_info)
        payload = {
            "source_image_temporal_context": [self.trace.image_context(ref) for ref in image_refs],
            "native_evidence": [],
            "data": result.data,
            "artifact_refs": [{"kind": ref.kind, "id": ref.id} for ref in result.artifact_refs],
        }
        if delegation is not None:
            from .openvl_delegation import assessment
            payload['delegation'] = delegation
            payload['delegated_analysis'] = assessment(result.data, delegation)
        if adjustment:
            payload["argument_adjustment"] = adjustment
        if not result.ok:
            payload["error"] = result.error
        pictures = []
        for part in result.evidence:
            if part.get("type") == "text":
                payload["native_evidence"].append(part["text"])
            elif part.get("type") == "image_url":
                url = part["image_url"]
                url = url["url"] if isinstance(url, dict) else url
                data_url, _, _ = self.memory.resolve_image(url)
                if data_url and self.kit.images != "none":
                    pictures.append(base64.b64decode(data_url.split(",", 1)[1]))
                    derived_ref = self.memory.state.images.url_to_ref.get(data_url)
                    if derived_ref and derived_ref not in image_refs:
                        self.trace.record_derivation(derived_ref, image_refs)
        payload["images"] = [f"openvl_{self.serial}_{i}" for i in range(len(pictures))]
        return ArmResult(payload, pictures)

    def after_tool(self, result):
        result = super().after_tool(result)
        # Include ordinary robo.show/display outputs so native tools can consume them.
        refs = []
        for data in result.images:
            url = image_part(data)["image_url"]["url"]
            refs.append(
                self.memory.state.images.url_to_ref.get(url)
                or self.memory.register_image(
                    url,
                    provenance=f"pyrualean:turn={len(self.kit.turns)}:step={self.kit.robo._backend.env_steps()}",
                )
            )
        if refs:
            result.result["openvl_image_refs"] = refs
        self._directory_visible_images.update(refs)
        for ref in self.bridge.refs[self.ref_start:]:
            image_ref = self.bridge.values[ref].get('source', {}).get('image_ref')
            if image_ref:
                self._directory_visible_images.add(image_ref)
        try:
            result.result["robot_trace"] = self.trace.finish(result, refs)
            result.result['evidence_temporal_context']['images'] = [
                self.trace.image_context(ref) for ref in refs
            ]
            result.result['evidence_temporal_context']['output_scope'] = {
                'cell': len(self.kit.turns),
                'phase': 'cell output may mix observations before and after actions',
                'actions': self.trace.find(kind='action', cell=len(self.kit.turns), limit=100),
                'constraint': 'Delivery at cell end does not refresh earlier measurements '
                              'printed in stdout or retained in Python variables.',
            }
        except Exception as exc:
            self.trace_error(exc)
            result.result["robot_trace_error"] = str(exc)
        inventory = self.memory.build_environment_update(successful="error" not in result.result)
        if inventory:
            delta = self.compact_inventory(inventory["content"])
            if delta:
                result.result["openvl_environment_update"] = delta
        self.archive_native()
        if self.feedback_mode != "full":
            from .openvl_feedback import present_feedback
            try:
                with (self.root / 'full_feedback.jsonl').open('a') as archive:
                    archive.write(json.dumps(result.result, default=str) + '\n')
            except OSError as exc:
                self.trace_error(exc)
                result.result['feedback_archive_error'] = str(exc)
            rows = []
            for ref in self.bridge.refs[self.ref_start:]:
                entry = self.bridge.entries.get(ref)
                if not entry: continue
                v = self.bridge.validity(ref)
                rows.append({'reference':ref, 'query':entry['query'], 'camera':entry['camera'],
                             'previous_ref':entry.get('previous_ref'),
                             'source_step':v.get('source_step'), 'usable':v.get('usable'),
                             'status':v.get('status')})
            result.result = present_feedback(result.result, rows, automatic=self.automatic_feedback)
        return result

    def compact_inventory(self, content, *, initial=False):
        """Only directory entries are deduplicated, never measurements or failures."""
        if not isinstance(content, str):
            return content
        lines = []
        recognized = False
        for line in content.splitlines():
            if not line.startswith('- '):
                continue
            recognized = True
            if self.directory_mode == 'evidence' and line.startswith('- trace_'):
                continue  # Full audit remains accessible via trace_find/evidence_context.
            ref = line[2:].split(':', 1)[0]
            if (self.directory_mode == 'evidence' and ref in self.memory.state.images.registry
                    and ref not in self._directory_visible_images):
                continue  # Saved action captures are discoverable through evidence_context.
            if line.startswith('- trace_'):
                line = line.split(':', 1)[0] + ': historical record; trace_read(id)'
            lines.append(line)
        if not lines:
            return '' if recognized else content  # Preserve unfamiliar upstream formats.
        new = lines if initial else [line for line in lines if line not in self._inventory_lines]
        self._inventory_lines.update(lines)
        if not new:
            return ''
        return ('OpenVL memory directory (new/changed entries; prior IDs remain available; '
                'not a freshness certificate):\n' + '\n'.join(new))

    def archive_native(self):
        """Persist images and numerical bindings without embedding arrays in prompts."""
        for ref, entry in self.memory.state.images.registry.items():
            url = entry.get("data_url")
            if ref in self.saved_images or not url:
                continue
            path = f"image_{len(self.saved_images)}.bin"
            (self.artifact_root / path).write_bytes(base64.b64decode(url.split(",", 1)[1]))
            self.saved_images[ref] = path
        for ref, value in self.memory.state.bindings.bindings.items():
            if ref in self.saved_bindings:
                continue
            stem = f"binding_{len(self.saved_bindings)}"
            arrays = {}

            def encode(item, arrays=arrays, stem=stem):
                if isinstance(item, np.ndarray):
                    key = f"array_{len(arrays)}"
                    arrays[key] = item
                    return {"npz": stem + ".npz", "key": key}
                if isinstance(item, np.generic):
                    return item.item()
                if isinstance(item, dict):
                    return {k: encode(v) for k, v in item.items()}
                if isinstance(item, (list, tuple)):
                    return [encode(v) for v in item]
                return item

            encoded = encode(value)
            (self.artifact_root / (stem + ".json")).write_text(json.dumps(encoded) + "\n")
            if arrays:
                np.savez_compressed(self.artifact_root / (stem + ".npz"), **arrays)
            self.saved_bindings[ref] = stem + ".json"
        manifest = {
            "images": self.saved_images,
            "image_metadata": {
                ref: {
                    key: sorted(value) if isinstance(value, set) else value
                    for key, value in entry.items()
                    if key != "data_url"
                }
                for ref, entry in self.memory.state.images.registry.items()
            },
            "bindings": self.saved_bindings,
            "descriptors": self.memory.describe_bindings(),
            "webpages": self.memory.state.urls.webpage_registry,
        }
        (self.artifact_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def summary(self):
        return {
            **super().summary(),
            "native_calls": self.native_calls,
            "native_bindings": len(self.saved_bindings),
            "native_images": len(self.saved_images),
            "trace_records": len(self.trace.rows),
            "trace_errors": list(self.trace.errors),
        }
