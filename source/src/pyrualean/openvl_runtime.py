"""Run the upstream OpenVLHarness agent with PyRUA robot capabilities.

MainAgent, ModelAdapter, MemoryStore, prompts and optional perception tools are
imported from OpenVLHarness. This module supplies the robot capability, host
termination/transport boundaries and PyRUA audit records; it contains no agent loop.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import json
import math
import subprocess
import threading
import time
from importlib import metadata, resources
from pathlib import Path
from typing import Any

from .openai_runtime import httpx_transport, image_part, openai_tools

ROBOT_INSTRUCTIONS = """
This question is one live robot episode. Execute the user's task through the
provided python/run_program tool with its persistent robo object. The robot API,
coordinate conventions and observation restrictions are in the user message.
One tool call may execute a complete multi-step robot program, including loops,
helper functions and checks of primitive return values. The one-tool-per-turn
rule does not limit the number of robot primitives inside that program. Return
for another model decision when new visual interpretation or replanning is needed.
Only listed tools are available. Call finish when your work is complete or stuck;
the benchmark judges success independently of your claim. A final text answer
also ends your policy. Images in memory are historical observations: after moving
the robot, inspect a fresh observation before relying on an earlier object's pose.
Zoom coordinates use 0–1000 within the selected image; robot pixel coordinates use
the robot API's image dimensions. A crop does not change the robot coordinate frame.
"""


def _request_shape(payload):
    """Count context strings and image occurrences without logging image bytes.

    Context characters include non-image string values (roles, call IDs and
    arguments as well as text); they are not a tokenizer estimate.
    """
    context = payload.get("messages", payload.get("input", []))
    chars, images, image_chars = 0, 0, 0

    def visit(value):
        nonlocal chars, images, image_chars
        if isinstance(value, str):
            if value.startswith("data:image/"):
                images += 1
                image_chars += len(value)
            else:
                chars += len(value)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(context)
    return {
        "context_messages": len(context) if isinstance(context, list) else 1,
        "context_string_chars": chars,
        "inline_images": images,
        "image_data_chars": image_chars,
    }


def _retryable_http(code, response):
    # Some gateways report a transient image-download timeout as HTTP 400.
    # Auth, model routing and context-length errors need a changed request.
    return (
        code in {408, 409, 425, 429}
        or code >= 500
        or (code == 400 and any(term in str(response).lower() for term in ("timed out", "timeout")))
    )


def _source_record() -> dict[str, Any]:
    import openvlharness
    from openvlharness.agent.agent import MainAgent
    from openvlharness.memory.store import MemoryStore
    from openvlharness.services.models import ModelAdapter

    root = Path(openvlharness.__file__).resolve().parent
    files = {}
    for cls in (MainAgent, ModelAdapter, MemoryStore):
        path = Path(inspect.getfile(cls)).resolve()
        files[cls.__name__] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    record: dict[str, Any] = {"package": str(root), "files": files}
    try:
        record["version"] = metadata.version("openvlharness")
    except metadata.PackageNotFoundError:
        record["version"] = "source checkout"
    if (root.parent / ".git").exists():
        for key, args in (("commit", ["rev-parse", "HEAD"]), ("dirty", ["status", "--porcelain"])):
            result = subprocess.run(
                ["git", "-C", str(root.parent), *args], capture_output=True, text=True, check=False
            )
            record[key] = bool(result.stdout.strip()) if key == "dirty" else result.stdout.strip()
    return record


class _Artifacts:
    """Keep image bytes once; JSON retains local image paths instead of base64."""

    def __init__(self, out: Path):
        self.out = out
        self.images: dict[str, str] = {}
        (out / "images").mkdir(parents=True, exist_ok=True)

    def externalize(self, value):
        if isinstance(value, str) and value.startswith("data:image/"):
            if value not in self.images:
                header, encoded = value.split(",", 1)
                data = base64.b64decode(encoded)
                suffix = ".png" if "image/png" in header else ".jpg"
                path = self.out / "images" / (hashlib.sha256(data).hexdigest() + suffix)
                path.write_bytes(data)
                self.images[value] = str(path)
            return self.images[value]
        if isinstance(value, dict):
            return {k: self.externalize(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.externalize(v) for v in value]
        return value

    def write(self, name, value):
        (self.out / name).write_text(
            json.dumps(self.externalize(value), indent=2, ensure_ascii=False, default=str) + "\n"
        )


class RobotCapability:
    """Implement upstream's Capability protocol using the existing ArmToolkit."""

    def __init__(self, schema, toolkit, stopped, event):
        from jsonschema import Draft7Validator

        self.schema = schema
        self.name = schema["function"]["name"]
        self.toolkit, self.stopped, self.event = toolkit, stopped, event
        self.validator = Draft7Validator(schema["function"]["parameters"])

    def public_schema(self):
        return copy.deepcopy(self.schema)

    def invoke(self, call, memory):
        from openvlharness.capabilities._base import failure
        from openvlharness.contracts import ArtifactRef, CapabilityResult

        reason = self.stopped()
        if reason:
            return failure(call, "episode_ended", reason)
        errors = list(self.validator.iter_errors(call.arguments))
        allowed = self.schema["function"]["parameters"].get("properties", {})
        if errors or set(call.arguments) - set(allowed):
            return failure(
                call,
                "invalid_tool_arguments",
                errors[0].message if errors else "Unsupported robot tool argument",
            )
        started = time.perf_counter()
        bridge = getattr(self.toolkit, "openvl_evidence", None)
        if bridge:
            bridge.bind(memory)
        bindings_before = len(bridge.refs) if bridge else 0
        errors_before = len(bridge.errors) if bridge else 0
        if self.toolkit.stage_verifier:
            self.toolkit.stage_verifier.memory = memory
        result = self.toolkit.execute_tool(call.name, call.arguments)
        if bridge and len(bridge.refs) > bindings_before:
            result.result["perception_bindings"] = bridge.refs[bindings_before:]
        if bridge and len(bridge.errors) > errors_before:
            result.result["perception_memory_errors"] = bridge.errors[errors_before:]
        # Robot exceptions are feedback to reason about. The capability itself
        # successfully executed and must preserve text AND images on failure.
        evidence = [{"type": "text", "text": result.content_blocks[0]["text"]}]
        refs = []
        labels = result.result.get("images") or []
        for i, data in enumerate(result.images):
            part = image_part(data)
            label = labels[i] if i < len(labels) else f"camera_{i}"
            ref = memory.register_image(
                part["image_url"]["url"], provenance=f"{call.call_id}:{label}"
            )
            refs.append(ref)
            evidence.extend([{"type": "text", "text": f"Observation {label}: {ref}"}, part])
        payload = {
            "ok": True,
            "tool_name": call.name,
            "tool_call_id": call.call_id,
            "result": result.result,
            "error": None,
            "meta": {"backend": "pyrualean.ArmToolkit"},
        }
        self.event(
            "robot_tool",
            tool=call.name,
            call_id=call.call_id,
            elapsed_s=time.perf_counter() - started,
            image_refs=refs,
            robot_ok="error" not in result.result and "exception" not in result.result,
            result=result.result,
        )
        return CapabilityResult(
            ok=True,
            data=result.result,
            evidence=evidence,
            payload=payload,
            artifact_refs=[ArtifactRef("image", ref) for ref in refs],
            prepared={"image_refs": refs},
            processor_info={"mode": "pyrualean_robot", "status": "ok"},
        )


class _LocalZoomClient:
    """Transport adapter for the unmodified upstream CPU crop implementation."""

    def execute(self, tool_name, args, *, tool_call_id):
        from openvlharness.services.envelope import _build_success_payload
        from openvlharness.services.tools.zoom_in_image_by_bbox.tool import Zoom_In_Tool

        return _build_success_payload(tool_name, tool_call_id, Zoom_In_Tool().execute(**args))


class _BoundedCapability:
    """Check host termination before entering an upstream perception capability."""

    def __init__(self, capability, stopped, remaining):
        self.capability, self.stopped, self.remaining = capability, stopped, remaining

    def public_schema(self):
        return self.capability.public_schema()

    def invoke(self, call, memory):
        from openvlharness.capabilities._base import failure
        from openvlharness.services.client import ToolClient

        reason = self.stopped()
        if reason:
            return failure(call, "episode_ended", reason)
        client = getattr(self.capability, "client", None)
        if isinstance(client, ToolClient):
            # Preserve upstream serialization/result processing, but do not allow
            # the router's default nine attempts to outlive the robot episode.
            client.timeout_seconds = max(0.001, min(client.timeout_seconds, self.remaining()))
            client.retry = 0
        return self.capability.invoke(call, memory)


def _build_agent(
    model,
    base_url,
    api_key,
    *,
    config_path,
    max_iterations,
    max_tokens,
    reasoning_effort,
    temperature,
    max_context_length,
    zoom,
    protocol=None,
    memory_emit_policy=None,
):
    import yaml
    from openvlharness.bootstrap import build_agent, load_agent_config
    from openvlharness.services.models import load_profile

    profiles = yaml.safe_load(
        resources.files("openvlharness.services").joinpath("model_registry.yaml").read_text()
    )["profiles"]
    alias = (
        model
        if model in profiles
        else next((name for name, p in profiles.items() if p["model_id"] == model), model)
    )
    if protocol not in {None, "chat_completions", "responses"}:
        raise ValueError("protocol must be chat_completions or responses")
    registered = alias in profiles and protocol is None
    served_model = profiles[alias]["model_id"] if registered else model
    if not registered:
        # Reuse an upstream provider-enforced API policy, not Qwen's tokenizer
        # or vLLM-specific sampling options. The wire model is always the actual
        # gateway model, and provenance explicitly records this custom binding.
        protocol = protocol or "chat_completions"
        alias = "kimi-k3" if protocol == "chat_completions" else "gpt-6-astra"
    profile = load_profile(alias)
    if config_path:
        config = load_agent_config(str(config_path))
        service = config["agent"]["model_service"]
        settings = config["services"]["models"][service]
        if load_profile(settings["alias"])["model_id"] != profile["model_id"]:
            raise ValueError("--model must match the OpenVLHarness config orchestrator")
        settings.update(base=base_url, key=api_key or "empty")
    else:
        settings = {"alias": alias, "base": base_url, "key": api_key or "empty"}
        config = {
            "agent_name": "MainAgent",
            "agent_config_version": "paper_v1",
            "agent": {"model_service": "orchestrator"},
            "memory": {
                "modules": [{"name": "image"}, {"name": "tool_context"}],
                "emit_policy": "on_change",
            },
            "capabilities": {"registries": ["paper"], "tools": {}},
            "services": {
                "client": {"base_url": "http://127.0.0.1:8101"},
                "server": {},
                "tools": {},
                "models": {"orchestrator": settings},
            },
        }
    if memory_emit_policy is not None:
        if memory_emit_policy not in {"always", "on_change"}:
            raise ValueError("memory_emit_policy must be always or on_change")
        config.setdefault("memory", {})["emit_policy"] = memory_emit_policy
    settings["max_tokens"] = max_tokens
    if reasoning_effort is not None and "reasoning_effort" in profile["allowed_options"]:
        settings["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        settings["temperature"] = temperature
    config["agent"]["max_iterations"] = max_iterations
    if max_context_length is not None:
        config["agent"]["max_context_length"] = max_context_length
    agent = build_agent(config)
    if not registered:
        if "coding" in config["capabilities"]["tools"]:
            raise ValueError(
                "A custom gateway model needs its own upstream router registration for the "
                "paper coding generator. Disable coding in YAML; robot python remains available."
            )
        agent.model = agent.adapter.model = served_model
        agent.adapter.profile["model_id"] = served_model
        # Custom gateways may accept effort even when the protocol template's
        # original model does not. Preserve the caller's explicit setting so
        # paired Qwen runs do not silently use different reasoning budgets.
        if reasoning_effort is not None:
            if reasoning_effort not in {"none", "minimal", "low", "medium", "high", "xhigh", "max"}:
                raise ValueError("Unsupported gateway reasoning_effort")
            agent.adapter.options["reasoning_effort"] = reasoning_effort
    if zoom and "Zoom_In_Tool" not in agent.capabilities:
        from openvlharness.capabilities.zoom import Capability

        cap = Capability(
            backend="zoom_in_image_by_bbox", client_config={}, client=_LocalZoomClient()
        )
        agent.capabilities[cap.name] = cap
    agent.system_prompt += "\n" + ROBOT_INSTRUCTIONS
    return agent, alias


def run_openvl_agent(
    prompt_text,
    image_paths,
    arm_toolkit,
    *,
    model,
    base_url,
    api_key,
    max_turns,
    budget_s,
    out_dir,
    arm="cells",
    config_path=None,
    protocol=None,
    memory_emit_policy=None,
    max_iterations=128,
    max_tokens=8192,
    reasoning_effort=None,
    temperature=None,
    max_context_length=None,
    zoom=True,
    structured_evidence=True,
    action_feedback=True,
    stage_verification=False,
    retries=2,
    request_timeout_s=180,
    on_timeout=None,
    should_stop=None,
    transport=None,
):
    """Execute upstream MainAgent.generate once for one robot episode.

    A host stop is rendered as an explicitly logged synthetic terminal response,
    so the original loop can finalize its trajectory without another provider call.
    It is never counted as model output or benchmark success.
    """
    started = time.perf_counter()
    if (
        not math.isfinite(budget_s)
        or budget_s <= 0
        or max_iterations < 1
        or retries < 0
        or not math.isfinite(request_timeout_s)
        or request_timeout_s <= 0
    ):
        raise ValueError("budget, iterations and timeout must be positive; retries nonnegative")
    try:
        from openvlharness.services.models import load_profile

        agent, alias = _build_agent(
            model,
            base_url,
            api_key,
            config_path=config_path,
            max_iterations=max_iterations,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_context_length=max_context_length,
            zoom=zoom,
            protocol=protocol,
            memory_emit_policy=memory_emit_policy,
        )
    except ImportError as exc:
        raise ImportError(
            "Install the openvl extra and the pinned OpenVLHarness source; "
            "see docs/guide/openvlharness.md"
        ) from exc
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    artifacts = _Artifacts(out)
    timeout_fired = threading.Event()
    timeout_lock = threading.Lock()
    requests, prompt_tokens, output_tokens, cached_tokens, reasoning_tokens = 0, 0, 0, 0, 0
    request_profile, request_latencies, request_metrics = [], [], []
    stop_reason = None

    def event(kind, **fields):
        record = artifacts.externalize({"type": kind, "t": time.perf_counter() - started, **fields})
        with (out / "openvlharness_events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def timeout():
        with timeout_lock:
            if timeout_fired.is_set():
                return
            timeout_fired.set()
            if on_timeout:
                on_timeout()

    def remaining():
        return budget_s - (time.perf_counter() - started)

    def stopped():
        nonlocal stop_reason
        if remaining() <= 0:
            timeout()
        if timeout_fired.is_set():
            stop_reason = "timeout"
        elif arm_toolkit.finish is not None:
            stop_reason = "finish"
        elif should_stop and should_stop():
            stop_reason = "environment_ended"
        elif arm_toolkit.robo._reported_success():
            stop_reason = "environment_success"
        elif arm_toolkit.turn_budget_exhausted:
            stop_reason = "call_budget"
        return stop_reason

    def terminal():
        text = f"<answer>PyRUA host ended the episode: {stop_reason}.</answer>"
        event("host_stop", reason=stop_reason, synthetic=True)
        if agent.adapter.protocol == "chat_completions":
            return {
                "choices": [
                    {"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                ],
                "usage": {},
            }
        return {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": text}],
                }
            ],
            "usage": {},
        }

    post = transport or httpx_transport(api_key)

    def request(payload):
        nonlocal requests, prompt_tokens, output_tokens, cached_tokens, reasoning_tokens
        for attempt in range(retries + 1):
            if stopped():
                return terminal()
            requests += 1
            arm_toolkit.model_turn = requests
            metrics = {
                "number": requests,
                "attempt": attempt + 1,
                **_request_shape(payload),
                "status_code": None,
                "ok": False,
            }
            event("request", **metrics, payload=payload)
            t0 = time.perf_counter()
            elapsed = None
            retryable = True
            try:
                code, response = post(
                    agent.adapter.endpoint, payload, min(request_timeout_s, remaining())
                )
                elapsed = time.perf_counter() - t0
                metrics["status_code"] = code
                event(
                    "response",
                    number=requests,
                    status_code=code,
                    elapsed_s=elapsed,
                    payload=response,
                )
                if code >= 400:
                    retryable = _retryable_http(code, response)
                    raise RuntimeError(f"Model HTTP {code}: {str(response)[:1000]}")
                if not isinstance(response, dict):
                    raise RuntimeError("Model returned a non-object JSON response")
                usage = response.get("usage") or {}
                metrics["ok"] = True
                metrics["usage"] = usage
                prompt_tokens += usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
                output_tokens += usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
                cached_tokens += (
                    usage.get("prompt_tokens_details", usage.get("input_tokens_details", {})) or {}
                ).get("cached_tokens", 0) or 0
                reasoning_tokens += (
                    usage.get("completion_tokens_details", usage.get("output_tokens_details", {}))
                    or {}
                ).get("reasoning_tokens", 0) or 0
                request_profile.append(prompt_tokens)
                return terminal() if stopped() else response
            except Exception as exc:
                import httpx

                if not isinstance(exc, (RuntimeError, httpx.HTTPError, OSError)):
                    raise
                metrics["error"] = str(exc)
                event(
                    "request_error",
                    number=requests,
                    attempt=attempt,
                    elapsed_s=time.perf_counter() - t0,
                    retryable=retryable,
                    error=str(exc),
                )
                if stopped():
                    return terminal()
                if attempt == retries or not retryable:
                    raise RuntimeError(f"OpenVLHarness model request failed: {exc}") from exc
            finally:
                metrics["elapsed_s"] = elapsed if elapsed is not None else time.perf_counter() - t0
                request_latencies.append(metrics["elapsed_s"])
                request_metrics.append(metrics)
                event("request_complete", **metrics)
            # Only failed, retryable attempts reach this point. Account for
            # backoff separately from the actual HTTP request duration.
            if not stopped():
                until = time.perf_counter() + min(2**attempt, remaining())
                while time.perf_counter() < until and not stopped():
                    time.sleep(min(0.1, max(0, until - time.perf_counter())))
        raise AssertionError("unreachable")

    agent.adapter.requester = request  # upstream's supported transport injection point
    if action_feedback:
        arm_toolkit.enable_execution_recovery()
        from .action_feedback import RECOVERY_INSTRUCTIONS

        agent.system_prompt += RECOVERY_INSTRUCTIONS
    if stage_verification:
        from .stage_verification import PROMPT as STAGE_PROMPT

        arm_toolkit.enable_stage_verification()
        agent.system_prompt += STAGE_PROMPT
    if structured_evidence:
        from .action_feedback import checked
        from .openvl_evidence import INSTRUCTIONS, PerceptionMemory, RobotPerceptionCapability

        if "tool_context" not in agent.memory_factory().modules:
            raise ValueError("Robot structured evidence requires native tool_context memory")
        arm_toolkit.openvl_evidence = PerceptionMemory(arm_toolkit, event)
        arm_toolkit._globals["checked"] = checked
        perception = RobotPerceptionCapability(arm_toolkit.openvl_evidence)
        agent.capabilities[perception.name] = perception
        agent.system_prompt += INSTRUCTIONS
    agent.capabilities = {
        name: _BoundedCapability(cap, stopped, remaining)
        for name, cap in agent.capabilities.items()
    }
    for schema in openai_tools(arm_toolkit.get_tools_spec()):
        cap = RobotCapability(schema, arm_toolkit, stopped, event)
        if cap.name in agent.capabilities:
            raise ValueError(f"OpenVLHarness tool conflicts with robot tool: {cap.name}")
        agent.capabilities[cap.name] = cap
    agent.json_schema = [c.public_schema() for c in agent.capabilities.values()]
    agent.adapter.json_schema = agent.json_schema
    provenance = _source_record()
    provenance.update(
        model_alias=alias,
        requested_model=model,
        model_binding="registered"
        if agent.model == load_profile(alias)["model_id"]
        else "custom_gateway",
        served_model=agent.model,
        protocol=agent.adapter.protocol,
        memory_emit_policy=agent.memory_factory().emit_policy,
        capabilities=list(agent.capabilities),
        structured_evidence=structured_evidence,
        action_feedback=action_feedback,
        stage_verification=stage_verification,
        config_path=str(config_path) if config_path else None,
        robot_adapter=__file__,
        host_termination="synthetic terminal response",
        max_iterations=max_iterations,
        max_turns=max_turns,
        model_options=agent.adapter.options,
        native_model_profile=agent.adapter.profile,
        robot_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    artifacts.write("openvlharness_source.json", provenance)
    (out / "openvlharness_system_prompt.txt").write_text(agent.system_prompt)
    event("start", provenance=provenance)
    timer = threading.Timer(max(0, remaining()), timeout)
    timer.daemon = True
    timer.start()
    try:
        result = agent.generate(
            [{"type": "text", "value": prompt_text}]
            + [{"type": "image", "value": str(Path(p).resolve())} for p in image_paths],
            task_id=out.name,
        )
    finally:
        timer.cancel()
    stopped()
    trajectory = result["extra_records"]["trajectory"]
    artifacts.write("openvlharness_trajectory.json", result)
    artifacts.write("messages.json", trajectory["messages"])
    (out / "last_message.md").write_text(result["prediction"] or "")
    errors = [trajectory["error"]] if trajectory.get("error") else []
    iteration_limit = result["prediction"] == agent.limit_msg
    status = "timeout" if stop_reason == "timeout" else "error" if errors else "completed"
    if iteration_limit:
        status = "completed"  # host maps this explicit ceiling to call_budget
        errors = []
    tools: dict[str, int] = {}
    for turn in trajectory["chat_history"]:
        for call in turn["assistant"]["tool_calls"]:
            tools[call["name"]] = tools.get(call["name"], 0) + 1
    generation = {
        "runtime": "openvlharness",
        "model": agent.model,
        "models": [agent.model],
        "status": status,
        "returncode": None if status == "timeout" else 2 if errors else 0,
        "stop_reason": stop_reason or ("iteration_limit" if iteration_limit else "final_answer"),
        "errors": errors,
        "warnings": [],
        "requests": requests,
        "input_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_tokens,
        "reasoning_output_tokens": reasoning_tokens,
        "total_tokens": prompt_tokens + output_tokens,
        "request_profile": request_profile,
        "request_latency_s": request_latencies,
        "request_metrics": request_metrics,
        "request_wait_s": sum(request_latencies),
        "failed_request_wait_s": sum(m["elapsed_s"] for m in request_metrics if not m["ok"]),
        "tool_calls": tools,
        "turns": len(arm_toolkit.turns),
        "execution_recovery": isinstance(
            arm_toolkit.robo._action_feedback,
            __import__("pyrualean.action_feedback", fromlist=["ExecutionFeedback"]).ExecutionFeedback,
        ),
        "max_turns_hit": iteration_limit,
        "compactions": 0,
        "images_sent": len(artifacts.images),
        "elapsed_s": time.perf_counter() - started,
        "source": provenance,
    }
    event("end", generation=generation)
    artifacts.write("openvlharness_generation.json", generation)
    return generation


def add_cli_arguments(parser):
    group = parser.add_argument_group("OpenVLHarness runtime")
    group.add_argument("--openvl-config", default=None, help="upstream agent YAML (optional)")
    group.add_argument(
        "--openvl-protocol",
        choices=("chat_completions", "responses"),
        default=None,
        help="custom gateway API; unknown models default to chat_completions",
    )
    group.add_argument("--openvl-max-iterations", type=int, default=128)
    group.add_argument("--openvl-max-tokens", type=int, default=8192)
    group.add_argument("--openvl-max-context-length", type=int, default=None)
    group.add_argument(
        "--openvl-memory-emit-policy",
        choices=("on_change", "always"),
        default=None,
        help="native inventory emission; default on_change, or the supplied YAML setting",
    )
    group.add_argument(
        "--openvl-no-zoom", action="store_true", help="omit the native CPU zoom tool"
    )
    group.add_argument("--openvl-no-structured-evidence", action="store_true")
    group.add_argument("--openvl-no-action-feedback", action="store_true")


def cli_options(args):
    return {
        "config_path": args.openvl_config,
        "protocol": args.openvl_protocol,
        "max_iterations": args.openvl_max_iterations,
        "max_tokens": args.openvl_max_tokens,
        "max_context_length": args.openvl_max_context_length,
        "memory_emit_policy": args.openvl_memory_emit_policy,
        "zoom": not args.openvl_no_zoom,
        "structured_evidence": not getattr(args, "openvl_no_structured_evidence", False),
        "action_feedback": not getattr(args, "openvl_no_action_feedback", False),
        "reasoning_effort": args.reasoning,
    }
