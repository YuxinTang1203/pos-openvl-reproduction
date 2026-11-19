# Copyright 2026 PyRUA-Lean Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""A plain OpenAI chat-completions agent loop for the code arms (``--runtime openai``).

The Codex and Claude runtimes hand the agent loop to a coding-agent product
with its own system prompt, tools and transcript format.  This runtime is the
third option: the loop lives here, talks to *any* OpenAI-compatible
``/v1/chat/completions`` endpoint (a gateway, a local SGLang / vLLM server, a
training framework's rollout engine) with function calling, and executes the
arm's tools in-process through :class:`pyrualean.arms.ArmToolkit` (no MCP
server, no subprocess).  It is what a small open model, a distillation
teacher and an RL rollout all share.

Protocol (one episode):

1. The first user message carries the full prompt text and the task card
   images as ``image_url`` data URIs (same content Codex / Claude get).
2. Each assistant reply may carry ``tool_calls``; they are executed in order
   through ``ArmToolkit.execute_tool`` and answered with one ``tool`` message
   each.  Images returned by a tool (``on-motion`` or ``robo.show()``) cannot
   travel in a ``tool`` message under the OpenAI schema, so they follow in one
   extra ``user`` message that names the tool call they belong to.
3. The episode ends when the agent calls ``finish``, when it replies without
   a tool call (after ``nudges`` reminders), when the decision budget is
   spent (``ArmToolkit`` refuses further ``python`` calls), when the wall
   clock ``budget_s`` runs out, or when the provider keeps failing.

The conversation queue is written to ``messages.json`` (images
replaced by references to PNG files under ``images/``), every request to
``openai_events.jsonl``, and the usual ``generation`` record (Codex's token
conventions: ``input_tokens`` includes cached tokens) goes into
``result.json`` by :mod:`pyrualean.play`.
Queued tool replies may never be submitted after termination; feedback delivery
events distinguish queued replies, attempted planner requests and completed responses.
"""

from __future__ import annotations

import base64
import sys
import io
import json
import math
import os
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .arms import ArmResult, ArmToolkit, CellDeadlineExceeded

#: Retries of one chat request on transport / 5xx / 429 errors (env override).
from .image_transport import is_image_fetch_timeout as _multimodal_download_timeout, image_manifest, lossless_alternate


DEFAULT_RETRIES = int(os.environ.get("PYRUALEAN_OPENAI_RETRIES", "10"))
#: A rate limit (HTTP 429) is waited out, not counted against ``retries``: backoff doubling from
#: 5 s to RATE_LIMIT_MAX_WAIT_S, with jitter, for as long as the episode's own time allows.  With
#: ~40 episodes on one gateway key, 39 of 162 episodes ended at their first request after ten
#: 429s two to twenty seconds apart.
RATE_LIMIT_MAX_WAIT_S = float(os.environ.get("PYRUALEAN_RATE_LIMIT_MAX_WAIT_S", "120"))
#: Seconds one chat request may take (a 40-call episode on a slow model needs it).
DEFAULT_TIMEOUT_S = float(os.environ.get("PYRUALEAN_OPENAI_TIMEOUT_S", "900"))
#: Reminders sent when the model replies without a tool call before the episode is over.
DEFAULT_NUDGES = 2
NUDGE_TEXT = (
    "The episode is not over: the simulator has not reported success and you have not "
    "called `finish`. Continue by calling the `python` tool, or call `finish`."
)
#: One message carries every image the turn produced; ``sources`` names the calls they came from.
IMAGE_NOTE = "Images returned by tool call {sources}; they are not part of the text result."

#: ``post(url, json_body, timeout_s) -> (status_code, parsed_json_or_text)``.
Transport = Callable[[str, dict[str, Any], float], tuple[int, Any]]


class OpenAIRuntimeError(RuntimeError):
    """The provider failed for good (after the retries)."""


def parse_harness_payload(response):
    """Accept a state report as a function payload or a strict JSON text object.

    Some compatible gateways return JSON text despite forced function selection.
    No returned function is executed here; domain validators still check every
    state update. Wrong/ambiguous tool calls never fall back to text.
    """
    try:
        message = response["choices"][0].get("message") or {}
        calls = message.get("tool_calls") or []
        if calls:
            if len(calls) != 1 or calls[0].get("function", {}).get("name") != "submit_harness_state":
                return None, "expected_one_state_report"
            arguments = calls[0]["function"].get("arguments")
            decoded = json.loads(arguments) if isinstance(arguments, str) else arguments
            payload = decoded.get("payload") if isinstance(decoded, dict) else None
            return (payload, "tool") if isinstance(payload, dict) else (None, "invalid_payload")
        content = message.get("content")
        if not isinstance(content, str):
            return None, "missing_state_report"
        decoded = json.loads(content)
        if isinstance(decoded, dict) and set(decoded) == {"payload"}:
            decoded = decoded["payload"]
        return (decoded, "text_json") if isinstance(decoded, dict) else (None, "invalid_json_object")
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return None, "malformed_state_report"


# -- images ---------------------------------------------------------------------


def _resize_png(data: bytes, max_side: int) -> tuple[bytes, float]:
    """Downscale ``data`` so its longer side is ``max_side``; returns (png, scale)."""

    if not max_side:
        return data, 1.0
    from PIL import Image

    image = Image.open(io.BytesIO(data))
    longest = max(image.size)
    if longest <= max_side:
        return data, 1.0
    scale = max_side / longest
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    )
    buffer = io.BytesIO()
    resized.save(buffer, format="PNG")
    return buffer.getvalue(), scale


def image_part(data: bytes) -> dict[str, Any]:
    """An OpenAI ``image_url`` content part holding a PNG as a data URI."""

    return {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode("ascii")},
    }


def openai_tools(spec: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the arm's MCP-style tool specs (``input_schema``) to OpenAI ``tools``."""

    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in spec
    ]


# -- transport ------------------------------------------------------------------


def httpx_transport(api_key: str | None) -> Transport:
    """The default transport: one ``httpx`` client, bearer auth, JSON in and out."""

    import httpx

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    client = httpx.Client(headers=headers, timeout=None, trust_env=True)

    def post(url: str, body: dict[str, Any], timeout_s: float) -> tuple[int, Any]:
        response = client.post(url, json=body, timeout=timeout_s)
        try:
            payload: Any = response.json()
        except ValueError:
            payload = response.text
        return response.status_code, payload

    return post


# -- the loop -------------------------------------------------------------------


class _Transcript:
    """The messages the model sees, plus a copy with images swapped for file references."""

    def __init__(self, out_dir: Path) -> None:
        self.messages: list[dict[str, Any]] = []  # what is sent to the model
        self.recorded: list[dict[str, Any]] = []  # same, images as {"image_ref": path}
        self.images_dir = out_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._image_index = 0
        self._round_starts: list[int] = []
        self.compactions = 0
        self.dropped_rounds = 0

    def _save_image(self, data: bytes, name: str) -> str:
        self._image_index += 1
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        path = self.images_dir / f"{self._image_index:03d}_{safe}"
        if not path.suffix:
            path = path.with_suffix(".png")
        path.write_bytes(data)
        return str(path)

    def add(self, message: dict[str, Any], images: list[tuple[str, bytes]] | None = None) -> None:
        """Append ``message``; ``images`` (name, png) become content parts of a user message."""

        sent = dict(message)
        recorded = dict(message)
        if images:
            parts_sent: list[dict[str, Any]] = []
            parts_recorded: list[dict[str, Any]] = []
            text = message.get("content")
            if isinstance(text, str) and text:
                parts_sent.append({"type": "text", "text": text})
                parts_recorded.append({"type": "text", "text": text})
            elif isinstance(text, list):
                parts_sent.extend(text)
                parts_recorded.extend(text)
            for name, data in images:
                parts_sent.append(image_part(data))
                parts_recorded.append(
                    {"type": "image_ref", "image_ref": self._save_image(data, name)}
                )
            sent["content"] = parts_sent
            recorded["content"] = parts_recorded
        self.messages.append(sent)
        self.recorded.append(recorded)

    def begin_round(self, prompt: str, window: int) -> int:
        """Retain the initial task and whole interaction rounds, including tool images.

        The archive is append-only. Only the active provider transcript is pruned;
        no tool result can survive without its assistant call (or vice versa).
        """
        self._round_starts.append(len(self.messages))
        self.add({"role": "user", "content": prompt})
        dropped = 0
        while len(self._round_starts) > window:
            self.drop_oldest_round()
            dropped += 1
        return dropped

    def drop_oldest_round(self) -> bool:
        if len(self._round_starts) < 2:
            return False
        start, end = self._round_starts[:2]
        del self.messages[start:end]
        self._round_starts = [i - (end - start) for i in self._round_starts[1:]]
        self.compactions += 1
        self.dropped_rounds += 1
        return True


def _usage_fields(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    cached = int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)
    reasoning = int(((usage.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0)
    return {"prompt": prompt, "completion": completion, "cached": cached, "reasoning": reasoning}


def _tool_result_text(result: ArmResult) -> str:
    """The text block of a tool result, exactly what the MCP server would send."""

    return result.content_blocks[0]["text"]


def _execute_with_deadline(
    arm_toolkit: ArmToolkit,
    name: str,
    arguments: dict[str, Any],
    remaining_s: float,
    on_timeout: Callable[[], None] | None,
) -> ArmResult:
    """Enforce the existing episode deadline for robot calls AND generated CPU loops.

    A timer still cancels blocking robot operations. A trace confined to generated
    cell/workspace frames also interrupts pure Python loops which never call robo.
    Recovery enforces only the already configured episode deadline; no new
    per-cell quota is introduced.
    Native/C calls remain subject to cooperative cancellation and the outer process
    watchdog; this is not a hostile-code isolation boundary.
    """
    if remaining_s <= 0:
        if on_timeout is not None:
            on_timeout()
        return ArmResult({"episode_timeout": True, "error": "episode execution deadline exceeded"})
    deadline = time.perf_counter() + remaining_s
    workspace = str(arm_toolkit.workspace.resolve()) + "/"
    previous = sys.gettrace()

    def trace(frame, event, arg):
        filename = frame.f_code.co_filename
        generated = filename.startswith("<cell ") or filename.startswith(workspace)
        if not generated:
            return previous(frame, event, arg) if previous else None
        if time.perf_counter() >= deadline:
            if on_timeout is not None:
                on_timeout()
            raise CellDeadlineExceeded("episode execution deadline exceeded during generated Python")
        return trace

    timer = threading.Timer(remaining_s, on_timeout) if on_timeout is not None else None
    if timer is not None:
        timer.daemon = True
        timer.start()
    sys.settrace(trace)
    try:
        return arm_toolkit.execute_tool(name, arguments)
    finally:
        sys.settrace(previous)
        if timer is not None:
            timer.cancel()


def _parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(raw, dict):
        return raw, None
    if raw is None or raw == "":
        return {}, None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError) as exc:
        return None, f"tool arguments are not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "tool arguments must be a JSON object"
    return parsed, None


def run_openai_agent(
    prompt_text: str,
    image_paths: list[str | Path],
    arm_toolkit: ArmToolkit,
    *,
    model: str,
    base_url: str,
    api_key: str | None,
    max_turns: int | None,
    budget_s: float,
    out_dir: str | Path,
    arm: str = "cells",
    temperature: float | None = None,
    max_tokens: int | None = 4096,
    extra_body: dict[str, Any] | None = None,
    tool_choice: str = "auto",
    image_max_side: int = 0,
    nudges: int = DEFAULT_NUDGES,
    preserve_reasoning_content: bool = False,
    retries: int = DEFAULT_RETRIES,
    request_timeout_s: float = DEFAULT_TIMEOUT_S,
    max_requests: int | None = None,
    auxiliary_budget_s: float | None = None,
    auxiliary_round_budget_s: float | None = None,
    auxiliary_stage_budget_s: float | None = None,
    planner_reserve_s: float | None = None,
    transport: Transport | None = None,
    stream: bool = True,
    batch_actions: bool = False,
    phase_tools: bool = False,
    on_timeout: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run one episode's agent loop; returns the ``generation`` record.

    ``status`` is ``completed`` (the agent finished, stopped calling tools, or
    spent its decision budget), ``timeout`` (``budget_s`` expired:
    ``on_timeout`` is called first) or ``error`` (the provider failed after
    ``retries``); ``returncode`` mirrors the Claude runtime: 0 clean, 2 after
    a provider error, ``None`` on timeout.  ``extra_body`` is merged into
    every request (``reasoning_effort``, ``chat_template_kwargs`` ...).
    ``tool_choice`` is OpenAI's: ``auto`` lets the model answer in text,
    ``required`` makes every reply a tool call (``python`` or ``finish``),
    which small thinking models need (they otherwise think and emit nothing).
    ``image_max_side`` downscales every image the model receives (0 keeps the
    native 1024 px that the prompt's pixel coordinates refer to).
    ``preserve_reasoning_content`` replays an assistant's ``reasoning_content``
    in the next request for providers that require historical thinking content.
    ``batch_actions`` adds optional guidance to combine locally decidable steps
    in persistent Python cells while retaining observation/verification boundaries.
    It changes the prompt, not the executor, and is disabled by default.
    ``phase_tools`` enables public RGB observation/motion phases and calibrated
    point measurement for native RGB-only RoboDojo cells; disabled by default.
    Auxiliary state maintenance shares the episode deadline, with an additional
    total wall allowance (default 20% of budget_s), per-round allowance, and
    planner reserve (default min(request_timeout_s, 10% of budget_s)). A harness
    can declare a minimum chain of maintenance calls: each gets a stage allowance
    (default min(120s, 10% of budget_s, request_timeout_s)), and the default round
    fits the complete chain. Explicitly incompatible allowances fail before any
    request; optional repair chains require another full chain's remaining time.
    Unstaged harnesses retain the default min(120s, 10% of budget_s) round.
    These are configurable safeguards, not tuned performance
    claims. Zero auxiliary allowance disables maintenance; raw tool feedback stays
    in the planner transcript. Timeouts/backoffs consume these same allowances.
    ``should_stop`` is asked between requests, between retries and during a rate
    limit's wait: once it answers true the loop ends there (``completed``) without
    another request.  It is how a host says that whoever owns the environment has
    ended the episode, so that nothing the model could still write would count.
    """

    out_dir = Path(out_dir)
    controller = arm_toolkit.general_harness
    state_harness = controller if hasattr(controller, "before_request") else None
    if not getattr(controller, "controls_loop", False):
        controller = None
    protected_fields = {"model", "messages", "tools", "tool_choice", "parallel_tool_calls"}
    if arm_toolkit.general_harness and extra_body and set(extra_body) & protected_fields:
        raise ValueError("harness extra_body cannot override the model, transcript or control tools")
    compatible_controller = bool(controller and controller.kind in {"reuseit", "rlm"})
    if controller and not compatible_controller and max_requests is None:
        max_requests = 128
    if max_requests is not None and (type(max_requests) is not int or max_requests < 1):
        raise ValueError("max_requests must be a positive integer")
    call_reliability = getattr(state_harness, "call_reliability", None)
    requirements = None
    declaration = getattr(state_harness, "maintenance_requirements", None)
    if declaration is not None:
        requirements = declaration()
        if (not isinstance(requirements, dict)
                or set(requirements) != {"min_calls", "role_followup_calls"}
                or type(requirements["min_calls"]) is not int
                or not 1 <= requirements["min_calls"] <= 16
                or not isinstance(requirements["role_followup_calls"], dict)
                or not requirements["role_followup_calls"]
                or any(not isinstance(role, str) or type(count) is not int
                       or not 0 <= count < requirements["min_calls"]
                       for role, count in requirements["role_followup_calls"].items())):
            raise ValueError("invalid harness maintenance_requirements")

    def wall_option(name: str, value: float | None, default: float) -> float:
        value = default if value is None else value
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be a finite nonnegative number")
        return float(value)

    auxiliary_budget_s = wall_option(
        "auxiliary_budget_s", auxiliary_budget_s, max(0.0, budget_s * .2))
    auxiliary_stage_budget_s = wall_option(
        "auxiliary_stage_budget_s", auxiliary_stage_budget_s,
        max(0.0, min(120.0, budget_s * .1, request_timeout_s)))
    minimum_chain_s = (
        requirements["min_calls"] * auxiliary_stage_budget_s if requirements else 0.0)
    auxiliary_round_budget_s = wall_option(
        "auxiliary_round_budget_s", auxiliary_round_budget_s,
        minimum_chain_s if requirements else max(0.0, min(120.0, budget_s * .1)))
    planner_reserve_s = wall_option(
        "planner_reserve_s", planner_reserve_s, max(0.0, min(request_timeout_s, budget_s * .1)))
    if requirements and auxiliary_budget_s > 0:
        if auxiliary_stage_budget_s <= 0:
            raise ValueError("staged maintenance requires auxiliary_stage_budget_s > 0")
        for name, value in (("auxiliary_round_budget_s", auxiliary_round_budget_s),
                            ("auxiliary_budget_s", auxiliary_budget_s)):
            if value < minimum_chain_s:
                raise ValueError(
                    f"{name}={value:g} cannot fit {requirements['min_calls']} required "
                    f"maintenance stages of {auxiliary_stage_budget_s:g}s; "
                    f"configure at least {minimum_chain_s:g}s or disable maintenance explicitly")
        if max_requests is not None and max_requests < requirements["min_calls"] + 1:
            raise ValueError("max_requests cannot fit required maintenance stages and a planner")
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "openai_events.jsonl"
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    # The deadline can be noticed by the tool timer and again by the loop; the robot is stopped
    # once either way, so make the hook itself fire once (the Claude runtime's contract).
    timeout_fired = False

    def stop_once() -> None:
        nonlocal timeout_fired
        if timeout_fired or on_timeout is None:
            return
        timeout_fired = True
        on_timeout()

    tools = openai_tools(arm_toolkit.get_tools_spec())
    transcript = _Transcript(out_dir)
    started = time.perf_counter()

    def event(kind: str, **fields: Any) -> None:
        with open(events_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"t": round(time.perf_counter() - started, 3), "type": kind, **fields},
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

    if transport is not None:
        post = transport
    elif stream:
        from .openai_stream import streaming_transport
        post = streaming_transport(api_key, event, should_stop)
    else:
        post = httpx_transport(api_key)

    # 1. The prompt + the card images, as one user message.
    card_images: list[tuple[str, bytes]] = []
    scales: list[float] = []
    for path in image_paths:
        data, scale = _resize_png(Path(path).read_bytes(), image_max_side)
        card_images.append((Path(path).name, data))
        scales.append(scale)
    from .execution_style import execution_prompt

    text = execution_prompt(prompt_text, batch_actions=batch_actions, arm=arm)
    if type(phase_tools) is not bool:
        raise ValueError("phase_tools must be boolean")
    if phase_tools:
        if arm != "cells":
            raise ValueError("phase_tools requires the cells arm")
        from .phase_tools import PROMPT as phase_prompt, install as install_phases
        install_phases(arm_toolkit)
        text += "\n\n" + phase_prompt
    if any(s != 1.0 for s in scales):
        factor = min(scales)
        text += (
            f"\n\nNote: every image you receive is downscaled by {factor:.3f} "
            "(multiply pixel coordinates you read off an image by "
            f"{1 / factor:.3f} before using them as 1024x1024 coordinates)."
        )
    transcript.add({"role": "user", "content": text}, card_images)

    totals = {"prompt": 0, "completion": 0, "cached": 0, "reasoning": 0}
    auxiliary_totals = dict.fromkeys(totals, 0)
    auxiliary_requests = 0
    request_profile: list[int] = []
    request_latency: list[float] = []
    tool_calls: dict[str, int] = {}
    tool_requests: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    requests = 0
    nudges_left = max(0, nudges)
    malformed_streak = 0
    max_malformed_streak = 3
    status = "completed"
    returncode: int | None = 0
    last_text = ""
    last_harness_context = None
    auxiliary_wall_s = 0.0
    auxiliary_round_deadline: float | None = None
    auxiliary_round_calls = 0
    feedback_delivery: dict[str, dict[str, Any]] = {}
    finish_reasons: list[str | None] = []
    #: Replies allowed after the decision budget ended the episode on the robot
    #: (the refused call told the model to call ``finish``; give it one reply to do so).
    post_budget_replies = 1

    def remaining() -> float:
        return budget_s - (time.perf_counter() - started)

    def over() -> bool:
        return bool(should_stop is not None and should_stop())

    if arm_toolkit.evidence_adapter and hasattr(arm_toolkit.evidence_adapter, "native_names"):
        arm_toolkit.evidence_adapter.remaining = remaining
        arm_toolkit.evidence_adapter.stopped = over

    def wait(seconds: float) -> None:
        """Sleep, in slices short enough to notice that the episode has been ended."""

        if should_stop is None:
            sleep(seconds)
            return
        until = time.perf_counter() + seconds
        while not over():
            left = until - time.perf_counter()
            if left <= 0:
                break
            sleep(min(1.0, left))

    def chat(
        override=None, *, role="planner", fatal=True, request_ceiling=None, deadline=None, attempt_timeout_s=None
    ) -> dict[str, Any] | None:
        """One chat request with retries; ``None`` when the provider is down for good."""

        nonlocal requests, auxiliary_requests
        body: dict[str, Any] = {
            "model": model,
            "messages": transcript.messages,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        if controller:
            body.update(parallel_tool_calls=False)
            if not compatible_controller:
                body["tool_choice"] = "required"
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens:  # 0 / None: do not send the field at all (o-series reject max_tokens)
            body["max_tokens"] = max_tokens
        if extra_body:
            body.update(extra_body)
        if override is not None:
            body = override
        if deadline is None:
            # A retry shares the original logical request deadline, including backoff.
            deadline = time.perf_counter() + min(request_timeout_s, remaining())
        attempt = 0
        limited = 0
        alternate_used = False

        def request_remaining() -> float:
            if deadline is None:
                return remaining()
            return min(remaining(), deadline - time.perf_counter())

        while True:
            attempt += 1
            left = request_remaining()
            if left <= 0 or over() or (max_requests is not None and requests >= max_requests):
                return None
            if request_ceiling is not None and requests >= request_ceiling:
                return None
            t0 = time.perf_counter()
            feedback_ids = [m["tool_call_id"] for m in body["messages"]
                            if role == "planner" and m.get("role") == "tool"
                            and m.get("tool_call_id") in feedback_delivery
                            and not feedback_delivery[m["tool_call_id"]]["response_requests"]]
            for call_id in feedback_ids:
                feedback_delivery[call_id]["submitted_requests"].append(requests + 1)
            if feedback_ids:
                event("planner_feedback_submitted", request=requests + 1, call_ids=feedback_ids)
            attempt_cap = min(request_timeout_s, left,
                              attempt_timeout_s if attempt_timeout_s is not None else request_timeout_s)
            event("request_start", request=requests + 1, attempt=attempt, role=role,
                  stream=stream and transport is None, **image_manifest(body),
                  timeout_s=attempt_cap, logical_remaining_s=left)
            try:
                code, payload = post(url, body, attempt_cap)
            except Exception as exc:  # noqa: BLE001 - transport errors are retried
                code, payload = 0, f"{type(exc).__name__}: {exc}"
            latency = time.perf_counter() - t0
            requests += 1
            if role != "planner":
                auxiliary_requests += 1
            if arm_toolkit.harnesspro_hook:
                arm_toolkit.harnesspro_hook.note_model_request()
            arm_toolkit.model_turn = requests
            ok = code == 200 and isinstance(payload, dict) and payload.get("choices")
            event(
                "request",
                role=role,
                attempt=attempt,
                status_code=code,
                latency_s=round(latency, 3),
                messages=len(body["messages"]),
                ok=bool(ok),
                error=None if ok else str(payload)[:2000],
            )
            if call_reliability:
                call_reliability.observe(role, latency, bool(ok),
                    timeout=code == 0 and 'Timeout' in str(payload).split(':', 1)[0])
            if ok:
                for call_id in feedback_ids:
                    feedback_delivery[call_id]["response_requests"].append(requests)
                if feedback_ids:
                    event("planner_feedback_response", request=requests, call_ids=feedback_ids)
                request_latency.append(round(latency, 3))
                return payload
            message = f"request {requests} attempt {attempt}: HTTP {code}: {str(payload)[:500]}"
            context_error = code in (400, 413) and any(s in str(payload).lower() for s in (
                "context_length_exceeded", "maximum context length", "context window", "context length"))
            if override is None and controller and context_error and transcript.drop_oldest_round():
                warnings.append(message + "; dropped one whole context round")
                event("context_trim", reason="provider_context_limit", dropped_rounds=1,
                      active_messages=len(transcript.messages))
                continue
            if code == 429 and not call_reliability:
                # rate limited: wait it out (only the episode's time limit ends this)
                if limited < 5:
                    warnings.append(message)
                limited += 1
                attempt -= 1
                pause = min(RATE_LIMIT_MAX_WAIT_S, 5.0 * 2.0 ** min(limited - 1, 5))
                wait(max(0.0, min(pause * (0.5 + random.random()), request_remaining())))
                continue
            # 4xx other than 408/429 is our fault (bad request, context overflow): no retry.
            retryable = (call_reliability.retryable(code, payload) if call_reliability
                         else code in (0, 408, 409, 425) or code >= 500)
            retryable = retryable or _multimodal_download_timeout(code, payload)
            retry_limit = call_reliability.max_retries if call_reliability else retries
            if call_reliability:
                event('request_retry_classified', role=role, attempt=attempt,
                      retryable=retryable, max_retries=retry_limit,
                      remaining_s=max(0.0, request_remaining()),
                      same_request_body=True, robot_replay=False)
            pause = (2.0 + random.random()) if call_reliability else min(60.0, 2.0 * attempt)
            # Auxiliary retries require one full configured attempt, not the
            # few seconds left over after the first request timed out.
            retry_floor = (min(request_timeout_s, attempt_timeout_s)
                           if attempt_timeout_s is not None else 0.0)
            slot_available = ((request_ceiling is None or requests < request_ceiling)
                and (max_requests is None or requests < max_requests))
            skip_reason = ('not_transient' if not retryable else
                           'attempt_limit' if attempt > retry_limit else
                           'reserved_request_slots' if not slot_available else
                           'insufficient_full_retry_time' if call_reliability and request_remaining() < pause + retry_floor else None)
            if skip_reason:
                if call_reliability:
                    event('request_retry_skipped', role=role, reason=skip_reason,
                          remaining_s=max(0.0, request_remaining()),
                          required_retry_s=retry_floor, backoff_s=pause,
                          robot_replay=False)
                (errors if fatal else warnings).append(message)
                return None
            warnings.append(message)
            if _multimodal_download_timeout(code, payload) and attempt >= 2 and not alternate_used:
                alternate_used = True
                try:
                    replacement, encoding_audit = lossless_alternate(body)
                    body = replacement
                    event('image_transport_recovery', role=role, **encoding_audit)
                except Exception as exc:
                    event('image_transport_recovery_failed', role=role, reason=type(exc).__name__+': '+str(exc), robot_replay=False)
            if call_reliability:
                event('request_retry_scheduled', role=role, next_attempt=attempt + 1,
                      backoff_s=pause, required_retry_s=retry_floor,
                      same_request_body=True, robot_replay=False)
            wait(max(0.0, min(pause, request_remaining())))

    def harness_model_call(role, prompt, payload, images=()):
        """Budgeted, audited auxiliary inference; it cannot execute robot tools."""
        nonlocal auxiliary_round_calls
        # Reserve both a request and wall time; a maintenance round's repairs and
        # retries all use one absolute deadline, never a fresh allowance.
        if (over() or auxiliary_round_deadline is None
                or time.perf_counter() >= auxiliary_round_deadline
                or (max_requests is not None and requests >= max_requests - 1)):
            event("harness_model_skipped", role=role, reason="shared_budget_or_stop")
            return None
        followups = 0
        stage_deadline = auxiliary_round_deadline
        request_ceiling = max_requests - 1 if max_requests is not None else None
        if requirements:
            if role not in requirements["role_followup_calls"]:
                raise ValueError(f"undeclared maintenance role: {role}")
            followups = requirements["role_followup_calls"][role]
            left = auxiliary_round_deadline - time.perf_counter()
            calls_available = (max_requests - requests - 1
                               if max_requests is not None else math.inf)
            # An invalid first belief never used its reserved audit call. Its
            # one existing repair + audit may share the remaining round time;
            # do not demand two fresh full-size stages after already spending
            # the belief stage. No new round, auxiliary credit or planner time.
            stage_cap = auxiliary_stage_budget_s
            reuse_unspent_audit = (
                not call_reliability and
                role == "pos_repair" and auxiliary_round_calls == 1
                and followups == 1 and left < stage_cap * 2
                and left >= stage_cap
            )
            if reuse_unspent_audit:
                stage_cap = left / 2
            dynamic_allocation = None
            if call_reliability:
                dynamic_allocation = call_reliability.allocate(
                    role, left, followups, auxiliary_stage_budget_s)
                if dynamic_allocation is None:
                    event('harness_stage_chain_skipped', role=role,
                          reason='insufficient_dynamic_followup_reservation', remaining_s=left)
                    return None
                stage_cap = dynamic_allocation['stage_cap_s']
                if calls_available < 2 + followups:
                    stage_cap = auxiliary_stage_budget_s
                    dynamic_allocation.update(stage_cap_s=stage_cap,
                                              retry_capacity_reserved=False)
            # Repairs after an actual audit still require the original complete
            # new chain. Request slots for the auditor/planner remain reserved.
            if (calls_available < 1 + followups
                    or (not call_reliability and auxiliary_round_calls and followups
                        and left < stage_cap * (1 + followups))):
                event("harness_stage_chain_skipped", role=role,
                      reason="incomplete_chain_budget", remaining_s=max(0.0, left),
                      required_calls=1 + followups)
                return None
            if reuse_unspent_audit:
                event("harness_repair_budget_reused", role=role,
                      reason="first_belief_rejected_before_audit",
                      remaining_s=max(0.0, left), stage_cap_s=stage_cap,
                      reserved_followup_s=followups * stage_cap)
            followup_reserve = (dynamic_allocation['minimum_followup_s']
                                if dynamic_allocation else followups * stage_cap)
            stage_deadline = min(time.perf_counter() + stage_cap,
                                 auxiliary_round_deadline - followup_reserve)
            if dynamic_allocation:
                event('harness_dynamic_stage_allocation', role=role,
                      remaining_round_s=left, **dynamic_allocation,
                      absolute_round_deadline_unchanged=True)
            if stage_deadline <= time.perf_counter():
                event("harness_stage_chain_skipped", role=role,
                      reason="no_time_after_followup_reservation")
                return None
            if request_ceiling is not None:
                request_ceiling -= followups
            event("harness_stage_start", role=role, reserved_followup_calls=followups,
                  timeout_s=stage_deadline - time.perf_counter())
        auxiliary_round_calls += 1
        schema_provider = getattr(state_harness, "response_schema", None)
        payload_schema = (schema_provider(role) if schema_provider
                          else {"type": "object", "additionalProperties": True})
        if not isinstance(payload_schema, dict) or payload_schema.get("type") != "object":
            raise ValueError("harness response_schema must be an object schema")
        auxiliary = _Transcript(out_dir / "harness_model" / f"request_{requests + 1}")
        auxiliary.add({"role": "system", "content": str(prompt) + (
            "\nReturn your JSON object in the payload field of submit_harness_state. "
            "This is a state-maintenance report, not a robot action or official task verdict."
        )})
        named_images = list(images.items()) if isinstance(images, dict) else list(images)
        auxiliary.add({"role": "user", "content": json.dumps(payload, ensure_ascii=False,
                                                                default=str) +
                       "\nAttached image labels in order: " +
                       json.dumps([name for name, _ in named_images])},
                      [(name, _resize_png(data, image_max_side)[0]) for name, data in named_images])
        body = dict(extra_body or {})
        body.update(model=model, messages=auxiliary.messages, tools=[{
            "type": "function", "function": {
                "name": "submit_harness_state", "description": "Submit the requested state JSON.",
                "parameters": {"type": "object", "properties": {
                    "payload": payload_schema},
                    "required": ["payload"], "additionalProperties": False}}}],
            tool_choice="auto" if compatible_controller else {"type": "function", "function": {"name": "submit_harness_state"}},
            parallel_tool_calls=False, max_tokens=max_tokens or 8192)
        if temperature is not None:
            body["temperature"] = temperature
        response = chat(body, role=role, fatal=False,
                        request_ceiling=request_ceiling, deadline=stage_deadline,
                        attempt_timeout_s=auxiliary_stage_budget_s if call_reliability else None)
        audit = {"role": role, "request": requests, "messages": auxiliary.recorded,
                 "response": response, "response_schema": payload_schema}
        with (out_dir / "harness_model_calls.jsonl").open("a") as handle:
            handle.write(json.dumps(audit, ensure_ascii=False, default=str) + "\n")
        if response is None:
            return None
        usage = _usage_fields(response.get("usage"))
        for key in totals:
            totals[key] += usage[key]
            auxiliary_totals[key] += usage[key]
        request_profile.append(totals["prompt"])
        event("harness_assistant", role=role, usage=usage,
              finish_reason=response["choices"][0].get("finish_reason"))
        parsed, representation = parse_harness_payload(response)
        if parsed is None:
            event("harness_model_invalid", role=role, reason=representation)
            invalid_report = getattr(state_harness, "invalid_state_report", None)
            if callable(invalid_report):
                return invalid_report(role, representation)
            if compatible_controller and role == "rlm_query":
                return {"_invalid_report": representation}
            return None
        event("harness_report", role=role, representation=representation)
        return parsed

    try:
        while True:
            if max_requests is not None and requests >= max_requests:
                warnings.append("model request limit reached")
                break
            if remaining() <= 0:
                status, returncode = "timeout", None
                break
            if over():
                break  # the episode was ended from outside: nothing more to write
            if arm_toolkit.turn_budget_exhausted and arm_toolkit.finish is None:
                if post_budget_replies <= 0:
                    break
                post_budget_replies -= 1
            if controller:
                dropped = transcript.begin_round(controller.request_prompt(), controller.options["window_rounds"])
                event("recap_context" if controller.kind == "recap_full" else "controller_context",
                      revision=controller.revision, phase=controller.phase,
                      dropped_rounds=dropped, active_messages=len(transcript.messages))
            if state_harness and not arm_toolkit.turn_budget_exhausted:
                allowance = min(auxiliary_budget_s - auxiliary_wall_s,
                                auxiliary_round_budget_s, remaining() - planner_reserve_s)
                schedule_plan = None
                planner = getattr(state_harness, "maintenance_plan", None)
                if callable(planner):
                    schedule_plan = planner(spent=auxiliary_wall_s, remaining=max(0.0, remaining()),
                                            total=budget_s, auxiliary=auxiliary_budget_s,
                                            round_limit=auxiliary_round_budget_s,
                                            minimum=minimum_chain_s)
                    allowance = min(allowance, schedule_plan["allowance_s"])
                    event("harness_schedule", **schedule_plan)
                if call_reliability and requirements and (schedule_plan is None or schedule_plan['due']):
                    scheduled_allowance = allowance
                    final_reserve = schedule_plan.get('reserve_s', 0.0) if schedule_plan else 0.0
                    available = max(0.0, min(auxiliary_budget_s - auxiliary_wall_s - final_reserve,
                                             remaining() - planner_reserve_s))
                    allowance = call_reliability.recovery_allowance(
                        allowance, available, auxiliary_round_budget_s,
                        auxiliary_stage_budget_s, requirements['min_calls'])
                    event('harness_recovery_capacity', scheduled_allowance_s=scheduled_allowance,
                          reserved_allowance_s=allowance, final_chain_reserve_s=final_reserve,
                          total_auxiliary_budget_s=auxiliary_budget_s,
                          total_budget_increased=False)
                maintenance_started = time.perf_counter()
                try:
                    minimum_calls = requirements["min_calls"] if requirements else 1
                    if ((schedule_plan is not None and not schedule_plan["due"])
                            or allowance <= 0
                            or (requirements and allowance < minimum_chain_s)
                            or (max_requests is not None
                                and max_requests - requests < minimum_calls + 1)):
                        event("harness_maintenance_skipped",
                              reason=(("scheduled_" + schedule_plan["reason"]) if schedule_plan is not None else
                                      "incomplete_maintenance_chain" if requirements
                                      else "auxiliary_allowance_or_planner_reserve"),
                              remaining_s=max(0.0, remaining()),
                              auxiliary_wall_s=round(auxiliary_wall_s, 3),
                              required_calls=minimum_calls, minimum_chain_s=minimum_chain_s)
                        context = (
                            "Auxiliary maintenance was skipped to preserve the main policy budget. "
                            "Previous maintained beliefs have not been refreshed against the "
                            "latest tool results; use the original tool feedback and images for "
                            "the next decision."
                        )
                    else:
                        auxiliary_round_deadline = maintenance_started + allowance
                        auxiliary_round_calls = 0
                        event("harness_maintenance_start", allowance_s=allowance,
                              planner_reserve_s=planner_reserve_s,
                              requirements=requirements)
                        context = state_harness.before_request(harness_model_call)
                        if getattr(state_harness, "fatal_error", None):
                            errors.append(state_harness.fatal_error)
                            event("harness_fatal_error", error=state_harness.fatal_error)
                            break
                    inject = bool(context and context != last_harness_context)
                    if inject:
                        transcript.add({"role": "user", "content": context})
                        last_harness_context = context
                    event("harness_context", harness_kind=state_harness.kind,
                          chars=len(context or ""), injected=inject)
                except Exception as exc:
                    warning = f"harness maintenance failed: {type(exc).__name__}: {exc}"
                    warnings.append(warning)
                    event("harness_maintenance_error", error=warning)
                    context = (
                        "Auxiliary maintenance failed. Previous maintained beliefs have not been "
                        "refreshed reliably; use the original tool feedback and images for the "
                        "next decision."
                    )
                    if context != last_harness_context:
                        transcript.add({"role": "user", "content": context})
                        last_harness_context = context
                finally:
                    if auxiliary_round_deadline is not None:
                        spent = time.perf_counter() - maintenance_started
                        auxiliary_wall_s += spent
                        event("harness_maintenance_end", elapsed_s=round(spent, 3),
                              auxiliary_wall_s=round(auxiliary_wall_s, 3))
                    auxiliary_round_deadline = None
            # A bounded, fresh decision view is computed without auxiliary
            # inference, including when scheduled maintenance was skipped.
            runtime_context = getattr(state_harness, "runtime_context", None)
            if callable(runtime_context):
                decision_context = runtime_context()
                if decision_context:
                    transcript.add({"role": "user", "content": decision_context})
                    event("runtime_decision_context", context=decision_context)
            response = chat()
            if response is None:
                if over():
                    pass  # ended from outside while a request was being retried
                elif remaining() <= 0:
                    status, returncode = "timeout", None
                elif not errors and max_requests is not None and requests >= max_requests:
                    warnings.append("model request limit reached during retry")
                else:
                    status, returncode = "error", 2
                break
            usage = _usage_fields(response.get("usage"))
            for key in totals:
                totals[key] += usage[key]
            request_profile.append(totals["prompt"])  # cumulative, as the Claude runtime reports it
            choice = response["choices"][0]
            finish_reasons.append(choice.get("finish_reason"))
            message = choice.get("message") or {}
            calls = message.get("tool_calls") or []
            content = message.get("content")
            # A reply with no tool calls must replay with a string content: providers reject
            # {"role": "assistant", "content": null} without tool_calls (a thinking model that
            # spent its whole budget reasoning produces exactly that).
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": content if (calls or isinstance(content, str)) else "",
            }
            if calls:
                assistant["tool_calls"] = calls
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            if preserve_reasoning_content and isinstance(message.get("reasoning_content"), str):
                assistant["reasoning_content"] = message["reasoning_content"]
            recorded_extra = {"reasoning_content": reasoning} if reasoning else {}
            if controller and hasattr(controller, "record_assistant"):
                controller.record_assistant(assistant)
            transcript.messages.append(assistant)
            transcript.recorded.append({**assistant, **recorded_extra})
            if isinstance(assistant["content"], str) and assistant["content"]:
                last_text = assistant["content"]
            event(
                "assistant",
                tool_calls=[c.get("function", {}).get("name") for c in calls],
                text_chars=len(assistant["content"] or "")
                if isinstance(assistant["content"], str)
                else 0,
                reasoning_chars=len(reasoning or ""),
                finish_reason=choice.get("finish_reason"),
                usage=usage,
            )
            if not calls:
                if (controller and not compatible_controller) or tool_choice == "required":
                    warnings.append(
                        f"request {requests}: no tool call despite tool_choice=required "
                        f"(finish_reason={choice.get('finish_reason')!r})"
                    )
                done = (
                    arm_toolkit.finish is not None
                    or arm_toolkit.robo._reported_success()
                    or arm_toolkit.turn_budget_exhausted
                )
                if done or nudges_left == 0:
                    break
                nudges_left -= 1
                transcript.add({"role": "user", "content": (
                    getattr(controller, "nudge", "Use one of the advertised controller tools, or finish with failure/stuck.")
                    if controller and controller.kind != "recap_full" else
                    "Call recap_decide with a revised plan, or finish with failure/stuck." if controller else NUDGE_TEXT)})
                event("nudge", left=nudges_left)
                continue
            # Images cannot ride in a `tool` message, and a `tool` message may not be separated
            # from its `tool_calls` by another role, so every call's images wait until the last
            # tool reply has been appended and then travel together in one `user` message.
            pending_images: list[tuple[str, bytes]] = []
            image_sources: list[str] = []
            malformed = 0
            for index, call in enumerate(calls):
                function = call.get("function") or {}
                name = function.get("name") or ""
                call_id = call.get("id") or f"call_{requests}_{index}"
                call["id"] = call_id  # the replayed assistant message must carry the id we answer
                tool_calls[name] = tool_calls.get(name, 0) + 1
                arguments, problem = _parse_arguments(function.get("arguments"))
                t0 = time.perf_counter()
                # Passive evaluation timestamps/ranges; never enter tool feedback or prompts.
                tool_started_unix = time.time()
                tool_ledger = getattr(arm_toolkit.robo, "_ledger", None)
                primitive_start_index = len(tool_ledger) if tool_ledger is not None else None
                if controller and len(calls) != 1:
                    malformed += 1
                    result = ArmResult({"error": f"{controller.kind} requires exactly one tool call per response; none of this batch was executed"})
                elif problem:
                    malformed += 1
                    result = ArmResult({"error": problem})
                else:
                    result = _execute_with_deadline(
                        arm_toolkit, name, arguments or {}, remaining(), stop_once
                    )
                elapsed = time.perf_counter() - t0
                tool_requests.append(
                    {"request": requests, "tool": name, "elapsed_s": round(elapsed, 3)}
                )
                transcript.add(
                    {"role": "tool", "tool_call_id": call_id, "content": _tool_result_text(result)}
                )
                feedback_delivery[call_id] = {"submitted_requests": [], "response_requests": []}
                event("feedback_queued", call_id=call_id, tool=name)
                names = result.result.get("images") or []
                call_images: list[tuple[str, bytes]] = []
                for position, data in enumerate(result.images):
                    label = names[position] if position < len(names) else f"image_{position}"
                    png, _ = _resize_png(data, image_max_side)
                    call_images.append(
                        (f"{label}.png" if not label.endswith(".png") else label, png)
                    )
                if call_images:
                    image_sources.append(
                        f"{call_id} ({', '.join(name for name, _ in call_images)})"
                    )
                    pending_images.extend(call_images)
                event(
                    "tool",
                    tool=name,
                    executed_tool=result.result.get("controller_executed_tool", result.result.get("recap_executed_tool", name)),
                    call_id=call_id,
                    elapsed_s=round(elapsed, 3),
                    started_t=round(t0 - started, 6),
                    started_at_unix=tool_started_unix,
                    primitive_start_index=primitive_start_index,
                    primitive_end_index=len(tool_ledger) if tool_ledger is not None else None,
                    python_cell=result.result.get("cell"),
                    ok="error" not in result.result and "exception" not in result.result,
                    images=len(call_images),
                    terminated=arm_toolkit.robo._reported_success(),
                )
            if pending_images:
                transcript.add(
                    {
                        "role": "user",
                        "content": IMAGE_NOTE.format(sources="; ".join(image_sources)),
                    },
                    pending_images,
                )
            # `finish` only ends the episode when the toolkit actually recorded it: a call whose
            # arguments did not parse never reached the tool, and the agent must see the error.
            if arm_toolkit.finish is not None:
                break
            if malformed and malformed == len(calls):
                malformed_streak += 1
                if malformed_streak > max_malformed_streak:
                    warnings.append(
                        f"{malformed_streak} consecutive replies whose tool arguments did not parse"
                    )
                    status, returncode = "malformed", 2
                    break
            else:
                malformed_streak = 0
    finally:
        if controller and hasattr(controller, "close"):
            controller.close()
        if status == "timeout":
            stop_once()
        (out_dir / "messages.json").write_text(
            json.dumps(transcript.recorded, indent=1, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        (out_dir / "last_message.md").write_text(last_text or "", encoding="utf-8")

    return {
        "runtime": "openai",
        "model": model,
        "models": [model],
        "base_url": base_url,
        "input_tokens": totals["prompt"],
        "cached_input_tokens": totals["cached"],
        "output_tokens": totals["completion"],
        "reasoning_output_tokens": totals["reasoning"],
        "total_tokens": totals["prompt"] + totals["completion"],
        "requests": requests,
        "auxiliary_requests": auxiliary_requests,
        "auxiliary_usage": auxiliary_totals,
        "auxiliary_wall_s": round(auxiliary_wall_s, 3),
        "execution_guard": {"episode_deadline": True},
        "feedback_delivery": feedback_delivery,
        "request_profile": request_profile,
        "request_latency_s": request_latency,
        "finish_reasons": finish_reasons,
        "turns": len(arm_toolkit.turns),
        "tool_calls": tool_calls,
        "tool_requests": tool_requests,
        "errors": errors,
        "warnings": warnings,
        "compactions": transcript.compactions,
        "dropped_context_rounds": transcript.dropped_rounds,
        "python_cells": arm_toolkit._cell_index,
        "recap_full": controller.context() if controller and controller.kind == "recap_full" else None,
        "controller": controller.context() if controller else None,
        "max_turns_hit": arm_toolkit.turn_budget_exhausted,
        "max_requests_hit": max_requests is not None and requests >= max_requests,
        "status": status,
        "returncode": returncode,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "messages": len(transcript.recorded),
        "images_sent": transcript._image_index,
        "options": {
            "stream": stream,
            "batch_actions": batch_actions,
            "phase_tools": phase_tools,
            "model": model,
            "max_turns": max_turns,
            "max_requests": max_requests,
            "auxiliary_budget_s": auxiliary_budget_s,
            "auxiliary_round_budget_s": auxiliary_round_budget_s,
            "auxiliary_stage_budget_s": auxiliary_stage_budget_s,
            "maintenance_requirements": requirements,
            "planner_reserve_s": planner_reserve_s,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "extra_body": extra_body or {},
            "tool_choice": "required" if controller and not compatible_controller else tool_choice,
            "image_max_side": image_max_side,
            "nudges": nudges,
            "retries": retries,
            "request_timeout_s": request_timeout_s,
            "preserve_reasoning_content": preserve_reasoning_content,
            "images": len(image_paths),
        },
    }


__all__ = [
    "DEFAULT_NUDGES",
    "DEFAULT_RETRIES",
    "DEFAULT_TIMEOUT_S",
    "IMAGE_NOTE",
    "NUDGE_TEXT",
    "OpenAIRuntimeError",
    "httpx_transport",
    "image_part",
    "openai_tools",
    "run_openai_agent",
]
