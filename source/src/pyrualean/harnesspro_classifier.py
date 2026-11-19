"""Call the released HarnessPro classifier runner with a chat-gateway Agent port.

Classification, retries, consensus, caching, overrides and fallback remain in
the upstream runner. Only transport and native Trajectory construction live here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from .openai_runtime import httpx_transport


def load_runner(root):
    path = root.resolve() / "scripts/run_tb2_harness_eval.py"
    name = "_pyrualean_harnesspro_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name, None)
            raise
    return sys.modules[name]


class GatewayClassifier:
    def __init__(self, classifier, *, base_url, api_key=None, transport=None):
        self.model = classifier.model
        self.reasoning_effort = classifier.reasoning_effort
        self.max_tokens = classifier.max_output_tokens
        self.url = base_url.rstrip("/")
        if not self.url.endswith("/chat/completions"):
            self.url += "/chat/completions"
        self.post = transport or httpx_transport(api_key)

    def run(
        self,
        workspace,
        instructions,
        *,
        output_schema=None,
        task_id="",
        harness_id="",
        kind="evaluate",
        timeout_s=None,
        env=None,
    ):
        from jsonschema import validate
        from rho.protocols import Trajectory

        started = time.monotonic()
        # The native runner owns retries; one adapter run is one HTTP request.
        code, body = self.post(
            self.url,
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": instructions + "\nJSON schema: " + json.dumps(output_schema),
                    }
                ],
                "max_tokens": self.max_tokens,
                "reasoning_effort": self.reasoning_effort,
                "response_format": {"type": "json_object"},
            },
            timeout_s or 120,
        )
        if code != 200:
            raise RuntimeError(f"classifier HTTP {code}")
        message = body["choices"][0]["message"]["content"]
        # Strict schema validation supplements the runner's profile/confidence checks.
        validate(json.loads(message), output_schema)
        usage = body.get("usage") or {}
        return Trajectory(
            id=uuid.uuid4().hex,
            kind=kind,
            task_id=task_id,
            harness_id=harness_id,
            instructions=instructions,
            events=[
                {
                    "type": "turn.completed",
                    "api_calls": 1,
                    "usage_model": self.model,
                    "usage_source": "pyrualean-chat-classifier",
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "cached_input_tokens": (usage.get("prompt_tokens_details") or {}).get(
                            "cached_tokens", 0
                        ),
                        "cache_write_input_tokens": 0,
                    },
                    "raw_usage": usage,
                }
            ],
            final_message=message,
            stdout=message,
            stderr="",
            workspace_diff={},
            workspace_deletions=frozenset(),
            exit_code=0,
            wall_time_s=time.monotonic() - started,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
        )


def classify(
    root: Path, policy, task: str, directory: Path, *, base_url, api_key=None, transport=None
):
    from rho.stores.trajectory import FilesystemTrajectoryStore

    if policy.adjuster is not None or policy.fresh_review is not None:
        raise ValueError("Robot classifier port does not support adjuster/fresh_review yet")
    runner = load_runner(root)
    agent = GatewayClassifier(
        policy.classifier, base_url=base_url, api_key=api_key, transport=transport
    )
    descriptor = SimpleNamespace(
        id=hashlib.sha256(task.encode()).hexdigest()[:16], query=lambda: task
    )
    selection = runner._classify_profile_with_model(
        task=descriptor,
        policy=policy,
        classifier_agent=agent,
        adjuster_agent=None,
        run_dir=directory,
        traj_store=FilesystemTrajectoryStore(directory / "trajectories"),
        harness_id="pyrualean-robot",
    )
    return selection, {
        "runner": runner.__file__,
        "runner_sha256": hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest(),
        "transport": "chat_completions (PyRUA gateway adapter)",
        "directory": str(directory),
        "source": selection.source,
    }
