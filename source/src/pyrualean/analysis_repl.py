"""Persistent, process-isolated, read-only RLM analysis REPL with parent RPC.

The child receives serializable public context only. It has no robot, filesystem,
network, provider credentials or toolkit. Subcalls execute in the parent through
the episode's budgeted callback. This is a narrow research REPL, not a Python
security boundary for hostile users or arbitrary packages.
"""
from __future__ import annotations

import ast
import builtins
import contextlib
import io
import json
import math
import multiprocessing
import os
import re
import sys
import time
import subprocess
from pathlib import Path
from multiprocessing.connection import Connection
from types import SimpleNamespace

SAFE_NAMES = ("abs all any bool dict enumerate filter float int isinstance len list map "
              "max min next range repr reversed round set sorted str sum tuple type zip "
              "Exception ValueError TypeError KeyError IndexError").split()


def validate_code(code):
    if not isinstance(code, str) or not code.strip() or len(code) > 16000:
        raise ValueError("analysis code requires 1..16000 characters")
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.With,
                             ast.AsyncWith, ast.AsyncFunctionDef, ast.Global, ast.Nonlocal)):
            raise ValueError(f"analysis REPL disallows {type(node).__name__}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("analysis REPL disallows private attributes")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("analysis REPL disallows private names")
    return tree


def _worker(connection):
    try:
        os.environ.clear()
        # A separate process permits hard termination of C-level work as well.
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (1024 ** 3, 1024 ** 3))
        except (ImportError, ValueError, OSError):
            pass
        # Raw modules expose transitive imports (e.g. re.enum.sys.modules), so
        # offer only narrow function facades rather than the module objects.
        namespace = {
            "__builtins__": {name: getattr(builtins, name) for name in SAFE_NAMES},
            "json": SimpleNamespace(loads=json.loads, dumps=json.dumps),
            "re": SimpleNamespace(**{name: getattr(re, name) for name in
                ("search", "match", "fullmatch", "findall", "finditer", "split",
                 "sub", "subn", "escape", "compile")}),
            "math": SimpleNamespace(**{name: getattr(math, name) for name in dir(math)
                                        if not name.startswith("_")}),
        }
        connection.send({"type": "ready"})
        while True:
            command = connection.recv()
            if command["op"] == "close":
                break
            limits = {"deadline": time.monotonic() + command["timeout_s"],
                      "lines": 0, "waiting": False}
            output = []
            output_chars = 0

            def emit(*args, sep=" ", end="\n"):
                nonlocal output_chars
                value = sep.join(str(x) for x in args) + end
                if output_chars + len(value) > command["max_output"]:
                    raise ValueError("analysis output limit reached; inspect a smaller slice")
                output.append(value)
                output_chars += len(value)

            def query(question=None, selected_context=None, image_ids=(), **aliases):
                if set(aliases) - {"prompt", "context"}:
                    raise ValueError("llm_query accepts question, selected_context, image_ids")
                if "prompt" in aliases:
                    if question is not None:
                        raise ValueError("provide question or prompt, not both")
                    question = aliases["prompt"]
                if "context" in aliases:
                    if selected_context is not None:
                        raise ValueError("provide selected_context or context, not both")
                    selected_context = aliases["context"]
                started = time.monotonic()
                payload = {"prompt": question, "context": selected_context,
                           "image_ids": list(image_ids or ())}
                if len(json.dumps(payload)) > command["max_context_chars"]:
                    raise ValueError("subquery context too large; select a smaller slice")
                limits["waiting"] = True
                try:
                    connection.send({"type": "query", **payload})
                    answer = connection.recv()
                finally:
                    # The trace hook fires BEFORE each line. Keep it suspended
                    # until after extending the deadline when an RPC returns.
                    limits["deadline"] += time.monotonic() - started
                    limits["waiting"] = False
                if answer.get("error"):
                    raise ValueError(answer["error"])
                return answer["answer"]

            def trace(_frame, event, _arg):
                if event == "line":
                    limits["lines"] += 1
                    if not limits["waiting"] and (
                        limits["lines"] > 100000 or time.monotonic() > limits["deadline"]
                    ):
                        raise TimeoutError("analysis execution limit reached")
                return trace

            namespace.update(context=command["context"], llm_query=query, print=emit)
            result = {}
            try:
                tree = validate_code(command["code"])
                tail = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
                sys.settrace(trace)
                # Catch native library prints too; model print is explicitly bounded.
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    exec(compile(tree, "<rlm-analysis>", "exec"), namespace)
                    if tail is not None:
                        value = eval(compile(ast.Expression(tail.value), "<rlm-analysis>", "eval"), namespace)
                        if value is not None:
                            emit(value)
            except BaseException as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"[:command["max_output"]]
            finally:
                sys.settrace(None)
            connection.send({"type": "result", "stdout": "".join(output), **result})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class AnalysisREPL:
    def __init__(self, *, timeout_s=5, max_output=12000, max_context_chars=100000):
        self.timeout_s = timeout_s
        self.max_output = max_output
        self.max_context_chars = max_context_chars
        self.process = None
        self.connection = None

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        parent, child = multiprocessing.Pipe()
        # Spawn re-imports the policy-server entrypoint (Torch/CUDA and other
        # heavy modules). A 1GB address-space cap then breaks receiving context.
        # Start this stdlib-only file directly, without site hooks or credentials.
        self.process = subprocess.Popen(
            [sys.executable, "-I", "-S", "-u", str(Path(__file__).resolve()),
             "--worker-fd", str(child.fileno())], pass_fds=(child.fileno(),),
            env={}, cwd="/", stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE)
        child.close()
        self.connection = parent
        if not parent.poll(3) or parent.recv().get("type") != "ready":
            raise RuntimeError("analysis worker did not become ready")

    def run(self, code, context, callback):
        validate_code(code)
        try:
            self._start()
            self.connection.send({"op": "exec", "code": code, "context": context,
                                  "timeout_s": self.timeout_s, "max_output": self.max_output,
                                  "max_context_chars": self.max_context_chars})
            deadline = time.monotonic() + self.timeout_s + 3
            while time.monotonic() < deadline:
                if not self.connection.poll(.05):
                    if self.process.poll() is not None:
                        raise RuntimeError("analysis worker exited")
                    continue
                message = self.connection.recv()
                if message.get("type") == "result":
                    return {k: v for k, v in message.items() if k != "type"}
                if message.get("type") != "query":
                    raise ValueError("unknown analysis RPC")
                started = time.monotonic()
                try:
                    answer = callback(message["prompt"], message["context"], message["image_ids"])
                    response = {"answer": answer}
                except Exception as exc:
                    response = {"error": f"{type(exc).__name__}: {exc}"}
                deadline += time.monotonic() - started
                self.connection.send(response)
            raise TimeoutError("analysis worker deadline reached")
        except (EOFError, BrokenPipeError, OSError, TimeoutError, RuntimeError) as exc:
            self.close()
            return {"error": f"{type(exc).__name__}: {exc}", "stdout": "",
                    "namespace_reset": True}

    def close(self):
        if self.connection is not None:
            self.connection.close()
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=1)
            if self.process.stderr is not None:
                self.process.stderr.close()
        self.connection = None
        self.process = None


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker-fd":
        raise SystemExit("analysis worker requires inherited RPC descriptor")
    _worker(Connection(int(sys.argv[2])))
