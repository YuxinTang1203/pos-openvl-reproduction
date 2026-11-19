"""Validate accounting boundaries used to judge latency experiments."""
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "latency_analysis", Path(__file__).resolve().parents[1] / "scripts/analyze_model_latency.py",
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def write_episode(tmp_path, events=None):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({
        "environment_success": False, "policy_wall_s": 30,
        "generation": {"request_latency_s": [10], "output_tokens": 100,
                       "reasoning_output_tokens": 80, "tool_requests": [{"elapsed_s": 3}]},
    }))
    if events is not None:
        (tmp_path / "openai_events.jsonl").write_text(
            "\n".join(json.dumps(event) for event in events) + "\n",
        )
    return path


def test_failed_waits_and_successful_stream_phases_are_accounted_separately(tmp_path):
    path = write_episode(tmp_path, [
        {"type": "request_start"},
        {"type": "stream_first_delta", "channel": "reasoning_content", "latency_s": 1},
        {"type": "request", "ok": False, "latency_s": 15},
        {"type": "request_start"},
        {"type": "stream_first_delta", "channel": "reasoning_content", "latency_s": 2},
        {"type": "stream_first_delta", "channel": "tool_calls", "latency_s": 9},
        {"type": "request", "ok": True, "latency_s": 10},
    ])
    report = analysis.summarize([path])
    assert report["totals"]["all_request_wait_s"] == 25
    assert report["totals"]["failed_request_wait_s"] == 15
    assert report["successful_requests"]["count"] == 1
    assert report["phase_coverage_requests"] == 1
    assert sum(p["total_s"] for p in report["phases"].values()) == 10
    assert report["request_wait_fraction"] == pytest.approx(25 / 30)
    assert report["official_successes"] == 0


def test_missing_or_out_of_order_stream_phases_do_not_fabricate_compute_time(tmp_path):
    path = write_episode(tmp_path, [
        {"type": "request_start"},
        {"type": "stream_first_delta", "channel": "tool_calls", "latency_s": 1},
        {"type": "stream_first_delta", "channel": "reasoning_content", "latency_s": 2},
        {"type": "request", "ok": True, "latency_s": 10},
        {"type": "request_start"},
        {"type": "request", "ok": True, "latency_s": 10},
    ])
    report = analysis.summarize([path])
    assert report["phase_coverage_requests"] == 0
    assert report["totals"]["all_request_wait_s"] == 20


def test_missing_events_falls_back_to_recorded_successful_waits(tmp_path):
    path = write_episode(tmp_path)
    report = analysis.summarize([path])
    assert report["totals"]["all_request_wait_s"] == 10
    assert report["cases"][0]["request_events_available"] is False
    assert report["phase_coverage_requests"] == 0
