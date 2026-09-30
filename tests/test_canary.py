"""Tests for the canary rollout gate (app.ml.canary).

The course's rollout gate is a progressive ramp with AUTO-ROLLBACK: release to a
small traffic share, check production metrics, and abort the rollout rather than
continue if error rate or tool failures breach their thresholds.

The decision logic lives here rather than in .gitlab-ci.yml for the same reason
the load-test SLO check lives in the locustfile: a rule buried in YAML cannot be
tested, and an untested rollback rule is discovered to be wrong during the
incident it was supposed to handle.

FAIL-CLOSED IS THE POINT. A canary step whose metrics are missing must ABORT,
not proceed. "The scrape returned nothing" and "the service is healthy" produce
the same empty result, and only one of them is safe to promote on.
"""
from __future__ import annotations

from app.ml import canary


def _healthy() -> dict:
    return {"errorRate": 0.001, "toolFailureRate": 0.004}


def test_healthy_step_proceeds():
    result = canary.evaluate_step(5, _healthy())

    assert result["verdict"] == "PROCEED"
    assert result["breaches"] == []


def test_error_rate_above_threshold_aborts():
    result = canary.evaluate_step(25, {"errorRate": 0.02, "toolFailureRate": 0.0})

    assert result["verdict"] == "ABORT"
    assert any("error rate" in b for b in result["breaches"])


def test_tool_failure_rate_above_threshold_aborts():
    result = canary.evaluate_step(25, {"errorRate": 0.0, "toolFailureRate": 0.05})

    assert result["verdict"] == "ABORT"
    assert any("tool failure" in b for b in result["breaches"])


def test_missing_metrics_abort_rather_than_proceed():
    """An empty scrape is not evidence of health."""
    result = canary.evaluate_step(5, {})

    assert result["verdict"] == "ABORT"
    assert any("no metric" in b.lower() or "missing" in b.lower() for b in result["breaches"])


def test_none_metrics_abort():
    assert canary.evaluate_step(5, None)["verdict"] == "ABORT"


def test_ramp_completes_when_every_step_is_healthy():
    run = canary.run_ramp(lambda _step: _healthy())

    assert run["verdict"] == "COMPLETED"
    assert [s["step"] for s in run["steps"]] == list(canary.CANARY_STEPS)
    assert run["rollback"] is False


def test_ramp_stops_at_the_first_breaching_step():
    def metrics_for(step: int) -> dict:
        return _healthy() if step < 50 else {"errorRate": 0.2, "toolFailureRate": 0.0}

    run = canary.run_ramp(metrics_for)

    assert run["verdict"] == "ABORTED"
    assert run["rollback"] is True
    assert run["abortedAt"] == 50
    # 100% must never have been attempted after the abort.
    assert [s["step"] for s in run["steps"]] == [5, 25, 50]


def test_ramp_aborts_when_the_metric_source_raises():
    """A broken scrape is a breach, not a pass."""
    def explode(_step: int) -> dict:
        raise RuntimeError("prometheus unreachable")

    run = canary.run_ramp(explode)

    assert run["verdict"] == "ABORTED"
    assert run["rollback"] is True
    assert run["abortedAt"] == canary.CANARY_STEPS[0]


def test_thresholds_match_the_course_gate():
    assert canary.MAX_ERROR_RATE == 0.01
    assert canary.MAX_TOOL_FAILURE_RATE == 0.02
    assert canary.CANARY_STEPS == (5, 25, 50, 100)
