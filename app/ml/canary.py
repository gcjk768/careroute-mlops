"""[MLOps] Pillar 5 — progressive canary rollout with metric-breach auto-rollback.

The course's production-rollout gate is a ramp, not a switch: release to a small
traffic share, read production metrics, and ABORT the rollout (rolling back to
the last stable release) rather than widen it if the metrics breach. This module
is the decision half of that.

WHY THE LOGIC IS HERE AND NOT IN .gitlab-ci.yml. The same reason the load-test
SLO check lives in the locustfile: a rule embedded in YAML cannot be unit
tested, and a rollback rule that has never been exercised is discovered to be
wrong during the incident it exists to handle. Here it is nine assertions in
tests/test_canary.py, and the CI job is a thin caller.

FAIL-CLOSED. A step whose metrics are missing, empty, or whose scrape raised is
an ABORT. "Prometheus returned nothing" and "the service is perfectly healthy"
are the same empty dict, and only one of them is safe to promote on — the same
absent-is-not-clean rule the PII gate and the security report follow.

Traffic shifting itself needs a real deployment target, which this project does
not yet have (see [[Infra-Dependent Work]]). The CI job says so plainly rather
than printing a fake ramp.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

# Traffic percentages, in order. From the reference pipeline's 5 -> 25 -> 50 -> 100.
CANARY_STEPS: tuple[int, ...] = (5, 25, 50, 100)

# Auto-rollback thresholds (course gate: error rate < 1%, tool failures < 2%).
MAX_ERROR_RATE = 0.01
MAX_TOOL_FAILURE_RATE = 0.02

# Metrics every step must supply. Absent -> abort.
_REQUIRED = ("errorRate", "toolFailureRate")


def evaluate_step(step: int, metrics: Mapping[str, float] | None) -> dict:
    """Decide whether the ramp may widen past `step`."""
    breaches: list[str] = []

    if not metrics:
        breaches.append(f"no metrics returned for the {step}% step (missing telemetry is not health)")
        return {"step": step, "verdict": "ABORT", "breaches": breaches, "metrics": {}}

    missing = [m for m in _REQUIRED if metrics.get(m) is None]
    if missing:
        breaches.append(f"missing metric(s) at {step}%: {', '.join(missing)}")

    error_rate = metrics.get("errorRate")
    if error_rate is not None and error_rate > MAX_ERROR_RATE:
        breaches.append(f"error rate {error_rate:.2%} > {MAX_ERROR_RATE:.2%} at {step}%")

    tool_failures = metrics.get("toolFailureRate")
    if tool_failures is not None and tool_failures > MAX_TOOL_FAILURE_RATE:
        breaches.append(f"tool failure rate {tool_failures:.2%} > {MAX_TOOL_FAILURE_RATE:.2%} at {step}%")

    return {
        "step": step,
        "verdict": "ABORT" if breaches else "PROCEED",
        "breaches": breaches,
        "metrics": dict(metrics),
    }


def run_ramp(metrics_for: Callable[[int], Mapping[str, float] | None]) -> dict:
    """Walk the ramp, stopping at the first breaching step.

    `metrics_for(step)` returns the observed metrics after traffic has been
    shifted to `step`%. A raising scrape is treated as a breach, not skipped:
    the one thing a rollout must never do on a broken signal is widen.
    """
    steps: list[dict] = []
    for step in CANARY_STEPS:
        try:
            observed = metrics_for(step)
        except Exception as exc:  # noqa: BLE001 - any scrape failure fails closed
            result = {
                "step": step,
                "verdict": "ABORT",
                "breaches": [f"metric source failed at {step}%: {type(exc).__name__}"],
                "metrics": {},
            }
        else:
            result = evaluate_step(step, observed)
        steps.append(result)
        if result["verdict"] == "ABORT":
            return {
                "verdict": "ABORTED",
                "abortedAt": step,
                "rollback": True,
                "steps": steps,
                "breaches": result["breaches"],
            }

    return {"verdict": "COMPLETED", "abortedAt": None, "rollback": False, "steps": steps, "breaches": []}
