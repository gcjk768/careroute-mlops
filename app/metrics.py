"""[MLOps] Operational + model telemetry (Prometheus exposition).

Course notes (Integrating & Deploying, "Logging and Monitoring") call for
inference latency, throughput, and error metrics scraped by Prometheus and
charted in Grafana. This module exposes exactly those as Prometheus metrics at
`/metrics` in main.py.

`prometheus_client` is an optional dependency: if it isn't installed every
helper here becomes a no-op and `/metrics` reports that it's disabled, so the
app still runs. Install it (see requirements.txt) to activate real metrics.
"""
from __future__ import annotations

from contextlib import contextmanager

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _ENABLED = True

    TRIAGE_REQUESTS = Counter(
        "careroute_triage_requests_total", "Triage requests received", ["outcome"]
    )
    GUARDRAIL_BLOCKS = Counter(
        "careroute_guardrail_blocks_total", "Requests blocked by the input guardrail"
    )
    OUTPUT_FLAGS = Counter(
        "careroute_output_flags_total", "LLM outputs suppressed by the output guardrail"
    )
    # [Agentic] Model-chosen tool calls through the registry gateway, by tool,
    # calling agent and outcome (ok / refused / invalid_arguments / error / unknown).
    TOOL_CALLS = Counter(
        "careroute_tool_calls_total",
        "Tool calls through the registry gateway, by tool, agent and outcome",
        ["tool", "agent", "outcome"],
    )
    # [AI-Security] Indirect injection: retrieved / remembered text dropped by the
    # untrusted-content screen before it could reach a prompt (LLM04/LLM08).
    UNTRUSTED_CONTENT_DROPS = Counter(
        "careroute_untrusted_content_drops_total",
        "Retrieved or remembered text dropped by the untrusted-content screen",
        ["channel"],
    )
    ESCALATIONS = Counter(
        "careroute_escalations_total", "Cases escalated to a clinician"
    )
    ROUTING_FALLBACKS = Counter(
        "careroute_routing_fallbacks_total",
        "Care-routing safe fallbacks, labelled by bounded reason",
        ["reason"],
    )
    ROUTING_DATA_FRESHNESS = Counter(
        "careroute_routing_data_freshness_total",
        "Hours-directory freshness observed during care routing",
        ["status"],
    )
    RATE_LIMITED = Counter(
        "careroute_rate_limited_total", "Requests rejected by the rate limiter (429)"
    )
    # [AI-Security] Abuse monitor: quarantines started, and requests refused
    # while a client was quarantined.
    ABUSE_EVENTS = Counter(
        "careroute_abuse_events_total",
        "Abuse-monitor events, by kind (quarantined / refused)",
        ["kind"],
    )
    PREDICT_LATENCY = Histogram(
        "careroute_model_predict_seconds", "Severity-model inference latency (incl. SHAP)"
    )
    # [MLOps][Agentic] Per-agent step observability: how long each worker took
    # and WHERE its answer came from (llm / model / fallback / deterministic).
    # Labelling by `source` lets an operator see, e.g., how often the classifier
    # is on the LLM path vs. the deterministic fallback and their latencies.
    AGENT_DURATION = Histogram(
        "careroute_agent_duration_seconds",
        "Per-agent step duration (seconds), labelled by agent and result source",
        ["agent", "source"],
    )
    # [Microservices] One count per gateway -> agent-container call, by outcome
    # (ok / error / rejected / breaker_open). The AgentDown alert reads this.
    AGENT_CALLS = Counter(
        "careroute_agent_calls_total",
        "Gateway calls to agent containers, by agent and outcome",
        ["agent", "outcome"],
    )
    # [MLOps] Pillar 3 — MODEL-level telemetry (not just system-level). The
    # prediction-class distribution + confidence histogram are the live signals
    # a Grafana panel / alert rule watches for serving skew: a shift in the
    # acuity mix or a slide toward low confidence is drift showing up in
    # production BEFORE labelled ground truth exists.
    PREDICTION_ACUITY = Counter(
        "careroute_model_predictions_total",
        "Severity-model predictions served, labelled by acuity class",
        ["acuity"],
    )
    PREDICT_CONFIDENCE = Histogram(
        "careroute_model_confidence",
        "Calibrated confidence of served predictions",
        buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    )
    MODEL_INFO = Gauge(
        "careroute_model_info",
        "Deployed model identity (value is always 1; the version is the label)",
        ["version"],
    )
    # [MLOps][HITL] Ground-truth loop: clinician escalation decisions labelled
    # by whether the clinician's final acuity AGREED with the model's. This is
    # live production accuracy on human-reviewed cases — the label source that
    # closes the monitor -> retrain loop.
    HITL_DECISIONS = Counter(
        "careroute_hitl_decisions_total",
        "Clinician escalation decisions, labelled by model agreement",
        ["agreement"],
    )
    # Export every label at 0 now: a series born on its first .inc() is first
    # scraped at 1, and increase() — the agreement tile — never sees that decision.
    for _agreement in ("agreed", "overridden", "unlabelled"):
        HITL_DECISIONS.labels(agreement=_agreement)

    # [MLOps][Agentic] Token + cost accounting. The efficiency half of the
    # course's "cost per task / p95 latency" gate. Cost is a separate counter
    # from tokens because an UNPRICED model still contributes tokens: adding a
    # zero to the cost counter for it would read as "this traffic was free".
    LLM_TOKENS = Counter(
        "careroute_llm_tokens_total",
        "LLM tokens consumed, by model and direction",
        ["model", "direction"],
    )
    LLM_COST_USD = Counter(
        "careroute_llm_cost_usd_total",
        "Estimated LLM spend in USD, by model (priced models only)",
        ["model"],
    )
    # [Agentic] LLM router: which tier each named task was routed to, and how
    # often the exact-match response cache answered instead of a provider.
    LLM_ROUTER_DECISIONS = Counter(
        "careroute_llm_router_decisions_total",
        "LLM router decisions, by task and model tier",
        ["task", "tier"],
    )
    LLM_CACHE = Counter(
        "careroute_llm_cache_total",
        "LLM response cache lookups, by task, outcome (hit / miss) and kind (exact / semantic)",
        ["task", "outcome", "kind"],
    )
    # [MLOps] LLM call latency — the other half of the efficiency gate, and the
    # last item left open when token/cost accounting closed Gap 2.
    #
    # AGENT_DURATION already times each agent, which is NOT the same measurement:
    # an agent that fell back to its deterministic path is fast and an agent that
    # waited out a dead provider's timeout is slow, and both are recorded under the
    # agent's name with nothing to separate them. Labelling by OUTCOME is what makes
    # the p95 readable — a timeout and a fast answer must not share a bucket, or the
    # number moves when the failure mix changes and nobody can say why.
    #
    # Buckets are stretched to 60s deliberately: the default Prometheus ladder tops
    # out at 10s, and every provider timeout in this app is longer than that, so a
    # timing-out chain would pile into +Inf and become unmeasurable exactly when the
    # measurement matters.
    LLM_LATENCY = Histogram(
        "careroute_llm_latency_seconds",
        "LLM call wall-clock latency, by model and outcome (ok / error / timeout)",
        ["model", "outcome"],
        buckets=(0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60),
    )
    # [MLOps] HARVEST — how complete each served answer was (see app/availability.py).
    # Yield needs no new instrument, only an honest denominator: it is computed in
    # the recording rules from careroute_triage_requests_total plus the requests
    # that never reached it (rate-limited, quarantined). Harvest cannot be, because
    # nothing in the existing counters knows whether the answer that was returned
    # was whole.
    ANSWER_HARVEST = Histogram(
        "careroute_answer_harvest",
        "Fraction of the APPLICABLE answer components a served case carried",
        buckets=(0.2, 0.4, 0.6, 0.8, 1.0),
    )
    ANSWER_COMPONENTS = Counter(
        "careroute_answer_components_total",
        "Answer components by state (present / degraded / not_applicable)",
        ["component", "state"],
    )
    # [Agentic][AI-Security] LLM gateway controls (app/llm.py). ROUTE: the model
    # each call actually went to and WHY (base tier, or the difficulty signal that
    # escalated it). GUARD: per-call input/output guardrail events — the request
    # guardrail screens the patient's words once; this screens every LLM call,
    # including prompts assembled from retrieved text, memory and tool output.
    LLM_ROUTE = Counter(
        "careroute_llm_route_total",
        "LLM calls by task, chosen model and routing reason (base / low_confidence / thin_evidence / rerun / override)",
        ["task", "model", "reason"],
    )
    LLM_GUARD = Counter(
        "careroute_llm_guard_total",
        "Per-LLM-call guardrail events, by task, stage (input / output) and action (redacted / blocked / flagged)",
        ["task", "stage", "action"],
    )
except Exception:  # prometheus_client not installed — degrade to no-ops  # noqa: BLE001 - telemetry must disable itself rather than break serving
    _ENABLED = False
    CONTENT_TYPE_LATEST = "text/plain"
    TRIAGE_REQUESTS = GUARDRAIL_BLOCKS = OUTPUT_FLAGS = ESCALATIONS = RATE_LIMITED = PREDICT_LATENCY = None
    UNTRUSTED_CONTENT_DROPS = TOOL_CALLS = ABUSE_EVENTS = None
    ROUTING_FALLBACKS = ROUTING_DATA_FRESHNESS = None
    AGENT_DURATION = AGENT_CALLS = None
    PREDICTION_ACUITY = PREDICT_CONFIDENCE = MODEL_INFO = HITL_DECISIONS = None
    LLM_TOKENS = LLM_COST_USD = LLM_ROUTER_DECISIONS = LLM_CACHE = LLM_LATENCY = None
    ANSWER_HARVEST = ANSWER_COMPONENTS = None
    LLM_ROUTE = LLM_GUARD = None

# [Agentic] Plan-and-Execute: which plan shape each case ran (full / emergency /
# red_flag / clarify / low_confidence / fallback — see agents/planner.py). A
# rising `fallback` share means the planner is proposing plans the transition
# graph rejects. Separate block so it merges cleanly with the one above.
PLAN_SHAPES = (
    Counter("careroute_plan_total", "Per-case orchestration plans, by plan shape", ["shape"])
    if _ENABLED else None
)

# [HITL] SLA + learning-loop metrics (app/store.py, app/memory/retrain_queue.py).
try:
    HITL_SLA_BREACHED = Counter(
        "careroute_hitl_sla_breached_total",
        "Escalations that passed CAREROUTE_HITL_SLA_MINUTES without a decision (counted once each)",
    )
    HITL_OPEN_OVERDUE = Gauge(
        "careroute_hitl_open_overdue", "Pending escalations currently past their SLA"
    )
    RETRAIN_QUEUE = Counter(
        "careroute_retrain_queue_total", "Clinician disagreements queued for retraining"
    )
except Exception:  # noqa: BLE001 - prometheus_client absent: no-ops, same as above
    HITL_SLA_BREACHED = HITL_OPEN_OVERDUE = RETRAIN_QUEUE = None


def set_gauge(gauge, value: float) -> None:
    """Set a gauge if metrics are enabled (no-op otherwise)."""
    if gauge is not None:
        gauge.set(value)


def enabled() -> bool:
    return _ENABLED


def inc(counter, **labels) -> None:
    """Increment a counter if metrics are enabled (no-op otherwise)."""
    if counter is None:
        return
    (counter.labels(**labels) if labels else counter).inc()


def observe_prediction(acuity: str, confidence: float, model_version: str) -> None:
    """[MLOps] Record one served prediction: class-distribution counter,
    confidence histogram, and the deployed model version as an info gauge.
    No-op when prometheus_client is unavailable."""
    if PREDICTION_ACUITY is None or PREDICT_CONFIDENCE is None or MODEL_INFO is None:
        return
    PREDICTION_ACUITY.labels(acuity=str(acuity or "unknown")).inc()
    PREDICT_CONFIDENCE.observe(min(max(float(confidence), 0.0), 1.0))
    if model_version:
        MODEL_INFO.labels(version=str(model_version)).set(1)


def observe_hitl_decision(agreement: str) -> None:
    """[MLOps][HITL] Record a clinician escalation decision by model agreement
    ('agreed' / 'overridden' / 'unlabelled'). No-op without prometheus_client."""
    if HITL_DECISIONS is None:
        return
    HITL_DECISIONS.labels(agreement=str(agreement or "unlabelled")).inc()


def observe_agent(agent: str, source: str, seconds: float) -> None:
    """[MLOps][Agentic] Record a single agent step's duration, labelled by the
    agent name and the source of its answer (llm/model/fallback/deterministic).
    No-op when prometheus_client is unavailable."""
    if AGENT_DURATION is None:
        return
    AGENT_DURATION.labels(agent=agent, source=str(source or "unknown")).observe(max(0.0, seconds))


def observe_agent_call(agent: str, outcome: str) -> None:
    """[Microservices] Count one gateway -> agent call. No-op without prometheus_client."""
    if AGENT_CALLS is None:
        return
    AGENT_CALLS.labels(agent=agent, outcome=outcome).inc()


def observe_llm_usage(model: str, prompt_tokens: int, completion_tokens: int,
                      cost_usd: float | None) -> None:
    """[MLOps][Agentic] Record one LLM call's token use and, when the model has
    a published rate, its estimated cost. `cost_usd=None` means UNPRICED: the
    tokens are still counted, the money deliberately is not. No-op without
    prometheus_client."""
    if LLM_TOKENS is None:
        return
    name = str(model or "unknown")
    if prompt_tokens:
        LLM_TOKENS.labels(model=name, direction="prompt").inc(max(0, prompt_tokens))
    if completion_tokens:
        LLM_TOKENS.labels(model=name, direction="completion").inc(max(0, completion_tokens))
    if cost_usd is not None and LLM_COST_USD is not None and cost_usd > 0:
        LLM_COST_USD.labels(model=name).inc(cost_usd)


def observe_llm_latency(model: str, seconds: float, outcome: str = "ok") -> None:
    """[MLOps] Record one LLM call's wall-clock latency and how it ended.

    Called for FAILED calls too, and that is the point: a provider chain's cost is
    dominated by the attempts that time out, and a latency metric that only records
    successes reports a system getting faster as it gets sicker. No-op without
    prometheus_client."""
    if LLM_LATENCY is None:
        return
    LLM_LATENCY.labels(model=str(model or "unknown"),
                       outcome=str(outcome or "unknown")).observe(max(0.0, seconds))


def observe_harvest(state) -> float | None:
    """[MLOps] Record one served answer's HARVEST and its component states.

    Returns the score so a caller can log it. Both the per-case histogram and
    the per-component counter are recorded, because they answer different
    questions: the histogram says how bad the degradation was, the counter says
    WHICH part degraded — and an operator needs the second one to act.
    No-op without prometheus_client; never raises, since a telemetry fault must
    not fail a triage that has already succeeded.
    """
    try:
        from .availability import harvest

        score, parts = harvest(state)
    except Exception:  # noqa: BLE001 - telemetry only; a triage already succeeded
        return None
    if ANSWER_HARVEST is None or ANSWER_COMPONENTS is None:
        return score
    for component, present in parts.items():
        label = "not_applicable" if present is None else ("present" if present else "degraded")
        ANSWER_COMPONENTS.labels(component=component, state=label).inc()
    if score is not None:
        ANSWER_HARVEST.observe(score)
    return score


@contextmanager
def timer(histogram):
    """Observe an elapsed-time histogram, or do nothing if metrics are off.

    Uses the histogram's own `.time()` context manager (monotonic clock), so no
    forbidden wall-clock call is made here."""
    if histogram is None:
        yield
        return
    with histogram.time():
        yield


def render() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    if not _ENABLED:
        return b"# prometheus_client not installed; metrics disabled\n", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST
