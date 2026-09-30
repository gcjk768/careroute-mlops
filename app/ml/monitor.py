"""[MLOps] Pillar 3 — model + data monitoring (course: "Monitoring and Logging").

    python -m app.ml.monitor

Builds a REFERENCE set (the validation-like distribution the model was signed off
on) and a CURRENT set (a shifted "production" sample — the same shift the in-code
PSI audit uses) and emits a monitoring report covering the metrics the module
names: data drift (PSI/KS), target drift, and classification performance
(accuracy / precision / recall / F1).

Primary path uses **Evidently AI** (the drift tool named in the course tool
table). If Evidently is not installed it degrades to a built-in PSI + accuracy
report, so the job always produces output — mirroring the project's "optional
dep, never fatal" contract. The report records WHICH backend ran and WHY
(`backend` + `backendStatus`), because "not installed" and "installed but
broken" are different problems and were previously logged identically.

Headline data drift is the WORST per-feature PSI (see fairness.data_drift);
`dataDriftPSIMean` / `dataDriftFeatureShare` / `driftedFeatures` carry the
breadth and the culprits.

The monitor -> retrain loop is CLOSED here (Pillar 4 / CT):

  * `CAREROUTE_MONITOR_SOURCE=live` points `current` at REAL scored traffic
    from the inference log (`app/ml/inference_log.py`) when enough rows exist,
    instead of the simulated shift.
  * After the report, the headline drift value is compared against
    `CAREROUTE_DRIFT_THRESHOLD`; on a breach, if a GitLab trigger token is
    configured (`CAREROUTE_PIPELINE_TRIGGER_TOKEN` + CI vars), the pipeline
    API is called to fire a retrain (`train:model` runs on the new pipeline).
    A pipeline that was ITSELF started by that trigger never re-triggers
    (loop guard), so drift -> retrain converges instead of cascading.

Outputs (under backend/monitoring/, override with CAREROUTE_MONITOR_DIR):
  * drift_report.html   — human-readable Evidently dashboard (if available)
  * drift_report.json   — machine-readable metrics (drift share, drift gate)
"""
from __future__ import annotations

import json
import logging
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from . import data, fairness
from .features import FEATURE_NAMES, zero_coverage_rate
from .model import get_model

logger = logging.getLogger("careroute.ml.monitor")

_DEFAULT_OUT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "monitoring")
)

# [MLOps] Seed for the REFERENCE ("signed-off") sample. Deliberately disjoint
# from every seed the model has already seen: 42 is the TRAINING set (model.py
# build_artifact, n=6000), 7 is the shifted "production" sample, 123 the
# calibration set and 99 the fairness-gate set. The reference used to be
# seed=42/n=2000 — VERIFIED byte-identical to the first 2000 rows of the seed-42
# training data, ~75% of which land in the train split — so `referenceAccuracy`
# was largely being measured on memorised rows. Measured against the served
# artifact: seed 42 -> 0.9465, seed 2024 -> 0.9145, i.e. the old baseline read
# 3.2 points high, and every comparison against it understated degradation.
# A monitoring baseline has to be data the model has never seen.
_REFERENCE_SEED = 2024


def out_dir() -> str:
    return os.environ.get("CAREROUTE_MONITOR_DIR", _DEFAULT_OUT)


def _live_report(dst: str) -> dict | None:
    """[MLOps] LIVE-traffic monitoring: when CAREROUTE_MONITOR_SOURCE=live and
    the inference log holds enough rows, compare the reference distribution
    against REAL served predictions. No ground-truth labels exist for live
    traffic, so this reports data drift + PREDICTION drift (the class mix the
    model is emitting) — performance metrics need the HITL ground-truth loop.
    Returns None (fall through to the synthetic paths) when not in live mode
    or the log is too small to be statistically meaningful."""
    if os.environ.get("CAREROUTE_MONITOR_SOURCE", "").lower() != "live":
        return None
    from . import inference_log
    from .data import ACUITY_INDEX_TO_CODE

    live = inference_log.load_recent()
    min_rows = int(os.environ.get("CAREROUTE_MONITOR_MIN_ROWS", "200"))
    if live is None or len(live[0]) < min_rows:
        n = 0 if live is None else len(live[0])
        logger.warning(
            "live monitoring requested but inference log has %d rows (< %d) — "
            "falling back to the synthetic shifted sample", n, min_rows,
        )
        return None

    X_cur, acuity_codes = live
    code_to_idx = {c: i for i, c in enumerate(ACUITY_INDEX_TO_CODE)}
    p_cur = np.array([code_to_idx.get(c, 2) for c in acuity_codes])

    X_ref, y_ref, _ = data.generate_dataset(n=2000, seed=_REFERENCE_SEED)
    p_ref = get_model().model.predict(X_ref)

    os.makedirs(dst, exist_ok=True)
    drift = fairness.data_drift_detail(X_ref, X_cur)
    metrics = {
        "backend": "builtin-psi",
        "currentSource": f"live-inference-log ({len(X_cur)} rows)",
        # Headline drift is the WORST feature's PSI, not the average across all
        # 29 — see fairness.data_drift for why the mean hid real shifts. The
        # breadth (how many features moved) is reported alongside it.
        "dataDriftPSI": drift["max"],
        "dataDriftPSIMean": drift["mean"],
        "dataDriftFeatureShare": drift["driftedShare"],
        "driftedFeatures": [FEATURE_NAMES[j] for j in drift["driftedFeatures"]],
        # Prediction drift: has the CLASS MIX the model emits shifted vs the
        # reference? (No live labels -> target drift is not computable.)
        "predictionDriftPSI": fairness.target_drift(p_ref, p_cur),
        # [MLOps] MODEL COVERAGE on REAL traffic — the most useful instance of
        # this metric. Each point is a live case that lit up ZERO symptom
        # features, i.e. one the model had no signal for. A rate climbing above
        # the reference baseline means patients are describing something
        # FEATURE_KEYWORDS does not cover; the fix is a new category + retrain.
        # This is the monitor that would have surfaced the minor-trauma blind
        # spot from production instead of leaving it to evaluation E5.
        "zeroCoverageRate": zero_coverage_rate(X_cur),
        "zeroCoverageRateReference": zero_coverage_rate(X_ref),
        "performance": {
            "referenceAccuracy": round(float(accuracy_score(y_ref, p_ref)), 4),
            **_labelled_performance(),
        },
    }
    return metrics


def _labelled_performance() -> dict:
    """[MLOps][XRAI] Production metrics on the cases a clinician actually
    labelled — the HITL ground-truth loop closed. Joins the inference log to
    `ground_truth.jsonl` by case id (inference_log.load_labelled) and reports
    accuracy, red-flag recall, per-subgroup accuracy, the accuracy gap and the
    Equal Opportunity gap, computed with the SAME fairness functions the
    training audit uses. Also the share of predictions served with a DEFAULTED
    age band: on those rows the age-aware mitigation could not fire, so a
    fairness number over live traffic must say how much of it was age-blind.
    Returns {"labelled": None, "note": ...} until enough labels exist."""
    from . import inference_log

    min_labelled = int(os.environ.get("CAREROUTE_MONITOR_MIN_LABELLED", "20"))
    joined = inference_log.load_labelled()
    n = 0 if joined is None else len(joined[1])
    if joined is None or n < min_labelled:
        return {
            "labelled": None,
            "note": f"labelled live sample too small ({n} < {min_labelled}) — "
                    "clinician decisions with a final acuity feed ground_truth.jsonl",
        }
    _x, y_true, y_pred, groups, age_provided = joined
    sub = fairness.subgroup_accuracies(y_true, y_pred, groups)
    return {
        "labelled": {
            "n": int(n),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "redFlagRecall": fairness.red_flag_recall(y_true, y_pred),
            "subgroups": sub,
            "fairnessGap": fairness.fairness_gap(sub),
            "equalOpportunityGap": fairness.equal_opportunity(y_true, y_pred, groups)["equalOpportunityGap"],
            "ageDefaultedShare": round(float(1.0 - age_provided.mean()), 4),
        },
        "note": f"labelled by clinicians via the HITL decision endpoint (n={n})",
    }


def _datasets():
    """Reference (stable) vs current (shifted 'production') samples + the served
    model's predictions on each — the inputs every monitoring backend needs."""
    X_ref, y_ref, _ = data.generate_dataset(n=2000, seed=_REFERENCE_SEED)  # reference (unseen — see above)
    X_cur, y_cur, _ = data.generate_dataset(n=2000, seed=7, shift=True)  # "production"
    model = get_model().model                                           # served RF
    return (
        X_ref, y_ref, model.predict(X_ref),
        X_cur, y_cur, model.predict(X_cur),
    )


def _import_evidently():
    """Import the Evidently API, DISTINGUISHING "absent" from "broken".

    Returns (modules | None, status-string). The status is recorded in the drift
    report JSON so the artifact says which backend actually ran and why.

    Why this matters: the old code caught bare `Exception` and logged every
    outcome as `INFO: evidently not available`. In this project's venv
    `import evidently.report` raises a TypeError (a pydantic/py3.13 class-layout
    conflict), NOT ModuleNotFoundError — so the NAMED PRIMARY BACKEND had never
    once run, while the log claimed the library simply wasn't installed. A
    genuinely absent optional dep is an INFO-level non-event; an installed
    dependency that explodes on import is a WARNING with a traceback.
    (The evidently PIN is deliberately left alone here — changing it needs a
    verification environment; see the report notes.)"""
    try:
        import pandas as pd
        from evidently import ColumnMapping
        from evidently.metric_preset import ClassificationPreset, DataDriftPreset
        from evidently.report import Report
    except ModuleNotFoundError as exc:
        logger.info("evidently not installed (%s); using built-in PSI fallback", exc.name)
        return None, f"evidently not installed ({exc.name}); built-in PSI fallback"
    except Exception as exc:  # broken optional dep must not kill the job, but must be LOUD
        logger.warning("evidently is installed but failed to import: %s", exc, exc_info=True)
        return None, (
            "evidently is installed but failed to import "
            f"({type(exc).__name__}: {exc}); built-in PSI fallback"
        )
    return (pd, ColumnMapping, ClassificationPreset, DataDriftPreset, Report), "evidently"


def _evidently_report(dst: str) -> tuple[dict | None, str]:
    """Try the Evidently path (course-named tool). Returns (metrics, status);
    metrics is None if Evidently is unavailable / its API differs, so the caller
    falls back — and `status` explains which, for the report JSON."""
    mods, status = _import_evidently()
    if mods is None:
        return None, status
    pd, ColumnMapping, ClassificationPreset, DataDriftPreset, Report = mods

    X_ref, y_ref, p_ref, X_cur, y_cur, p_cur = _datasets()

    def _frame(X, y, p):
        df = pd.DataFrame(X, columns=FEATURE_NAMES)
        df["target"] = y
        df["prediction"] = p
        return df

    ref_df, cur_df = _frame(X_ref, y_ref, p_ref), _frame(X_cur, y_cur, p_cur)
    mapping = ColumnMapping(
        target="target", prediction="prediction", numerical_features=list(FEATURE_NAMES)
    )
    try:
        report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
        report.run(reference_data=ref_df, current_data=cur_df, column_mapping=mapping)
        os.makedirs(dst, exist_ok=True)
        report.save_html(os.path.join(dst, "drift_report.html"))
        as_dict = report.as_dict()
    except Exception as exc:  # optional monitoring dependency; falls back to the built-in metric
        logger.warning("evidently run failed; using built-in PSI fallback", exc_info=True)
        return None, (
            f"evidently imported but the report run failed ({type(exc).__name__}: {exc}); "
            "built-in PSI fallback"
        )

    # Pull the headline drift numbers out of Evidently's result structure.
    drift = {}
    for m in as_dict.get("metrics", []):
        if m.get("metric") == "DatasetDriftMetric":
            res = m.get("result", {})
            drift = {
                "datasetDrift": res.get("dataset_drift"),
                "driftShare": res.get("share_of_drifted_columns"),
                "driftedColumns": res.get("number_of_drifted_columns"),
            }
    return {
        "backend": "evidently",
        "backendStatus": status,
        **drift,
        # [MLOps] Model coverage — see zero_coverage_rate() in features.py.
        # Reported on every backend so the signal does not disappear depending
        # on which monitoring library happens to be installed.
        "zeroCoverageRate": zero_coverage_rate(X_cur),
        "zeroCoverageRateReference": zero_coverage_rate(X_ref),
    }, status


def _psi_fallback(dst: str) -> dict:
    """Built-in monitoring report when Evidently is absent: PSI data drift +
    target drift + accuracy/F1 on reference vs shifted current. main() writes
    the HTML dashboard so the `drift_report.html` artifact always exists
    (parity with Evidently's HTML — avoids a missing-artifact warning in CI)."""
    X_ref, y_ref, p_ref, X_cur, y_cur, p_cur = _datasets()
    os.makedirs(dst, exist_ok=True)
    drift = fairness.data_drift_detail(X_ref, X_cur)
    metrics = {
        "backend": "builtin-psi",
        # Headline drift is the WORST feature's PSI, not the average across all
        # 29 — see fairness.data_drift. `dataDriftFeatureShare` carries the
        # breadth the mean used to (badly) stand in for, and `driftedFeatures`
        # names the culprits so the report points somewhere.
        "dataDriftPSI": drift["max"],
        "dataDriftPSIMean": drift["mean"],
        "dataDriftFeatureShare": drift["driftedShare"],
        "driftedFeatures": [FEATURE_NAMES[j] for j in drift["driftedFeatures"]],
        "targetDriftPSI": fairness.target_drift(y_ref, y_cur),
        # [MLOps] Model coverage — see zero_coverage_rate() in features.py.
        "zeroCoverageRate": zero_coverage_rate(X_cur),
        "zeroCoverageRateReference": zero_coverage_rate(X_ref),
        "performance": {
            "referenceAccuracy": round(float(accuracy_score(y_ref, p_ref)), 4),
            "currentAccuracy": round(float(accuracy_score(y_cur, p_cur)), 4),
            "referenceF1Macro": round(float(f1_score(y_ref, p_ref, average="macro")), 4),
            "currentF1Macro": round(float(f1_score(y_cur, p_cur, average="macro")), 4),
        },
    }
    return metrics  # HTML is written once in main(), after the drift gate


def _write_html(dst: str, m: dict) -> None:
    """Minimal, dependency-free HTML drift dashboard for the non-Evidently
    paths (PSI fallback + live-traffic). Renders whatever metrics are present."""
    rows = [(k, v) for k, v in m.items() if not isinstance(v, dict) and k != "backend"]
    rows += [(f"performance / {k}", v) for k, v in (m.get("performance") or {}).items()]
    trs = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>CareRoute drift report</title><style>"
        "body{font-family:system-ui,sans-serif;margin:2rem;color:#111}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.4rem .8rem;text-align:left}"
        ".muted{color:#666}</style></head><body>"
        "<h2>CareRoute model monitoring — data drift &amp; performance</h2>"
        "<p class='muted'>Backend: built-in PSI (Evidently unavailable). Reference vs shifted "
        "&quot;production&quot; sample.</p>"
        f"<table><tr><th>Metric</th><th>Value</th></tr>{trs}</table>"
        "<p class='muted'>A shifted sample is expected to drift; in production this signal "
        "would trigger evaluation/retraining (monitor &rarr; CT loop).</p></body></html>"
    )
    with open(os.path.join(dst, "drift_report.html"), "w", encoding="utf-8") as fh:
        fh.write(html)


# Secondary retrain triggers. Target drift reuses the conventional PSI cut; the
# accuracy drop is an absolute fall against the reference split, well clear of
# the ~3pp sampling wobble a healthy monitor shows run to run.
_TARGET_DRIFT_THRESHOLD = 0.25
_ACCURACY_DROP_THRESHOLD = 0.10


def _retrain_reason(gate: dict) -> str:
    """`drift:<metric>=<value>` for every signal that breached, so the triggered
    pipeline records WHY it was retrained."""
    breached = [s for s in gate.get("signals") or [] if s.get("breached")]
    if not breached:
        return f"drift:{gate['metric']}={gate['value']}"
    return "drift:" + ",".join(f"{s['metric']}={s['value']}" for s in breached)


def _secondary_signal(metric: str, value, threshold: float, breach_reason: str) -> dict:
    """A retrain signal that fails OPEN when absent — see `_drift_gate`."""
    if value is None:
        return {"metric": metric, "value": None, "threshold": threshold,
                "breached": False, "reason": "not measured by this backend"}
    breached = float(value) >= threshold
    return {"metric": metric, "value": round(float(value), 4), "threshold": threshold,
            "breached": breached, "reason": breach_reason if breached else "within tolerance"}


def _drift_gate(metrics: dict) -> dict:
    """[MLOps][CT] Compare the headline drift value to the retrain threshold.
    Evidently reports a drifted-column SHARE in [0,1] (default gate 0.5); the
    PSI backends report a PSI value (default gate 0.25 — the conventional
    'significant shift' cut, shared with fairness.DRIFT_PSI_THRESHOLD).
    `CAREROUTE_DRIFT_THRESHOLD` overrides either.

    A MISSING metric is a BREACH, not a pass. The old expression was
    `value is not None and value >= threshold`, so a backend that produced no
    headline number scored a clean gate — the quietest possible way for drift
    detection to be switched off, and indistinguishable in the report from
    "measured, no drift". Now it breaches with an explicit reason and main()
    exits non-zero, because a monitor that cannot measure is broken, not calm."""
    if metrics.get("backend") == "evidently":
        value, kind, default = metrics.get("driftShare"), "driftShare", 0.5
    else:
        value, kind, default = (
            metrics.get("dataDriftPSI"), "dataDriftPSI", fairness.DRIFT_PSI_THRESHOLD,
        )
    threshold = float(os.environ.get("CAREROUTE_DRIFT_THRESHOLD", default))
    if value is None:
        logger.error(
            "drift metric %s missing from the %r report — failing the gate loudly",
            kind, metrics.get("backend"),
        )
        return {
            "metric": kind, "value": None, "threshold": threshold,
            "breached": True, "reason": "drift metric unavailable", "signals": [],
        }
    breached = float(value) >= threshold

    # [MLOps][CT] The CI/CD deck names four reasons to retrain: dips in accuracy,
    # data drift, TARGET drift and CONCEPT drift. Only the first of the drifts
    # gated here; the other signals were computed, published in the report, and
    # then ignored — so a pipeline whose label mix had shifted, or whose accuracy
    # had collapsed against the reference, still scored a clean gate as long as
    # the INPUT features looked familiar. Those are different failures with
    # different causes, and each one on its own justifies a retrain.
    #
    # Secondary signals fail OPEN when absent, deliberately: the headline metric
    # already fails closed above, and treating "target drift was never computed"
    # as "the labels have drifted" would fire a retrain on every backend that
    # does not produce it.
    signals = [{
        "metric": kind, "value": value, "threshold": threshold, "breached": breached,
        "reason": "above threshold" if breached else "below threshold",
    }]
    signals.append(_secondary_signal(
        "targetDriftPSI", metrics.get("targetDriftPSI"), _TARGET_DRIFT_THRESHOLD,
        "label distribution shifted",
    ))
    perf = metrics.get("performance") or {}
    ref_acc, cur_acc = perf.get("referenceAccuracy"), perf.get("currentAccuracy")
    drop = (float(ref_acc) - float(cur_acc)) if (ref_acc is not None and cur_acc is not None) else None
    signals.append(_secondary_signal(
        "accuracyDrop", drop, _ACCURACY_DROP_THRESHOLD,
        "accuracy fell against the reference (concept drift)",
    ))

    return {
        # Headline stays dataDriftPSI so the retrain reason and the printed line
        # read the same as before.
        "metric": kind, "value": value, "threshold": threshold,
        "breached": any(s["breached"] for s in signals),
        "reason": "above threshold" if breached else "below threshold",
        "signals": signals,
    }


def _trigger_retrain(gate: dict) -> str:
    """[MLOps][CT] Close the loop: on a drift breach, fire a new pipeline via
    the GitLab trigger API so `train:model` retrains + re-gates + re-registers.

    Requires CI context (CI_API_V4_URL / CI_PROJECT_ID) and a pipeline trigger
    token in `CAREROUTE_PIPELINE_TRIGGER_TOKEN` (Settings -> CI/CD -> Pipeline
    trigger tokens). LOOP GUARD: a pipeline that was itself started by a
    trigger never re-triggers, so one breach yields exactly one retrain.
    Returns a human-readable status line for the summary."""
    if os.environ.get("CI_PIPELINE_SOURCE") == "trigger":
        return "skipped (this pipeline WAS the triggered retrain — loop guard)"
    api = os.environ.get("CI_API_V4_URL")
    project = os.environ.get("CI_PROJECT_ID")
    token = os.environ.get("CAREROUTE_PIPELINE_TRIGGER_TOKEN")
    if not (api and project and token):
        return "skipped (set CAREROUTE_PIPELINE_TRIGGER_TOKEN + run in CI to auto-retrain)"
    # Only ever call the TLS GitLab API endpoint — CI_API_V4_URL is always https;
    # reject anything else so a mis-set variable can't redirect the trigger.
    if not api.startswith("https://"):
        return "skipped (CI_API_V4_URL is not an https endpoint)"
    ref = os.environ.get("CAREROUTE_RETRAIN_REF") or os.environ.get("CI_DEFAULT_BRANCH", "main")
    try:
        from urllib.parse import urlencode
        from urllib.request import urlopen

        payload = urlencode({
            "token": token,
            "ref": ref,
            # Name the signal(s) that ACTUALLY breached, not just the headline
            # metric. With three triggers now feeding this gate, "drift:
            # dataDriftPSI=0.01" on a retrain fired by target drift would send
            # whoever investigates to the wrong place.
            "variables[RETRAIN_REASON]": _retrain_reason(gate),
        }).encode("utf-8")
        # `api` is the trusted, https-validated GitLab CI_API_V4_URL, not user input
        # (https-only guard above), so this is not an SSRF / file:// sink.
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected, bandit.B310-1
        with urlopen(f"{api}/projects/{project}/trigger/pipeline", payload, timeout=30) as resp:  # noqa: S310  # nosec B310
            body = json.loads(resp.read().decode("utf-8"))
        return f"TRIGGERED retrain pipeline #{body.get('id')} on '{ref}'"
    except Exception as exc:  # noqa: BLE001 - optional monitoring dependency; falls back to the built-in metric
        logger.warning("retrain trigger failed: %s", exc, exc_info=False)
        return f"FAILED to trigger retrain ({type(exc).__name__}) — investigate manually"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    dst = out_dir()
    metrics = _live_report(dst)
    if metrics is None:
        metrics, backend_status = _evidently_report(dst)
        if metrics is None:
            metrics = _psi_fallback(dst)
            # Which backend actually ran, and WHY the named primary one didn't —
            # recorded in the artifact so "evidently" being in the tool table
            # can't be mistaken for evidently having run.
            metrics["backendStatus"] = backend_status

    # [MLOps][CT] The drift gate + (conditional) retrain trigger — the step that
    # turns this report from an open loop into a closed monitor -> CT loop.
    gate = _drift_gate(metrics)
    metrics["driftGate"] = gate
    unavailable = gate.get("reason") == "drift metric unavailable"
    if unavailable:
        # Never retrain off a broken monitor: a missing number is not evidence
        # the data moved, so the correct response is to fail, not to retrain.
        retrain_status = "not attempted (drift metric unavailable — fix the monitor)"
    elif gate["breached"]:
        retrain_status = _trigger_retrain(gate)
    else:
        retrain_status = "not needed (below threshold)"
    metrics["retrain"] = retrain_status

    if metrics.get("backend") != "evidently":
        _write_html(dst, metrics)
    with open(os.path.join(dst, "drift_report.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n=== CareRoute model monitoring ===")
    print(f"  backend : {metrics['backend']}")
    if metrics.get("backendStatus") and metrics["backend"] != "evidently":
        print(f"  why     : {metrics['backendStatus']}")
    print(f"  source  : {metrics.get('currentSource', 'synthetic shifted sample')}")
    print(f"  output  : {dst}")
    print(f"  metrics : {json.dumps({k: v for k, v in metrics.items() if k not in ('backend', 'driftGate', 'retrain')})}")
    print(f"  drift   : {gate['metric']}={gate['value']} vs threshold {gate['threshold']}"
          f" -> {'BREACHED' if gate['breached'] else 'ok'} ({gate.get('reason', '')})")
    # [MLOps] Model coverage, printed beside drift because it answers a question
    # drift cannot: not "has the input distribution moved?" but "is the model
    # blind to any of it?". Deliberately reported, never gated — see
    # features.has_no_feature_coverage for why it must not force escalation.
    cov, cov_ref = metrics.get("zeroCoverageRate"), metrics.get("zeroCoverageRateReference")
    if cov is not None:
        delta = "" if cov_ref is None else f" (reference {cov_ref})"
        note = " <- investigate: live traffic the model has no features for" if (
            cov_ref is not None and cov > cov_ref + 0.05
        ) else ""
        print(f"  coverage: zero-feature rate {cov}{delta}{note}")
    print(f"  retrain : {retrain_status}")
    if unavailable:
        # NON-ZERO exit: a breach on a real number is an expected, reportable
        # outcome (the shifted sample is meant to drift, and the CT loop handles
        # it), but a monitor that produced NO drift number is a malfunction and
        # has to be visible as a job failure, not a green report saying "ok".
        print(
            f"MONITOR FAILED: no {gate['metric']} value was produced by the "
            f"'{metrics['backend']}' backend — drift is UNMEASURED, not absent.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
