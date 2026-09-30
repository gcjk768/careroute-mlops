"""[MLOps] Standalone training entrypoint — the SPLIT between training and serving.

    python -m app.ml.train

Serving (`app/ml/model.get_model`) only ever LOADS a persisted artifact; this
module is the one place that BUILDS one. It:

  1. builds the deployed model end-to-end (seeded, reproducible),
  2. VALIDATES it against release-gate thresholds (min accuracy, red-flag
     recall, an ABSOLUTE subgroup-fairness ceiling and a calibration-ECE
     ceiling) — a failing gate aborts the release with a non-zero exit code,
  3. PERSISTS a versioned, content-addressed joblib artifact + a sidecar SHA-256
     integrity hash (tamper-evidence / rollback), and
  4. logs params + metrics and REGISTERS the model in the MLflow Model Registry
     ("CareRouteTriageRF") when MLflow is installed — MLflow is OPTIONAL, its
     absence is logged and ignored, never fatal.

Run it in CI/CD before deploy so the container ships a validated artifact that
`get_model()` loads at boot instead of retraining on every process start.
"""
from __future__ import annotations

import logging
import sys

from . import lifecycle, run_context
from .model import (
    MAX_CALIBRATION_ECE,
    MAX_EQUAL_OPPORTUNITY_GAP,
    MAX_FAIRNESS_GAP,
    MIN_ACCURACY,
    MIN_EXPLANATION_AGREEMENT,
    MIN_RED_FLAG_RECALL,
    MIN_SUBGROUP_SEVERE_SUPPORT,
    build_artifact,
    save_artifact,
)

logger = logging.getLogger("careroute.ml.train")


class ReleaseGateError(AssertionError):
    """Raised when a freshly trained model fails a release-gate threshold."""


def subgroup_red_flag_shortfalls(audit: dict) -> list[str]:
    """[Responsible-AI][Safety] Every age-band x sex subgroup with at least
    MIN_SUBGROUP_SEVERE_SUPPORT held-out emergencies must reach the population
    red-flag floor, on the raw forest AND on the served pipeline (calibrated +
    raise-only thresholds). Returns one message per shortfall; [] passes.

    The population gate alone shipped 4e409ad346f3 with 65+ Male at 0.897.
    A missing block raises: no evidence is not a pass."""
    block = audit.get("subgroupRedFlagRecall")
    if not block:
        raise ReleaseGateError(
            "audit carries no subgroupRedFlagRecall block — a model with no per-subgroup "
            "red-flag evidence is not release-ready"
        )
    out = []
    for pipeline, label in (("raw", "raw forest"), ("served", "served pipeline")):
        for group, v in sorted((block.get(pipeline) or {}).items()):
            if v["nSevere"] >= MIN_SUBGROUP_SEVERE_SUPPORT and v["recall"] < MIN_RED_FLAG_RECALL:
                out.append(f"{group} red-flag recall {v['recall']:.3f} on the {label} "
                           f"(n={v['nSevere']}) below {MIN_RED_FLAG_RECALL}")
    return out


def validate(audit: dict) -> None:
    """[MLOps] Release gate: block persistence/registration of a model that does
    not clear the minimum accuracy AND the safety-critical red-flag recall."""
    acc = audit["overallAccuracy"]
    recall = audit["redFlagRecall"]
    if acc < MIN_ACCURACY:
        raise ReleaseGateError(
            f"overall accuracy {acc:.3f} below release-gate minimum {MIN_ACCURACY}"
        )
    if recall < MIN_RED_FLAG_RECALL:
        raise ReleaseGateError(
            f"red-flag recall {recall:.3f} below release-gate minimum {MIN_RED_FLAG_RECALL}"
        )
    # ...and in every subgroup, not just on average (see subgroup_red_flag_shortfalls).
    shortfalls = subgroup_red_flag_shortfalls(audit)
    if shortfalls:
        raise ReleaseGateError("; ".join(shortfalls))
    # The fairness mitigation must never make the subgroup gap worse.
    if audit["fairnessGapAfter"] > audit["fairnessGapBefore"] + 1e-9:
        raise ReleaseGateError(
            f"fairness gap widened after mitigation: "
            f"{audit['fairnessGapBefore']:.3f} -> {audit['fairnessGapAfter']:.3f}"
        )
    # ...and it must clear an ABSOLUTE ceiling, not just beat the straw man.
    # The relative check above compares against the AGE-BLIND baseline, whose gap
    # is ~0.549 — so a real regression from 0.17 to 0.45 satisfied it and was
    # only caught by ai-security:fairness-gate, several stages later, after this
    # artifact had already been persisted and MLflow-registered. Same 0.35 that
    # gate uses (one shared constant in model.py), enforced BEFORE persistence.
    gap = audit["fairnessGapAfter"]
    if gap > MAX_FAIRNESS_GAP:
        raise ReleaseGateError(
            f"subgroup fairness gap {gap:.3f} exceeds release-gate maximum {MAX_FAIRNESS_GAP}"
        )
    # [XRAI] Calibration ceiling. `confidence` drives the HITL escalation gate,
    # so an over-confident model routes patients wrongly; ECE was measured and
    # printed but never gated anywhere in the pipeline until now.
    ece = (audit.get("calibration") or {}).get("ece")
    if ece is None:
        raise ReleaseGateError(
            "audit carries no calibration.ece — a model with no calibration "
            "evidence is not release-ready"
        )
    if ece > MAX_CALIBRATION_ECE:
        raise ReleaseGateError(
            f"calibration ECE {ece:.4f} exceeds release-gate maximum {MAX_CALIBRATION_ECE}"
        )
    # [XRAI] Explanation-agreement floor. The SHAP bars a patient sees are
    # only evidence if an independent attribution method points at the same
    # features; a model whose explanations two methods disagree about is not
    # release-ready, and one with no agreement evidence at all is not either.
    agreement = (audit.get("explanationAgreement") or {}).get("reportedTop1Agreement")
    if agreement is None:
        raise ReleaseGateError(
            "audit carries no explanation agreement block — a model with no "
            "explanation-faithfulness evidence is not release-ready"
        )
    if agreement < MIN_EXPLANATION_AGREEMENT:
        raise ReleaseGateError(
            f"explanation agreement (SHAP vs surrogate, leading reported symptom) {agreement:.3f} "
            f"below release-gate minimum {MIN_EXPLANATION_AGREEMENT}"
        )
    # [Responsible-AI] Equal Opportunity ceiling AFTER post-processing, on the
    # calibrated pipeline a patient receives. The metric fairness.py calls the
    # one that matters most in triage was reported for months and gated nowhere.
    eo_after = ((audit.get("postProcessing") or {}).get("after") or {}).get("equalOpportunityGap")
    if eo_after is None:
        raise ReleaseGateError(
            "audit carries no postProcessing.after.equalOpportunityGap — a model with no "
            "post-mitigation fairness evidence is not release-ready"
        )
    if eo_after > MAX_EQUAL_OPPORTUNITY_GAP:
        raise ReleaseGateError(
            f"equal opportunity gap after mitigation {eo_after:.3f} exceeds release-gate "
            f"maximum {MAX_EQUAL_OPPORTUNITY_GAP}"
        )


def _mlflow_log(payload: dict, artifact_path: str, artifact_hash: str) -> str:
    """[MLOps] Pillar 1 — experiment tracking + Model Registry.

    Logs this run's params/metrics and REGISTERS the model as `CareRouteTriageRF`.
    The DESTINATION is wherever `MLFLOW_TRACKING_URI` points — nothing else in
    this function changes with the backend:
      * unset             -> a local ./mlruns store (browse it with `mlflow ui`)
      * a GitLab project   -> GitLab's built-in, MLflow-compatible Model Registry
        (set CI/CD vars MLFLOW_TRACKING_URI + MLFLOW_TRACKING_TOKEN — see README)
      * a standalone MLflow server -> that server

    That single-URI indirection is the whole point of a tracking layer: the same
    training code writes to a laptop, to GitLab, or to a central server. Strictly
    best-effort — any failure (mlflow absent, registry unreachable) is logged and
    swallowed so tracking can never block a release. Returns a short status line
    for the training summary so the outcome is visible, not silent.
    """
    audit = payload["audit"]
    try:
        import mlflow
        import mlflow.sklearn
    except Exception:  # noqa: BLE001 - optional experiment-tracking dependency
        logger.info("mlflow not installed; skipping experiment tracking")
        return "skipped (mlflow not installed)"

    # Where the registry lives -- shared with `python -m app.ml.promote` so the two
    # entry points cannot disagree. See lifecycle.pin_tracking_uri for the Windows
    # percent-encoding trap this exists to avoid.
    tracking_uri = lifecycle.pin_tracking_uri(mlflow)
    logger.info("mlflow tracking uri: %s", tracking_uri)
    try:
        mlflow.set_experiment("careroute-triage")
        with mlflow.start_run(run_name=payload["versionId"]):
            mlflow.log_param("n_estimators", payload["served"].n_estimators)
            mlflow.log_param("model_version", payload["modelVersion"])
            mlflow.log_param("version_id", payload["versionId"])
            mlflow.log_param("calibration", audit["calibration"]["method"])
            # Tags carry the lineage hashes -> a registered model traces back to
            # its exact training data + serialized artifact (Pillar 2 hook).
            mlflow.set_tag("data_sha256", payload["dataSha256"])
            mlflow.set_tag("model_sha256", payload["modelSha256"])
            if artifact_hash:
                mlflow.set_tag("artifact_sha256", artifact_hash)
            # CODE VERSION + ENVIRONMENT. The tracking deck asks a run to record
            # three things — code version, dataset version, environment — and
            # only the dataset version was here. Without the commit, tracing a
            # misbehaving registered model backwards stops at a version string.
            # Best-effort: every field degrades to "unknown" rather than failing
            # a training run over its own provenance record.
            mlflow.set_tags(run_context.mlflow_tags())
            mlflow.log_metric("overall_accuracy", audit["overallAccuracy"])
            mlflow.log_metric("red_flag_recall", audit["redFlagRecall"])
            mlflow.log_metric("fairness_gap_before", audit["fairnessGapBefore"])
            mlflow.log_metric("fairness_gap_after", audit["fairnessGapAfter"])
            mlflow.log_metric("calibration_ece", audit["calibration"]["ece"])
            mlflow.log_metric("calibration_brier", audit["calibration"]["brier"])
            # `registered_model_name` creates/versions the model in the registry.
            # (The pin is mlflow==3.16.1 -- see backend/requirements-mlops.txt;
            # 3.16 fixes a known-exploited CRITICAL. From 3.15 log_model saves
            # with skops, which refuses unlisted types: the tree internals are
            # named below, or registration fails and training only logs it.)
            # SIGNATURE + INPUT EXAMPLE (deck 07 "Experiment Tracking", and what
            # every Workshop 2 submission logged). Without a signature the
            # registry cannot validate a caller's input shape at serve time and
            # the model page shows no schema; without an example a reviewer
            # cannot see what one row looks like. Both are derived from the
            # SAME featurizer the API uses (ml/features.extract_features), so
            # the recorded schema is the served schema, not a hand-typed copy.
            import numpy as np
            from mlflow.models import infer_signature

            from .features import extract_features

            input_example = np.vstack([
                extract_features("crushing chest pain radiating to the left arm", "40-64", "Male"),
                extract_features("mild sore throat since yesterday, no fever", "18-39", "Female"),
            ])
            signature = infer_signature(input_example, payload["served"].predict(input_example))
            mlflow.sklearn.log_model(
                payload["served"],
                name="model",
                registered_model_name="CareRouteTriageRF",
                signature=signature,
                input_example=input_example,
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
            if artifact_path:
                mlflow.log_artifact(artifact_path)
            run_id = mlflow.active_run().info.run_id
        # STAGES (deck 01 "Model stages", deck 02): registering a version and
        # leaving it in an undifferentiated pile cannot answer the one question a
        # registry exists for. Done AFTER the run closes so the version exists.
        # `challenger`, never `champion` — promotion is deploy:promote-production's
        # manual gate, and auto-promoting would make shadow deployment decorative.
        # See app/ml/lifecycle.py for why aliases rather than the slide's
        # transition_model_version_stage (removed in the pinned MLflow 3.x).
        version = lifecycle.record_stage(mlflow.MlflowClient(), run_id, payload, audit)
        logger.info("mlflow run logged + model registered for %s", payload["versionId"])
        staged = f", v{version} -> @{lifecycle.STAGING_ALIAS}" if version else ""
        return f"registered CareRouteTriageRF{staged} -> {tracking_uri}"
    except Exception as exc:  # noqa: BLE001 - optional experiment-tracking dependency
        logger.warning("mlflow logging skipped: %s", exc, exc_info=False)
        return f"skipped (error: {type(exc).__name__})"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    payload = build_artifact()
    audit = payload["audit"]

    try:
        validate(audit)
    except ReleaseGateError as exc:
        print(f"\nRELEASE GATE FAILED: {exc}\n", file=sys.stderr)
        return 1

    artifact_path, artifact_hash = save_artifact(payload)
    mlflow_status = _mlflow_log(payload, artifact_path, artifact_hash)

    cal = audit["calibration"]
    print("\n=== CareRoute triage model — training summary ===")
    print(f"  model version    : {payload['modelVersion']}")
    print(f"  overall accuracy : {audit['overallAccuracy']:.4f}  (gate >= {MIN_ACCURACY})")
    print(f"  red-flag recall  : {audit['redFlagRecall']:.4f}  (gate >= {MIN_RED_FLAG_RECALL})")
    worst = {k: min((v["recall"] for v in audit["subgroupRedFlagRecall"][k].values()
                     if v["recall"] is not None), default=1.0) for k in ("raw", "served")}
    print(f"  worst subgroup   : red-flag recall raw {worst['raw']:.4f} / served {worst['served']:.4f}"
          f"  (gate >= {MIN_RED_FLAG_RECALL} per subgroup, n >= {MIN_SUBGROUP_SEVERE_SUPPORT})")
    print(f"  sex-flip rate    : raw {audit['counterfactual']['sexFlipRate']:.4f}"
          f" / served {audit['counterfactualServed']['sexFlipRate']:.4f}")
    print(f"  fairness gap     : {audit['fairnessGapBefore']:.4f} -> {audit['fairnessGapAfter']:.4f}"
          f"  (gate <= {MAX_FAIRNESS_GAP})")
    print(f"  calibration      : {cal['method']}  ECE={cal['ece']:.4f}  Brier={cal['brier']:.4f}"
          f"  (gate ECE <= {MAX_CALIBRATION_ECE})")
    print(f"  data sha256      : {payload['dataSha256'][:16]}...")
    print(f"  model sha256     : {payload['modelSha256'][:16]}...")
    print(f"  artifact         : {artifact_path or '(not persisted)'}")
    if artifact_hash:
        print(f"  artifact sha256  : {artifact_hash[:16]}...")
    print(f"  experiment track : {mlflow_status}")
    print("  release gate     : PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
