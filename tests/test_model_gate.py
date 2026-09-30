"""[MLOps] MODEL-QUALITY GATE — the hard release gate on the deployed artifact.

This is the CI gate that must NOT be allowed to fail (see .gitlab-ci.yml:
test:model-gate is a blocking job that gates build/deploy). It reads the audit
that `get_model().fairness()` surfaces for the loaded/served model and asserts
the release thresholds:

  [MLOps]          overall accuracy >= 0.75 (the training-time release gate),
  [Responsible-AI] red-flag recall >= 0.95 — the SAFETY-CRITICAL floor. The
                   training gate in app/ml/train.py enforces the SAME 0.95
                   (aligned so a model that would fail this CI gate is never
                   persisted or MLflow-registered in the first place); this job
                   re-asserts it on the DEPLOYED artifact, because a missed
                   P1/P2 emergency is the highest-severity failure mode.
  [Responsible-AI] subgroup fairness gap <= 0.35 (absolute ceiling), and
  [XRAI]           calibration ECE <= 0.05 — `confidence` drives the HITL
                   escalation gate, so a mis-calibrated model routes wrongly.

Every threshold is IMPORTED from app.ml.model, not restated here: when the CI
gate and the training gate keep their own copies of a number, they drift, and
the training gate silently becomes the weaker of the two.

It also spot-checks the integrity/calibration blocks are present, so a model
shipped without tamper-evidence or calibration metadata fails the gate too.

FIRST, though, it asserts the model under test was LOADED FROM A PERSISTED
ARTIFACT. That is not ceremony: TriageModel.__init__ falls back to training a
fresh model in-process when no compatible artifact loads, and that fallback used
to be invisible here — so this "release gate on the deployed artifact" would
pass with a model it had just built itself and no artifact anywhere. The gate
then proved nothing about what CI was about to ship. In the pipeline the
artifact arrives via `needs: ["train:model"]` + CAREROUTE_MODEL_DIR.
"""
from __future__ import annotations

import os

import pytest

from app.ml.model import (
    MAX_CALIBRATION_ECE,
    MAX_FAIRNESS_GAP,
    MIN_ACCURACY,
    MIN_RED_FLAG_RECALL,
    get_model,
)

# Escape hatch for a developer with no artifact on disk yet. Deliberately an
# EXPLICIT opt-in and deliberately NOT set anywhere in .gitlab-ci.yml — in CI a
# missing artifact means train:model did not hand one over, which is a pipeline
# failure, not something to skip past.
_ALLOW_FRESH = os.environ.get("CAREROUTE_ALLOW_FRESH_MODEL_IN_GATE") == "1"


def _gated_model():
    """The model this gate grades — asserted to be a PERSISTED artifact."""
    model = get_model()
    if not model.loaded_from_artifact:
        if _ALLOW_FRESH:
            pytest.skip(
                "CAREROUTE_ALLOW_FRESH_MODEL_IN_GATE=1: grading a model trained "
                "in-process, so this run does NOT gate a deployable artifact"
            )
        pytest.fail(
            "release gate has no artifact to grade: get_model() trained a fresh "
            f"model instead of loading one from CAREROUTE_MODEL_DIR="
            f"{os.environ.get('CAREROUTE_MODEL_DIR', 'models')!r}. In CI this "
            "means train:model's artifact did not arrive (check `needs:`); "
            "locally, run `python -m app.ml.train` first."
        )
    return model


def test_gate_grades_a_persisted_artifact():
    """The provenance assertion itself, as its own named test — so a missing
    artifact reads as 'the gate had nothing to grade', not as a quality
    regression buried inside an accuracy assertion."""
    model = _gated_model()
    assert model.artifact_path, "loaded artifact reported no path"
    assert os.path.exists(model.artifact_path), (
        f"recorded artifact path does not exist: {model.artifact_path}"
    )


def test_model_meets_release_thresholds():
    audit = _gated_model().fairness()

    acc = audit["overallAccuracy"]
    recall = audit["redFlagRecall"]

    assert acc >= MIN_ACCURACY, (
        f"overall accuracy {acc:.4f} below release-gate minimum {MIN_ACCURACY}"
    )
    assert recall >= MIN_RED_FLAG_RECALL, (
        f"red-flag recall {recall:.4f} below safety-critical minimum {MIN_RED_FLAG_RECALL}"
    )


def test_model_meets_fairness_and_calibration_ceilings():
    """[Responsible-AI][XRAI] The two gates that used to be un-enforced here.

    Fairness was only ever checked RELATIVELY (against the age-blind baseline's
    ~0.549 gap), so a regression to 0.45 passed; calibration was checked as
    `ece >= 0.0`, which is true of every float ever measured. Both are now
    absolute ceilings shared with app/ml/train.py's release gate."""
    audit = _gated_model().fairness()

    gap = audit["fairnessGapAfter"]
    assert gap <= MAX_FAIRNESS_GAP, (
        f"subgroup fairness gap {gap:.4f} exceeds release-gate maximum {MAX_FAIRNESS_GAP} "
        f"(per subgroup: {audit.get('subgroups')})"
    )

    ece = audit["calibration"]["ece"]
    assert ece <= MAX_CALIBRATION_ECE, (
        f"calibration ECE {ece:.4f} exceeds release-gate maximum {MAX_CALIBRATION_ECE} — "
        "confidence drives the HITL escalation gate, so this mis-routes patients"
    )


def test_model_carries_integrity_and_calibration_metadata():
    """A deployable artifact must be tamper-evident (integrity hashes) and
    calibrated (ECE/Brier) — a model missing either is not release-ready."""
    audit = _gated_model().fairness()

    integrity = audit.get("integrity") or {}
    assert integrity.get("dataSha256"), "missing training-data integrity hash"
    assert integrity.get("modelSha256"), "missing serialized-model integrity hash"

    calibration = audit.get("calibration") or {}
    assert "ece" in calibration and "brier" in calibration, "missing calibration metrics"
    assert isinstance(calibration["ece"], float) and calibration["ece"] >= 0.0

    assert audit.get("modelVersion"), "missing content-addressed model version"


def test_every_subgroup_meets_the_red_flag_floor():
    """[Responsible-AI][Safety] The population floor, per age-band x sex subgroup,
    on the raw forest and the served pipeline. 4e409ad346f3 cleared the
    population gate at 0.9747 with 65+ Male at 0.897; this is what stops that."""
    from app.ml.train import subgroup_red_flag_shortfalls

    audit = _gated_model().fairness()
    assert "subgroupRedFlagRecall" in audit, "artifact predates the per-subgroup red-flag audit"
    assert subgroup_red_flag_shortfalls(audit) == []
