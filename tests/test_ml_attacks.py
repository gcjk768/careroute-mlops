"""[AI-Security] TRAINING-PHASE and PRIVACY attacks on the classical triage model.

AIC Day 1 ("Traditional AI systems: attack and defense") splits the classical ML
threat model three ways by the stage the attacker reaches:

  * **Evasion** (deployment phase) — already gated by `test_robustness.py`, which
    runs a real ART HopSkipJump attack and asserts a robustness floor.
  * **Poisoning** (training phase) — covered here.
  * **Privacy** (query access) — covered here.

The deck's point is that the deployed RandomForest is an attack surface in its own
right, independent of anything LLM-shaped. CareRoute's model decides clinical
acuity from patient records, so both remaining classes are live risks: a poisoned
training set silently suppresses red-flag detection, and a membership-inference
attack against a model trained on patient records is a PDPA disclosure.

**These tests gate a DEFENCE, not a demonstration.** Showing that an attack works
proves nothing about this system. Each test below asserts that a control already
in the pipeline detects or bounds the attack.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.ml import data
from app.ml import model as model_module
from app.ml.model import MIN_RED_FLAG_RECALL, build_artifact
from app.ml.train import ReleaseGateError, validate

# Indices into data.ACUITY_INDEX_TO_CODE. P1/P2 are the "red flag" acuities whose
# recall the release gate protects; P5 is the benign class an attacker wants them
# relabelled as.
RED_FLAG_CLASSES = [0, 1]  # P1_RESUSCITATION, P2_EMERGENT
BENIGN_CLASS = 4  # P5_SELF_CARE

# Fraction of red-flag training labels the simulated attacker flips. Until
# 2026-09-24 this was 10%, which took red-flag recall to ~0.944 (0.915 on an older
# build) and was rejected. With min_samples_leaf=5 (model._MIN_SAMPLES_LEAF) a
# leaf averages at least five rows, so a minority of flipped labels no longer
# decides one: measured recall is 0.989 / 0.984 / 0.973 / 0.958 / 0.785 at
# 10 / 20 / 30 / 40 / 50% flipped. Up to 40% the poison does not push recall under
# the floor; 50% is the smallest tested dose that does, so it is the dose the
# gate has to catch.
POISON_RATE = 0.50
POISON_SEED = 1234


def _poisoned_generate_dataset(rate: float):
    """Wrap data.generate_dataset so it returns a LABEL-FLIPPED training set.

    This is the deck's "label flipping" poisoning (AIC Day 1, p27), aimed in the
    one direction that matters clinically: red flags relabelled as self-care. The
    features are untouched, which is what makes the attack realistic — poisoned
    rows stay plausible and would survive a schema or range check.
    """
    original = data.generate_dataset

    def generate(*args, **kwargs):
        X, y, groups = original(*args, **kwargs)
        y = y.copy()
        red_flag_rows = np.where(np.isin(y, RED_FLAG_CLASSES))[0]
        victims = np.random.default_rng(POISON_SEED).choice(
            red_flag_rows, size=int(len(red_flag_rows) * rate), replace=False
        )
        y[victims] = BENIGN_CLASS
        return X, y, groups

    return generate


def test_clean_training_run_passes_the_release_gate():
    """Control. Without it, the poisoning test below proves only that the gate
    rejects everything, which is the failure mode a gate is most likely to have."""
    audit = build_artifact()["audit"]

    validate(audit)  # raises ReleaseGateError if any threshold is breached

    assert audit["redFlagRecall"] >= MIN_RED_FLAG_RECALL


def test_label_flip_poisoning_is_caught_by_the_release_gate(monkeypatch):
    """[AI-Security] A poisoned training set must never reach the registry.

    The attacker is assumed to have training-data control (the deck's "attacker's
    capabilities" axis) but not code access, so the pipeline runs unmodified and
    the only defence is the gate itself.
    """
    monkeypatch.setattr(
        model_module.data, "generate_dataset", _poisoned_generate_dataset(POISON_RATE)
    )

    audit = build_artifact()["audit"]

    assert audit["redFlagRecall"] < MIN_RED_FLAG_RECALL, (
        f"a {POISON_RATE:.0%} red-flag label flip left red-flag recall at "
        f"{audit['redFlagRecall']:.3f}, still above the {MIN_RED_FLAG_RECALL} gate — "
        "the poison would have shipped"
    )
    with pytest.raises(ReleaseGateError, match="red-flag recall"):
        validate(audit)


def test_poisoning_degrades_red_flag_recall_monotonically():
    """The gate catching 50% is only reassuring if the damage scales with the dose.

    A non-monotonic response would mean the 50% result was luck of one seed rather
    than the attack working. The smaller doses clear the gate because recall stays
    above the floor: the leaf-size regularisation absorbs them (see POISON_RATE).
    """
    recalls = []
    original = data.generate_dataset
    try:
        for rate in (0.30, 0.40, 0.50):
            model_module.data.generate_dataset = _poisoned_generate_dataset(rate)
            recalls.append(build_artifact()["audit"]["redFlagRecall"])
    finally:
        model_module.data.generate_dataset = original

    assert recalls == sorted(recalls, reverse=True), (
        f"red-flag recall did not fall monotonically with poison rate: {recalls}"
    )
    assert recalls[-1] < MIN_RED_FLAG_RECALL


# ---------------------------------------------------------------------------
# Privacy attacks. ART is a DEV-only dependency (requirements-dev.txt), so the
# import is skipped INSIDE the test rather than at module level: a module-level
# `importorskip` would also skip the poisoning gates above, which need only
# numpy and must run everywhere.
# ---------------------------------------------------------------------------

# Attack accuracy of 0.5 means the attacker learns nothing. Measured at ~0.55 on
# the deployed forest, so 0.60 leaves headroom for seed noise while still failing
# if a future change (deeper trees, less regularisation, a smaller training set)
# starts memorising patients. Expressed as ACCURACY, not advantage, because that
# is what ART reports and what a reviewer can reproduce.
MAX_MEMBERSHIP_ATTACK_ACCURACY = 0.60


def test_membership_inference_advantage_stays_near_chance():
    """[AI-Security / PDPA] The model must not reveal WHO was in its training set.

    The training data here is synthetic, so a successful attack leaks nothing
    today. The gate exists because the same forest, retrained on real records
    under the same hyperparameters, would leak exactly as much — and by then the
    question is no longer testable in public CI.
    """
    pytest.importorskip("art", reason="adversarial-robustness-toolbox (art) not installed")
    from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
    from art.estimators.classification import SklearnClassifier

    forest = model_module.get_model().model
    X, y, _ = data.generate_dataset(n=6000, seed=42)
    members, non_members = (X[:1000], y[:1000]), (X[3000:4000], y[3000:4000])

    attack = MembershipInferenceBlackBox(SklearnClassifier(model=forest), attack_model_type="rf")
    # Train the attacker on the first half of each pool, score it on the second.
    attack.fit(members[0][:600], members[1][:600], non_members[0][:600], non_members[1][:600])
    member_pred = attack.infer(members[0][600:], members[1][600:])
    non_member_pred = attack.infer(non_members[0][600:], non_members[1][600:])

    correct = int(np.sum(member_pred)) + int(len(non_member_pred) - np.sum(non_member_pred))
    accuracy = correct / (len(member_pred) + len(non_member_pred))

    assert accuracy <= MAX_MEMBERSHIP_ATTACK_ACCURACY, (
        f"membership-inference attack reached {accuracy:.3f} accuracy "
        f"(chance = 0.5, ceiling = {MAX_MEMBERSHIP_ATTACK_ACCURACY}) — the model is "
        "memorising its training rows"
    )


# --------------------------------------------------------------------------
# Privacy attack 2 — MODEL EXTRACTION (query access). Measured, accepted risk.
# --------------------------------------------------------------------------
def test_model_extraction_probe_reproduces_the_documented_curve():
    """SECURITY.md quotes an agreement curve for a surrogate trained on the
    deployed model's answers. This pins that the curve is produced by code, has
    the shape the accepted-risk argument relies on (more queries -> a better
    clone), and reaches the level the register calls "cheap to clone". A future
    change that made the model materially harder to steal would show up here as
    a pleasant surprise; one that made the register's numbers false would fail.
    """
    from app.ml import extraction

    served = model_module.TriageModel(build_artifact())
    result = extraction.probe((100, 500, 2000), model=served, holdout_n=1000)
    agreements = [row["agreement"] for row in result["rows"]]

    assert result["rows"][0]["queries"] == 100 and result["rows"][-1]["queries"] == 2000
    assert agreements[-1] >= agreements[0], "a larger query budget must not clone worse"
    assert agreements[-1] >= 0.90, "SECURITY.md: ~2000 queries reproduce the model — no longer true"
