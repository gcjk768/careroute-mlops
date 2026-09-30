"""[Responsible-AI] Fairness GATE using Fairlearn (a tool from the XRAI course).

Uses Fairlearn's MetricFrame to compute per-subgroup accuracy of the DEPLOYED
severity model on a fresh evaluation set, and asserts the accuracy spread across
demographic subgroups stays within tolerance. This is a genuine CI gate: it
passes for the age-aware + rebalanced model (subgroup spread ~0.15) but would
FAIL if the model regressed to the age-blind baseline (spread ~0.5).
"""
from __future__ import annotations

import numpy as np
from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score

from app.ml import data
from app.ml.model import MAX_EQUAL_OPPORTUNITY_GAP, MAX_FAIRNESS_GAP, get_model

# Max acceptable accuracy spread (max-min) across subgroups. IMPORTED, not
# restated: this gate runs in `ai-security`, several stages AFTER the artifact
# has been persisted and MLflow-registered, so app/ml/train.py now enforces the
# same ceiling up front. Two hand-copied 0.35s would eventually disagree and the
# earlier (training-time) gate is the one that would quietly become weaker.
FAIRNESS_TOLERANCE = MAX_FAIRNESS_GAP


def test_subgroup_accuracy_parity_within_tolerance():
    # Fresh, unseen evaluation set (different seed from training).
    X, y, groups = data.generate_dataset(n=2500, seed=99)
    model = get_model().model
    pred = model.predict(X)

    mf = MetricFrame(
        metrics=accuracy_score,
        y_true=y,
        y_pred=pred,
        sensitive_features=np.asarray(groups),
    )
    spread = float(mf.by_group.max() - mf.by_group.min())

    assert mf.overall >= 0.75, f"overall accuracy too low: {mf.overall:.3f}"
    assert spread <= FAIRNESS_TOLERANCE, (
        f"subgroup accuracy spread {spread:.3f} exceeds tolerance {FAIRNESS_TOLERANCE} "
        f"(per group: {mf.by_group.to_dict()})"
    )


def test_equal_opportunity_gap_after_post_processing_within_tolerance():
    """[Responsible-AI] Equal Opportunity — severe-class TRUE-POSITIVE RATE parity —
    on the pipeline a patient actually receives (calibrated argmax + the raise-only
    group thresholds), verified independently with Fairlearn's MetricFrame."""
    from fairlearn.metrics import MetricFrame, true_positive_rate

    X, y, groups = data.generate_dataset(n=2500, seed=99)
    model = get_model()
    proba = model.calibrated.predict_proba(X)
    pred = model.apply_post_processing(proba, groups)
    severe_true = np.isin(y, (0, 1)).astype(int)
    severe_pred = np.isin(pred, (0, 1)).astype(int)
    mf = MetricFrame(
        metrics=true_positive_rate,
        y_true=severe_true,
        y_pred=severe_pred,
        sensitive_features=np.asarray(groups),
    )
    gap = float(mf.by_group.max() - mf.by_group.min())
    assert gap <= MAX_EQUAL_OPPORTUNITY_GAP, (
        f"equal opportunity gap {gap:.3f} exceeds tolerance {MAX_EQUAL_OPPORTUNITY_GAP} "
        f"(TPR per group: {mf.by_group.to_dict()})"
    )
