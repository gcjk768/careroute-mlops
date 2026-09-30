"""Unit tests for the fairness / drift metric helpers (app.ml.fairness)."""
from __future__ import annotations

import numpy as np

from app.ml import fairness


def test_subgroup_accuracies_and_gap():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2])
    groups = ["A", "A", "A", "B", "B", "B"]
    sub = fairness.subgroup_accuracies(y_true, y_pred, groups)
    assert set(sub) == {"A", "B"}
    assert sub["A"]["n"] == 3
    gap = fairness.fairness_gap(sub)
    assert 0.0 <= gap <= 1.0


def test_red_flag_recall_perfect_and_missed():
    y_true = np.array([0, 1, 2, 3, 4])
    assert fairness.red_flag_recall(y_true, y_true) == 1.0
    missed = fairness.red_flag_recall(y_true, np.array([2, 2, 2, 3, 4]))  # both severe missed
    assert missed == 0.0


def test_demographic_parity_and_equal_opportunity():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 3, 3, 0, 2, 2, 3])
    groups = ["F", "F", "F", "F", "M", "M", "M", "M"]
    dp = fairness.demographic_parity(y_pred, groups)
    assert "byGroup" in dp and "statisticalParityDifference" in dp
    eo = fairness.equal_opportunity(y_true, y_pred, groups)
    assert "byGroup" in eo and "equalOpportunityGap" in eo


def test_equalized_odds_reports_both_error_directions():
    """The metric exists to catch what Equal Opportunity cannot see.

    Group M's severe cases are all caught (TPR 1.0, same as F) while two of its
    non-severe cases are over-triaged into severe. Equal Opportunity is blind to
    that; Equalized Odds must surface it as an FPR gap.
    """
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 0, 1])
    groups = ["F", "F", "F", "F", "M", "M", "M", "M"]

    eo = fairness.equal_opportunity(y_true, y_pred, groups)
    odds = fairness.equalized_odds(y_true, y_pred, groups)

    assert eo["equalOpportunityGap"] == 0.0  # TPR is identical, so EO sees nothing
    assert odds["tprGap"] == 0.0
    assert odds["fprGap"] == 1.0  # F over-triages 0 of 2, M over-triages 2 of 2
    # The headline is the MAX of the two gaps, never the mean: averaging would
    # report 0.5 here and let a total FPR disparity read as "half fair".
    assert odds["equalizedOddsGap"] == 1.0


def test_disparate_impact_is_the_ratio_form_of_demographic_parity():
    """A small parity DIFFERENCE can still be a large RATIO at low base rates."""
    y_pred = np.array([0, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # F: 1 of 10 urgent
                       0, 0, 0, 2, 2, 2, 2, 2, 2, 2])  # M: 3 of 10 urgent
    groups = ["F"] * 10 + ["M"] * 10

    dp = fairness.demographic_parity(y_pred, groups)
    di = fairness.disparate_impact(y_pred, groups)

    assert dp["statisticalParityDifference"] == 0.2  # looks small
    assert di["disparateImpactRatio"] == 0.3333  # 0.1 / 0.3 -- a 3x disparity
    assert di["meetsFourFifthsRule"] is False


def test_disparate_impact_is_one_when_rates_match():
    y_pred = np.array([0, 2, 2, 2, 0, 2, 2, 2])
    groups = ["F"] * 4 + ["M"] * 4

    di = fairness.disparate_impact(y_pred, groups)

    assert di["disparateImpactRatio"] == 1.0
    assert di["meetsFourFifthsRule"] is True


def test_psi_zero_for_identical_and_positive_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=1000)
    assert fairness.population_stability_index(ref, ref) == 0.0 or fairness.population_stability_index(ref, ref) < 1e-6
    shifted = ref + 2.0
    assert fairness.population_stability_index(ref, shifted) > 0.1


# --------------------------------------------------------------------------
# REGRESSION GUARD — binary-feature PSI.
#
# 27 of the 29 features are 0/1 flags. np.quantile over a binary column returns
# {0., 1.}, so `np.unique(...)` had size 2 (NOT < 2) and the binary fallback
# never ran; np.histogram with two edges is ONE bin, which collects 100% of both
# samples -> PSI exactly 0.0 for ANY binary feature. Measured: a 5% -> 95%
# prevalence flip (the most violent shift a flag can undergo) scored 0.0, so
# data drift was structurally undetectable and the CT retrain trigger could
# never fire. These tests pin the proportion branch that now handles it.
# --------------------------------------------------------------------------
def test_psi_detects_a_binary_prevalence_flip():
    rng = np.random.default_rng(0)
    ref = (rng.random(2000) < 0.05).astype(float)
    cur = (rng.random(2000) < 0.95).astype(float)
    psi = fairness.population_stability_index(ref, cur)
    assert psi > 0.25, f"binary 5%->95% flip scored {psi} (was 0.0 before the fix)"


def test_psi_zero_for_identical_binary_distributions():
    ref = (np.arange(1000) % 20 == 0).astype(float)  # 5% prevalence
    assert fairness.population_stability_index(ref, ref) < 1e-6


def test_psi_on_a_constant_reference_column_still_scores_a_shift():
    ref = np.zeros(500)
    assert fairness.population_stability_index(ref, ref) < 1e-6
    assert fairness.population_stability_index(ref, np.ones(500)) > 0.25


def test_data_drift_reports_the_worst_feature_not_the_mean():
    """One violently drifted flag among many stable ones must SURFACE. Averaging
    is what buried it: 1/29 features at PSI 5 averages to 0.17, under the 0.25
    gate. `data_drift` therefore reports the MAX per-feature PSI."""
    rng = np.random.default_rng(1)
    ref = (rng.random((2000, 10)) < 0.5).astype(float)
    cur = ref.copy()
    cur[:, 3] = (rng.random(2000) < 0.97).astype(float)  # one feature flips hard
    worst = fairness.data_drift(ref, cur)
    per_feature = fairness.feature_psis(ref, cur)
    assert worst == round(max(per_feature), 4)
    assert worst > 0.25
    assert worst > round(float(np.mean(per_feature)), 4)


def test_data_drift_detail_reports_share_and_named_features():
    rng = np.random.default_rng(2)
    ref = (rng.random((1500, 8)) < 0.4).astype(float)
    cur = ref.copy()
    cur[:, 5] = (rng.random(1500) < 0.95).astype(float)
    detail = fairness.data_drift_detail(ref, cur)
    assert detail["max"] == fairness.data_drift(ref, cur)
    assert detail["driftedFeatureCount"] == 1
    assert detail["driftedFeatures"] == [5]
    assert 0.0 < detail["driftedShare"] < 1.0
    assert detail["threshold"] == fairness.DRIFT_PSI_THRESHOLD


def test_data_drift_is_zero_for_identical_matrices():
    rng = np.random.default_rng(3)
    ref = (rng.random((500, 6)) < 0.3).astype(float)
    assert fairness.data_drift(ref, ref) == 0.0
    assert fairness.data_drift_detail(ref, ref)["driftedFeatureCount"] == 0


def test_data_target_concept_drift():
    ref_X = np.zeros((50, 4))
    cur_X = np.ones((50, 4))
    assert fairness.data_drift(ref_X, cur_X) >= 0.0
    ref_y = np.array([0, 1, 2, 3, 4] * 10)
    cur_y = np.array([0, 0, 0, 0, 0] * 10)
    assert fairness.target_drift(ref_y, cur_y) >= 0.0
    assert fairness.concept_drift(0.9, 0.8) == round(0.9 - 0.8, 4)
