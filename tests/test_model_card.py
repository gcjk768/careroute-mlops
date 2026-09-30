"""[Responsible-AI] The model card must agree with the model it describes.

Same argument as `tests/test_datasheet.py`, applied to `MODEL_CARD.md`: a
governance artefact that silently goes stale is worse than none — it is a
confident, wrong answer to "how does this model behave?". PDPC §4.30 asks for
traceability, and traceability to numbers nobody re-checks is decoration.

The card is prose, so it is not generated. Instead every FALSIFIABLE number in
it is asserted against the LIVE training audit here, so a retrain that moves a
metric fails the build until the card is updated.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sklearn")

from app.ml.model import (  # noqa: E402
    MAX_CALIBRATION_ECE,
    MAX_EQUAL_OPPORTUNITY_GAP,
    MAX_FAIRNESS_GAP,
    MIN_ACCURACY,
    MIN_EXPLANATION_AGREEMENT,
    MIN_RED_FLAG_RECALL,
    get_model,
)

CARD = Path(__file__).resolve().parents[1] / "MODEL_CARD.md"


@pytest.fixture(scope="module")
def card() -> str:
    return CARD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def audit() -> dict:
    return get_model().fairness()


def test_identity_and_integrity_match(card, audit):
    assert audit["modelVersion"] in card
    assert audit["integrity"]["dataSha256"] in card
    assert f"Artifact schema **{audit['integrity']['schema']}**" in card


def test_headline_metrics_match(card, audit):
    assert f"| Overall accuracy | {audit['overallAccuracy']} |" in card
    assert f"| Red-flag recall (P1/P2 — the safety-critical metric) | {audit['redFlagRecall']} |" in card
    assert f"(age-blind baseline) | {audit['fairnessGapBefore']} |" in card
    assert f"| Subgroup accuracy gap — after mitigation | {audit['fairnessGapAfter']} |" in card


def test_named_fairness_metrics_match(card, audit):
    dp = audit["demographicParity"]["statisticalParityDifference"]
    assert f"| Demographic Parity (statistical parity difference) | {dp} |" in card
    assert f"| Equal Opportunity gap (severe-class TPR) | {audit['equalOpportunity']['equalOpportunityGap']} |" in card
    assert f"| Equalized Odds gap (max of TPR / FPR gap) | {audit['equalizedOdds']['equalizedOddsGap']} |" in card
    assert f"| Disparate Impact ratio (four-fifths rule) | {audit['disparateImpact']['disparateImpactRatio']} |" in card
    cf = audit["counterfactual"]
    assert f"| Counterfactual fairness — sex-flip rate / mean acuity delta | {cf['sexFlipRate']} / {cf['meanAcuityDelta']} |" in card


def test_calibration_and_drift_match(card, audit):
    cal, drift = audit["calibration"], audit["drift"]
    assert f"| Calibration — ECE / Brier (isotonic, fresh n = 2000 sample) | {cal['ece']} / {cal['brier']} |" in card
    assert f"PSI data **{drift['data']}**, target **{drift['target']}**" in card
    assert f"(accuracy drop) **{drift['concept']}**" in card


def test_cross_validation_table_matches(card, audit):
    cv = audit["crossValidation"]
    assert f"| Accuracy | {cv['accuracy']['mean']} | {cv['accuracy']['std']} |" in card
    assert f"| Red-flag recall | {cv['redFlagRecall']['mean']} | {cv['redFlagRecall']['std']} |" in card
    assert f"| Subgroup accuracy gap | {cv['fairnessGap']['mean']} | {cv['fairnessGap']['std']} |" in card


def test_post_processing_table_matches(card, audit):
    pp = audit["postProcessing"]
    before, after = pp["before"], pp["after"]
    assert f"| Equal Opportunity gap | {before['equalOpportunityGap']} | {after['equalOpportunityGap']} |" in card
    assert f"| Equalized Odds gap | {before['equalizedOddsGap']} | {after['equalizedOddsGap']} |" in card
    assert f"| Red-flag recall | {before['redFlagRecall']} | {after['redFlagRecall']} |" in card
    assert f"| Overall accuracy | {before['overallAccuracy']} | {after['overallAccuracy']} |" in card
    assert f"| False-positive-rate gap | {before['fprGap']} | {after['fprGap']} |" in card
    assert f"{pp['raisedShare'] * 100:.1f}% of held-out cases are" in card


def test_explanation_agreement_matches(card, audit):
    ag = audit["explanationAgreement"]
    assert (f"leading reported symptom **{ag['reportedTop1Agreement']}** "
            f"(on the {ag['nMultiSymptomRows']}") in card
    assert f"feature agreement@3 over all flags **{ag['top3Overlap']}**" in card
    assert f"rank\ncorrelation of magnitudes **{ag['spearman']}**" in card
    assert f"sign agreement on reported symptoms **{ag['signAgreementPresent']}**" in card
    assert f"on {ag['n']}\nheld-out rows" in card


def test_release_gate_table_matches_the_enforced_constants(card, audit):
    rows = [
        (f"| Overall accuracy | ≥ {MIN_ACCURACY} | {audit['overallAccuracy']} |"),
        (f"| Red-flag recall | ≥ {MIN_RED_FLAG_RECALL} | {audit['redFlagRecall']} |"),
        (f"| Subgroup accuracy gap | ≤ {MAX_FAIRNESS_GAP} | {audit['fairnessGapAfter']} |"),
        (f"| Calibration ECE | ≤ {MAX_CALIBRATION_ECE} | {audit['calibration']['ece']} |"),
        (f"| Explanation agreement (leading reported symptom) | ≥ {MIN_EXPLANATION_AGREEMENT} | "
         f"{audit['explanationAgreement']['reportedTop1Agreement']} |"),
        (f"| Equal Opportunity gap after post-processing | ≤ {MAX_EQUAL_OPPORTUNITY_GAP} | "
         f"{audit['postProcessing']['after']['equalOpportunityGap']} |"),
    ]
    for row in rows:
        assert row in card, f"model card release-gate table is stale: {row}"


def test_card_does_not_claim_a_lime_dependency(card):
    assert "optional LIME" not in card
    assert "`lime` package is **not** a dependency" in card


def test_out_of_distribution_rates_match(card, audit):
    ood = audit["outOfDistribution"]
    assert f"flags {ood['heldOutFlagRate'] * 100:.1f}% of held-out cases" in card
    assert f"{ood['shiftedFlagRate'] * 100:.1f}% of" + chr(10) + "the shifted sample" in card
    assert f"{ood['implausibleComboFlagRate'] * 100:.1f}% of implausible random symptom combinations" in card
    assert f"check flags {ood['invalidVectorDetectionRate'] * 100:.0f}% of bounded feature-space perturbations" in card


def test_subgroup_red_flag_table_and_served_counterfactual_match(card, audit):
    """[Responsible-AI] The per-subgroup red-flag table and the served-pipeline
    sex-flip row, against the live audit."""
    block = audit["subgroupRedFlagRecall"]
    for group, raw in block["raw"].items():
        served = block["served"][group]
        assert f"| {group} | {raw['nSevere']} | {raw['recall']} | {served['recall']} |" in card, group
    worst = [min(v["recall"] for v in block[k].values()) for k in ("raw", "served")]
    assert (f"| Red-flag recall, every subgroup with ≥ {block['minSevereSupport']} emergencies "
            f"(raw / served, worst) | ≥ {MIN_RED_FLAG_RECALL} | {worst[0]} / {worst[1]} |") in card
    cf = audit["counterfactualServed"]
    assert (f"served pipeline (all 4 age bands) — sex-flip rate / mean acuity delta | "
            f"{cf['sexFlipRate']} / {cf['meanAcuityDelta']} |") in card


def test_datasheet_names_the_weakest_subgroup(audit):
    """DATASHEET.md said "65+ Female" for a model whose weakest subgroup was 65+ Male."""
    sheet = (CARD.parent / "DATASHEET.md").read_text(encoding="utf-8")
    weakest = min(audit["subgroups"], key=lambda s: s["accuracy"])
    name = weakest["name"].replace(" · ", " ")  # the prose writes "65+ Male"
    assert f"**{name} remains the weakest subgroup**" in sheet
    assert f"accuracy {weakest['accuracy']}" in sheet
