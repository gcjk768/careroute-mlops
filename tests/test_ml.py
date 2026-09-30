"""[Responsible-AI][MLOps] Tests for the real ML severity model.

Covers: feature extraction, the seeded synthetic dataset, model prediction +
real SHAP explanation, low-confidence on vague input (drives escalation), and
the fairness/drift audit (mitigation does not worsen the gap; drift is real).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.ml import data, fairness
from app.ml import features
from app.ml.features import FEATURE_NAMES, extract_features
from app.ml.model import get_model


def test_features_extract_expected_flags():
    x = extract_features("severe crushing chest pain and shortness of breath")
    assert x.shape == (len(FEATURE_NAMES),)
    assert x[FEATURE_NAMES.index("chest_pain")] == 1.0
    assert x[FEATURE_NAMES.index("breathless")] == 1.0
    assert x[FEATURE_NAMES.index("rash")] == 0.0


def test_dataset_has_subgroups_and_valid_labels():
    X, y, groups = data.generate_dataset(n=600, seed=1)
    assert X.shape == (600, len(FEATURE_NAMES))
    assert set(np.unique(y)).issubset({0, 1, 2, 3, 4})
    assert any("65+" in g for g in groups)  # the biased subgroup is represented


def test_model_predicts_severe_with_real_shap_explanation():
    pred = get_model().predict(
        "severe crushing chest pain radiating to my left arm, sweating and short of breath"
    )
    assert pred["acuity_code"] in {"P1_RESUSCITATION", "P2_EMERGENT"}
    assert 0.0 <= pred["confidence"] <= 1.0
    # Real SHAP: a non-empty, signed feature-contribution list.
    assert pred["explanation"]
    assert all("feature" in c and "weight" in c for c in pred["explanation"])
    assert any(c["weight"] != 0.0 for c in pred["explanation"])


def test_model_low_confidence_on_vague_input():
    # Vague, feature-sparse input must be low-confidence so the pipeline escalates.
    pred = get_model().predict("feeling a bit off and not really myself today")
    assert pred["confidence"] < 0.6


def test_fairness_audit_is_real_and_mitigation_does_not_worsen_gap():
    audit = get_model().fairness()
    for key in (
        "overallAccuracy", "redFlagRecall", "fairnessGapBefore", "fairnessGapAfter",
        "subgroups", "drift", "modelVersion", "updatedAt",
    ):
        assert key in audit
    assert len(audit["subgroups"]) >= 4
    assert 0.0 <= audit["overallAccuracy"] <= 1.0
    # Safety-critical severe classes must be caught with high recall.
    assert audit["redFlagRecall"] >= 0.95
    # Mitigation (up-weighting the under-served subgroup) must not make it worse.
    assert audit["fairnessGapAfter"] <= audit["fairnessGapBefore"] + 1e-9
    # Drift metrics are real, non-negative floats.
    for value in audit["drift"].values():
        assert isinstance(value, float) and value >= 0.0


def test_psi_zero_for_identical_distributions():
    ref = np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=float)
    assert abs(fairness.population_stability_index(ref, ref)) < 1e-6


# --------------------------------------------------------------------------
# Model-coverage telemetry (formerly `is_suspected_evasion`).
#
# These tests exist because the old function was uncalled, untested, and made a
# claim it could not support: it reported "suspected adversarial evasion" for
# any input lighting up zero symptom features. Evaluation E5 disproved that —
# three legitimate minor-trauma complaints were flagged, purely because the
# feature space had no trauma categories. The renamed function measures MODEL
# COVERAGE and must never be used as a gate.
# --------------------------------------------------------------------------
def test_a_denied_symptom_does_not_light_its_feature():
    """The model's evidence read "fever" for a patient who wrote "no fever":
    the extractor matched substrings with no notion of denial, so a mild sore
    throat classified as P3 on a symptom the patient had ruled out. Same rule
    as the classifier's keyword path: a cue (no/not/without/denies/never/n't)
    in the same clause, within three tokens before the phrase, is a denial."""
    from app.ml.features import FEATURE_NAMES, extract_features

    def fired(text: str) -> set[str]:
        x = extract_features(text)
        return {name for name, flag in zip(FEATURE_NAMES, x, strict=False) if flag > 0.5}

    assert "high_fever" not in fired("mild sore throat and a slight cough for two days, no fever")
    assert {"sore_throat", "cough"} <= fired("mild sore throat and a slight cough for two days, no fever")
    # The denial is clause-bound: it does not leak past the comma.
    assert {"high_fever", "chest_pain"} & fired("no fever, but severe chest pain") == {"chest_pain"}
    # ...nor past a conjunction: what follows "and"/"but" is a new statement.
    assert "cough" in fired("no fever and a dry cough")
    assert "sore_throat" in fired("high fever and not improving and scratchy throat")
    assert "abdominal_pain" in fired("baby not moving and belly pain")
    # Keywords that are themselves phrased as a negation still fire.
    assert "unresponsive" in fired("my father is not responding and has no pulse")
    assert "high_fever" in fired("I don't think it is serious but I do have a fever")


def test_ordinary_words_do_not_fuzzy_match_a_symptom():
    """The typo tier accepted any token within one edit of a keyword, so
    ordinary words lit emergency features: "cooked" read as "choked" sent
    "I cooked dinner and now my stomach hurts a little" to P1 with "choking"
    as evidence. An edit-distance match must share the keyword's first three
    letters and be a reasonably long word; the doubled-letter and truncation
    tiers are untouched."""
    from app.ml.features import FEATURE_NAMES, N_SYMPTOM_FEATURES, extract_features

    def fired(text: str) -> set[str]:
        x = extract_features(text)
        return {FEATURE_NAMES[i] for i in range(N_SYMPTOM_FEATURES) if x[i] > 0.5}

    for word in ("cooked", "stubbed", "brushing", "couch", "lives", "never", "sitting", "scalp",
                 "tough", "rough", "cruise", "burst", "grace", "lodge", "grasping", "hitting",
                 "hanged", "bagged", "confusion", "gives"):
        assert not fired(f"I was {word} at home today"), word
    assert fired("I cooked dinner and now my stomach hurts a little") == {"abdominal_pain"}
    # Typos the tier exists for still match.
    assert "chest_pain" in fired("chest paiin since this morning")
    assert "breathless" in fired("I am breathles and dizzy")


def test_the_dinner_sentence_is_not_read_as_choking():
    """The typo-tier regression this guards: cooked~choked made it P1 via choking."""
    pytest.importorskip("sklearn")
    from app.ml.model import get_model

    result = get_model().predict("I cooked dinner and now my stomach hurts a little")
    assert result["acuity_code"] != "P1_RESUSCITATION"
    assert "choking / airway obstruction" not in result["evidence"]


def test_mild_abdominal_pain_alone_is_not_an_emergency():
    """Was a strict xfail (2026-09-24): `abdominal_pain` never appeared ALONE in
    the synthetic data, so the forest extrapolated this sentence toward P2.
    Fixed 2026-09-25 by the `["abdominal_pain"]` P4 profile in data.py."""
    pytest.importorskip("sklearn")
    from app.ml.model import get_model

    result = get_model().predict("I cooked dinner and now my stomach hurts a little")
    assert result["acuity_code"] in {"P4_NON_URGENT", "P5_SELF_CARE", "P3_URGENT"}


def test_a_contracted_denial_is_a_denial_in_every_tier():
    """"I don't have a fever" was read as fever by the typo tier: its
    tokeniser dropped the apostrophe and its window check had no n't rule."""
    from app.ml.features import FEATURE_NAMES, extract_features

    def fired(text: str) -> set[str]:
        x = extract_features(text)
        return {name for name, flag in zip(FEATURE_NAMES, x, strict=False) if flag > 0.5}

    assert "high_fever" not in fired("I don't have a fever, just a mild sore throat")
    assert "high_fever" not in fired("I haven't got a fever, just a sore throat")
    assert "sore_throat" in fired("I don't have a fever, just a mild sore throat")


def test_never_had_it_this_bad_and_cannot_stop_are_reports():
    """"never" and "n't" are denial cues, but "never had chest pain like this"
    and "can't stop coughing" report the symptom; the cue is an intensifier."""
    from app.ml.features import FEATURE_NAMES, extract_features

    def fired(text: str) -> set[str]:
        x = extract_features(text)
        return {name for name, flag in zip(FEATURE_NAMES, x, strict=False) if flag > 0.5}

    assert "chest_pain" in fired("I have never had chest pain like this before")
    assert "headache" in fired("never had a headache this bad")
    assert "cough" in fired("I can't stop coughing")
    assert "vomiting" in fired("I couldn't stop vomiting all night")
    assert "high_fever" not in fired("I have never had a fever")


def test_no_feature_coverage_flags_input_the_model_cannot_see():
    """Gibberish carries real words but lights up nothing the model knows."""
    assert features.has_no_feature_coverage("asdkfj qwoieru zxcvbnm lkjhgfds") is True


def test_covered_clinical_input_is_not_flagged():
    assert features.has_no_feature_coverage("crushing chest pain and sweating") is False


def test_minor_trauma_is_covered_since_the_e5_fix():
    """REGRESSION GUARD for the defect E5 found. Each of these lit up zero
    features and scored an identical 0.448 confidence before the trauma
    categories were added — under CONFIDENCE_THRESHOLD, so the HITL gate turned
    each into a clinician interruption."""
    for text in (
        "I have a small bruise on my knee from bumping into a table.",
        "Minor cut on my finger from cooking, it is small and clean.",
        "I sprained my ankle playing football, there is mild swelling but I can walk.",
    ):
        assert features.has_no_feature_coverage(text) is False, text


def test_trivially_short_input_is_not_a_coverage_gap():
    """Under three words is 'empty', not 'uncovered' — flagging it would make
    the live zero-coverage rate meaningless."""
    assert features.has_no_feature_coverage("hi") is False


def test_zero_coverage_rate_is_a_fraction_of_rows():
    covered = features.extract_features("chest pain")
    uncovered = features.extract_features("asdkfj qwoieru zxcvbnm")
    X = np.vstack([covered, uncovered, uncovered, uncovered])
    assert features.zero_coverage_rate(X) == 0.75


def test_zero_coverage_rate_handles_an_empty_matrix():
    assert features.zero_coverage_rate(np.zeros((0, len(FEATURE_NAMES)))) == 0.0


def test_evasion_helper_is_gone():
    """The old name asserted adversarial INTENT from a model-coverage fact, and
    its verdict flipped on retrain. Real evasion screening is guardrail.py's
    job, which does it independently of the model's feature space."""
    assert not hasattr(features, "is_suspected_evasion")


def test_a_mild_companion_symptom_never_lowers_a_high_fever_to_self_care():
    """Live 2026-09-25: "high fever with body aches" served P5, and the 65+
    "39.5 for three days ... drinking less" case served P5 too — the aches only
    ever appeared in P5 training rows, so adding them pulled the fever down."""
    pytest.importorskip("sklearn")
    from app.ml.model import get_model

    m = get_model()
    urgent = {"P1_RESUSCITATION", "P2_EMERGENT", "P3_URGENT"}
    assert m.predict("High fever with body aches.", "18-39", "Male")["acuity_code"] in urgent
    elder = "High fever of 39.5 for three days with body aches and bad sore throat, drinking less."
    assert m.predict(elder, "65+", "Female")["acuity_code"] in {"P1_RESUSCITATION", "P2_EMERGENT"}


def test_a_lingering_cough_is_primary_care_not_the_emergency_department():
    """Live MAP4: a 65+ week-long cough with no fever or breathlessness served
    P2 ("persistent" only ever sat beside fever). The elderly rule lifts P4 to
    P3, not to an emergency."""
    pytest.importorskip("sklearn")
    from app.ml.model import get_model

    text = "Persistent cough for one week with mild phlegm, no fever, no breathlessness."
    assert get_model().predict(text, "65+", "Female")["acuity_code"] == "P3_URGENT"
    assert get_model().predict(text, "40-64", "Female")["acuity_code"] == "P4_NON_URGENT"
