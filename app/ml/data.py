"""Deterministic synthetic triage dataset.

Generates (text + demographics -> acuity) samples with a realistic, MITIGABLE
fairness problem:

- The **65+ band is under-represented** in the data (fewer training examples).
- The 65+ band follows a clinically realistic **age-adjusted acuity rule**
  (moderate symptoms escalate one level for the elderly).

A model with too few 65+ examples underfits that rule → lower accuracy on 65+ →
a real fairness gap. The mitigation in model.py (oversampling 65+ to parity) lets
the model learn the rule, which measurably shrinks the gap. All randomness is
seeded, so every downstream metric is reproducible.
"""
from __future__ import annotations

import numpy as np

from .features import AGE_BANDS, FEATURE_KEYWORDS, extract_features

ACUITY_INDEX_TO_CODE = [
    "P1_RESUSCITATION", "P2_EMERGENT", "P3_URGENT", "P4_NON_URGENT", "P5_SELF_CARE",
]

SEXES = ["Female", "Male"]
_BIASED_BAND = "65+"
# 65+ deliberately under-represented (last weight is smallest).
_BAND_WEIGHTS = np.array([0.30, 0.30, 0.28, 0.12])

_ACUITY_PROFILES: dict[int, list[list[str]]] = {
    0: [["chest_pain"], ["breathless"], ["stroke_signs"], ["severe_bleeding"], ["anaphylaxis"],
        # Airway and arrest presentations. Without these the P1 band was
        # defined entirely by medical red flags and carried no trauma at all.
        ["unresponsive"], ["choking"], ["major_trauma"],
        ["major_trauma", "severe_bleeding"]],
    # NO MIXED-BAND COMBINATIONS INVOLVING A P1 CATEGORY: a P1 marker paired
    # with a lesser one and labelled P2 says the marker is sometimes not an
    # emergency. ["wheeze_asthma","breathless"] at P2 while ["breathless"] is
    # P1 was the example, and it is removed on that consistency argument
    # alone — measured on its own it moved red-flag recall very little. The
    # gate failure had a different cause; see `_distractor_pool`.
    1: [["suicidal"], ["seizure"], ["severe_pain"], ["severe_pain", "abdominal_pain"],
        # Time-critical but not airway: torsion, obstetric emergencies, sudden
        # visual loss and arrhythmia all lose organs or lives on a delay.
        ["syncope"], ["palpitations"], ["wheeze_asthma"], ["vision_loss"],
        ["testicular_pain"], ["pregnancy_concern"], ["allergic_reaction"],
        ["burn_injury"], ["fracture"], ["limb_swelling"],
        ["pregnancy_concern", "abdominal_pain"], ["allergic_reaction", "rash"]],
    2: [["high_fever"], ["high_fever", "persistent"], ["vomiting", "dehydrated"],
        ["abdominal_pain", "vomiting"], ["persistent", "high_fever"],
        # Same-day primary/urgent care — the bulk of real walk-in volume.
        ["foreign_body"], ["urinary"], ["diarrhoea"], ["diarrhoea", "dehydrated"],
        ["urinary", "high_fever"], ["bite_sting"], ["mental_distress"],
        ["back_pain", "severe_pain"],
        # Fever with poor fluid intake is urgent. It had no profile, so "39.5
        # for three days ... drinking less" (live test 2026-09-24) fell through
        # to the milder companions and came out P5. Chosen over a persistent-
        # cough P4 profile, which fixed a 65+ case but broke the subgroup gate.
        ["high_fever", "dehydrated"],
        # Flu-like high fever is same-day care. Body aches existed only in P5
        # profiles and fever never met them, so for this unseen pair the forest
        # split on the aches first: "high fever with body aches" served P5 and
        # the 65+ "39.5 for three days ... drinking less" case served P5 too —
        # a mild companion symptom LOWERED the acuity (2026-09-25).
        ["high_fever", "muscle_ache"], ["high_fever", "sore_throat"]],
    3: [["sore_throat"], ["cough"], ["headache"], ["rash"], ["sore_throat", "cough"],
        # Sprain/strain sits at P4: routine primary care, not self-care, because
        # a fracture has to be excluded before it can be sent home.
        ["sprain_strain"], ["sprain_strain", "bruise"],
        ["back_pain"], ["eye_problem"], ["ear_problem"], ["dental"],
        ["joint_pain"], ["skin_infection"], ["constipation"], ["menstrual"],
        ["ear_problem", "high_fever"],
        # A lingering mild symptom is routine primary care. "persistent" used
        # to appear only beside fever (P3), so any persistent symptom read as
        # urgent and a 65+ week-long cough served P2 (live MAP4). With these
        # rows the elderly rule lifts it one level, to P3, as designed.
        ["persistent", "cough"], ["persistent", "sore_throat"],
        # Mild abdominal pain on its own. It only ever appeared with a severe
        # or vomiting companion, so the forest extrapolated a lone stomach ache
        # toward P2 (the known gap xfailed in tests/test_ml.py until now).
        ["abdominal_pain"]],
    4: [["cold_symptoms"], ["cold_symptoms", "cough"], ["cold_symptoms", "sore_throat"],
        # Minor wounds and isolated bruising are textbook self-care.
        ["minor_wound"], ["bruise"], ["minor_wound", "bruise"],
        ["muscle_ache"], ["heartburn"], ["fatigue"], ["muscle_ache", "fatigue"],
        ["cold_symptoms", "muscle_ache"]],
}

_ACUITY_PRIOR = np.array([0.08, 0.12, 0.30, 0.32, 0.18])

_VAGUE_PHRASES = [
    "feeling a bit off", "not feeling myself", "generally unwell", "tired and run down",
    "a vague ache", "feeling under the weather", "some discomfort", "just not right",
]


def _compose_text(categories: list[str], rng: np.random.Generator) -> str:
    return " and ".join(rng.choice(FEATURE_KEYWORDS[c]) for c in categories)


#: Categories used by each acuity band's profiles, for the co-occurring-symptom
#: draw below.
_BAND_CATEGORIES: dict[int, list[str]] = {
    band: sorted({c for profile in profiles for c in profile})
    for band, profiles in _ACUITY_PROFILES.items()
}


def _distractor_pool(y: int) -> list[str]:
    """Categories that could plausibly co-occur with an acuity-`y` presentation.

    The second symptom below used to be drawn UNIFORMLY from every category.
    That was survivable at 22 categories; at 51 it is not. Roughly twenty of
    the new categories sit in the P4/P5 band, so a uniform draw began handing
    P1 rows a mild companion symptom — "chest pain and period cramps",
    "unresponsive and ear discharge" — at 15% of severe cases. Those rows are
    not clinically plausible, and the model learned from them that mild
    features argue against an emergency: red-flag recall fell to 0.936 against
    a 0.95 gate. Restricting the draw to the case's own band and its immediate
    neighbours lifts it to 0.987 AND improves overall accuracy, because the
    training rows stop contradicting each other.
    """
    return sorted({
        category
        for band in (y - 1, y, y + 1)
        if band in _BAND_CATEGORIES
        for category in _BAND_CATEGORIES[band]
    })


def generate_dataset(n: int = 7000, seed: int = 42, shift: bool = False):
    """Return (X, y, subgroups). X via extract_features (train == inference)."""
    rng = np.random.default_rng(seed)
    prior = _ACUITY_PRIOR.copy()
    band_weights = _BAND_WEIGHTS.copy()
    if shift:
        prior = np.array([0.16, 0.20, 0.30, 0.22, 0.12])  # more severe under load
    prior = prior / prior.sum()
    band_weights = band_weights / band_weights.sum()

    texts, bands, sexes, labels = [], [], [], []

    for _ in range(n):
        band = str(rng.choice(AGE_BANDS, p=band_weights))
        sex = str(rng.choice(SEXES))

        # ~8% vague/ambiguous cases -> low-acuity label scattered across P3/P4/P5,
        # keeping the feature-sparse region uncertain (vague input -> low conf).
        if rng.random() < 0.08:
            texts.append(str(rng.choice(_VAGUE_PHRASES)))
            bands.append(band)
            sexes.append(sex)
            labels.append(int(rng.choice([2, 3, 4])))
            continue

        y = int(rng.choice(5, p=prior))
        profile = _ACUITY_PROFILES[y][rng.integers(len(_ACUITY_PROFILES[y]))]
        cats = list(profile)
        if rng.random() < (0.30 if shift else 0.15):
            cats.append(str(rng.choice(_distractor_pool(y))))

        # [Responsible-AI] Age-adjusted acuity for the elderly: moderate (P3/P4)
        # presentations escalate one level for the 65+ band. This is the
        # subgroup-specific rule the model must learn from the age feature.
        if band == _BIASED_BAND and y in (2, 3) and rng.random() < 0.85:
            y = y - 1  # more urgent (lower index)

        texts.append(_compose_text(cats, rng))
        bands.append(band)
        sexes.append(sex)
        labels.append(y)

    X = np.vstack([extract_features(t, b, s) for t, b, s in zip(texts, bands, sexes, strict=True)])
    y = np.asarray(labels, dtype=int)
    subgroups = [f"{b} · {s}" for b, s in zip(bands, sexes, strict=True)]
    return X, y, subgroups
