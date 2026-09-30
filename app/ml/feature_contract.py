"""[MLOps] TRAIN/SERVE SKEW gate — the cheap half of a feature store.

Decks 01-02 put a feature store (Feast) in the toolchain, and the property it
is bought for is train/serve consistency: the features a model is scored on in
production are the features it was trained on. CareRoute already has that
property structurally — `ml/data.py` (training) and `ml/model.py` +
`agents/classifier.py` (serving) all call the SAME `features.extract_features`.
What it did not have is anything that NOTICES when the two stop agreeing, and
"they call the same function" is a property of today's source, not of the
artifact currently being served.

WHAT THE EXISTING CHECK MISSES
------------------------------
`model._is_compatible` already rejects an artifact whose `featureNames` differ
from the current code's. That catches a change in feature LAYOUT — a column
added, removed or reordered — and it is what caught the E5 minor-trauma fix,
which added three whole categories (26 -> 29 features).

It cannot catch a change in feature MEANING: keywords added to an EXISTING
category. Same names, same dimension, different function. Nothing makes that
the less likely shape of the next fix — the same E5 finding could have been
answered by adding "twisted my ankle" to an existing category rather than a new
one, and the neighbouring module's 2026-09-12 fix (limb/joint vocabulary added
to the guardrail's lexicon) is exactly that shape. An artifact trained before
such an edit loads without complaint and is then scored on features whose
semantics moved underneath it — no error anywhere, and a confident wrong
prediction as the only symptom.

THE CONTRACT
------------
A fingerprint over both the extractor's CONFIGURATION (the keyword table, the
age bands, the unknown-age default) and its BEHAVIOUR (the vectors it returns
for a fixed probe set). Recorded into the model artifact at training time
(`model.build_artifact`), re-computed from the code at load time, and compared.
Any edit that changes what the extractor produces — a keyword, a threshold, the
fuzzy matcher, the unknown-age default — moves the fingerprint, and the
artifact is rejected and retrained rather than served features it never saw.

The probes must LIGHT UP every symptom category, or an edit to an unprobed one
is invisible to the hash; `tests/test_feature_contract.py` asserts exactly that
coverage, so the probe set cannot silently rot as categories are added.

Run it:  python -m app.ml.feature_contract            (print the contract)
         python -m app.ml.feature_contract --check    (gate the served artifact)
"""
from __future__ import annotations

import hashlib
import json
import sys

#: (text, age_band, sex) probes, one per symptom category plus the edge cases
#: that have their own branch in `extract_features`. Hand-written and fixed:
#: deriving them from FEATURE_KEYWORDS would move the probe and the vector in
#: lockstep, and a fingerprint that changes with its own input proves nothing.
PROBES: tuple[tuple[str, str | None, str | None], ...] = (
    ("crushing chest pain radiating to the left arm", "40-64", "Male"),
    ("cannot breathe, gasping for air", "65+", "Female"),
    ("face droop and slurred speech since this morning", "65+", "Male"),
    ("severe bleeding that won't stop bleeding through the dressing", "18-39", "Male"),
    ("throat swelling after a bee sting, used my epipen", "18-39", "Female"),
    ("i want to die and keep thinking about self harm", "18-39", "Female"),
    ("had a seizure and convulsion this morning", "40-64", "Male"),
    ("high fever and temperature for days, getting worse", "0-17", "Female"),
    ("vomiting and nauseous, cannot keep fluids down, dehydrated", "0-17", "Male"),
    ("severe pain in my abdominal pain area, excruciating", "40-64", "Female"),
    ("headache and migraine with dizziness and lightheaded", "18-39", "Female"),
    ("itchy skin rash and a sore throat with a cough", "0-17", "Male"),
    ("runny nose, congestion and sneezing from a common cold", "18-39", "Male"),
    ("minor cut and a graze on my knee with some bruising", "40-64", "Male"),
    ("sprained my ankle, twisted my knee, pulled a muscle", "18-39", "Female"),
    # The categories added by the 22 -> 51 widening (0daf5e2) that no probe above
    # already fires, one probe each:
    ("found him unconscious and unresponsive on the floor", "65+", "Male"),
    ("choking on food, something stuck in my throat", "40-64", "Female"),
    ("hit by a car in a road accident", "18-39", "Male"),
    ("fainted and passed out at work", "18-39", "Female"),
    ("palpitations, my heart racing and pounding", "40-64", "Female"),
    ("wheezing badly, asthma attack and inhaler not working", "0-17", "Male"),
    ("sudden vision loss, cannot see out of one eye", "65+", "Female"),
    ("testicular pain and a swollen testicle", "18-39", "Male"),
    ("30 weeks pregnant and bleeding with contractions", "18-39", "Female"),
    ("allergic reaction with hives and swollen lips", "0-17", "Female"),
    ("scalded my hand, burnt with hot oil", "40-64", "Male"),
    ("broken bone, fractured my wrist, bone sticking out", "0-17", "Male"),
    ("swollen calf and leg swelling after a long flight", "40-64", "Female"),
    ("my child swallowed a coin, object inside", "0-17", "Male"),
    ("painful urination, burning when i pee", "18-39", "Female"),
    ("diarrhoea and loose stools since last night", "18-39", "Male"),
    ("lower back pain, backache for a week", "40-64", "Male"),
    ("red eye and eye pain, eyes hurt in the light", "18-39", "Female"),
    ("earache and ear pain after swimming", "0-17", "Female"),
    ("toothache, tooth pain keeping me awake", "40-64", "Female"),
    ("panic attack, anxiety attack, cannot cope", "18-39", "Male"),
    ("knee pain and shoulder pain, joint pain when climbing stairs", "65+", "Female"),
    ("an abscess and infected wound on my arm", "40-64", "Male"),
    ("constipated, cannot pass motion for five days", "65+", "Male"),
    ("heavy period with period cramps and menstrual pain", "18-39", "Female"),
    ("body aches and muscle ache, aching all over", "18-39", "Male"),
    ("heartburn and acid reflux after meals, gastric", "40-64", "Female"),
    ("no energy, lethargic and tired all the time, fatigue", "65+", "Female"),
    # Edge cases with their own branch, each of which has silently changed
    # behaviour in this project before:
    ("", None, None),                                   # empty text
    ("chest paiin and siezure", None, None),            # fuzzy/typo matcher
    ("mild sore throat since yesterday", "unknown", "Other"),   # unknown band/sex defaults
    ("a" * 600, "0-17", "Female"),                      # text_length clamp
)


def vectors() -> list[list[float]]:
    """The extractor's output over the probe set, rounded to 6 decimals.

    Rounded because the hash must be stable across platforms: the numeric
    features are divisions, and an exact float repr is not something to make a
    release gate depend on.
    """
    from .features import extract_features

    return [
        [round(float(v), 6) for v in extract_features(text, band, sex)]
        for text, band, sex in PROBES
    ]


def fingerprint() -> str:
    """SHA-256 over the extractor's CONFIGURATION and its BEHAVIOUR.

    Two halves, because neither is sufficient on its own:

    * **Configuration** — the keyword table, the age bands and the unknown-age
      default. Catches a keyword added to an existing category, which changes
      what the model sees in production and need not change any probe.
    * **Behaviour** — the probe vectors. Catches a change in the matcher logic
      (the fuzzy tiers, the clamps, the one-hot construction) that leaves every
      table untouched.

    What neither catches: a logic change whose effect no probe happens to
    exercise. That is why probe COVERAGE is asserted in the tests rather than
    left to whoever last edited this list.
    """
    from .features import _DEFAULT_BAND, AGE_BANDS, FEATURE_KEYWORDS, FEATURE_NAMES

    material = json.dumps(
        {
            "featureNames": list(FEATURE_NAMES),
            "keywords": {name: list(words) for name, words in FEATURE_KEYWORDS.items()},
            "ageBands": list(AGE_BANDS),
            "defaultBand": _DEFAULT_BAND,
            "vectors": vectors(),
        },
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def contract() -> dict:
    """What gets recorded into the model artifact at training time."""
    from .features import FEATURE_NAMES

    return {
        "featureNames": list(FEATURE_NAMES),
        "nFeatures": len(FEATURE_NAMES),
        "probeCount": len(PROBES),
        "fingerprint": fingerprint(),
    }


def check(recorded: dict | None) -> dict:
    """Compare an artifact's recorded contract against the current code.

    Returns `{"ok", "reason", "expected", "actual"}`. FAILS CLOSED on a missing
    contract: an artifact that cannot state what its features meant is exactly
    the case this gate exists for, and rejecting it costs a retrain, not a
    wrong answer. (Every rejection path here already leads to a fresh, seeded
    rebuild in `TriageModel.__init__`, so there is no outage to trade against.)
    """
    current = contract()
    if not isinstance(recorded, dict) or not recorded.get("fingerprint"):
        return {"ok": False, "reason": "artifact records no feature contract",
                "expected": current["fingerprint"], "actual": None}
    if list(recorded.get("featureNames") or []) != current["featureNames"]:
        return {"ok": False, "reason": "feature layout changed (names differ)",
                "expected": current["fingerprint"], "actual": recorded.get("fingerprint")}
    if recorded["fingerprint"] != current["fingerprint"]:
        return {"ok": False,
                "reason": "feature MEANING changed: same names, different extractor output",
                "expected": current["fingerprint"], "actual": recorded["fingerprint"]}
    return {"ok": True, "reason": "train/serve feature contract matches",
            "expected": current["fingerprint"], "actual": recorded["fingerprint"]}


def main(argv: list[str] | None = None) -> int:
    """Print the contract, or gate the artifact that would actually be served."""
    argv = sys.argv[1:] if argv is None else argv
    if "--check" not in argv:
        json.dump(contract(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    from .model import _load_artifact

    payload, path = _load_artifact()
    if payload is None:
        # No artifact means nothing is being served from one, so there is no
        # skew to report. Silent success here would read as "checked and fine",
        # so say which of the two it is.
        print("SKIPPED — no persisted artifact to check (the model would be trained in-process)")
        return 0
    verdict = check(payload.get("featureContract"))
    print(f"artifact: {path}")
    print(f"expected fingerprint: {verdict['expected']}")
    print(f"artifact fingerprint: {verdict['actual']}")
    print(("PASSED — " if verdict["ok"] else "FAILED — ") + verdict["reason"])
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
