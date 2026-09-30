"""Feature extraction for the severity model.

The SAME function is used at training time (app/ml/data.py) and at inference
time (app/ml/model.py), so the SHAP explanation the clinician sees is computed
over exactly the features the model was trained on.

Features:
- interpretable symptom-category flags (human-readable SHAP),
- two numeric signals (symptom count, description length),
- demographic one-hots (age band, sex) — these let the model learn clinically
  realistic subgroup rules (e.g. age-adjusted acuity for the elderly) and are
  what the fairness audit stratifies on. At LIVE inference, when demographics
  aren't collected, age defaults to a neutral, well-represented adult band
  ("40-64") so the input stays in a region the model was trained on.
"""
from __future__ import annotations

import re

import numpy as np

# Ordered symptom-category -> trigger keywords. Order is stable and defines the
# model's feature layout (do not reorder without retraining).
FEATURE_KEYWORDS: dict[str, list[str]] = {
    "chest_pain": ["chest pain", "chest tightness", "chest pressure", "crushing", "tight chest"],
    "breathless": ["can't breathe", "cannot breathe", "breathless", "shortness of breath", "struggling to breathe", "gasping"],
    "stroke_signs": ["face droop", "facial droop", "slurred speech", "weakness one side", "numb arm", "sudden confusion"],
    "severe_bleeding": ["severe bleeding", "heavy bleeding", "won't stop bleeding", "coughing up blood", "vomiting blood"],
    "anaphylaxis": ["anaphylaxis", "throat closing", "throat swelling", "tongue swelling", "epipen"],
    "suicidal": ["suicid", "kill myself", "end my life", "self harm", "self-harm", "want to die", "hurt myself"],
    "seizure": ["seizure", "convulsion", "fitting"],
    "high_fever": ["high fever", "fever", "temperature"],
    "persistent": ["persistent", "for days", "several days", "worsening", "getting worse", "not improving"],
    "vomiting": ["vomit", "throwing up", "nausea", "nauseous"],
    "dehydrated": ["dehydrat", "can't keep fluids", "not drinking", "drinking less",
                   "barely drinking", "fewer wet diapers"],
    "severe_pain": ["severe pain", "worst pain", "excruciating", "unbearable pain"],
    "abdominal_pain": ["abdominal pain", "stomach pain", "stomach hurts", "belly pain",
                       "tummy pain", "tummy hurts"],
    "headache": ["headache", "migraine"],
    "dizziness": ["dizzy", "dizziness", "lightheaded", "light-headed"],
    "rash": ["rash", "itchy skin"],
    "sore_throat": ["sore throat", "throat hurts", "scratchy throat"],
    "cough": ["cough"],
    "cold_symptoms": ["runny nose", "blocked nose", "congestion", "common cold", "sneezing"],
    # Minor trauma. Added after evaluation E5 (app/evals/plan.E5_HITL_TRIGGER)
    # measured specificity for the first time and found 3 of 6 must-NOT-escalate
    # cases being escalated. Root cause: the 19 categories above cover medical
    # presentations only, so EVERY minor-injury complaint lit up zero symptom
    # flags, collapsed to the same feature vector, and scored an identical 0.448
    # confidence — below CONFIDENCE_THRESHOLD, which the HITL gate turns into a
    # clinician interruption. Minor trauma is a large share of real walk-in
    # volume, so its absence was a genuine blind spot, not a fixture artefact.
    # "cut" was a bare keyword here and tier-1 matching is plain substring, so
    # "aCUTe pain" scored as a minor wound. Replaced with the phrasings a
    # patient actually uses; "acute" no longer fires anything.
    "minor_wound": ["minor cut", "small cut", "a cut", "cut my", "cut on my", "deep cut",
                    "laceration", "graze", "grazed", "scrape", "scraped", "blister"],
    "bruise": ["bruise", "bruised", "bruising", "contusion", "banged", "bumped into", "knocked my"],
    "sprain_strain": ["sprain", "sprained", "twisted my ankle", "twisted my knee", "strained", "pulled muscle", "pulled a muscle"],
    # ----------------------------------------------------------------------
    # Presentation categories 23-51, added 2026-09-21 for the same reason the
    # minor-trauma block above was: the feature space decided what the model
    # could see, and everything outside it collapsed to one zero vector.
    #
    # Found on the live console — "something went up my ass" lit up ZERO
    # features, scored 0.43, and Reflection's low-confidence backstop nudged it
    # to P2. Not a wrong answer about the symptom; no answer, because there was
    # no foreign-body category to answer with. The same hole covered urinary,
    # dental, eye, ear, obstetric and mental-health complaints — all ordinary
    # walk-in volume.
    #
    # Keyword rules learned the hard way (tier-1 matching is plain substring,
    # see `_keyword_match`, so a short keyword fires inside longer words):
    #   - no bare "ear"  -> matches "heart", "clear", "fear"
    #   - no bare "uti"  -> matches "routine"
    #   - no bare "pus"  -> matches "push"
    #   - no bare "boil" -> matches "boiling water" (burn_injury)
    #   - no bare "burn" -> matches "burning when I pee" (urinary)
    #   - "stopped breathing", not "not breathing" -> the latter fires on the
    #     very common "not breathing properly", which is breathlessness, not
    #     an arrest.
    # Every keyword below is >= 5 characters or multi-word for that reason.
    # ----------------------------------------------------------------------

    # --- airway / immediately life-threatening ---
    "unresponsive": ["unconscious", "unresponsive", "won't wake up", "wont wake up",
                     "will not wake up", "stopped breathing", "no pulse", "not responding"],
    "choking": ["choking", "choked", "food stuck in throat", "something stuck in my throat",
                "cannot swallow at all", "can't swallow at all"],
    "major_trauma": ["hit by a car", "road accident", "traffic accident", "motorcycle accident",
                     "fell from height", "fell down the stairs", "crush injury", "deep wound",
                     "stabbed", "gunshot"],

    # --- time-critical but not airway ---
    "syncope": ["fainted", "fainting", "passed out", "blacked out", "lost consciousness",
                "collapsed"],
    "palpitations": ["palpitations", "heart racing", "heart pounding", "irregular heartbeat",
                     "skipped beats", "racing pulse"],
    "wheeze_asthma": ["wheezing", "wheeze", "asthma attack", "inhaler not working",
                      "using my inhaler", "asthma flare"],
    "vision_loss": ["sudden vision loss", "lost my vision", "cannot see", "can't see",
                    "double vision", "blurred vision", "curtain over my eye"],
    "testicular_pain": ["testicle pain", "testicular pain", "scrotal pain", "swollen testicle",
                        "pain in my groin"],
    "pregnancy_concern": ["weeks pregnant", "pregnant and bleeding", "contractions",
                          "waters broke", "reduced fetal movement", "baby not moving"],
    "allergic_reaction": ["allergic reaction", "hives", "swollen lips", "swollen face",
                          "face swelling", "came out in welts"],
    "burn_injury": ["burnt", "burned my", "scalded", "scald", "boiling water", "chemical burn",
                    "steam burn", "hot oil"],
    "fracture": ["broken bone", "fracture", "fractured", "bone sticking out", "deformed",
                 "cannot bear weight", "can't bear weight"],
    "limb_swelling": ["swollen leg", "swollen calf", "leg swelling", "swollen ankle",
                      "one leg bigger"],

    # --- urgent, same-day ---
    # `foreign_body` is the category the live P2 case had no way to express.
    "foreign_body": ["swallowed", "stuck inside", "went up my", "object inside", "inserted",
                     "lodged", "stuck in my", "foreign body"],
    "urinary": ["painful urination", "burning when i pee", "burning when i urinate",
                "hurts when i pee", "hurts to pee",
                "cannot pass urine", "can't pass urine", "blood in urine", "urine infection",
                "urinary", "peeing a lot", "passing urine often",
                # "Burning sensation when urinating" lit nothing (live test 2026-09-24).
                "urinating", "when i pee", "dysuria"],
    "diarrhoea": ["diarrhoea", "diarrhea", "loose stools", "loose motion", "watery stools",
                  "running stomach", "passing motion many times"],
    "back_pain": ["back pain", "backache", "back hurts", "lower back", "slipped disc",
                  "back is killing"],
    "eye_problem": ["eye pain", "eye hurts", "eyes hurt", "red eye", "something in my eye",
                    "eye discharge", "swollen eyelid", "eyes are itchy"],
    "ear_problem": ["earache", "ear pain", "ear hurts", "ears hurt", "ear discharge",
                    "blocked ear", "ringing in my ear", "ears are blocked"],
    "dental": ["toothache", "tooth pain", "tooth hurts", "teeth hurt", "my tooth",
               "dental abscess", "gum swelling", "wisdom tooth", "swollen gums"],
    "mental_distress": ["panic attack", "anxiety attack", "cannot cope", "can't cope",
                        "very anxious", "feeling depressed", "cannot sleep at all"],
    "bite_sting": ["dog bite", "cat bite", "animal bite", "bitten by", "bee sting",
                   "insect bite", "snake bite", "wasp sting"],

    # --- routine primary care ---
    "joint_pain": ["joint pain", "knee pain", "knee hurts", "shoulder pain", "shoulder hurts",
                   "joints hurt", "arthritis", "stiff joints", "hip pain",
                   # "Painful swollen big toe" lit nothing (live test 2026-09-24).
                   "big toe", "toe pain", "swollen toe", "painful toe", "gout", "swollen joint"],
    "skin_infection": ["abscess", "a boil", "boils", "infected wound", "oozing pus",
                       "filled with pus", "swollen and red"],
    "constipation": ["constipation", "constipated", "cannot pass motion", "can't pass motion",
                     "hard stools", "not passed motion"],
    "menstrual": ["period pain", "period cramps", "menstrual", "heavy period",
                  "missed my period"],

    # --- self-care band ---
    "muscle_ache": ["body ache", "body aches", "muscle ache", "aching all over",
                    "sore muscles", "muscle soreness"],
    "heartburn": ["heartburn", "acid reflux", "gastric", "indigestion", "reflux",
                  "burping a lot"],
    "fatigue": ["no energy", "lethargic", "tired all the time", "fatigue", "exhausted",
                "worn out"],
}

AGE_BANDS = ["0-17", "18-39", "40-64", "65+"]
_DEFAULT_BAND = "40-64"  # neutral, well-represented band used when age is unknown

_NUMERIC_FEATURES = ["symptom_count", "text_length"]
_DEMO_FEATURES = [f"age_{b.replace('-', '_').replace('+', 'p')}" for b in AGE_BANDS] + ["sex_female"]

FEATURE_NAMES: list[str] = list(FEATURE_KEYWORDS.keys()) + _NUMERIC_FEATURES + _DEMO_FEATURES

FEATURE_LABELS: dict[str, str] = {
    "chest_pain": "chest pain", "breathless": "breathlessness", "stroke_signs": "stroke (FAST) signs",
    "severe_bleeding": "severe bleeding", "anaphylaxis": "anaphylaxis signs", "suicidal": "self-harm risk",
    "seizure": "seizure activity", "high_fever": "fever", "persistent": "persistent / worsening",
    "vomiting": "vomiting / nausea", "dehydrated": "dehydration", "severe_pain": "severe pain",
    "abdominal_pain": "abdominal pain", "headache": "headache", "dizziness": "dizziness", "rash": "rash",
    "sore_throat": "sore throat", "cough": "cough", "cold_symptoms": "cold symptoms",
    "minor_wound": "minor wound / cut", "bruise": "bruising", "sprain_strain": "sprain / strain",
    "unresponsive": "unresponsive / not rousable", "choking": "choking / airway obstruction",
    "major_trauma": "major trauma", "syncope": "fainting / blackout",
    "palpitations": "palpitations", "wheeze_asthma": "wheeze / asthma",
    "vision_loss": "vision disturbance", "testicular_pain": "testicular / groin pain",
    "pregnancy_concern": "pregnancy concern", "allergic_reaction": "allergic reaction",
    "burn_injury": "burn / scald", "fracture": "suspected fracture",
    "limb_swelling": "limb swelling", "foreign_body": "foreign body",
    "urinary": "urinary symptoms", "diarrhoea": "diarrhoea", "back_pain": "back pain",
    "eye_problem": "eye problem", "ear_problem": "ear problem", "dental": "dental problem",
    "mental_distress": "mental distress", "bite_sting": "bite / sting",
    "joint_pain": "joint pain", "skin_infection": "skin infection",
    "constipation": "constipation", "menstrual": "menstrual symptoms",
    "muscle_ache": "muscle aches", "heartburn": "heartburn / reflux", "fatigue": "fatigue",
    "symptom_count": "number of symptoms", "text_length": "description length",
    "age_0_17": "age 0-17", "age_18_39": "age 18-39", "age_40_64": "age 40-64",
    "age_65p": "age 65+", "sex_female": "sex: female",
}


# Number of symptom-category flags at the FRONT of the feature vector — the
# adversarial-evasion signal (below) inspects exactly this slice.
N_SYMPTOM_FEATURES: int = len(FEATURE_KEYWORDS)

# --------------------------------------------------------------------------
# [AI-Security] A1 adversarial-input hardening of the keyword matcher.
#
# A naive `substring` match is trivially defeated by character-level noise a
# real patient (or an adversary probing the model) produces: doubled letters
# ("chest paiin"), truncations ("breathin"), or a one-off typo ("siezure").
# The helpers below add THREE cheap, deterministic matching tiers on top of the
# original exact-substring path, so trivial perturbations still light up the
# right symptom feature WITHOUT changing the feature vector's layout/dimension
# (the model is trained on the SAME extractor via app/ml/data.py, so any change
# here is applied consistently at train and inference time).
# --------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z]+")
_FUZZY_MIN_LEN = 5   # only fuzzy-match reasonably long tokens (short ones over-fire)
_PREFIX_MIN = 6      # shared leading chars that count as "the same word, truncated"
# The one-edit tier is for typos, not for neighbouring English words. Without
# these two bounds it read "cooked" as "choked" (a P1 choking feature on "I
# cooked dinner and my stomach hurts"), "stubbed" as "stabbed", "couch" as
# "cough", "never" as "fever", "lives" as "hives". A typo keeps the start of the
# word; a different word usually does not.
_EDIT_MIN_LEN = 6    # one-edit matches only for words at least this long
_EDIT_PREFIX = 3     # ...that share the keyword's first three letters


_KEYWORD_TOKENS = frozenset(
    tok.replace("'", "")
    for keywords in FEATURE_KEYWORDS.values()
    for kw in keywords
    for tok in re.findall(r"[a-z']+", kw)
)


def _collapse_repeats(s: str) -> str:
    """Collapse any run of a repeated character to a single instance, so
    doubling perturbations normalise to canon: 'paiin'->'pain', 'feverr'->'fever'."""
    out: list[str] = []
    prev = ""
    for ch in s:
        if ch == prev:
            continue
        out.append(ch)
        prev = ch
    return "".join(out)


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):  # intentional: prefix ends at the shorter string
        if x != y:
            break
        n += 1
    return n


def _edit_distance_le1(a: str, b: str) -> bool:
    """True iff Levenshtein(a, b) <= 1. Linear, allocation-free early-outs."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    i = 0
    while i < min(la, lb) and a[i] == b[i]:
        i += 1
    if la == lb:          # single substitution -> remainder must be identical
        return a[i + 1:] == b[i + 1:]
    if la < lb:           # single insertion into `a`
        return a[i:] == b[i + 1:]
    return a[i + 1:] == b[i:]  # single deletion from `a`


def _same_word(kw_tok: str, tok: str) -> bool:
    """Is text token `tok` the keyword token, a truncation of it, or a one-edit
    typo of it? An exact keyword token of ANOTHER category is never a typo
    ("confusion" is a stroke keyword, not a misspelt "contusion")."""
    if tok == kw_tok:
        return True
    if len(tok) < _FUZZY_MIN_LEN or tok in _KEYWORD_TOKENS:
        return False
    if _common_prefix_len(kw_tok, tok) >= _PREFIX_MIN:
        return True
    return (
        len(tok) >= _EDIT_MIN_LEN
        and _common_prefix_len(kw_tok, tok) >= _EDIT_PREFIX
        and _edit_distance_le1(kw_tok, tok)
    )


def _token_fuzzy_match(kw_tok: str, text_tokens: list[str]) -> bool:
    return any(_same_word(kw_tok, tok) for tok in text_tokens)


# Negation. "no fever" is the patient ruling a symptom OUT, and a substring
# match read it as the symptom being reported: a mild sore throat classified as
# P3 on the strength of a fever the patient had denied, with "fever" listed as
# evidence. Same rule as the classifier's keyword path (app/agents/classifier.py
# `_is_negated`), so the two reads of one sentence agree: a cue in the SAME
# clause, within _NEGATION_WINDOW tokens before the phrase, is a denial.
# "no fever, but severe chest pain" still reports the chest pain (clause-bound)
# and "no fever and a dry cough" still reports the cough (window-bound).
# Keywords that are themselves phrased as denials ("no pulse", "not responding")
# are unaffected: the cue must come BEFORE the phrase. The forcing red-flag
# table in app/redflags.py runs separately, so this can never suppress a
# Safety-Override escalation.
_NEGATION_CUES = frozenset({
    "no", "not", "without", "denies", "denied", "deny", "never", "negative",
})
_NEGATION_WINDOW = 3
#: A conjunction starts a new statement: "not improving and scratchy throat"
#: reports the throat, "no fever but a bad cough" reports the cough. "or" is
#: deliberately NOT a boundary, so "no fever or chills" denies both.
_NEGATION_BOUNDARY = frozenset({"and", "but"})
_CLAUSE_SPLIT = re.compile(r"[.,;:!?]")
_CUE_RE = re.compile(r"[a-z']+")


#: "can't stop coughing" reports the cough; the cue negates "stop", not the
#: symptom. "never had chest pain like this before" reports the chest pain; the
#: cue is an intensifier when the phrase is followed by a comparison.
_INTENSIFIER_BEFORE = frozenset({"stop", "stopping", "stopped"})
_INTENSIFIER_AFTER = ("like this", "this bad", "so bad", "before", "as bad")


def _window(tokens: list[str]) -> list[str]:
    """The tokens a denial cue may sit in: after the last conjunction, at most
    the negation window long."""
    for index in range(len(tokens) - 1, -1, -1):
        if tokens[index] in _NEGATION_BOUNDARY:
            tokens = tokens[index + 1:]
            break
    return tokens[-_NEGATION_WINDOW:]


def _is_cue(tok: str) -> bool:
    return tok in _NEGATION_CUES or tok.endswith("n't")


def _denied_by(before: list[str], after: str) -> bool:
    """Denial decision from the tokens BEFORE a phrase and the text AFTER it
    (within the clause). Shared by the substring tiers and the typo tier."""
    window = _window(before)
    if not window or not any(_is_cue(tok) for tok in window):
        return False
    if window[-1] in _INTENSIFIER_BEFORE:
        return False
    return not ("never" in window and any(marker in after for marker in _INTENSIFIER_AFTER))


def _denied(clause: str, start: int, end: int | None = None) -> bool:
    """True when the phrase at clause[start:end] is denied rather than reported."""
    after = clause[end:] if end is not None else ""
    return _denied_by(_CUE_RE.findall(clause[:start]), after)


def _substring_reported(needle: str, haystack: str) -> bool:
    """Is `needle` present in `haystack` at least once WITHOUT being denied?"""
    start = haystack.find(needle)
    while start != -1:
        if not _denied(haystack, start, start + len(needle)):
            return True
        start = haystack.find(needle, start + 1)
    return False


def _keyword_match(kw: str, text_lower: str, text_norm: str, text_tokens: list[str]) -> bool:
    """Tiered, perturbation-robust keyword match (exact -> doubling -> fuzzy),
    reported (not denied) in at least one clause of the text."""
    if _substring_reported(kw, text_lower):        # tier 1: exact (original behaviour)
        return True
    kw_norm = _collapse_repeats(kw)
    if _substring_reported(kw_norm, text_norm):    # tier 2: doubling-robust substring
        return True
    # tier 3: token-level fuzzy. Every keyword token must match a text token;
    # short (<5) tokens must be present verbatim so we don't over-fire. Text
    # tokens keep their apostrophes for the cue check ("don't") and are matched
    # bare ("dont"); keyword tokens are tokenised the same way ("won't" -> "wont").
    # The denial check reads the tokens before the FIRST keyword token's match
    # and the tokens after it.
    kw_tokens = [t.replace("'", "") for t in _CUE_RE.findall(kw_norm)]
    bare = [t.replace("'", "") for t in text_tokens]
    if not kw_tokens:
        return False
    for kt in kw_tokens:
        if len(kt) < _FUZZY_MIN_LEN:
            if kt not in bare:
                return False
        elif not _token_fuzzy_match(kt, bare):
            return False
    first = kw_tokens[0]
    for index, tok in enumerate(bare):
        if _same_word(first, tok) and not _denied_by(text_tokens[:index], " ".join(bare[index + 1:])):
            return True
    return False


def extract_features(text: str, age_band: str | None = None, sex: str | None = None) -> np.ndarray:
    """Return the model's feature vector for symptom text (+ optional demographics)."""
    t = (text or "").lower()
    # Matching is per CLAUSE so a denial cannot reach past a comma (see the
    # negation note above). Pre-compute each clause's normalised forms once so
    # the tiered matcher stays cheap even when called thousands of times during
    # dataset generation.
    clauses = [
        (c, c_norm, _CUE_RE.findall(c_norm))
        for c in _CLAUSE_SPLIT.split(t)
        if c.strip()
        for c_norm in (_collapse_repeats(c),)
    ]
    flags: list[float] = []
    count = 0.0
    for keywords in FEATURE_KEYWORDS.values():
        hit = 1.0 if any(
            _keyword_match(kw, c, c_norm, c_tokens)
            for c, c_norm, c_tokens in clauses
            for kw in keywords
        ) else 0.0
        flags.append(hit)
        count += hit
    symptom_count = min(count, 6.0) / 6.0
    text_length = min(len(t), 400) / 400.0

    band = age_band if age_band in AGE_BANDS else _DEFAULT_BAND
    age_onehot = [1.0 if band == b else 0.0 for b in AGE_BANDS]
    sex_female = 1.0 if sex == "Female" else 0.0

    return np.asarray([*flags, symptom_count, text_length, *age_onehot, sex_female], dtype=float)


def symptom_flag_count(x: np.ndarray) -> int:
    """Number of active symptom-category flags in a feature vector `x`."""
    return int(np.asarray(x, dtype=float)[:N_SYMPTOM_FEATURES].sum())


def has_no_feature_coverage(text: str, age_band: str | None = None, sex: str | None = None) -> bool:
    """[MLOps] MODEL-COVERAGE signal — telemetry only. NEVER use this as a gate.

    True when the input carries real content (>= 3 words) yet lights up ZERO
    symptom features even after the perturbation-robust matching above. That
    means one thing only: **the model has no signal for this input**, because
    nothing the patient wrote maps onto a category in FEATURE_KEYWORDS.

    WHY THIS IS NOT A SECURITY SIGNAL (it used to claim to be)
    ----------------------------------------------------------
    This was `is_suspected_evasion`, and it read a zero-feature vector as
    evidence of adversarial obfuscation. Evaluation E5 disproved that. Three
    ordinary complaints — a bruise, a minor cut, a sprained ankle — were all
    flagged, not because a patient was evading anything but because the feature
    space had no minor-trauma categories at all. After those categories were
    added and the model retrained, the same three sentences flag False.

    A verdict that flips when you RETRAIN THE MODEL, with the input unchanged,
    is not measuring the input. It is measuring coverage. So the name, the
    docstring and the intended use were all wrong, and the function was never
    wired in — which is the only reason the mislabelling was harmless.

    Genuine evasion screening belongs to `app/guardrail.py`, which does it
    independently of the model: Unicode normalisation, leetspeak folding,
    decoded-variant checks, injection patterns, and a graded clinical-relevance
    score. On those same three trauma sentences guardrail scored 0.083-0.10
    (clinical) versus 0.0 for gibberish — the correct separation, and it does
    not move when the model is retrained.

    WHY IT MUST NOT GATE
    --------------------
    1. Redundant. Zero features -> the model has no signal -> low confidence ->
       the existing CONFIDENCE_THRESHOLD rule in the HITL worker already
       escalates. `data.py` reinforces this deliberately: the `_VAGUE_PHRASES`
       rows are labelled across P3/P4/P5 to keep the feature-sparse region
       uncertain.
    2. It would have HIDDEN the defect E5 caught. Had this forced escalation,
       the three trauma cases would have escalated "by design", the specificity
       failure would have looked like a safety feature working, and the model
       would still be blind to minor trauma today.

    WHAT IT IS GOOD FOR
    -------------------
    Aggregate monitoring. A rising `zero_coverage_rate` over live traffic is the
    early warning that real patients are describing something the model cannot
    see — the class of defect E5 only caught by luck. See `ml/monitor.py`.
    """
    if len((text or "").split()) < 3:
        return False  # trivially short input is "empty", not "uncovered"
    return symptom_flag_count(extract_features(text, age_band, sex)) == 0


def zero_coverage_rate(X: np.ndarray) -> float:
    """[MLOps] Fraction of rows in a feature matrix that light up NO symptom
    feature — the aggregate form of `has_no_feature_coverage`.

    This is the number worth watching. Computed from the feature vectors the
    inference log already stores, so it needs no schema change and can be run
    over historical logs. A rate that climbs against the reference distribution
    means live traffic is drifting into territory the model has no features
    for; the fix is a new feature category plus a retrain, not a threshold
    change.
    """
    arr = np.asarray(X, dtype=float)
    if arr.size == 0 or arr.shape[0] == 0:
        return 0.0
    uncovered = (arr[:, :N_SYMPTOM_FEATURES].sum(axis=1) == 0).sum()
    return round(float(uncovered) / float(arr.shape[0]), 4)


def label(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " "))
