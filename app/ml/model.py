"""The trained severity model: real predictions, real SHAP, calibrated
confidence, real fairness/drift + integrity hashing.

MLOps split: training lives in `app/ml/train.py` (a standalone entrypoint that
BUILDS + validates + persists a versioned, integrity-hashed artifact and logs it
to MLflow). At serving time `get_model()` LOADS that persisted artifact from
`CAREROUTE_MODEL_DIR` when a compatible one exists, and only trains from scratch
as a seeded, reproducible fallback (so a loaded model is behaviourally identical
to a freshly trained one). `predict()` returns acuity + calibrated confidence +
a coarse confidence band + a signed SHAP urgency explanation + a single-edit
counterfactual. The training audit also measures how far a LIME-style local
surrogate agrees with SHAP (`explanationAgreement`).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import logging
import os
import threading
from datetime import UTC, datetime
from glob import glob

import numpy as np
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import data, fairness, feature_contract
from .data import ACUITY_INDEX_TO_CODE
from .features import AGE_BANDS, FEATURE_NAMES, extract_features
from .features import label as feature_label

logger = logging.getLogger("careroute.ml")


def _predict_timer():
    """[MLOps] Observe model-inference latency when Prometheus is available;
    a no-op context manager otherwise (keeps the ml layer import-light)."""
    try:
        from .. import metrics
        return metrics.timer(metrics.PREDICT_LATENCY)
    except Exception:  # noqa: BLE001 - optional explainer/persistence dependency; serving must not fail on it
        from contextlib import nullcontext
        return nullcontext()

# Maps acuity index -> urgency direction for the SHAP explanation:
# P1(+1) .. P3(0) .. P5(-1). A feature's urgency contribution is the SHAP-weighted
# sum across classes, so a POSITIVE weight means "pushed toward more urgent".
_URGENCY_WEIGHT = np.array([1.0, 0.5, 0.0, -0.5, -1.0])

# [MLOps] Artifact identity + schema. The concrete `modelVersion` is CONTENT-
# ADDRESSED (derived from a hash of the training data + serialized model — see
# `build_artifact`) rather than a hard-coded "v1", so retraining produces a new,
# non-clobbering artifact and rollbacks are unambiguous. `_ARTIFACT_SCHEMA` is
# bumped whenever the payload layout / feature space changes so an incompatible
# old artifact is rejected on load and safely rebuilt.
_MODEL_FAMILY = "careroute-triage-rf"
_ARTIFACT_SCHEMA = 5  # 5: served scoring is sex-blind (_SexBlind)
_N_ESTIMATORS = 200

# [Responsible-AI][Safety] Cost-sensitive class weights. A MISSED emergency
# (false-negative on the severe P1/P2 classes) is the highest-severity failure
# mode, and P1/P2 are the minority classes in the acuity prior — so the ensemble
# is trained to weight a severe misclassification 2x a routine one. This lifts
# red-flag recall clear of the CI safety gate (>=0.95) with margin, at a
# negligible overall-accuracy cost, and is applied to BOTH the served RF (which
# the fairness audit/gate measures) and the calibrated estimator (whose argmax
# is the acuity a patient actually receives) so the safety gain is real end-to-end.
_SEVERE_CLASS_WEIGHT = {0: 2.0, 1: 2.0, 2: 1.0, 3: 1.0, 4: 1.0}

# [Responsible-AI] Leaf-size regularisation — the subgroup red-flag fix
# (2026-09-24). With the sklearn default of 1 row per leaf, the forest memorised
# single rows. That hurt 65+ most: age-band oversampling duplicates the 15% of
# elderly P3-profile cases that the generator leaves at P3, and a fully grown tree
# can give each duplicate a pure P3 leaf of its own. A 65+ "vomiting +
# dehydrated" case, which is P2 in 10 of 13 training rows, was scored P3 at 0.98.
# Held-out red-flag recall for 65+ Male was 0.897 (26/29).
# Up-weighting 65+ emergencies (1.5x-4x) moved it not at all, because a pure leaf
# ignores weights, which is how the leaf size was found to be the cause. Five
# rows per leaf, measured on the seeded build (raw forest, held-out split):
# 65+ Male 0.897 -> 0.966, accuracy 0.9227 -> 0.9213, population red-flag recall
# 0.9747 -> 0.9873, accuracy gap 0.149 -> 0.101; 5-fold CV accuracy 0.907 -> 0.910
# and worst subgroup per fold 0.846-0.919 -> 0.929-0.978. 3 and 8 were worse on CV.
# It is a property of the whole model, not a per-group rule, so sex plays no part.
_MIN_SAMPLES_LEAF = 5

# [MLOps] Release-gate thresholds — asserted by the training entrypoint
# (app/ml/train.py) and mirrored by the CI fairness/ML tests. The red-flag
# recall floor MATCHES the CI test:model-gate (0.95): the training gate must
# never be weaker than the release gate, or a model that will fail CI gets
# persisted and MLflow-registered first.
MIN_ACCURACY = 0.75
MIN_RED_FLAG_RECALL = 0.95
# [Responsible-AI] The same 0.95 floor also applies to EVERY age-band x sex
# subgroup (train.subgroup_red_flag_shortfalls), on the raw forest and on the
# served pipeline. A subgroup is gated once it has this many held-out emergencies:
# 20 is the smallest n at which one miss still clears 0.95 (19/20). Below it a
# single case decides pass/fail, so the number is reported but not gated. The
# held-out split currently has 29-48 emergencies per subgroup, so all are gated.
MIN_SUBGROUP_SEVERE_SUPPORT = 20
# [XRAI] Calibration ceiling. `confidence` drives the HITL escalation gate, so a
# mis-calibrated model does not just look wrong — it routes wrong. Calibration
# was measured and reported but never GATED: the CI check was `ece >= 0.0`,
# which is true of every float, and train.validate() never looked at it at all.
# 0.05 is the conventional "well calibrated" bar for a 5-class problem and sits
# comfortably above the current isotonic ECE of 0.0216.
MAX_CALIBRATION_ECE = 0.05
# [Responsible-AI] ABSOLUTE subgroup-accuracy ceiling. The training gate only
# compared the mitigated model against the age-blind straw-man baseline (gap
# ~0.549), so ANY number under that passed — a regression from 0.17 to 0.45
# cleared train.py's gate and was caught only by ai-security:fairness-gate, a
# LATER stage, i.e. after the artifact had already been persisted and MLflow-
# registered. This is the same 0.35 that gate asserts (tests/test_fairness_gate.py
# imports THIS constant, so the two can no longer drift apart), moved earlier so
# a fairness regression never reaches the registry. Current gap: 0.1007.
MAX_FAIRNESS_GAP = 0.35
# [XRAI] Explanation-agreement floor: on held-out rows that report two or more
# symptoms, how often the SHAP attribution and an independent LIME-style local
# surrogate name the SAME reported symptom as the strongest driver. Two methods
# that disagree about which symptom drove a prediction mean the explanation
# shown to a patient is not evidence; the release gate refuses such a model.
# Measured 0.86-0.98 across five sampling seeds on the seeded 51-category build
# (0.99-1.00 on the 22-category one it replaced; ~60 multi-symptom rows per
# sample, so one seed moves it a few points); the floor sits below that with
# margin so a seed-level wobble cannot fail the gate while a genuine collapse
# (e.g. a corrupted explainer, which would sit near the 0.4-0.5 chance level
# of a two- or three-way pick) does.
# Until 2026-09-22 this floored feature agreement@3 over ALL symptom flags at
# 0.4 — see model._explanation_agreement for why that number fell to 0.31 when
# the categories went 22 -> 51 without the explanations getting any worse.
MIN_EXPLANATION_AGREEMENT = 0.75
# [Responsible-AI] Equal Opportunity ceiling AFTER post-processing. fairness.py
# itself calls the severe-class TPR gap "the metric that matters most" in triage
# (a gap means one group's emergencies are missed more often), yet nothing
# gated it: 0.079 was measured and merely reported. Gated on the calibrated,
# post-processed pipeline — the acuity a patient actually receives.
MAX_EQUAL_OPPORTUNITY_GAP = 0.10

# [XRAI] Coarse confidence-band cut points (calibrated confidence -> label).
_BAND_HIGH = 0.70
_BAND_MODERATE = 0.40

# Neutral, well-represented age band used when probing the sex counterfactual,
# so any output change is attributable to sex alone.
_DEFAULT_AUDIT_BAND = "40-64"

# Representative symptom probes spanning the acuity range, for the
# counterfactual-fairness audit (sex-flip stability).
_COUNTERFACTUAL_PROBES = [
    "crushing chest pain radiating to the arm",
    "can't breathe and lips going blue",
    "severe bleeding that won't stop",
    "high fever and vomiting for three days",
    "persistent headache getting worse",
    "sore throat and a cough",
    "mild runny nose and sneezing",
    "abdominal pain and nausea",
    "dizzy and lightheaded when standing",
    "sudden slurred speech and arm weakness",
]


def feature_vector_valid(x: np.ndarray) -> bool:
    """[AI-Security] Domain constraints every REAL feature vector satisfies.

    `extract_features` only ever produces: 0/1 symptom and demographic flags,
    exactly one age band set, `symptom_count` equal to min(flags, 6) / 6, and
    `text_length` in [0, 1]. A vector that breaks any of these did not come from
    text — it was constructed or perturbed in feature space, which is what an
    evasion attack does. Measured: 6 of 6 in-budget HopSkipJump examples
    violate it, 0 held-out rows do."""
    from .features import AGE_BANDS as _BANDS
    from .features import N_SYMPTOM_FEATURES

    v = np.asarray(x, dtype=float).ravel()
    if v.shape[0] != len(FEATURE_NAMES) or not np.all(np.isfinite(v)):
        return False
    age_cols = [FEATURE_NAMES.index("age_" + b.replace("-", "_").replace("+", "p")) for b in _BANDS]
    flag_cols = list(range(N_SYMPTOM_FEATURES)) + age_cols + [FEATURE_NAMES.index("sex_female")]
    flags = v[flag_cols]
    if np.any(np.abs(flags - np.round(flags)) > 1e-6) or np.any((flags < 0) | (flags > 1)):
        return False
    if round(float(v[age_cols].sum())) != 1:
        return False
    expected_count = min(float(v[:N_SYMPTOM_FEATURES].sum()), 6.0) / 6.0
    if abs(v[FEATURE_NAMES.index("symptom_count")] - expected_count) > 1e-6:
        return False
    text_length = v[FEATURE_NAMES.index("text_length")]
    return bool(0.0 <= text_length <= 1.0)


def _group_key(age_band: str | None, sex: str | None) -> str:
    """The "<band> · <sex>" subgroup label for a served request, using the SAME
    defaults extract_features applies (unknown age → the neutral band; anything
    but "Female" is encoded as not-female), so serving and audit agree."""
    band = age_band if age_band in AGE_BANDS else _DEFAULT_AUDIT_BAND
    return f"{band} · {'Female' if sex == 'Female' else 'Male'}"


def _shap_matrix(explainer, x_row: np.ndarray) -> np.ndarray:
    """Return SHAP values as a (n_features, n_classes) matrix, robust to the
    several output shapes shap has used across versions."""
    raw = explainer.shap_values(x_row, check_additivity=False)
    if isinstance(raw, list):  # list[n_classes] of (1, n_features)
        return np.stack([np.asarray(c)[0] for c in raw], axis=1)
    arr = np.asarray(raw)
    if arr.ndim == 3:  # (1, n_features, n_classes)
        return arr[0]
    if arr.ndim == 2:  # (1, n_features) — single output
        return arr[0][:, None]
    return arr


class TriageModel:
    def __init__(self, payload: dict | None = None) -> None:
        self.model: RandomForestClassifier
        self.calibrated = None
        self.explainer = None
        # [Responsible-AI] Post-processing: per-group P(severe) thresholds
        # (fairness.equal_opportunity_thresholds), applied raise-only in _predict.
        self.severe_thresholds: dict[str, float] = {}
        self.feature_names = FEATURE_NAMES
        self._audit: dict = {}
        # [MLOps] PROVENANCE. Whether this instance LOADED a persisted artifact
        # or silently BUILT one is not an implementation detail — the CI release
        # gate (tests/test_model_gate.py) grades `fairness()`, which is the audit
        # dict pickled at TRAINING time. With the fallback below invisible, that
        # gate happily graded a model it had just trained in-process, so
        # "release gate on the deployed artifact" passed with no artifact
        # present at all. These two attributes let the gate assert what it claims
        # to be testing.
        self.loaded_from_artifact = False
        self.artifact_path: str | None = None
        # [MLOps] LOAD a persisted, compatible artifact if one exists; otherwise
        # BUILD (seeded, reproducible) and persist so the next boot loads it.
        if payload is None:
            payload, path = _load_artifact()
            if payload is None:
                payload = build_artifact()
                saved_path, _ = save_artifact(payload)
                self.artifact_path = saved_path or None
                logger.info("no compatible artifact found — trained fresh: %s",
                            payload["modelVersion"])
            else:
                self.loaded_from_artifact = True
                self.artifact_path = path
                logger.info("loaded model artifact: %s", payload["modelVersion"])
        self._apply(payload)

    # ------------------------------------------------------------------
    def _apply(self, payload: dict) -> None:
        """Rehydrate a (loaded or freshly built) artifact into a serving model.
        The heavy tree/calibrator estimators are persisted; the SHAP explainer
        is rebuilt from them here so it never has to be pickled."""
        self.model = payload["served"]          # raw RF — SHAP + fairness/audit
        self.calibrated = _SexBlind(payload["calibrated"])  # calibrated, sex-blind proba -> confidence
        self._audit = payload["audit"]
        # Trained with every core (fine for a one-off fit); SERVED with one.
        # A forest that keeps n_jobs=-1 spawns and tears down a worker pool on
        # every predict_proba call — 10-26 s per prediction on a laptop, several
        # predictions per case — for a 58-feature row that takes milliseconds.
        _serve_single_threaded(self.model)
        _serve_single_threaded(self.calibrated.estimator)
        for calibrated in getattr(self.calibrated, "calibrated_classifiers_", []) or []:
            _serve_single_threaded(getattr(calibrated, "estimator", None))
        self.feature_names = FEATURE_NAMES
        self.explainer = shap.TreeExplainer(self.model)
        self.severe_thresholds = dict(payload.get("severeThresholds") or {})
        # [AI-Security] Novelty detector + its threshold (see build_artifact).
        self.novelty = payload.get("novelty")
        self.novelty_threshold = float(payload.get("noveltyThreshold", float("-inf")))

    # ------------------------------------------------------------------
    def predict(self, text: str, age_band: str | None = None, sex: str | None = None,
                *, case_id: str | None = None) -> dict:
        """Predict acuity + confidence + a SHAP-derived urgency explanation.

        Optional demographics (age_band, sex) activate the age-aware acuity rule
        for the elderly; when omitted, the model uses a neutral adult baseline.
        `case_id` is telemetry only: it lets a later clinician label be joined
        back to this exact prediction (live labelled monitoring).
        """
        with _predict_timer():
            return self._predict(text, age_band, sex, case_id=case_id)

    def _predict(self, text: str, age_band: str | None, sex: str | None,
                 *, case_id: str | None = None) -> dict:
        x = extract_features(text, age_band, sex)
        x_row = x.reshape(1, -1)
        # [XRAI] Confidence comes from the CALIBRATED classifier (isotonic/sigmoid
        # over the RF) so `confidence` is a trustworthy probability, not a raw RF
        # vote share. Argmax of the calibrated proba selects the acuity.
        proba = self.calibrated.predict_proba(x_row)[0]
        idx = int(np.argmax(proba))
        # [Responsible-AI] POST-PROCESSING (raise-only group thresholds). If this
        # patient's demographic group trails on severe-class recall and the
        # calibrated P(severe) clears the group's threshold, a non-severe argmax
        # is raised to P2; confidence then reports P(severe), the probability the
        # decision now rests on. Never lowers an acuity.
        group = _group_key(age_band, sex)
        threshold = float(self.severe_thresholds.get(group, fairness.NEVER_RAISE))
        p_severe = float(proba[:2].sum())
        post_applied = bool(idx > 1 and p_severe >= threshold)
        if post_applied:
            idx = 1
            confidence = p_severe
        else:
            confidence = float(proba[idx])
        acuity_code = ACUITY_INDEX_TO_CODE[idx]

        # Signed per-feature urgency contribution from real SHAP values.
        sv = _shap_matrix(self.explainer, x_row)  # (n_features, n_classes)
        if sv.shape[1] == len(_URGENCY_WEIGHT):
            contrib = sv @ _URGENCY_WEIGHT
        else:  # unexpected shape — fall back to the predicted-class column
            contrib = sv[:, min(idx, sv.shape[1] - 1)]
        max_abs = float(np.max(np.abs(contrib))) or 1.0

        # Patient-facing explanation: the SHAP contributions of the symptoms the
        # patient actually REPORTED (present features), ranked by magnitude.
        # (SHAP also scores absent features, but surfacing "you don't have X" is
        # confusing for a patient — so we scope to reported symptoms. Demographic
        # features are audit inputs and are never shown.)
        present = [
            j for j in range(len(contrib))
            if x[j] > 0.5 and not self.feature_names[j].startswith(("age_", "sex_"))
        ]
        ranked = sorted(present, key=lambda j: abs(contrib[j]), reverse=True)
        explanation = [
            {"feature": feature_label(self.feature_names[j]), "weight": round(float(contrib[j] / max_abs), 3)}
            for j in ranked[:6]
            if abs(contrib[j]) > 1e-9
        ]
        evidence = [feature_label(self.feature_names[j]) for j in ranked[:3]]

        self._record_prediction(
            x, acuity_code, confidence,
            case_id=case_id, age_provided=age_band in AGE_BANDS,
        )
        return {
            "acuity_code": acuity_code,
            "confidence": round(confidence, 3),
            "evidence": evidence,
            "explanation": explanation or [{"feature": "no strong signal", "weight": 0.0}],
            # [XRAI] "What would change this?" — the smallest single symptom edit
            # in each direction that flips the calibrated acuity (Day 2 "why-not /
            # how-to-be-that"; PDPC 5.5). None when no single edit flips it.
            "counterfactual": self._counterfactual(x, idx),
            # [AI-Security] Out-of-distribution verdict. `flagged` makes the
            # classifier doubt its own confidence; it never changes the acuity.
            "outOfDistribution": self._ood_verdict(x_row),
            # [Responsible-AI] Whether the post-processing raise fired for this
            # patient's group, and the threshold it was judged against.
            "postProcessing": {"applied": post_applied, "group": group, "threshold": round(threshold, 2)},
            # [AI-Security] A3: a COARSE confidence label. The frontend shows this
            # band to anonymous patients (clinicians see the precise `confidence`),
            # reducing the fine-grained probability signal exposed to a would-be
            # model-extraction attacker.
            "confidence_band": _confidence_band(confidence),
            # [MLOps] Stable, content-addressed model identity for traceability.
            "modelVersion": self._audit.get("modelVersion", ""),
        }

    def _record_prediction(self, x: np.ndarray, acuity_code: str, confidence: float,
                           *, case_id: str | None = None, age_provided: bool = False) -> None:
        """[MLOps] Pillar 3 — per-prediction telemetry: Prometheus model metrics
        (class distribution / confidence / model version) + a de-identified
        inference-log row that monitor.py can use as the LIVE `current` sample.
        Strictly best-effort — serving must never fail on telemetry."""
        version = self._audit.get("modelVersion", "")
        try:
            from .. import metrics
            metrics.observe_prediction(acuity_code, confidence, version)
        except Exception:
            logger.debug("prediction metric not recorded", exc_info=True)
        try:
            from . import inference_log
            inference_log.log_prediction(
                x, acuity_code, confidence, version,
                datetime.now(UTC).isoformat(),
                case_id=case_id, age_provided=age_provided,
            )
        except Exception:
            logger.debug("inference-log row not written", exc_info=True)

    def _ood_verdict(self, x_row: np.ndarray) -> dict:
        valid = feature_vector_valid(x_row[0])
        score = None
        novel = False
        if self.novelty is not None:
            try:
                score = float(self.novelty.score_samples(x_row)[0])
                novel = score < self.novelty_threshold
            except Exception:  # noqa: BLE001 - a detector fault must not fail serving; treated as not novel
                logger.debug("novelty scoring skipped", exc_info=False)
        return {"validVector": valid, "noveltyScore": round(score, 4) if score is not None else None,
                "novel": bool(novel), "flagged": bool(novel or not valid)}

    def apply_post_processing(self, proba: np.ndarray, subgroups: list[str]) -> np.ndarray:
        """Batch form of the serving-time rule (fairness.apply_severe_thresholds)
        with this model's thresholds — used by the audit and the gate tests."""
        return fairness.apply_severe_thresholds(proba, subgroups, self.severe_thresholds)

    def _counterfactual(self, x: np.ndarray, idx: int) -> dict:
        """[XRAI] Bounded SINGLE-EDIT counterfactual search on the calibrated model.

        For every symptom flag, toggle it once (absent→present, present→absent),
        keep the derived symptom_count consistent, and batch-predict the whole
        neighbourhood in ONE call. Among the edits that change the calibrated
        argmax, report the one with the largest probability shift toward the new
        class in each direction — the smallest, most convincing change a patient
        can act on ("would move to P2 if breathlessness were also reported").
        Demographic flags are NEVER toggled: age and sex are not something a
        patient can change, and an explanation that tells them so is not
        actionable. Deterministic; no optimisation library; ~27 evaluations.
        """
        from .features import N_SYMPTOM_FEATURES

        base = np.asarray(x, dtype=float)
        count_col = FEATURE_NAMES.index("symptom_count")
        Z = np.tile(base, (N_SYMPTOM_FEATURES, 1))
        for j in range(N_SYMPTOM_FEATURES):
            Z[j, j] = 1.0 - Z[j, j]
        Z[:, count_col] = np.minimum(Z[:, :N_SYMPTOM_FEATURES].sum(axis=1), 6.0) / 6.0
        proba = self.calibrated.predict_proba(Z)
        new_idx = proba.argmax(axis=1)

        def _best(direction: int) -> dict | None:
            cands = [
                j for j in range(N_SYMPTOM_FEATURES)
                if new_idx[j] != idx and np.sign(idx - new_idx[j]) == direction
            ]
            if not cands:
                return None
            # Strongest flip first (probability of the new class); tie → lower index.
            j = max(cands, key=lambda k: (float(proba[k, new_idx[k]]), -k))
            to_present = base[j] < 0.5
            return {
                "feature": FEATURE_NAMES[j],
                "label": feature_label(FEATURE_NAMES[j]),
                "to": "present" if to_present else "absent",
                "targetAcuity": ACUITY_INDEX_TO_CODE[int(new_idx[j])],
                "targetConfidence": round(float(proba[j, new_idx[j]]), 3),
            }

        more = _best(+1)   # new index LOWER than idx == more urgent
        less = _best(-1)
        current = ACUITY_INDEX_TO_CODE[idx].split("_")[0]

        def _phrase(edit: dict) -> str:
            target = edit["targetAcuity"].split("_")[0]
            if edit["to"] == "present":
                return f"would move from {current} to {target} if {edit['label']} were also reported"
            return f"would move from {current} to {target} if {edit['label']} were not present"

        parts = [_phrase(e) for e in (more, less) if e]
        sentence = ("This assessment " + "; and ".join(parts) + ".") if parts else (
            "No single reported symptom, added or removed, would change this assessment on its own."
        )
        return {
            "method": "single-edit search over symptom flags on the calibrated model; demographics never edited",
            "current": ACUITY_INDEX_TO_CODE[idx],
            "moreUrgent": more,
            "lessUrgent": less,
            "sentence": sentence,
        }

    def fairness(self) -> dict:
        return dict(self._audit)


# --------------------------------------------------------------------------
# [XRAI] Confidence-band + calibration helpers
# --------------------------------------------------------------------------
def _confidence_band(confidence: float) -> str:
    """Map a calibrated confidence in [0,1] to a coarse label."""
    if confidence >= _BAND_HIGH:
        return "high"
    if confidence >= _BAND_MODERATE:
        return "moderate"
    return "low"


def _expected_calibration_error(proba: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """[XRAI] Confidence-ECE: weighted gap between confidence and accuracy across
    equal-width confidence bins (0 == perfectly calibrated)."""
    proba = np.asarray(proba)
    y = np.asarray(y_true)
    conf = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = max(len(y), 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            continue
        ece += (m.sum() / n) * abs(correct[m].mean() - conf[m].mean())
    return round(float(ece), 4)


def _multiclass_brier(proba: np.ndarray, y_true: np.ndarray, n_classes: int = 5) -> float:
    """[XRAI] Multiclass Brier score (mean squared error vs one-hot truth)."""
    onehot = np.eye(n_classes)[np.asarray(y_true)]
    return round(float(np.mean(np.sum((np.asarray(proba) - onehot) ** 2, axis=1))), 4)


# --------------------------------------------------------------------------
# [Responsible-AI] Counterfactual-fairness audit (model-free helpers so the
# training builder can run them before any TriageModel instance exists).
# --------------------------------------------------------------------------
def _raw_acuity_index(model, text: str, age_band: str, sex: str) -> int:
    x = extract_features(text, age_band, sex).reshape(1, -1)
    return int(np.argmax(model.predict_proba(x)[0]))


def _global_explanation(
    served, calibrated, X: np.ndarray, rs, *, sample_n: int = 300, top_k: int = 10, pdp_k: int = 3,
) -> dict:
    """[XRAI] GLOBAL explanation of the deployed forest, to sit beside the
    per-prediction (local) SHAP values the API already returns.

    Two views of the same question, "what does the model rely on overall?":
      * mean |SHAP| per feature over a held-out sample (model-agnostic,
        additive, the same TreeExplainer that explains one prediction);
      * impurity feature importance from the forest itself (cheap, but biased
        toward high-cardinality features — reported so the two can be compared).
    Plus PARTIAL DEPENDENCE for the top features: the average calibrated
    P(urgent = P1/P2) when that feature is forced to each grid value, holding
    every other feature at its observed value. Most features are binary flags,
    so the grid is {0, 1} and the curve is two numbers a clinician can read.
    """
    idx = rs.choice(len(X), size=min(sample_n, len(X)), replace=False)
    sample = np.asarray(X[idx], dtype=float)

    raw = shap.TreeExplainer(served).shap_values(sample, check_additivity=False)
    if isinstance(raw, list):                      # list[n_classes] of (n, features)
        arr = np.stack([np.asarray(c) for c in raw], axis=-1)
    else:
        arr = np.asarray(raw)
        if arr.ndim == 2:                          # (n, features) — single output
            arr = arr[:, :, None]
    mean_abs = np.abs(arr).mean(axis=(0, 2))       # (features,)
    impurity = np.asarray(served.feature_importances_, dtype=float)
    order = np.argsort(mean_abs)[::-1]

    features = [
        {
            "feature": FEATURE_NAMES[i],
            "label": feature_label(FEATURE_NAMES[i]),
            "meanAbsShap": round(float(mean_abs[i]), 4),
            "impurityImportance": round(float(impurity[i]), 4),
        }
        for i in order[:top_k]
    ]

    partial_dependence = []
    for i in order[:pdp_k]:
        observed = np.unique(sample[:, i])
        grid = observed.tolist() if len(observed) <= 5 else np.quantile(sample[:, i], [0, 0.25, 0.5, 0.75, 1]).tolist()
        p_urgent = []
        for value in grid:
            modified = sample.copy()
            modified[:, i] = value
            proba = calibrated.predict_proba(modified)
            p_urgent.append(round(float(proba[:, :2].sum(axis=1).mean()), 4))
        partial_dependence.append({
            "feature": FEATURE_NAMES[i],
            "label": feature_label(FEATURE_NAMES[i]),
            "grid": [round(float(v), 3) for v in grid],
            "pUrgent": p_urgent,
        })

    return {
        "method": "mean |SHAP| (TreeExplainer) and impurity importance over a held-out sample; "
                  "partial dependence of calibrated P(P1/P2) for the top features",
        "sampleN": len(sample),
        "features": features,
        "partialDependence": partial_dependence,
    }


def _local_surrogate_weights(predict_proba, x: np.ndarray, class_idx: int, rs,
                             *, n_samples: int = 1000, expected_flips: float = 2.0,
                             ridge: float = 1e-2) -> np.ndarray:
    """[XRAI] LIME-style LOCAL linear surrogate, implemented in-repo.

    LIME's recipe (Ribeiro et al. 2016): perturb the instance, query the black
    box, fit a simple model on the neighbourhood weighted by proximity, read its
    coefficients as the local explanation. Here every binary flag (symptoms +
    demographics) is flipped independently, the two derived numerics
    (symptom_count, text_length) are recomputed/kept so the perturbed rows stay
    valid feature vectors, proximity is an exponential kernel on Hamming
    distance (LIME's default kernel width, 0.75 * sqrt(d)), and the surrogate
    is a weighted ridge regression of P(class_idx). Returns one coefficient per
    feature (0 for the derived numerics). Deterministic for a given `rs`. No
    external dependency — the `lime` package was never installed, so the
    previous "LIME explanation" was always an empty list.

    The neighbourhood is sized by `expected_flips` — the mean number of flags
    changed per perturbed row — NOT by a per-flag probability. A fixed
    per-flag rate scales the neighbourhood with the feature space: at 0.2 it
    flipped ~5 of 27 flags when there were 22 symptom categories and ~11 of 56
    once there were 51 (2026-09-21), so a "local" neighbour carried eleven
    symptoms, the proximity kernel (width ∝ sqrt(d), so it grew only 1.4x)
    discarded most rows (effective sample size 82 of 300), and a 57-coefficient
    ridge fitted on that ranked the two reported symptoms of a row in the wrong
    order 30% of the time (SHAP vs surrogate, top-1 over reported symptoms:
    0.70; it had been 0.99 at 22 categories). Two expected flips at 1000 rows
    brings that back to 0.86-0.98 depending on the sampled rows. A real patient
    reports one to three symptoms, so
    two flips is what "nearby" means here whatever the category count.
    """
    from .features import N_SYMPTOM_FEATURES

    d = len(FEATURE_NAMES)
    flag_cols = [j for j, name in enumerate(FEATURE_NAMES) if name not in ("symptom_count", "text_length")]
    flip_p = min(0.5, expected_flips / len(flag_cols))
    base = np.asarray(x, dtype=float)
    Z = np.tile(base, (n_samples, 1))
    flips = rs.random((n_samples, len(flag_cols))) < flip_p
    Z[:, flag_cols] = np.where(flips, 1.0 - Z[:, flag_cols], Z[:, flag_cols])
    Z[0] = base  # the instance itself is always in the neighbourhood
    count_col = FEATURE_NAMES.index("symptom_count")
    Z[:, count_col] = np.minimum(Z[:, :N_SYMPTOM_FEATURES].sum(axis=1), 6.0) / 6.0

    target = np.asarray(predict_proba(Z))[:, class_idx]
    hamming = np.abs(Z[:, flag_cols] - base[flag_cols]).sum(axis=1)
    kernel_width = 0.75 * np.sqrt(len(flag_cols))
    w = np.exp(-(hamming ** 2) / (kernel_width ** 2))

    # Weighted ridge on the binary design (intercept + flags).
    A = np.hstack([np.ones((n_samples, 1)), Z[:, flag_cols]])
    Aw = A * w[:, None]
    coef = np.linalg.solve(A.T @ Aw + ridge * np.eye(A.shape[1]), Aw.T @ target)
    out = np.zeros(d, dtype=float)
    out[flag_cols] = coef[1:]
    return out


def _explanation_agreement(served, X: np.ndarray, rs, *, sample_n: int = 160, top_k: int = 3,
                           n_samples: int = 1000) -> dict:
    """[XRAI] Agreement between SHAP (the served explanation) and the local
    surrogate above, on `sample_n` held-out rows, for each row's PREDICTED class.

    Uses the explanation-disagreement metrics of Krishna et al. (2022), "The
    Disagreement Problem in Explainable ML", computed over the symptom flags:
      * reported-symptom agreement — over the symptoms the row REPORTS, do both
        methods name the same one as the strongest driver? Counted on rows
        that report two or more symptoms (with one there is nothing to rank);
        (gated, see below);
      * feature agreement @k  — |top-k(|SHAP|) ∩ top-k(|surrogate|)| / k over
        ALL symptom flags, reported or not (reported, not gated, see below);
      * rank correlation     — Spearman of |SHAP| vs |surrogate| over all
        symptom flags: do they order importance the same way?
      * sign agreement       — over the symptoms the row REPORTS, do both say
        the symptom pushed toward the class or away from it? (Absent symptoms
        are excluded here on purpose: SHAP scores the absence while the
        surrogate scores a hypothetical presence, so their signs differ by
        construction.)
    Two independent attribution methods pointing at the same symptoms is the
    faithfulness/stability evidence the syllabus asks for; reported-symptom
    agreement is floored by MIN_EXPLANATION_AGREEMENT in the release gate.

    Why the gate moved off feature agreement@3 (2026-09-22): the exclusion the
    sign metric always applied to absent symptoms holds for ranking too. For a
    symptom the row does not report, SHAP scores its absence (≈0: absence is
    the baseline) while the surrogate scores what ADDING it would do, which for
    "unresponsive" or "major_trauma" is enormous. With 22 categories the
    reported symptom still topped both lists often enough for @3 to read
    0.575; with 51 there are ~50 such hypotheticals per row and the
    surrogate's top-1 was an absent symptom on 59 of 80 rows, so @3 fell to
    0.31 while the explanations a patient sees had not changed. Comparing the
    two methods where they measure the same thing — the reported symptoms —
    reads 0.99-1.00 at 22 categories and 0.86-0.98 at 51 (five sampling
    seeds). @3 stays in the audit as the literal Krishna et al. figure, with
    that caveat attached."""
    from scipy.stats import spearmanr

    from .features import N_SYMPTOM_FEATURES

    explainer = shap.TreeExplainer(served)
    idx = rs.choice(len(X), size=min(sample_n, len(X)), replace=False)
    flags = slice(0, N_SYMPTOM_FEATURES)
    overlaps: list[float] = []
    rhos: list[float] = []
    reported_hits: list[float] = []
    sign_hits = 0
    sign_total = 0
    for i in idx:
        x = np.asarray(X[i], dtype=float)
        cls = int(np.argmax(served.predict_proba(x.reshape(1, -1))[0]))
        sv = _shap_matrix(explainer, x.reshape(1, -1))
        shap_vec = sv[flags, min(cls, sv.shape[1] - 1)]
        sur_vec = _local_surrogate_weights(served.predict_proba, x, cls, rs, n_samples=n_samples)[flags]
        top_shap = set(np.argsort(np.abs(shap_vec))[::-1][:top_k].tolist())
        top_sur = set(np.argsort(np.abs(sur_vec))[::-1][:top_k].tolist())
        overlaps.append(len(top_shap & top_sur) / top_k)
        rho = spearmanr(np.abs(shap_vec), np.abs(sur_vec)).correlation
        if rho is not None and not np.isnan(rho):
            rhos.append(float(rho))
        reported = np.where(x[flags] > 0.5)[0]
        if len(reported) >= 2:
            lead_shap = reported[int(np.argmax(np.abs(shap_vec[reported])))]
            lead_sur = reported[int(np.argmax(np.abs(sur_vec[reported])))]
            reported_hits.append(float(lead_shap == lead_sur))
        for j in reported:
            sign_total += 1
            sign_hits += int(np.sign(shap_vec[j]) == np.sign(sur_vec[j]))
    return {
        "method": "Krishna et al. (2022) disagreement metrics between TreeExplainer SHAP and an "
                  "in-repo LIME-style local linear surrogate (perturbed neighbourhood, proximity-"
                  "weighted ridge), per held-out row, for the predicted class, over symptom flags",
        "n": len(idx),
        "topK": top_k,
        "perturbations": n_samples,
        # Gated: same leading REPORTED symptom, on rows reporting >= 2 symptoms.
        "reportedTop1Agreement": round(float(np.mean(reported_hits)) if reported_hits else 0.0, 4),
        "nMultiSymptomRows": len(reported_hits),
        # Reported only: over all flags, so absent-symptom hypotheticals (surrogate)
        # are ranked against absence attributions (SHAP) — see the docstring.
        "top3Overlap": round(float(np.mean(overlaps)) if overlaps else 0.0, 4),
        "spearman": round(float(np.mean(rhos)) if rhos else 0.0, 4),
        "signAgreementPresent": round(sign_hits / sign_total, 4) if sign_total else None,
    }


def _served_acuity_index(calibrated, thresholds: dict[str, float], text: str, age_band: str, sex: str) -> int:
    """The acuity index a patient receives: calibrated argmax + the raise-only
    group threshold (same rule as TriageModel._predict)."""
    x = extract_features(text, age_band, sex).reshape(1, -1)
    return int(fairness.apply_severe_thresholds(
        calibrated.predict_proba(x), [_group_key(age_band, sex)], thresholds)[0])


def _counterfactual_audit(predict, sample: list[str] | None = None, bands=(_DEFAULT_AUDIT_BAND,)) -> dict:
    """[Responsible-AI] Counterfactual-fairness check (XRAI Day 2).

    For each probe symptom, hold everything constant and FLIP the protected
    attribute `sex` (Female↔Male). A fair model produces the same acuity, so we
    report the flip rate and the mean absolute acuity delta — both should be ~0.
    (Age is not flipped: the model is designed to escalate the elderly on
    clinical grounds, so an age counterfactual is expected to change output.)

    `predict(text, band, sex) -> acuity index`. The raw-forest audit probes the
    neutral band only. The served-pipeline audit probes EVERY band, because the
    post-processing thresholds are keyed on band x sex, so that is where the
    served pipeline could act on sex when the forest does not.
    """
    probes = sample or _COUNTERFACTUAL_PROBES
    flips = 0
    deltas: list[int] = []
    for text in probes:
        for band in bands:
            f = predict(text, band, "Female")
            m = predict(text, band, "Male")
            flips += f != m
            deltas.append(abs(f - m))
    n = max(len(deltas), 1)
    return {
        "attribute": "sex",
        "n": len(deltas),
        "sexFlipRate": round(flips / n, 4),
        "meanAcuityDelta": round(float(np.mean(deltas)) if deltas else 0.0, 4),
    }


# --------------------------------------------------------------------------
# [MLOps] Reproducible training BUILDER + versioned, integrity-hashed artifact
# --------------------------------------------------------------------------
def _sha256_hex(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return h.hexdigest()


class _SexBlind:
    """[Responsible-AI] The calibrated model, scored with sex held out.

    `predict_proba` averages the model over sex_female = 0 and 1, so the served
    probability cannot depend on sex. The forest does use the sex one-hot, and
    a 0.04 wobble in P(severe) across it ("dizzy and lightheaded when
    standing", 18-39: Female 0.288, Male 0.328) straddled the 0.30 raise-only
    threshold, so flipping sex alone moved a patient P4 -> P2 (served sex-flip
    rate 0.05 on model 2fccc97c3aa3). Sex now acts only through the explicit,
    audited per-group thresholds. Wraps at use; the persisted artifact keeps
    the plain scikit-learn estimator (no custom class in the pickle).
    """

    def __init__(self, estimator) -> None:
        self.estimator = estimator
        self._col = FEATURE_NAMES.index("sex_female")

    def predict_proba(self, X) -> np.ndarray:
        X = np.array(X, dtype=float, ndmin=2)
        female, male = X.copy(), X.copy()
        female[:, self._col], male[:, self._col] = 1.0, 0.0
        return (self.estimator.predict_proba(female) + self.estimator.predict_proba(male)) / 2.0

    def __getattr__(self, name):
        if name == "estimator":  # not yet set (copy/unpickle): no recursion
            raise AttributeError(name)
        return getattr(self.estimator, name)


def _serialize_models(served, calibrated) -> bytes:
    """Serialize the served + calibrated estimators to bytes (for hashing)."""
    import joblib
    buf = io.BytesIO()
    joblib.dump({"served": served, "calibrated": calibrated}, buf)
    return buf.getvalue()




def _fit_served(X_tr: np.ndarray, y_tr: np.ndarray, g_tr: np.ndarray):
    """The served-model PIPELINE: age-band oversampling (pre-processing fairness
    mitigation) + cost-sensitive RandomForest. One function so build_artifact and
    the K-fold cross-validation train exactly the same thing.

    [Responsible-AI] OVERSAMPLE every under-represented age band to parity so the
    model can learn the age-adjusted rule for the elderly and shrink the fairness
    gap. NOTE: iterate bands in a STABLE sorted order (not raw `set(...)`, whose
    iteration order over strings varies with PYTHONHASHSEED per process). The
    order determines how the seeded rng assigns oversample rows to bands, so a raw
    set made X_bal — and thus the trained model and its red-flag recall —
    NON-reproducible across processes (recall wobbled ~0.949-0.955 around the
    0.95 safety gate). Sorting makes the artifact deterministic as documented.
    Returns (fitted_model, X_bal, y_bal)."""
    bands_tr = np.array([g.split(" · ")[0] for g in g_tr])
    counts = {b: int((bands_tr == b).sum()) for b in sorted(set(bands_tr.tolist()))}
    target = max(counts.values())
    rs = np.random.default_rng(0)
    extra_rows = [
        rs.choice(np.where(bands_tr == b)[0], size=target - c, replace=True)
        for b, c in counts.items() if c < target
    ]
    if extra_rows:
        extra_idx = np.concatenate(extra_rows)
        X_bal = np.vstack([X_tr, X_tr[extra_idx]])
        y_bal = np.concatenate([y_tr, y_tr[extra_idx]])
    else:
        X_bal, y_bal = X_tr, y_tr
    served = RandomForestClassifier(
        n_estimators=_N_ESTIMATORS, random_state=42, n_jobs=-1,
        class_weight=_SEVERE_CLASS_WEIGHT,  # [Safety] cost-sensitive: catch P1/P2
        min_samples_leaf=_MIN_SAMPLES_LEAF,  # [Responsible-AI] see _MIN_SAMPLES_LEAF
    )
    served.fit(X_bal, y_bal)
    return served, X_bal, y_bal


def _cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, folds: int = 5) -> dict:
    """[XRAI] K-fold reliability evidence (Day 1: "train/test PLUS K-fold mean ± std").

    Stratified so every fold keeps the acuity prior (P1/P2 are minority classes);
    each fold trains the SAME served pipeline via `_fit_served`, so the numbers
    describe the model that ships, not a simplified stand-in. Calibration is
    excluded (it is a separate estimator whose ECE is gated on its own fresh
    sample). Reports mean, std and the per-fold values for accuracy, red-flag
    recall and the subgroup accuracy gap — the three quantities the release gate
    thresholds, so a reviewer can see how much margin each gate really has."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    acc, recall, gap = [], [], []
    for tr, te in skf.split(X, y):
        model, _, _ = _fit_served(X[tr], y[tr], groups[tr])
        pred = model.predict(X[te])
        acc.append(float(accuracy_score(y[te], pred)))
        recall.append(fairness.red_flag_recall(y[te], pred))
        gap.append(fairness.fairness_gap(fairness.subgroup_accuracies(y[te], pred, groups[te].tolist())))

    def _stat(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=float)
        return {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
            "perFold": [round(float(v), 4) for v in arr],
        }

    return {
        "method": f"StratifiedKFold({folds}, shuffle=True, random_state=42) over the served "
                  "pipeline (age-band oversampling + cost-sensitive RandomForest); "
                  f"(min_samples_leaf={_MIN_SAMPLES_LEAF}); calibration excluded",
        "folds": folds,
        "n": len(y),
        "accuracy": _stat(acc),
        "redFlagRecall": _stat(recall),
        "fairnessGap": _stat(gap),
    }



def build_artifact() -> dict:
    """[MLOps] Train the deployed model end-to-end and return a persistable,
    content-addressed, integrity-hashed artifact payload. Fully seeded, so the
    training data + the content-addressed `versionId` are reproducible and a
    loaded artifact is behaviourally identical to a freshly built one. Used by
    both the serving fallback and app/ml/train.py."""
    # 6000 still, after the symptom categories went 22 -> 51. Raising this to
    # 14000 was tried when red-flag recall dipped and made it WORSE, which is
    # what pointed at the real cause — implausible generated rows, fixed in
    # `data._distractor_pool`, not a shortage of them.
    X, y, groups = data.generate_dataset(n=6000, seed=42)
    groups_arr = np.asarray(groups)
    X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
        X, y, groups_arr, test_size=0.25, stratify=y, random_state=42
    )

    # Baseline (un-mitigated) model — AGE-BLIND: trained on symptom features only,
    # so it structurally cannot account for the age-adjusted acuity of the
    # elderly. This is the "before" model for the fairness gap.
    demo_start = FEATURE_NAMES.index("age_0_17")
    baseline = RandomForestClassifier(n_estimators=_N_ESTIMATORS, random_state=42, n_jobs=-1)
    baseline.fit(X_tr[:, :demo_start], y_tr)

    # [Responsible-AI] Mitigation (pre-processing): oversample under-represented
    # age bands, then fit the cost-sensitive forest — see `_fit_served`. This is
    # the DEPLOYED model.
    served, X_bal, y_bal = _fit_served(X_tr, y_tr, g_tr)
    # Seeded sampler for the audit-time subsamples below (global explanation,
    # surrogate agreement). Separate from the oversampler's rng on purpose so
    # adding an audit never changes which rows the mitigation duplicated.
    rs = np.random.default_rng(0)

    # [XRAI] Probability calibration: an ISOTONIC CalibratedClassifierCV (3-fold)
    # over an identically-configured RF, so served `confidence` is a calibrated
    # probability rather than a raw RF vote share (isotonic keeps feature-sparse,
    # ambiguous input appropriately UNDER-confident, which the HITL gate needs).
    # This is a SEPARATE estimator: the raw `served` RF — and thus the SHAP
    # explainer and the fairness audit below, which run on `served` — is left
    # completely unchanged, so the fairness gate is unaffected by calibration.
    _CAL_METHOD = "isotonic"
    calibrated = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=_N_ESTIMATORS, random_state=42, n_jobs=-1,
            class_weight=_SEVERE_CLASS_WEIGHT,  # [Safety] same cost-sensitivity as served
            min_samples_leaf=_MIN_SAMPLES_LEAF,
        ),
        method=_CAL_METHOD, cv=3,
    )
    calibrated.fit(X_bal, y_bal)
    calibrated_raw, calibrated = calibrated, _SexBlind(calibrated)  # audit what patients receive

    # --- Fairness audit (before = age-blind, after = age-aware + rebalanced) ---
    pred_before = baseline.predict(X_te[:, :demo_start])
    pred_after = served.predict(X_te)
    g_te_list = g_te.tolist()
    sub_before = fairness.subgroup_accuracies(y_te, pred_before, g_te_list)
    sub_after = fairness.subgroup_accuracies(y_te, pred_after, g_te_list)

    # --- Drift audit against a shifted "production" sample ---
    X_shift, y_shift, _ = data.generate_dataset(n=1500, seed=7, shift=True)

    # --- [AI-Security] Out-of-distribution detection (AIC Day 1: adversarial-
    #     example detection). An isolation forest over the TRAINING features
    #     scores how unusual an input is; the threshold is the 1st percentile of
    #     held-out scores, so ~1% of legitimate cases are flagged by construction.
    #     Evaluated on: held-out rows, the shifted sample, implausible random
    #     symptom combinations, and bounded feature-space perturbations (the
    #     shape of an in-budget evasion attack) for the validity check.
    novelty = IsolationForest(n_estimators=_N_ESTIMATORS, contamination="auto", random_state=42)
    novelty.fit(X_bal)
    held_scores = novelty.score_samples(X_te)
    novelty_threshold = float(np.quantile(held_scores, 0.01))
    from .features import N_SYMPTOM_FEATURES as _NSYM
    ood_rng = np.random.default_rng(0)
    implausible = X_te[ood_rng.choice(len(X_te), size=300, replace=False)].copy()
    implausible[:, :_NSYM] = (ood_rng.random((300, _NSYM)) < 0.35).astype(float)
    implausible[:, FEATURE_NAMES.index("symptom_count")] = np.minimum(implausible[:, :_NSYM].sum(axis=1), 6.0) / 6.0
    perturbed = X_te[:300].copy()
    perturbed[:, :_NSYM] += ood_rng.uniform(-0.25, 0.25, size=(300, _NSYM))
    out_of_distribution = {
        "method": "isolation forest novelty score over training features (threshold = 1st percentile "
                  "of held-out scores) + feature-vector validity check (binary flags, one age band, "
                  "consistent symptom count); flagged inputs have their confidence capped below the "
                  "escalation threshold, acuity unchanged",
        "noveltyThreshold": round(novelty_threshold, 4),
        "heldOutFlagRate": round(float((held_scores < novelty_threshold).mean()), 4),
        "shiftedFlagRate": round(float((novelty.score_samples(X_shift) < novelty_threshold).mean()), 4),
        "implausibleComboFlagRate": round(float((novelty.score_samples(implausible) < novelty_threshold).mean()), 4),
        "invalidVectorDetectionRate": round(float(np.mean([not feature_vector_valid(r) for r in perturbed])), 4),
        "note": "measured 2026-09-16 on 6 real in-budget HopSkipJump examples: isolation forest 0/6, "
                "validity check 6/6 — the novelty score is not an adversarial-example detector",
    }
    ref_acc = accuracy_score(y_te, pred_after)
    prod_acc = accuracy_score(y_shift, served.predict(X_shift))

    # --- Calibration reliability metrics on a FRESH, unseen sample ---
    X_cal, y_cal, _ = data.generate_dataset(n=2000, seed=123)
    proba_cal = calibrated.predict_proba(X_cal)
    calibration = {
        "method": _CAL_METHOD,
        "ece": _expected_calibration_error(proba_cal, y_cal),
        "brier": _multiclass_brier(proba_cal, y_cal),
    }

    # --- [Responsible-AI] POST-PROCESSING mitigation: per-group severe thresholds
    #     derived on the UNSEEN calibration sample, audited on the held-out split.
    #     "before" is the calibrated argmax a patient would otherwise receive;
    #     "after" is the same with the raise-only rule applied.
    _, _, g_cal = data.generate_dataset(n=2000, seed=123)
    severe_thresholds = fairness.equal_opportunity_thresholds(
        proba_cal[:, :2].sum(axis=1), proba_cal.argmax(axis=1) <= 1, y_cal, g_cal,
    )
    proba_te = calibrated.predict_proba(X_te)
    pred_cal_before = proba_te.argmax(axis=1)
    pred_cal_after = fairness.apply_severe_thresholds(proba_te, g_te_list, severe_thresholds)

    def _pp_block(pred: np.ndarray) -> dict:
        return {
            "overallAccuracy": round(float(accuracy_score(y_te, pred)), 4),
            "redFlagRecall": fairness.red_flag_recall(y_te, pred),
            "equalOpportunityGap": fairness.equal_opportunity(y_te, pred, g_te_list)["equalOpportunityGap"],
            "equalizedOddsGap": fairness.equalized_odds(y_te, pred, g_te_list)["equalizedOddsGap"],
            "fprGap": fairness.equalized_odds(y_te, pred, g_te_list)["fprGap"],
            "disparateImpactRatio": fairness.disparate_impact(pred, g_te_list)["disparateImpactRatio"],
            "severeRate": round(float(np.isin(pred, (0, 1)).mean()), 4),
        }

    post_processing = {
        "method": "raise-only group-specific P(severe) thresholds (post-processing); a P3–P5 "
                  "calibrated argmax is raised to P2 when the patient's subgroup trails on "
                  "severe-class TPR and P(severe) clears the group's threshold; derived on the "
                  "unseen calibration sample, audited on the held-out split; deterministic",
        "thresholds": severe_thresholds,
        "raisedShare": round(float((pred_cal_after != pred_cal_before).mean()), 4),
        "before": _pp_block(pred_cal_before),
        "after": _pp_block(pred_cal_after),
    }

    # --- [MLOps][AI-Security A4] Integrity hashes: training data + serialized
    #     model. Recorded in metadata and surfaced in the model card/audit so
    #     tampering (e.g. a poisoned artifact swap) is detectable. The model
    #     VERSION is derived from these hashes -> content-addressed, no clobber.
    data_hash = _sha256_hex(
        X.tobytes(), y.tobytes(), ("|".join(FEATURE_NAMES)).encode("utf-8")
    )
    model_bytes = _serialize_models(served, calibrated_raw)
    model_hash = _sha256_hex(model_bytes)
    # The VERSION is content-addressed to the (deterministic) training data +
    # training config, so an identical retrain is idempotent (same version, same
    # file) while ANY data/config change yields a fresh, non-clobbering artifact.
    # `model_hash` is the serialized-artifact integrity hash recorded alongside it
    # (it varies run-to-run due to sklearn's non-deterministic pickle, so it is
    # NOT used for the stable identity — it is the tamper-evidence checksum).
    version_id = _sha256_hex(
        data_hash.encode(), _CAL_METHOD.encode(), str(_N_ESTIMATORS).encode(),
        str(_ARTIFACT_SCHEMA).encode(), f"leaf{_MIN_SAMPLES_LEAF}".encode(),
    )[:12]
    model_version = f"{_MODEL_FAMILY}-{version_id} (rf/{_N_ESTIMATORS}, cal/{_CAL_METHOD})"

    audit = {
        "overallAccuracy": round(float(accuracy_score(y_te, pred_after)), 4),
        "redFlagRecall": fairness.red_flag_recall(y_te, pred_after),
        "fairnessGapBefore": fairness.fairness_gap(sub_before),
        "fairnessGapAfter": fairness.fairness_gap(sub_after),
        "subgroups": [
            {"name": name, "accuracy": v["accuracy"], "n": v["n"]}
            for name, v in sub_after.items()
        ],
        "demographicParity": fairness.demographic_parity(pred_after, g_te_list),
        "equalOpportunity": fairness.equal_opportunity(y_te, pred_after, g_te_list),
        # Equal Opportunity constrains only the TPR; Equalized Odds adds the FPR
        # so over-triaging a group cannot pass as fairness. Disparate Impact is
        # the ratio form of Demographic Parity (the four-fifths rule) — reported,
        # not gated, because age-adjusted acuity is correct medicine here.
        "equalizedOdds": fairness.equalized_odds(y_te, pred_after, g_te_list),
        "disparateImpact": fairness.disparate_impact(pred_after, g_te_list),
        # Sex-flip on the raw forest (neutral band) AND on the served pipeline
        # (calibrated + raise-only band x sex thresholds, every band) — the
        # second is what a patient receives, and the only place sex is a key.
        "counterfactual": _counterfactual_audit(lambda t, b, s: _raw_acuity_index(served, t, b, s)),
        "counterfactualServed": _counterfactual_audit(
            lambda t, b, s: _served_acuity_index(calibrated, severe_thresholds, t, b, s), bands=AGE_BANDS),
        # [Responsible-AI][Safety] Red-flag recall per age-band x sex subgroup,
        # on the raw forest (the audit headline) and the served pipeline. Gated
        # in train.validate: every subgroup with enough emergencies >= 0.95.
        "subgroupRedFlagRecall": {
            "floor": MIN_RED_FLAG_RECALL,
            "minSevereSupport": MIN_SUBGROUP_SEVERE_SUPPORT,
            "raw": fairness.subgroup_red_flag_recall(y_te, pred_after, g_te_list),
            "served": fairness.subgroup_red_flag_recall(y_te, pred_cal_after, g_te_list),
        },
        # [Responsible-AI] Post-processing mitigation, before/after on the
        # calibrated pipeline a patient actually receives. Gated in train.validate
        # (MAX_EQUAL_OPPORTUNITY_GAP on `after`).
        "postProcessing": post_processing,
        # [AI-Security] Out-of-distribution / adversarial-input detection rates.
        "outOfDistribution": out_of_distribution,
        # [XRAI] Global view: what the deployed model relies on overall, and how
        # P(urgent) moves with its top features. Local SHAP explains one case;
        # this explains the model. Served on /api/fairness and the Governance page.
        "globalExplanation": _global_explanation(served, calibrated, X_te, rs),
        # [XRAI] Faithfulness/stability evidence: how far a LIME-style local
        # surrogate agrees with the SHAP attribution on held-out rows. Gated
        # in train.validate (MIN_EXPLANATION_AGREEMENT).
        "explanationAgreement": _explanation_agreement(served, X_te, rs),
        "drift": {
            "data": fairness.data_drift(X_tr, X_shift),
            "target": fairness.target_drift(y_tr, y_shift),
            "concept": fairness.concept_drift(ref_acc, prod_acc),
        },
        "calibration": calibration,
        # [XRAI] K-fold reliability: mean ± std over 5 stratified folds of the
        # served pipeline, beside the single held-out split reported above.
        "crossValidation": _cross_validate(X, y, groups_arr),
        # [MLOps][AI-Security] Integrity block surfaced to the audit/model card.
        "integrity": {
            "schema": _ARTIFACT_SCHEMA,
            "versionId": version_id,
            "dataSha256": data_hash,
            "modelSha256": model_hash,
        },
        "modelVersion": model_version,
        "updatedAt": datetime.now(UTC).isoformat(),
    }
    logger.info(
        "ml model built: acc=%.3f gap %.3f->%.3f redflag_recall=%.3f ece=%.3f ver=%s",
        audit["overallAccuracy"], audit["fairnessGapBefore"], audit["fairnessGapAfter"],
        audit["redFlagRecall"], calibration["ece"], version_id,
    )

    return {
        "schemaVersion": _ARTIFACT_SCHEMA,
        "modelVersion": model_version,
        "versionId": version_id,
        "featureNames": list(FEATURE_NAMES),
        # [MLOps] TRAIN/SERVE SKEW: what the extractor produced at TRAINING
        # time, hashed. `featureNames` above pins the layout; this pins the
        # meaning. See app/ml/feature_contract.py.
        "featureContract": feature_contract.contract(),
        "served": served,
        "calibrated": calibrated_raw,
        "severeThresholds": severe_thresholds,
        "novelty": novelty,
        "noveltyThreshold": novelty_threshold,
        "audit": audit,
        "dataSha256": data_hash,
        "modelSha256": model_hash,
        "createdAt": audit["updatedAt"],
    }


def _models_dir() -> str:
    return os.environ.get("CAREROUTE_MODEL_DIR", "models")


def _compatibility(payload: object) -> tuple[bool, str]:
    """(usable, reason). A loaded payload is served only if it is a
    schema-matching artifact whose feature space is identical to the current
    code's — otherwise it is rejected and rebuilt.

    The `featureNames` check catches a change in feature LAYOUT. The contract
    check catches a change in feature MEANING — a keyword or matcher edit that
    leaves every name and the dimension intact while changing what the numbers
    mean. That is the edit this repository actually makes, and until the
    contract existed it loaded silently. See app/ml/feature_contract.py.
    """
    if not isinstance(payload, dict):
        return False, "not an artifact payload"
    if payload.get("schemaVersion") != _ARTIFACT_SCHEMA:
        return False, f"artifact schema {payload.get('schemaVersion')} != {_ARTIFACT_SCHEMA}"
    if list(payload.get("featureNames") or []) != list(FEATURE_NAMES):
        return False, "feature layout changed (names differ)"
    if "served" not in payload or "calibrated" not in payload:
        return False, "artifact is missing its estimators"
    verdict = feature_contract.check(payload.get("featureContract"))
    return bool(verdict["ok"]), verdict["reason"]


def _is_compatible(payload: object) -> bool:
    return _compatibility(payload)[0]


def _serve_single_threaded(estimator: object) -> None:
    """Set n_jobs=1 on an estimator that has it (and on a calibrator's base
    estimator). Best-effort: an estimator without the attribute is left alone."""
    if estimator is None:
        return
    if hasattr(estimator, "n_jobs"):
        with contextlib.suppress(Exception):  # a read-only estimator attribute is not a serving failure
            estimator.n_jobs = 1
    inner = getattr(estimator, "estimator", None)
    if inner is not None and inner is not estimator:
        _serve_single_threaded(inner)


def _sidecar_matches(path: str) -> bool:
    """True when `path` hashes to the digest recorded in `path.sha256`."""
    try:
        with open(path + ".sha256", encoding="utf-8") as fh:
            recorded = fh.read().split()[0].lower()
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except (OSError, IndexError):
        return False
    return h.hexdigest() == recorded


def _load_artifact() -> tuple[dict | None, str | None]:
    """[MLOps] Load the newest COMPATIBLE persisted artifact, if any. Returns
    (payload, path) — the PATH so callers can record provenance (see
    TriageModel.loaded_from_artifact). Any error (missing dir,
    unreadable/legacy/incompatible file) degrades to (None, None) so the caller
    trains a fresh model — persistence is never allowed to break serving."""
    try:
        import joblib
    except Exception:  # noqa: BLE001 - optional explainer/persistence dependency; serving must not fail on it
        return None, None
    models_dir = _models_dir()
    candidates = sorted(
        glob(os.path.join(models_dir, f"{_MODEL_FAMILY}-*.joblib")),
        key=lambda p: os.path.getmtime(p), reverse=True,
    )
    for path in candidates:
        # [AI-Security A4] Verify the file against the `.sha256` sidecar that
        # save_artifact wrote BEFORE unpickling it: joblib.load executes code, so
        # a swapped or corrupted artifact must be refused unread. A missing
        # sidecar is refused too, or deleting it would switch the check off.
        # ponytail: the sidecar sits beside the file, so this catches corruption
        # and a careless swap, not an attacker who can rewrite both; signing is
        # the upgrade if that threat matters.
        if not _sidecar_matches(path):
            logger.warning("model artifact failed its sha256 check, skipped: %s", path)
            continue
        try:
            payload = joblib.load(path)
        except Exception:
            logger.warning("unreadable model artifact skipped: %s", path, exc_info=True)
            continue
        usable, reason = _compatibility(payload)
        if usable:
            logger.info("model artifact loaded: %s", path)
            return payload, path
        # The REASON matters: "rejected and retrained" and "rejected because the
        # features silently changed meaning" look identical in a log that only
        # says `skipping incompatible artifact`, and only one of them is a
        # train/serve skew event somebody needs to know about.
        logger.warning("skipping incompatible artifact: %s — %s", path, reason)
    return None, None


def save_artifact(payload: dict) -> tuple[str, str]:
    """[MLOps] Persist a built artifact under its content-addressed name and write
    a sidecar SHA-256 of the serialized file for tamper-evidence. Best-effort:
    returns ('', '') if persistence is unavailable, never raising."""
    try:
        models_dir = _models_dir()
        os.makedirs(models_dir, exist_ok=True)
        path = os.path.join(models_dir, f"{_MODEL_FAMILY}-{payload['versionId']}.joblib")
        # [AI-Security] A PLAIN pickle (protocol 4), not joblib.dump: joblib splices
        # raw numpy buffers into the opcode stream, so modelscan and Fickling cannot
        # parse the file and "scan" it as 0 files. joblib.load reads plain pickles,
        # so serving is unchanged. Protocol 5 adds BYTEARRAY8, which Fickling lacks.
        import pickle  # nosec B403  # our own training output; _load_artifact checks the SHA-256 sidecar first
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=4)  # nosemgrep: python.lang.security.deserialization.pickle.avoid-pickle
        with open(path, "rb") as fh:
            artifact_hash = hashlib.sha256(fh.read()).hexdigest()
        with open(path + ".sha256", "w", encoding="utf-8") as fh:
            fh.write(f"{artifact_hash}  {os.path.basename(path)}\n")
        logger.info("model artifact persisted: %s (sha256=%s)", path, artifact_hash[:12])
        return path, artifact_hash
    except Exception:  # noqa: BLE001 - optional explainer/persistence dependency; serving must not fail on it
        logger.warning("model artifact persistence skipped", exc_info=False)
        return "", ""


# --------------------------------------------------------------------------
# Lazy, thread-safe singleton
# --------------------------------------------------------------------------
_model: TriageModel | None = None
_lock = threading.Lock()


def get_model() -> TriageModel:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = TriageModel()
    return _model
