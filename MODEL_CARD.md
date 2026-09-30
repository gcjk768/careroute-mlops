# Model Card — CareRoute Triage Severity Classifier

*Following the Google "Model Cards" pattern and the traceability requirements of the Singapore PDPC
Model AI Governance Framework (§4.30). This documents the ML model in
[`app/ml/model.py`](./app/ml/model.py); it is a Practice-Module prototype, **not** a certified
medical device.*

Every falsifiable number below is checked against the live training audit by
`tests/test_model_card.py`, so a retrain that moves a metric fails the build rather than
quietly leaving this page wrong. Live values: `GET /api/fairness`.

## Model details
- **Name / version:** `careroute-triage-rf-33e265607d69 (rf/200, cal/isotonic)` — the version is
  **content-addressed** (a hash of the training data + training config), not a hand-typed `v1`, so
  an identical retrain is idempotent and any data/config change yields a new, non-clobbering
  artifact. Artifact schema **5**.
- **Type:** Multi-class classifier (scikit-learn `RandomForestClassifier`, 200 trees,
  `class_weight` 2× on P1/P2, `min_samples_leaf=5`) + an isotonic `CalibratedClassifierCV` (3-fold) whose argmax is the
  served acuity.
- **Owner:** CareRoute AI team (NUS-ISS Architecting AI Systems, Team 3, Proposal 3)
- **Output:** One of five acuity levels — `P1_RESUSCITATION`, `P2_EMERGENT`, `P3_URGENT`,
  `P4_NON_URGENT`, `P5_SELF_CARE` — with a calibrated confidence, a coarse confidence band, a SHAP
  explanation over reported symptoms, and a single-edit **counterfactual**.
- **Explainer:** `shap.TreeExplainer` (exact tree SHAP) locally; mean |SHAP| + impurity importance
  + partial dependence globally; an in-repo LIME-style local linear surrogate used only for the
  explanation-agreement audit. (The `lime` package is **not** a dependency and never was.)
- **Integrity:** `dataSha256` `f623aa41afef5f71b447dee53a69bc22b849e0f1de5611b9aae78b441e5f951f`;
  the serialized artifact is hashed to a `.sha256` sidecar, and `_load_artifact` re-hashes the
  file and refuses it (before unpickling) when the sidecar is missing or does not match. The sidecar
  sits beside the file, so this catches corruption and accidental swaps, not an attacker who can
  rewrite both.

## Intended use
- **Intended:** Decision-**support** for triage prioritisation, always behind a deterministic
  safety-override and a human-in-the-loop escalation path. The model **assists**; a clinician decides.
- **Out of scope:** Autonomous diagnosis or treatment, use as a standalone medical device, use on
  populations or presentations unlike the synthetic training distribution, any non-triage task.

## Factors
- **Demographic factors modelled:** age band (`0-17`, `18-39`, `40-64`, `65+`) and sex. Age is used
  as a clinically-justified factor (elderly escalation); sex is a **protected** attribute the model
  should not act on (verified by the counterfactual audit).
- **Serving-time caveat:** when a patient does not give an age band, `extract_features` substitutes
  the neutral `40-64` band, so the age-aware behaviour cannot fire for that case. The live monitor
  reports the share of predictions served that way (`performance.labelled.ageDefaultedShare`).

## Training data
- **Source:** Deterministic **synthetic** dataset (`app/ml/data.py`), seeded and reproducible.
  Full provenance in [`DATASHEET.md`](./DATASHEET.md).
- **Features:** interpretable symptom-category flags + symptom count + description length + demographic
  one-hots. The *same* `extract_features()` is used at train and inference time.
- **Designed bias (for the fairness demo):** the 65+ band is deliberately under-represented **and**
  follows an age-adjusted acuity rule, creating a real, *mitigable* fairness gap.

## Evaluation & metrics
Held-out 25% split (n = 1500), unless stated otherwise.

| Metric | Value |
|---|---|
| Overall accuracy | 0.9047 |
| Red-flag recall (P1/P2 — the safety-critical metric) | 0.9969 |
| Subgroup accuracy gap — before mitigation (age-blind baseline) | 0.5151 |
| Subgroup accuracy gap — after mitigation | 0.1432 |
| Demographic Parity (statistical parity difference) | 0.3777 |
| Equal Opportunity gap (severe-class TPR) | 0.0286 |
| Equalized Odds gap (max of TPR / FPR gap) | 0.1406 |
| Disparate Impact ratio (four-fifths rule) | 0.2886 |
| Counterfactual fairness — sex-flip rate / mean acuity delta | 0.0 / 0.0 |
| Counterfactual fairness, served pipeline (all 4 age bands) — sex-flip rate / mean acuity delta | 0.0 / 0.0 |
| Calibration — ECE / Brier (isotonic, fresh n = 2000 sample) | 0.0246 / 0.1301 |

The first counterfactual row flips sex on the raw forest at the neutral 40-64 band. The second
flips it on the pipeline a patient receives (calibrated argmax + the raise-only thresholds, which
are keyed on age band × sex) for 10 probes × 4 bands = 40 pairs; `TriageModel.predict` reproduces
it end-to-end (`tests/test_subgroup_red_flag_gate.py`).

**Sex-blind scoring (schema 5).** The served probability is the calibrated model averaged over
sex_female = 0 and 1 (`_SexBlind` in `app/ml/model.py`), so sex acts only through the explicit
per-group thresholds. Model `2fccc97c3aa3` scored "dizzy and lightheaded when standing" (18-39) at
P(severe) 0.288 for a woman and 0.328 for a man, either side of the 0.30 raise threshold: P4 versus
P2 for identical text, a served sex-flip rate of 0.05. It is now 0.0, and the worst served subgroup
red-flag recall rose from 0.962 to 0.981.

**A judgement, not a defect.** "30 weeks pregnant, heavy vaginal bleeding with abdominal pain" is
served P1 although the obstetric red-flag rule sets P2: the rule is a floor, and heavy antepartum
bleeding with pain (possible placental abruption) warrants resuscitation-level priority.

**Red-flag recall per subgroup** (held-out P1/P2 cases caught; gated at ≥ 0.95 for every subgroup
with ≥ 20 held-out emergencies, on both pipelines):

| Subgroup | Emergencies (n) | Raw forest | Served pipeline |
|---|---|---|---|
| 0-17 · Female | 47 | 1.0 | 1.0 |
| 0-17 · Male | 47 | 1.0 | 1.0 |
| 18-39 · Female | 42 | 1.0 | 1.0 |
| 18-39 · Male | 46 | 1.0 | 1.0 |
| 40-64 · Female | 32 | 1.0 | 1.0 |
| 40-64 · Male | 32 | 1.0 | 1.0 |
| 65+ · Female | 38 | 1.0 | 1.0 |
| 65+ · Male | 35 | 0.9714 | 0.9714 |

The previous model (`careroute-triage-rf-4e409ad346f3`) cleared the population floor at 0.9747
while 65+ Male sat at **0.8966** (26 of 29). The cause was fully grown trees memorising single
oversampled rows, and the fix is `min_samples_leaf=5` (see `_MIN_SAMPLES_LEAF` in `app/ml/model.py`).
With 29-53 emergencies per subgroup, one case is worth 1.9-3.5 points, so these per-subgroup figures
are noisy. For model `0d11ff43a26c` the worst subgroup per cross-validation fold was 0.929-0.978, so a
subgroup can still dip below 0.95 on another sample — and one did while this model was being chosen:
a candidate dataset that added a persistent-cough P4 profile left 40-64 Male at 0.941 (n = 34) and
was rejected by this gate. (On 2026-09-25 that profile went in as part of a larger set that passes;
two of the eight candidates tried that day failed the same gate.)

The current model (2026-09-24) retrains on three fixes from a live scenario test: keywords that lit
no feature ("burning sensation when urinating", "swollen big toe", "drinking less") and a P3 profile
for fever with poor fluid intake, which "39.5 for three days ... drinking less" (served P5) lacked.

The current model (2026-09-25) adds five training profiles for three extrapolation faults: symptom
pairs the data never contained, where the forest let the MILDER feature decide. "High fever with body
aches" served P5 (aches were only ever in P5 rows), and so did the 65+ "39.5 ... drinking less" case;
a 65+ week-long cough served P2 ("persistent" only ever sat beside fever); a lone mild stomach ache
drifted toward P2 (the strict xfail in `tests/test_ml.py`, now a passing test). New profiles: P3
`high_fever`+`muscle_ache`, `high_fever`+`sore_throat`; P4 `persistent`+`cough`,
`persistent`+`sore_throat`, `abdominal_pain` alone. Accuracy moved 0.912 → 0.9047 because the held-out
set now holds more realistic look-alike cases; red-flag recall rose 0.9906 → 0.9969.
**5-fold stratified cross-validation** of the served pipeline (mean ± std):

| Metric | Mean | Std |
|---|---|---|
| Accuracy | 0.9127 | 0.0092 |
| Red-flag recall | 0.9851 | 0.0085 |
| Subgroup accuracy gap | 0.1308 | 0.0263 |
**Post-processing fairness mitigation** (raise-only, deterministic per-group P(severe) thresholds
derived on an unseen calibration sample; a P3–P5 calibrated argmax is raised to P2 when the
patient's subgroup trails on severe-class recall and P(severe) clears that group's threshold). It
can only ever make a case *more* urgent, so red-flag recall cannot fall. 0.1% of held-out cases are
raised.

| Metric | Before | After |
|---|---|---|
| Equal Opportunity gap | 0.0286 | 0.0286 |
| Equalized Odds gap | 0.1406 | 0.1406 |
| Red-flag recall | 0.9969 | 0.9969 |
| Overall accuracy | 0.9073 | 0.9067 |
| False-positive-rate gap | 0.1406 | 0.1406 |
With the leaf-size fix the calibrated argmax already has a small Equal Opportunity gap, so the
raise-only rule changes few held-out cases (nine here). When it does more, the FPR gap can widen: catching a
trailing group's missed emergencies means escalating more of that group's non-emergencies. Both
directions are reported rather than averaged, because averaging would let one hide behind the other.
When the leaf-size fix landed (model `0d11ff43a26c`), the raw forest's Equalized Odds gap rose from 0.1053 to 0.1404, on the FPR side: the
forest flags more non-emergencies in some subgroups, the cost of catching more of their emergencies.

**Explanation faithfulness** — SHAP vs the in-repo local surrogate, over symptom flags on 160
held-out rows (Krishna et al. 2022 disagreement metrics): same leading reported symptom **0.913** (on the 69
rows that report two or more symptoms — the gated figure), feature agreement@3 over all flags **0.4062**, rank
correlation of magnitudes **0.7196**, sign agreement on reported symptoms **0.9318**.
Feature agreement@3 is reported, not gated: over all 51 flags it ranks what the surrogate says *adding* an
absent symptom would do against SHAP scoring its absence (≈0), so it fell from 0.575 to ~0.39 when the
categories went 22 → 51 while the two methods still agree on which reported symptom leads.

**Out-of-distribution and adversarial-input detection.** An isolation forest over the training
features flags 1.0% of held-out cases (its threshold is that percentile by construction), 1.7% of
the shifted sample and 100.0% of implausible random symptom combinations. A feature-vector validity
check flags 100% of bounded feature-space perturbations. On six real in-budget HopSkipJump examples
the isolation forest caught **none** and the validity check caught **all six**, so only the validity
check is claimed as adversarial-example detection. A flagged case keeps its acuity; its confidence
is capped below the escalation threshold so a person decides.

**Drift** vs a shifted "production" sample — PSI data **0.0743**, target **0.1602**, concept
(accuracy drop) **0.0193**. The shifted sample is *synthetic*, generated at training time; live
drift against real traffic comes from `app/ml/monitor.py` in `CAREROUTE_MONITOR_SOURCE=live` mode.

### Release gates (enforced in `app/ml/train.py` before the artifact is persisted or registered)
| Gate | Threshold | Current |
|---|---|---|
| Overall accuracy | ≥ 0.75 | 0.9047 |
| Red-flag recall | ≥ 0.95 | 0.9969 |
| Red-flag recall, every subgroup with ≥ 20 emergencies (raw / served, worst) | ≥ 0.95 | 0.9714 / 0.9714 |
| Subgroup accuracy gap | ≤ 0.35 | 0.1432 |
| Calibration ECE | ≤ 0.05 | 0.0246 |
| Explanation agreement (leading reported symptom) | ≥ 0.75 | 0.913 |
| Equal Opportunity gap after post-processing | ≤ 0.1 | 0.0286 |
The same thresholds are re-asserted on the **deployed artifact** in CI by
`tests/test_model_gate.py` and `tests/test_fairness_gate.py` (the latter re-derives the fairness
numbers independently with Fairlearn's `MetricFrame`), so the training gate can never become the
weaker of the two.

## Ethical considerations
- Under-triage of a subgroup is the primary harm; the pre-processing (age-band oversampling) and
  post-processing (raise-only group thresholds) mitigations both target it directly, and the
  deterministic red-flag override is a hard safety net independent of the model.
- Disparate Impact is **reported, not gated**: "urgent" here is a clinical finding, not a benefit
  being allocated — the 65+ band *should* be flagged urgent more often — so a 0.8 ratio floor would
  gate against this project's own fairness mitigation.
- Patient identifiers are redacted before inference/storage; explanations are scoped to reported
  symptoms; every decision is recorded to a tamper-evident, hash-chained audit trail.

## Limitations
- Trained on synthetic data — **not** validated on real clinical populations.
- Free-text understanding is bounded by the keyword feature set and the (optional) LLM intake step.
- Confidence is an **isotonic-calibrated** probability, gated at ECE ≤ 0.05. It is calibrated
  against the *synthetic* label distribution — it is **not** a validated clinical risk score.
- The counterfactual is a **single-edit** search over symptom flags on the calibrated model. It
  describes the model, not medicine, and does not consider combinations of changes.
- Model extraction is an accepted, unmitigated risk (see `SECURITY.md`).

## Maintenance
- The API **loads a persisted, content-addressed `joblib` artifact** at startup (`models/`; the file
  is re-hashed against its `.sha256` sidecar and refused on a mismatch or a missing sidecar); it does not retrain on startup. Retraining is explicit — `python -m
  app.ml.train` locally, or the `train:model` CI job — and each run persists a new versioned artifact
  and, when MLflow is reachable, logs metrics + registers the model (`CareRouteTriageRF`).
- Drift is monitored via PSI (Evidently when it imports; the report JSON records which backend ran)
  and gated in CI by a Fairlearn subgroup-parity test. Live **labelled** performance comes from
  clinician ground-truth decisions joined to the inference log by case id.
