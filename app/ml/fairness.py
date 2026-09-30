"""[Responsible-AI][MLOps] Fairness audit + drift metrics for the severity model.

- `subgroup_accuracies` / `fairness_gap` — stratified performance and the
  max-min accuracy spread across demographic subgroups.
- `red_flag_recall` — recall on the safety-critical severe classes (P1/P2).
- `population_stability_index` — a standard drift metric; we report data,
  target, and concept drift. (Evidently AI is the production-grade drop-in;
  PSI is implemented here to keep the dependency footprint light.)
  `data_drift` reports the WORST per-feature PSI, not the mean — see its
  docstring for why the mean made drift undetectable on this feature space.
"""
from __future__ import annotations

import numpy as np


def subgroup_accuracies(y_true: np.ndarray, y_pred: np.ndarray, subgroups: list[str]) -> dict[str, dict]:
    """Per-subgroup accuracy + support count."""
    groups = np.asarray(subgroups)
    out: dict[str, dict] = {}
    for name in sorted(set(subgroups)):
        mask = groups == name
        n = int(mask.sum())
        if n == 0:
            continue
        acc = float((y_pred[mask] == y_true[mask]).mean())
        out[name] = {"accuracy": round(acc, 4), "n": n}
    return out


def fairness_gap(sub_acc: dict[str, dict]) -> float:
    """Max-min accuracy spread across subgroups (0 == perfectly equal)."""
    if not sub_acc:
        return 0.0
    accs = [v["accuracy"] for v in sub_acc.values()]
    return round(max(accs) - min(accs), 4)


def red_flag_recall(y_true: np.ndarray, y_pred: np.ndarray, severe_indices=(0, 1)) -> float:
    """Recall on the severe classes (P1/P2) — the safety-critical metric."""
    severe = np.isin(y_true, severe_indices)
    if severe.sum() == 0:
        return 1.0
    caught = np.isin(y_pred[severe], severe_indices)
    return round(float(caught.mean()), 4)


def subgroup_red_flag_recall(y_true: np.ndarray, y_pred: np.ndarray, subgroups: list[str],
                             severe_indices=(0, 1)) -> dict[str, dict]:
    """Red-flag recall per subgroup + its support (held-out P1/P2 cases).
    `recall` is None for a subgroup with no emergencies, not a vacuous 1.0."""
    groups = np.asarray(subgroups)
    severe = np.isin(y_true, severe_indices)
    out: dict[str, dict] = {}
    for name in sorted(set(subgroups)):
        pos = (groups == name) & severe
        n = int(pos.sum())
        recall = round(float(np.isin(y_pred[pos], severe_indices).mean()), 4) if n else None
        out[name] = {"recall": recall, "nSevere": n}
    return out


# --------------------------------------------------------------------------
# [Responsible-AI] Named classification-fairness metrics (XRAI Day 2).
#
# The accuracy gap above answers "is the model equally CORRECT across groups?".
# The two metrics below are the ones the course names explicitly and answer a
# different, complementary question — "does the model TREAT groups equally?":
#   - Demographic Parity (a.k.a. Statistical Parity): equal rate of the
#     favourable/positive outcome across subgroups, regardless of ground truth.
#     Here the "positive" outcome is being flagged urgent (P1/P2). Reported as
#     the max-min spread of subgroup positive-rates (Statistical Parity Diff).
#   - Equal Opportunity: equal TRUE-POSITIVE RATE across subgroups for the
#     safety-critical severe classes — i.e. a genuinely urgent patient is just
#     as likely to be caught whether they are elderly or not. Reported as the
#     max-min TPR spread. In healthcare triage this is the metric that matters
#     most: an equal-opportunity gap means one group's emergencies get missed
#     more often than another's.
# --------------------------------------------------------------------------
def demographic_parity(y_pred: np.ndarray, subgroups: list[str], positive_indices=(0, 1)) -> dict:
    """Per-subgroup positive (urgent) rate + Statistical Parity Difference."""
    groups = np.asarray(subgroups)
    rates: dict[str, float] = {}
    for name in sorted(set(subgroups)):
        mask = groups == name
        if mask.sum() == 0:
            continue
        rates[name] = round(float(np.isin(y_pred[mask], positive_indices).mean()), 4)
    spread = round(max(rates.values()) - min(rates.values()), 4) if rates else 0.0
    return {"byGroup": rates, "statisticalParityDifference": spread}


def equal_opportunity(y_true: np.ndarray, y_pred: np.ndarray, subgroups: list[str], severe_indices=(0, 1)) -> dict:
    """Per-subgroup true-positive rate on the severe classes + max-min TPR gap."""
    groups = np.asarray(subgroups)
    tpr: dict[str, float] = {}
    for name in sorted(set(subgroups)):
        mask = (groups == name) & np.isin(y_true, severe_indices)
        if mask.sum() == 0:
            continue
        tpr[name] = round(float(np.isin(y_pred[mask], severe_indices).mean()), 4)
    gap = round(max(tpr.values()) - min(tpr.values()), 4) if tpr else 0.0
    return {"byGroup": tpr, "equalOpportunityGap": gap}


def equalized_odds(y_true: np.ndarray, y_pred: np.ndarray, subgroups: list[str],
                   severe_indices=(0, 1)) -> dict:
    """Equalized Odds — equal TPR **and** equal FPR across subgroups (XRAI Day 2).

    Equal Opportunity above constrains only the true-positive rate, which a model
    can satisfy while over-triaging one group into urgent care: catching every
    elderly emergency by sending half the well elderly to resuscitation scores a
    perfect equal-opportunity gap. Equalized Odds adds the false-positive rate, so
    both error directions have to match.

    Both matter clinically and they are not interchangeable: an FPR gap is a
    queue-and-cost harm (a group is disproportionately escalated), a TPR gap is a
    missed-emergency harm. Reported separately rather than as one number, because
    averaging them would let a large gap in one hide behind parity in the other.
    The headline `equalizedOddsGap` is therefore the MAX of the two, not the mean.
    """
    groups = np.asarray(subgroups)
    severe_true = np.isin(y_true, severe_indices)
    severe_pred = np.isin(y_pred, severe_indices)
    tpr: dict[str, float] = {}
    fpr: dict[str, float] = {}
    for name in sorted(set(subgroups)):
        in_group = groups == name
        positives = in_group & severe_true
        negatives = in_group & ~severe_true
        if positives.sum():
            tpr[name] = round(float(severe_pred[positives].mean()), 4)
        if negatives.sum():
            fpr[name] = round(float(severe_pred[negatives].mean()), 4)
    tpr_gap = round(max(tpr.values()) - min(tpr.values()), 4) if tpr else 0.0
    fpr_gap = round(max(fpr.values()) - min(fpr.values()), 4) if fpr else 0.0
    return {
        "tprByGroup": tpr,
        "fprByGroup": fpr,
        "tprGap": tpr_gap,
        "fprGap": fpr_gap,
        "equalizedOddsGap": round(max(tpr_gap, fpr_gap), 4),
    }


# --------------------------------------------------------------------------
# [Responsible-AI] POST-PROCESSING mitigation (XRAI Day 2: pre / in / post).
#
# Group-specific severe thresholds. The served acuity is the calibrated argmax;
# for a subgroup whose severe-class TRUE-POSITIVE RATE trails the best group,
# a P3–P5 argmax is RAISED to P2 whenever calibrated P(severe) clears that
# group's threshold. Properties, all deliberate:
#   * raise-only  — it can catch a missed emergency, never dismiss one, so the
#                   monotone-escalation safety rule and red-flag recall hold;
#   * deterministic — Fairlearn's ThresholdOptimizer reaches exact parity by
#                   RANDOMISING decisions near the threshold; a patient's acuity
#                   must not depend on a coin flip, so the thresholds here are
#                   fixed numbers derived once at training time;
#   * bounded     — a threshold never drops below `floor`, which caps the FPR
#                   cost of the raise (reported before/after in the audit).
# --------------------------------------------------------------------------
NEVER_RAISE = 1.0


def equal_opportunity_thresholds(p_severe: np.ndarray, pred_severe: np.ndarray, y_true: np.ndarray,
                                 subgroups: list[str], *, severe_indices=(0, 1),
                                 floor: float = 0.30, step: float = 0.01) -> dict[str, float]:
    """Per-group P(severe) thresholds that lift every trailing group's severe
    TPR to the best group's, as far as the `floor` allows.

    `p_severe`    calibrated P(P1 or P2) per row,
    `pred_severe` whether the un-mitigated decision already flags the row severe,
    `y_true`      acuity index, `subgroups` "<band> · <sex>" per row.
    A group already at (or above) the target keeps NEVER_RAISE (1.0). For a
    trailing group the threshold is the HIGHEST value on a `step` grid whose
    raise rule reaches the target TPR; if no value down to `floor` reaches it,
    `floor` is used (best effort, bounded)."""
    groups = np.asarray(subgroups)
    p_severe = np.asarray(p_severe, dtype=float)
    pred_severe = np.asarray(pred_severe, dtype=bool)
    severe_true = np.isin(y_true, severe_indices)

    def _tpr(mask_group: np.ndarray, threshold: float) -> float | None:
        pos = mask_group & severe_true
        if not pos.any():
            return None
        flagged = pred_severe | (p_severe >= threshold)
        return float(flagged[pos].mean())

    names = sorted(set(subgroups))
    base = {g: _tpr(groups == g, NEVER_RAISE) for g in names}
    measured = [v for v in base.values() if v is not None]
    target = max(measured) if measured else 1.0
    out: dict[str, float] = {}
    grid = np.arange(0.99, floor - 1e-9, -step)
    for g in names:
        tpr0 = base[g]
        if tpr0 is None or tpr0 >= target - 1e-12:
            out[g] = NEVER_RAISE
            continue
        chosen = floor
        for t in grid:
            if _tpr(groups == g, float(t)) >= target - 1e-12:
                chosen = float(t)
                break
        out[g] = round(chosen, 2)
    return out


def apply_severe_thresholds(proba: np.ndarray, subgroups: list[str], thresholds: dict[str, float],
                            *, raise_to: int = 1) -> np.ndarray:
    """Apply the post-processing rule to calibrated probabilities. Returns the
    acuity INDEX per row: the argmax, or `raise_to` (P2) when the row's group
    threshold is cleared by P(severe) and the argmax was non-severe. Groups
    without a threshold are left untouched."""
    proba = np.asarray(proba, dtype=float)
    idx = proba.argmax(axis=1)
    p_severe = proba[:, :2].sum(axis=1)
    out = idx.copy()
    for i, g in enumerate(subgroups):
        t = thresholds.get(g, NEVER_RAISE)
        if idx[i] > raise_to and p_severe[i] >= t:
            out[i] = raise_to
    return out


# The "four-fifths rule": US EEOC guidance treats a selection-rate ratio below
# 0.8 as evidence of adverse impact. It is a legal convention, not a statistical
# threshold, and it is reported here rather than gated on -- see the docstring.
DISPARATE_IMPACT_FLOOR = 0.8


def disparate_impact(y_pred: np.ndarray, subgroups: list[str], positive_indices=(0, 1)) -> dict:
    """Disparate Impact — the RATIO of subgroup positive rates (XRAI Day 2).

    Demographic Parity above reports the same quantity as a DIFFERENCE. The ratio
    is not redundant: at low base rates a difference of 0.05 looks negligible and
    can still be a 3x disparity. Both are reported because each is misleading
    alone -- the ratio is unstable when a rate approaches zero, which is exactly
    where the difference is most trustworthy.

    **Reported, not gated.** "Urgent" here is a clinical finding, not a benefit
    being allocated: the elderly SHOULD be flagged urgent more often, because
    age-adjusted acuity is the correct medicine and is why the age-aware model
    exists at all. Enforcing a 0.8 ratio across age bands would gate against the
    mitigation this project deliberately made. The number is surfaced so a
    reviewer can see the disparity and judge whether it is clinical or unfair --
    a judgement no threshold can make.
    """
    groups = np.asarray(subgroups)
    rates: dict[str, float] = {}
    for name in sorted(set(subgroups)):
        mask = groups == name
        if mask.sum() == 0:
            continue
        rates[name] = round(float(np.isin(y_pred[mask], positive_indices).mean()), 4)
    highest = max(rates.values()) if rates else 0.0
    ratio = round(min(rates.values()) / highest, 4) if rates and highest > 0 else 1.0
    return {
        "byGroup": rates,
        "disparateImpactRatio": ratio,
        "fourFifthsFloor": DISPARATE_IMPACT_FLOOR,
        "meetsFourFifthsRule": bool(ratio >= DISPARATE_IMPACT_FLOOR),
    }


# Conventional PSI cut points: < 0.1 no meaningful shift, 0.1-0.25 moderate,
# >= 0.25 "significant shift — investigate". The CT retrain gate in monitor.py
# uses the same number, so the two agree by construction.
DRIFT_PSI_THRESHOLD = 0.25


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10, eps: float = 1e-6) -> float:
    """PSI between a reference and a current 1-D distribution.

    LOW-CARDINALITY COLUMNS ARE HANDLED UP FRONT, before any quantile binning.
    This is load-bearing, not defensive: 27 of the 29 features are 0/1 flags,
    and for a binary column `np.quantile(ref, linspace(0,1,11))` returns only
    {0., 1.}. The old guard tested `edges.size < 2`, but that array has size 2 —
    so the fallback was dead code and np.histogram ran with TWO edges, i.e. ONE
    bin holding 100% of both samples. PSI was then exactly 0.0 for every binary
    feature no matter how far it moved (measured: a 5% -> 95% prevalence flip
    scored 0.0), which made input drift structurally undetectable and left the
    CT retrain trigger unable to ever fire.

    So: <= 2 distinct reference values -> compare the CATEGORY PROPORTIONS
    directly. Anything richer keeps the quantile-binned histogram PSI.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    levels = np.unique(reference)
    if levels.size <= 2:  # binary / constant feature — proportion PSI
        e = np.array([float((reference == v).mean()) for v in levels])
        a = np.array([float((current == v).mean()) for v in levels])
        # Mass in `current` that falls outside the reference's value set is its
        # own bucket, so a brand-new level reads as drift instead of vanishing.
        residual = float(1.0 - a.sum())
        if residual > 1e-12:
            e = np.append(e, 0.0)
            a = np.append(a, residual)
    else:
        edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
        e_counts, _ = np.histogram(reference, bins=edges)
        a_counts, _ = np.histogram(current, bins=edges)
        e = e_counts / max(e_counts.sum(), 1)
        a = a_counts / max(a_counts.sum(), 1)
    e = np.clip(e, eps, None)
    a = np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psis(reference_X: np.ndarray, current_X: np.ndarray) -> list[float]:
    """Per-feature PSI, in feature order."""
    n_features = reference_X.shape[1]
    return [
        population_stability_index(reference_X[:, j], current_X[:, j])
        for j in range(n_features)
    ]


def data_drift(reference_X: np.ndarray, current_X: np.ndarray) -> float:
    """WORST (max) per-feature PSI — overall input-distribution drift.

    Deliberately the MAX, not the mean. This model has 29 features and a real
    shift usually moves one or two of them; averaging over the other 27 divides
    the signal by ~29 and buries it under the 0.25 gate. A single feature
    flipping from 5% to 95% prevalence is a production incident on its own —
    the headline number has to say so. Measured on the standard reference-vs-
    shifted monitoring pair: max 0.0714 vs mean 0.0103 (a 7x difference), and
    that is on top of the binary-PSI fix above, without which the old mean read
    0.0035 because 27 of the 29 features contributed a hard-coded zero.

    Use `data_drift_detail()` when you also want the breadth (how MANY features
    moved), which is the complementary question the mean was a poor proxy for.
    """
    return round(float(max(feature_psis(reference_X, current_X), default=0.0)), 4)


def data_drift_detail(
    reference_X: np.ndarray, current_X: np.ndarray, threshold: float = DRIFT_PSI_THRESHOLD
) -> dict:
    """Full input-drift picture: worst feature (severity), share of features over
    the threshold (breadth), and which ones — so a report can name the culprit
    instead of printing one averaged scalar."""
    psis = feature_psis(reference_X, current_X)
    drifted = [j for j, p in enumerate(psis) if p >= threshold]
    n = max(len(psis), 1)
    return {
        "max": round(float(max(psis, default=0.0)), 4),
        "mean": round(float(np.mean(psis)) if psis else 0.0, 4),
        "threshold": threshold,
        "driftedFeatureCount": len(drifted),
        "driftedShare": round(len(drifted) / n, 4),
        "driftedFeatures": drifted,
    }


def target_drift(reference_y: np.ndarray, current_y: np.ndarray, n_classes: int = 5) -> float:
    """PSI on the label distribution — target/prior drift."""
    ref = np.bincount(reference_y, minlength=n_classes) / max(len(reference_y), 1)
    cur = np.bincount(current_y, minlength=n_classes) / max(len(current_y), 1)
    eps = 1e-6
    ref = np.clip(ref, eps, None)
    cur = np.clip(cur, eps, None)
    return round(float(np.sum((cur - ref) * np.log(cur / ref))), 4)


def concept_drift(reference_acc: float, current_acc: float) -> float:
    """Accuracy drop on the drifted sample — a proxy for concept drift."""
    return round(max(0.0, float(reference_acc - current_acc)), 4)
