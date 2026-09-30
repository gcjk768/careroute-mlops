"""[AI-Security] Model-extraction probe — AIC Day 1, privacy attacks at query time.

Model extraction (model stealing) is the third of the deck's classical attack
classes: an attacker with only QUERY access trains a surrogate on the deployed
model's answers until the surrogate reproduces its decisions. `SECURITY.md`
records this as an *accepted risk* and quotes an agreement curve (~100 / 500 /
2000 queries). Until 2026-09-15 nothing in the repository produced those
numbers, so the claim was prose. This module is the producing code:

    python -m app.ml.extraction

Attacker model (deliberately the strongest black-box case):
  * knows the feature schema (it is published in MODEL_CARD.md) and can draw
    inputs from the same distribution the model serves — `data.generate_dataset`
    is used as that distribution, exactly as the fairness audit does;
  * sees ONLY what the API returns for each query: the acuity label (argmax of
    the calibrated probabilities, which is what `TriageModel.predict` exposes) —
    never probabilities, never the trees;
  * trains a random forest on (query, answer) pairs and is scored by how often
    it agrees with the deployed model on a fresh, unseen sample.

Agreement is against the deployed model's OWN answers, not ground truth: a
clone that matches the model's mistakes is a successful clone.

`tests/test_ml_attacks.py` pins the curve so the SECURITY.md table can never
drift from what the code measures. The defence position is unchanged and stated
there: the rate limiter buys time and noise, not prevention, and the model is
not the asset.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Sequence

import numpy as np

from . import data

DEFAULT_QUERY_BUDGETS: tuple[int, ...] = (100, 500, 2000)


def probe(
    query_budgets: Sequence[int] = DEFAULT_QUERY_BUDGETS,
    *,
    seed: int = 7,
    holdout_n: int = 2000,
    model=None,
) -> dict:
    """Train a surrogate at each query budget and measure agreement.

    Returns {"rows": [{"queries", "agreement"}], "holdoutN", "seed",
    "modelVersion"}. `model` defaults to the served `TriageModel`; tests pass a
    freshly built one so the probe never depends on a local artifact.
    """
    from sklearn.ensemble import RandomForestClassifier

    if model is None:
        from .model import get_model

        model = get_model()

    budgets = sorted(int(b) for b in query_budgets)
    if not budgets or budgets[0] < 1:
        raise ValueError("query_budgets must be positive integers")

    # What the attacker can send: inputs from the served distribution.
    X_query, _, _ = data.generate_dataset(n=budgets[-1], seed=seed)
    # What the attacker gets back: the label, nothing else.
    y_oracle = _oracle_labels(model, X_query)

    # A fresh sample the attacker never queried, scored against the deployed
    # model's own answers.
    X_hold, _, _ = data.generate_dataset(n=holdout_n, seed=seed + 1)
    y_hold = _oracle_labels(model, X_hold)

    rows = []
    for budget in budgets:
        surrogate = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)
        surrogate.fit(X_query[:budget], y_oracle[:budget])
        agreement = float(np.mean(surrogate.predict(X_hold) == y_hold))
        rows.append({"queries": budget, "agreement": round(agreement, 4)})

    return {
        "rows": rows,
        "holdoutN": int(holdout_n),
        "seed": int(seed),
        "modelVersion": _model_version(model),
        "attackerSees": "acuity label only (no probabilities, no model access)",
    }


def _model_version(model) -> str:
    """The content-addressed version the audit records (TriageModel keeps it in
    its audit dict, not as an attribute)."""
    try:
        return str(model.fairness().get("modelVersion") or "unknown")
    except Exception:  # noqa: BLE001 - a version label must never fail the probe
        return "unknown"


def _oracle_labels(model, X: np.ndarray) -> np.ndarray:
    """The label the API would return for each row — calibrated argmax, exactly
    as `TriageModel._predict` selects the acuity."""
    return np.asarray(model.calibrated.predict_proba(X)).argmax(axis=1)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    budgets = [int(a) for a in args] if args else list(DEFAULT_QUERY_BUDGETS)
    result = probe(budgets)

    print("Model-extraction probe (AIC Day 1 — privacy: model stealing)")
    print(f"deployed model : {result['modelVersion']}")
    print(f"attacker sees  : {result['attackerSees']}")
    print(f"scored on      : {result['holdoutN']} unseen rows, agreement with the deployed model's answers")
    print()
    print(f"{'queries':>8}  {'agreement':>9}")
    for row in result["rows"]:
        print(f"{row['queries']:>8}  {row['agreement']*100:>8.1f}%")
    print()
    print(json.dumps(result))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
