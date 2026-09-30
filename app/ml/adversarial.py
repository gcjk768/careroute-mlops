"""[AI-Security] Adversarial re-training — AIC Day 1's model-side defence.

The deck names two defences that change the model itself (as opposed to guarding
its inputs and outputs):

  * **Ensemble learning** — already satisfied: the deployed model is a 200-tree
    RandomForest, and the deck's own argument for ensembles is that an attack
    tuned to one decision boundary does not transfer to many.
  * **Adversarial re-training** — "inject adversarial examples during training
    with correct labels". That is what this module does.

WHY THIS IS A MODULE AND NOT THE DEPLOYED TRAINING PATH
-------------------------------------------------------
Hardening is not free, and the trade is one a clinical model cannot make blind:
adversarial re-training buys robustness by fitting the boundary around perturbed
points, which costs some clean accuracy. CareRoute's release gate holds a
red-flag-recall floor of 0.95, so a hardening step that shaved recall would be
rejected by the gate anyway — correctly.

So this is built as a MEASURABLE EXPERIMENT the pipeline can run and report,
rather than a silent change to the served model. Promoting it is a decision that
needs the measurement in hand, and the measurement is what this provides.

WHAT IT MEASURED (2026-09-12, n=3000, 40 attack attempts)
----------------------------------------------------------
    baseline forest       clean 0.9093   robust 0.7917
    adversarially trained clean 0.9093   robust 0.4167

Two things make that table unusable as evidence, and both are the finding:

1. **Only 6 of 40 attack attempts landed inside the budget.** "Re-train on
   adversarial examples" yielded SIX rows against ~2,250 training rows. Clean
   accuracy is byte-identical before and after, which is what six rows out of
   2,256 should do.

2. **The metric's own run-to-run noise is larger than most effects it could
   detect.** The SAME baseline model, measured twice by `robust_accuracy`,
   scored 0.8333 and 0.7917 — HopSkipJump is stochastic and 24 attacked samples
   is a small denominator. A comparison whose instrument moves by 0.04 on a
   fixed model cannot be read confidently.

So this is not "adversarial re-training harmed the model". It is "at a
CI-affordable attack budget this experiment cannot measure whether it helped or
harmed", and a defence adopted — or rejected — on that basis would be a guess
wearing a number.

The deployed model is therefore unchanged, on the basis of a measurement rather
than an omission. Re-running with a friendlier seed until the defence "works"
would be the easy move, and is exactly what AIC Day 1's red-teaming methodology
section exists to rule out. Making this conclusive needs one to two orders of
magnitude more attack budget than CI can carry.

`adversarial-robustness-toolbox` is a dev/test dependency; import it lazily so
`app.ml` stays importable at serving time without it.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Perturbation budget in L-infinity. Features live in [0, 1] and a symptom flag
# needs a >= 0.5 change to flip, so 0.25 keeps "adversarial" meaning a small
# nudge rather than a genuinely different clinical presentation. Same constant
# as tests/test_robustness.py, deliberately: a defence measured against a looser
# budget than the gate uses would report a robustness it has not earned.
EPSILON = 0.25


def _classifier(model):
    from art.estimators.classification import SklearnClassifier

    return SklearnClassifier(model=model, clip_values=(0.0, 1.0))


def generate_adversarial_examples(
    model, X: np.ndarray, *, n: int = 40, max_iter: int = 8, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Craft in-budget adversarial variants of `n` rows sampled from `X`.

    Returns `(x_adv, source_idx)` — the adversarial rows AND the index into `X` of
    the clean row each one came from.

    Returning the indices is not a convenience. The in-budget filter drops rows,
    so the survivors no longer line up positionally with the sample that produced
    them; a caller that re-derived the sample and truncated it would hand row 5's
    label to row 2's attack. That is silently mislabelled training data — the
    poisoning failure this module's whole point is to avoid.

    Black-box HopSkipJump, matching the attack the robustness gate uses: hardening
    against a weaker attack than the one you are graded on is self-deception.
    Only perturbations within EPSILON are kept — a larger change is a different
    presentation, not an attack.
    """
    from art.attacks.evasion import HopSkipJump

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    x0 = X[idx].astype(np.float32)

    attack = HopSkipJump(
        classifier=_classifier(model), targeted=False,
        max_iter=max_iter, max_eval=200, init_eval=10, init_size=10,
    )
    x_adv = attack.generate(x=x0)
    in_budget = np.max(np.abs(x_adv - x0), axis=1) <= EPSILON
    return x_adv[in_budget], idx[in_budget]


def adversarially_retrain(
    model, X: np.ndarray, y: np.ndarray, *, n: int = 40, seed: int = 42, **forest_kwargs
) -> tuple[RandomForestClassifier, int]:
    """Return a model retrained on the data PLUS adversarial examples.

    The adversarial rows carry the label of the clean row they were derived from
    — "with correct labels", per the deck. Labelling them by what the victim model
    predicts would teach the new model to reproduce the original's mistake, which
    is the one way to make this step actively harmful.
    """
    x_adv, source_idx = generate_adversarial_examples(model, X, n=n, seed=seed)
    if len(x_adv) == 0:
        return model, 0

    # `source_idx` comes back from the generator rather than being re-derived
    # here, so each adversarial row is labelled by the exact clean row it was
    # perturbed from even though the in-budget filter dropped some.
    X_aug = np.vstack([X, x_adv])
    y_aug = np.concatenate([y, y[source_idx]])

    params = {"n_estimators": 200, "n_jobs": -1, **forest_kwargs}
    seed = params.pop("random_state", 42)  # explicit, so the seed is visible at the call
    hardened = RandomForestClassifier(random_state=seed, **params)
    hardened.fit(X_aug, y_aug)
    return hardened, len(x_adv)


def robust_accuracy(model, X: np.ndarray, *, n: int = 24, seed: int = 7) -> float:
    """Fraction of in-budget evasion attempts the model survives.

    The same definition the robustness gate uses: a flip only counts as an evasion
    if the perturbation stayed within EPSILON.
    """
    from art.attacks.evasion import HopSkipJump

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    x0 = X[idx].astype(np.float32)

    attack = HopSkipJump(
        classifier=_classifier(model), targeted=False,
        max_iter=8, max_eval=200, init_eval=10, init_size=10,
    )
    x_adv_all = attack.generate(x=x0)
    linf = np.max(np.abs(x_adv_all - x0), axis=1)
    evaded = (model.predict(x_adv_all) != model.predict(x0)) & (linf <= EPSILON)
    return 1.0 - float(np.mean(evaded))
