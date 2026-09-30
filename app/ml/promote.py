"""[MLOps] Transition a registry version to Production — the deck's final stage.

    python -m app.ml.promote            # promote the current @challenger
    python -m app.ml.promote --version 7
    python -m app.ml.promote --show     # who is champion / challenger right now

This is the half of Gap 1 that `train.py` deliberately does not do. Training
stages a gated build as `@challenger`; moving `@champion` is a separate, explicit
act, because that is what `deploy:promote-production` (a `when: manual` job) is
for. Splitting them is the difference between a registry that records a decision
and one that merely records a build.

FAILS LOUDLY, unlike the tracking code in `train.py`. The asymmetry is deliberate:
a registry write that fails during training must not destroy a model that passed
its release gate, but a promotion that silently fails is far worse than one that
errors — it leaves an operator believing a new version is serving when the old one
still is, which is exactly the state a rollback is supposed to rescue you from.
"""
from __future__ import annotations

import argparse
import sys

from . import lifecycle


def _client():
    import mlflow  # imported here so --help works without the optional dependency

    # Must resolve the SAME store train.py writes to. Building a bare MlflowClient()
    # here silently picked MLflow's default file store instead, which 3.x refuses
    # outright -- the promotion CLI could not see a registry that training had just
    # written to.
    lifecycle.pin_tracking_uri(mlflow)
    return mlflow.MlflowClient()


def show(client) -> int:
    for alias in (lifecycle.PRODUCTION_ALIAS, lifecycle.STAGING_ALIAS):
        try:
            mv = client.get_model_version_by_alias(lifecycle.REGISTERED_MODEL, alias)
            tags = getattr(mv, "tags", {}) or {}
            print(f"  @{alias:11} version {mv.version}"
                  f"  stage={tags.get(lifecycle.STAGE_TAG, '?')}"
                  f"  acc={tags.get('overall_accuracy', '?')}"
                  f"  red_flag_recall={tags.get('red_flag_recall', '?')}")
        except Exception:  # noqa: BLE001 - an unset alias is a legitimate state, not an error
            print(f"  @{alias:11} (not set)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", help="registry version to promote (default: current @challenger)")
    ap.add_argument("--show", action="store_true", help="print the current aliases and exit")
    args = ap.parse_args(argv)

    client = _client()
    if args.show:
        return show(client)

    version = args.version
    if version is None:
        try:
            version = str(client.get_model_version_by_alias(
                lifecycle.REGISTERED_MODEL, lifecycle.STAGING_ALIAS).version)
        except Exception as exc:  # noqa: BLE001 - nothing staged is a real, reportable failure
            print(f"no @{lifecycle.STAGING_ALIAS} to promote ({type(exc).__name__}). "
                  f"Train a model first, or pass --version.", file=sys.stderr)
            return 1

    # Refuse to promote a version that never cleared the gate. The alias path cannot
    # reach one, but --version is a human typing a number, and this is the last point
    # at which that typo is cheap.
    try:
        tags = client.get_model_version(lifecycle.REGISTERED_MODEL, version).tags or {}
    except Exception as exc:  # noqa: BLE001
        print(f"version {version} not found ({type(exc).__name__})", file=sys.stderr)
        return 1
    if tags.get("release_gate") == "failed":
        print(f"REFUSING: version {version} is tagged release_gate=failed", file=sys.stderr)
        return 1

    result = lifecycle.promote(client, version)
    print(f"  {lifecycle.REGISTERED_MODEL} @{lifecycle.PRODUCTION_ALIAS}: "
          f"{result['previous']} -> {result['promoted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
