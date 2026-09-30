"""[MLOps] Registry lifecycle stages — Development -> Staging -> Production.

Closes Gap 1 in [[Lecture Alignment]]: the decks teach model *stages* in two
places, and this project registered `CareRouteTriageRF` on every gated build and
then never moved it anywhere. A registry where every version sits in the same
undifferentiated pile cannot answer the only question a registry exists for —
*which version is serving?*

THE SLIDE'S API DOES NOT EXIST ANY MORE, AND THAT IS THE INTERESTING PART.
Both decks show `MlflowClient.transition_model_version_stage(..., stage="Production")`.
MLflow deprecated stages in 2.9 and removed them in 3.x, and this project pins
`mlflow==3.13.0` (`requirements-mlops.txt`). Writing the call exactly as taught
would pass on a developer machine with an older MLflow and fail in CI — the worst
shape of bug, because the thing that breaks is the release step.

MLflow's replacement is **aliases** (a moving pointer to one version) plus
**tags** (immutable facts about a version), and that is what this module uses.
Aliases exist from 2.9 onward, so this works on both sides of the 3.x boundary.
The mapping is deliberate, not cosmetic:

    deck's stage   here                    meaning
    Development    (no alias, tagged)      built, but did not clear the gate
    Staging        alias `challenger`      cleared the release gate; eligible to be shadowed
    Production     alias `champion`        actually serving

WHY A GATED BUILD BECOMES `challenger` AND NEVER `champion`.
`validate()` has already refused to persist anything below the accuracy and
red-flag-recall floors, so every version that reaches the registry is by
construction good enough to *try*. It is not thereby good enough to *serve*:
`deploy:shadow-model` exists to run the candidate beside the primary, and
`deploy:promote-production` is a manual gate on purpose. Auto-promoting on a
passed gate would make both of those decorative — the pipeline would already have
done the thing they exist to authorise. Promotion is a separate, deliberate call
(`promote()`, driven by `python -m app.ml.promote`).

Everything below the MLflow boundary is pure and unit-tested; the client calls
are thin and best-effort, because experiment tracking must never be able to fail
a training run that the release gate already passed.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def pin_tracking_uri(mlflow: Any) -> str:
    """Point MLflow at this project's store when MLFLOW_TRACKING_URI is unset.

    Lifted out of `train.py` so `promote.py` cannot get it wrong -- and it did:
    the first version of the promotion CLI built a bare `MlflowClient()`, got
    MLflow's default file store, and died with "the filesystem tracking backend is
    in maintenance mode" while training worked fine. Two entry points into the same
    registry must agree on where that registry IS, so there is now one place to say
    it.

    mlflow 3.x changed the default from `./mlruns` to `sqlite:///<cwd>/mlflow.db`
    and builds it by PERCENT-ENCODING the cwd. On a path containing a space
    ("C:/Users/James Koh/...") the encoded "%20" is then used as a literal
    filesystem path, so every run died with
        PermissionError: [WinError 5] Access is denied: 'C:\\Users\\James%20Koh'
    Every other form fails too: a bare path string parses its drive letter as a URI
    scheme ("c:"), and `Path.as_uri()` yields a correct file:// URI that mlflow 3.x
    now refuses without MLFLOW_ALLOW_FILE_STORE=true. `as_posix()` keeps forward
    slashes and leaves spaces intact, which sqlite/SQLAlchemy accept verbatim.
    """
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        store_dir = Path(__file__).resolve().parents[2]
        store_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri("sqlite:///" + (store_dir / "mlflow.db").as_posix())
    return mlflow.get_tracking_uri()

REGISTERED_MODEL = "CareRouteTriageRF"

#: A version that cleared the release gate. Eligible for shadow + promotion.
STAGING_ALIAS = "challenger"
#: The version actually served. Moved only by an explicit promotion.
PRODUCTION_ALIAS = "champion"

#: Human-readable stage, recorded as a tag so the registry UI shows the deck's
#: vocabulary even though the stage *API* is gone.
STAGE_TAG = "stage"
DEVELOPMENT, STAGING, PRODUCTION = "Development", "Staging", "Production"


def stage_for(passed_release_gate: bool) -> str:
    """The stage a freshly registered version belongs in.

    Pure, so the lifecycle rule is testable without a tracking server — the same
    reason `canary.py` keeps its ramp logic out of the YAML. A rule that can only
    be exercised by running the thing it governs is a rule nobody checks.
    """
    return STAGING if passed_release_gate else DEVELOPMENT


def version_tags(payload: dict, audit: dict, passed_release_gate: bool) -> dict[str, str]:
    """Facts pinned to the VERSION rather than the run.

    A run records how a model was produced; a version has to answer "what is this,
    and may it serve?" to someone reading the registry months later with no access
    to the CI job. Both hashes are here so a registry entry can be tied back to the
    exact artifact and dataset without trusting the version number.
    """
    return {
        STAGE_TAG: stage_for(passed_release_gate),
        "release_gate": "passed" if passed_release_gate else "failed",
        "overall_accuracy": f"{audit['overallAccuracy']:.4f}",
        "red_flag_recall": f"{audit['redFlagRecall']:.4f}",
        "calibration_ece": f"{audit['calibration']['ece']:.4f}",
        "data_sha256": payload["dataSha256"],
        "model_sha256": payload["modelSha256"],
        "version_id": payload["versionId"],
    }


def _version_for_run(client: Any, run_id: str) -> str | None:
    """The registry version created by this run.

    `log_model(registered_model_name=...)` creates the version as a side effect and
    does not hand back its number in a way that is stable across MLflow versions,
    so it is looked up by run id instead of parsed out of a return value.
    """
    try:
        versions = client.search_model_versions(f"run_id='{run_id}'")
    except Exception:  # noqa: BLE001 - optional tracking dependency
        logger.warning("could not search model versions for run %s", run_id, exc_info=False)
        return None
    if not versions:
        logger.warning("no registered version found for run %s", run_id)
        return None
    # Newest first: a re-registration of the same run should win over its predecessor.
    return str(max(versions, key=lambda v: int(v.version)).version)


def record_stage(client: Any, run_id: str, payload: dict, audit: dict,
                 *, passed_release_gate: bool = True) -> str | None:
    """Tag the new version and, if it cleared the gate, point `challenger` at it.

    Best-effort by design: returns None instead of raising. A model that passed its
    release gate and was persisted must not be un-built because a registry write
    failed. The caller reports the outcome in the training summary.
    """
    version = _version_for_run(client, run_id)
    if version is None:
        return None
    for key, value in version_tags(payload, audit, passed_release_gate).items():
        try:
            client.set_model_version_tag(REGISTERED_MODEL, version, key, value)
        except Exception:  # noqa: BLE001 - optional tracking dependency
            logger.warning("tag %s not written to version %s", key, version, exc_info=False)
    if passed_release_gate:
        try:
            client.set_registered_model_alias(REGISTERED_MODEL, STAGING_ALIAS, version)
            logger.info("registry: %s@%s -> version %s", REGISTERED_MODEL, STAGING_ALIAS, version)
        except Exception:  # noqa: BLE001 - optional tracking dependency
            logger.warning("alias %s not moved", STAGING_ALIAS, exc_info=False)
    return version


def promote(client: Any, version: str) -> dict[str, str]:
    """Move `champion` to `version` — the deck's transition to Production.

    Returns what changed, including the version being replaced, because a promotion
    that cannot name its predecessor is a promotion you cannot roll back. The
    previous champion keeps its tags and stays in the registry; only the pointer
    moves, which is the entire advantage aliases have over the stage API they
    replaced.
    """
    previous = None
    try:
        previous = str(client.get_model_version_by_alias(REGISTERED_MODEL, PRODUCTION_ALIAS).version)
    except Exception:  # noqa: BLE001 - no champion yet is the expected first-run state
        logger.info("no current %s alias; this is the first promotion", PRODUCTION_ALIAS)
    client.set_registered_model_alias(REGISTERED_MODEL, PRODUCTION_ALIAS, version)
    try:
        client.set_model_version_tag(REGISTERED_MODEL, version, STAGE_TAG, PRODUCTION)
        if previous and previous != version:
            # The outgoing version is demoted to Staging, not to Development: it
            # cleared the same release gate and stays a legitimate rollback target.
            client.set_model_version_tag(REGISTERED_MODEL, previous, STAGE_TAG, STAGING)
    except Exception:  # noqa: BLE001 - optional tracking dependency
        logger.warning("stage tags not updated after promotion", exc_info=False)
    return {"promoted": version, "previous": previous or "(none)"}
