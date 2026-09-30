"""[MLOps] Pillar 1 — what an experiment run must record beyond its metrics.

The MLflow deck's "What Should an Experiment Track?" slide names three things:

    Code Version   — git commit and source reference
    Dataset Version — which data produced this model
    Environment    — Python, OS, libraries

CareRoute logged only the second one (`data_sha256`). A run in the registry could
therefore say what the model scored and which data it saw, but not which commit
built it or which scikit-learn trained it — and those are precisely the questions
asked when a registered model misbehaves months later and the trail has to be
walked backwards. A version string alone ends that trail.

Everything here is BEST EFFORT. No git binary, a source tarball with no `.git`,
a library whose metadata will not resolve — each degrades to `"unknown"` and
never raises. The model is the deliverable; its provenance record must not be
able to fail the training run that produces it.
"""
from __future__ import annotations

import platform
import subprocess  # nosec B404  # only fixed-argv read-only git calls, no shell
import sys
from importlib.metadata import PackageNotFoundError, version

# Libraries whose version materially changes a trained model or its serialised
# form. Keep the list short — this is provenance, not a lockfile.
_TRACKED_LIBRARIES = ("scikit-learn", "numpy", "scipy", "joblib", "mlflow")

_UNKNOWN = "unknown"


def _version_of(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return _UNKNOWN


def _git(*args: str) -> str:
    """Run a read-only git command, or return "unknown"."""
    try:
        # S603/S607 are suppressed below because every call site passes fixed
        # literal arguments — no caller input reaches this — and resolving `git`
        # from PATH is the point: CI images place it differently from a dev box.
        out = subprocess.check_output(("git", *args), stderr=subprocess.DEVNULL, text=True, timeout=10)  # noqa: S603, S607  # nosec B603 B607
    except Exception:  # noqa: BLE001 - no git, no repo, timeout: all mean "unknown"
        return _UNKNOWN
    return out.strip() or _UNKNOWN


def _libraries() -> dict[str, str]:
    libs: dict[str, str] = {}
    for name in _TRACKED_LIBRARIES:
        try:
            libs[name] = _version_of(name)
        except Exception:  # noqa: BLE001 - a broken metadata backend is not a failed run
            libs[name] = _UNKNOWN
    return libs


def training_run_context() -> dict:
    """Code version + environment for the current training run.

    Flat and string-valued (apart from `libraries`) so it can be logged as
    MLflow tags without stringifying a nested structure into a repr.
    """
    return {
        "gitCommit": _git("rev-parse", "HEAD"),
        "gitBranch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "gitDirty": "true" if _git("status", "--porcelain") not in ("", _UNKNOWN) else "false",
        "pythonVersion": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": _libraries(),
    }


def mlflow_tags() -> dict[str, str]:
    """The same context flattened to snake_case scalar tags for MLflow."""
    ctx = training_run_context()
    tags = {
        "git_commit": ctx["gitCommit"],
        "git_branch": ctx["gitBranch"],
        "git_dirty": ctx["gitDirty"],
        "python_version": ctx["pythonVersion"],
        "platform": ctx["platform"],
    }
    for name, ver in ctx["libraries"].items():
        tags[f"lib_{name.replace('-', '_')}"] = ver
    return tags
