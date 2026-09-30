"""[MLOps][DVC] Materialize the deterministic training dataset to a versionable
artifact + integrity metadata, so the EXACT data a model was trained on is
tracked (by DVC) and auditable — this closes the data-lineage gap.

    python -m app.ml.export_dataset            # (re)write data/ + print sha256
    python -m app.ml.export_dataset --check    # re-export & assert hash unchanged

The dataset is synthetic and fully seeded, so it is reproducible from code; what
DVC versions is a *materialized snapshot* of it plus a small metadata record. The
`dataSha256` computed here is IDENTICAL to the one `build_artifact()` records in
the model audit (same n / seed / feature space), so a model version traces back
to an exact dataset version — and the CI `data:integrity` gate asserts exactly
that equality.

Outputs (under backend/data/, override with CAREROUTE_DATA_DIR):
  * triage_dataset.npz        — X, y, subgroups (DVC-tracked; git-ignored)
  * triage_dataset.meta.json  — n/seed/feature space + dataSha256 (git-committed)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile

import numpy as np

from . import data
from .features import FEATURE_NAMES

# Keep in LOCKSTEP with build_artifact() in model.py (n=6000, seed=42), so the
# exported dataset hash equals the model audit's dataSha256.
_N = 6000
_SEED = 42

_DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data")
)


def data_dir() -> str:
    return os.environ.get("CAREROUTE_DATA_DIR", _DEFAULT_DATA_DIR)


def _data_sha256(X: np.ndarray, y: np.ndarray) -> str:
    """The same content-addressed data hash build_artifact() records, so the
    exported snapshot and the trained model share one dataset identity."""
    h = hashlib.sha256()
    h.update(X.tobytes())
    h.update(y.tobytes())
    h.update(("|".join(FEATURE_NAMES)).encode("utf-8"))
    return h.hexdigest()


def _statistics(X: np.ndarray, y: np.ndarray, subgroups: list[str]) -> dict:
    """Schema-level statistics for the snapshot.

    Rounded to 6 places so the record is byte-for-byte reproducible across
    machines — float repr differences would otherwise churn a git-committed file.
    """
    return {
        "labelBalance": {str(label): int(count) for label, count in
                         zip(*np.unique(y, return_counts=True), strict=True)},
        "subgroupBalance": {str(g): int(c) for g, c in
                            zip(*np.unique(np.asarray(subgroups), return_counts=True), strict=True)},
        "features": {
            name: {
                "min": round(float(X[:, i].min()), 6),
                "max": round(float(X[:, i].max()), 6),
                "mean": round(float(X[:, i].mean()), 6),
                "std": round(float(X[:, i].std()), 6),
            }
            for i, name in enumerate(FEATURE_NAMES)
        },
    }


def build_snapshot() -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    X, y, subgroups = data.generate_dataset(n=_N, seed=_SEED)
    data_hash = _data_sha256(X, y)
    meta = {
        "family": "careroute-triage",
        "n": _N,
        "seed": _SEED,
        "rows": int(X.shape[0]),
        "featureCount": int(X.shape[1]),
        "featureNames": list(FEATURE_NAMES),
        # [MLOps] Pillar 2 — the data-versioning deck asks a snapshot to record
        # schema, STATISTICS, SPLITS and PROVENANCE, not just the bytes. Without
        # statistics, a changed hash tells you THAT the data moved and never HOW.
        "statistics": _statistics(X, y, subgroups),
        # The split the model actually trains on (model.py). Recorded here so the
        # snapshot describes the experiment it feeds, not just the pool.
        "split": {"testSize": 0.25, "randomState": 42, "stratified": True},
        # DETERMINISTIC provenance only. This file is git-committed and
        # `--check` re-derives it, so a timestamp or git SHA would dirty it on
        # every export and turn a real diff into noise. Volatile provenance
        # (which commit trained this) lives on the MLflow run — see
        # app/ml/run_context.py, which is where the tracking deck puts it.
        "provenance": {
            "generator": "app.ml.data.generate_dataset",
            "seed": _SEED,
            "synthetic": True,
        },
        "dataSha256": data_hash,
    }
    return X, y, subgroups, meta


def _write_npz(path: str, **arrays) -> None:
    """np.savez, but byte-identical on every machine, because the DVC pointer pins the md5.

    Two things made the bytes vary with identical data (data:version, 26-27 Sep 2026):
    DEFLATE output depends on the zlib build (a python:3.12-slim update changed it), and
    zipfile stamps the writing OS into every entry (create_system 0 on Windows, 3 on Unix).
    So: STORED (~3.2 MB, held by DVC, not git), create_system pinned to Unix, and numpy's
    own fixed 1980 timestamp. Otherwise exactly what np.savez writes; np.load reads it.
    """
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, arr in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy")
            info.create_system = 3
            with zf.open(info, "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(arr), allow_pickle=True)


def write_snapshot(out_dir: str | None = None) -> dict:
    out_dir = out_dir or data_dir()
    os.makedirs(out_dir, exist_ok=True)
    X, y, subgroups, meta = build_snapshot()
    _write_npz(os.path.join(out_dir, "triage_dataset.npz"),
               X=X, y=y, subgroups=np.asarray(subgroups))
    with open(os.path.join(out_dir, "triage_dataset.meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def check_snapshot(out_dir: str | None = None) -> int:
    """Re-derive the dataset and assert its hash matches the committed metadata
    (reproducibility gate). Returns a process exit code."""
    out_dir = out_dir or data_dir()
    _, _, _, current = build_snapshot()
    meta_path = os.path.join(out_dir, "triage_dataset.meta.json")
    if not os.path.exists(meta_path):
        # No baseline yet — write one and succeed (first run establishes it).
        write_snapshot(out_dir)
        print(f"wrote initial dataset metadata: {current['dataSha256'][:16]}...")
        return 0
    with open(meta_path, encoding="utf-8") as fh:
        committed = json.load(fh)
    if committed.get("dataSha256") != current["dataSha256"]:
        print(
            "DATA DRIFT / LINEAGE MISMATCH:\n"
            f"  committed dataSha256 : {committed.get('dataSha256')}\n"
            f"  current   dataSha256 : {current['dataSha256']}\n"
            "The versioned dataset no longer matches what the code generates.",
            file=sys.stderr,
        )
        return 1
    print(f"dataset reproducible — sha256 {current['dataSha256'][:16]}... OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Export/version the triage dataset.")
    ap.add_argument("--check", action="store_true", help="assert hash unchanged")
    args = ap.parse_args()
    if args.check:
        return check_snapshot()
    meta = write_snapshot()
    print("=== CareRoute dataset snapshot ===")
    print(f"  dir         : {data_dir()}")
    print(f"  rows        : {meta['rows']}  x {meta['featureCount']} features")
    print(f"  seed        : {meta['seed']}")
    print(f"  dataSha256  : {meta['dataSha256']}")
    print("  next        : `dvc add backend/data/triage_dataset.npz` to version it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
