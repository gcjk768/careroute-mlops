"""[MLOps] DATA-VALIDATION GATE — "validate data before you train on it".

    python -m app.ml.validate_data

Loads the versioned dataset snapshot (`triage_dataset.npz` + its metadata) and
asserts the schema + quality invariants the training code assumes. Runs as a
BLOCKING CI job (`data:validate`) between `data:version` and `train:model`
consumers, so a malformed / drifted / corrupted snapshot can never silently
become a released model. Dependency-free on purpose (no Great Expectations /
pandera) — the checks are few, explicit, and fast.

Checks:
  1. schema  — feature count and names match the code's FEATURE_NAMES exactly
  2. domain  — features are in [0, 1] (binary/one-hot encoding), no NaN/inf
  3. labels  — every label is a valid acuity index; every class is present
               with a minimum support (a class-starved dataset can't train)
  4. balance — no class exceeds a dominance ceiling (degenerate distribution)
  5. lineage — the recorded dataSha256 matches a recomputation from the
               loaded arrays (the snapshot itself is internally consistent)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

from .data import ACUITY_INDEX_TO_CODE
from .export_dataset import data_dir
from .features import FEATURE_NAMES

_MIN_CLASS_SUPPORT = 100      # every acuity class needs at least this many rows
_MAX_CLASS_SHARE = 0.60       # no single class may dominate past this share


def _fail(msg: str) -> None:
    print(f"  FAIL  {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    d = data_dir()
    npz_path = os.path.join(d, "triage_dataset.npz")
    meta_path = os.path.join(d, "triage_dataset.meta.json")
    print("=== CareRoute data-validation gate ===")
    print(f"  snapshot : {npz_path}")
    if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
        _fail("snapshot missing — run `python -m app.ml.export_dataset` first")

    snap = np.load(npz_path, allow_pickle=False)
    X, y = snap["X"], snap["y"]
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)

    # 1. schema
    if X.shape[1] != len(FEATURE_NAMES):
        _fail(f"feature count {X.shape[1]} != code's {len(FEATURE_NAMES)}")
    if meta.get("featureNames") != list(FEATURE_NAMES):
        _fail("feature names in metadata do not match the code's feature space")
    print(f"  ok    schema: {X.shape[0]} rows x {X.shape[1]} features")

    # 2. domain
    if not np.isfinite(X).all():
        _fail("non-finite values (NaN/inf) in feature matrix")
    if X.min() < 0.0 or X.max() > 1.0:
        _fail(f"feature values outside [0,1]: min={X.min()}, max={X.max()}")
    print("  ok    domain: finite, all features in [0, 1]")

    # 3 + 4. labels
    n_classes = len(ACUITY_INDEX_TO_CODE)
    if y.min() < 0 or y.max() >= n_classes:
        _fail(f"labels outside 0..{n_classes - 1}: min={y.min()}, max={y.max()}")
    counts = np.bincount(y.astype(int), minlength=n_classes)
    for idx, count in enumerate(counts):
        if count < _MIN_CLASS_SUPPORT:
            _fail(f"class {ACUITY_INDEX_TO_CODE[idx]} has {count} rows (< {_MIN_CLASS_SUPPORT})")
        if count / len(y) > _MAX_CLASS_SHARE:
            _fail(f"class {ACUITY_INDEX_TO_CODE[idx]} dominates: {count / len(y):.0%} > {_MAX_CLASS_SHARE:.0%}")
    dist = ", ".join(f"{ACUITY_INDEX_TO_CODE[i]}={c}" for i, c in enumerate(counts))
    print(f"  ok    labels: {dist}")

    # 5. lineage (snapshot internal consistency)
    h = hashlib.sha256()
    h.update(X.tobytes())
    h.update(y.tobytes())
    h.update(("|".join(FEATURE_NAMES)).encode("utf-8"))
    if h.hexdigest() != meta.get("dataSha256"):
        _fail("dataSha256 in metadata does not match the snapshot arrays")
    print(f"  ok    lineage: dataSha256 {meta['dataSha256'][:16]}... consistent")

    print("  data-validation gate: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
