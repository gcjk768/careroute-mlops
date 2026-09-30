"""Tests for the dataset snapshot's provenance record (app.ml.export_dataset).

The DVC deck's "What Data Versioning Adds" slide asks a snapshot to record
**schema, statistics, splits and provenance** — not just the bytes. CareRoute's
metadata had schema (feature names) and the content hash, but no statistics, no
record of the split the model actually trains on, and no generator provenance.
Without statistics there is nothing to compare a future snapshot against except
"the hash changed", which tells you *that* the data moved but never *how*.

**Everything recorded here must be deterministic.** The metadata file is
git-committed and `data:version --check` re-derives it, so a timestamp or a git
SHA would make the file churn on every export and turn a real diff into noise.
Volatile provenance (which commit trained this) belongs on the MLflow run — see
`app/ml/run_context.py`, which is where the tracking deck puts it.
"""
from __future__ import annotations

import json

from app.ml import export_dataset


def test_snapshot_records_label_statistics():
    _X, _y, _g, meta = export_dataset.build_snapshot()

    balance = meta["statistics"]["labelBalance"]
    assert sum(balance.values()) == meta["rows"]
    assert all(count > 0 for count in balance.values()), "every acuity class must be represented"


def test_snapshot_records_per_feature_statistics():
    _X, _y, _g, meta = export_dataset.build_snapshot()

    stats = meta["statistics"]["features"]
    assert set(stats) == set(meta["featureNames"])
    for name, s in stats.items():
        assert s["min"] <= s["mean"] <= s["max"], name


def test_snapshot_records_subgroup_balance():
    """Fairness is gated on subgroups, so the snapshot must say what it holds."""
    _X, _y, _g, meta = export_dataset.build_snapshot()

    subgroups = meta["statistics"]["subgroupBalance"]
    assert subgroups
    assert sum(subgroups.values()) == meta["rows"]


def test_snapshot_records_the_split_the_model_trains_on():
    _X, _y, _g, meta = export_dataset.build_snapshot()

    split = meta["split"]
    assert split["testSize"] == 0.25
    assert split["randomState"] == 42
    assert split["stratified"] is True


def test_snapshot_records_deterministic_provenance():
    _X, _y, _g, meta = export_dataset.build_snapshot()

    prov = meta["provenance"]
    assert prov["generator"] == "app.ml.data.generate_dataset"
    assert prov["seed"] == meta["seed"]


def test_metadata_is_byte_for_byte_reproducible():
    """The whole point: two exports of the same code must be identical, or the
    committed snapshot churns and `data:version --check` becomes meaningless."""
    _X1, _y1, _g1, first = export_dataset.build_snapshot()
    _X2, _y2, _g2, second = export_dataset.build_snapshot()

    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_no_volatile_fields_leaked_into_the_metadata():
    """A timestamp or git SHA here would dirty the committed file on every run."""
    _X, _y, _g, meta = export_dataset.build_snapshot()
    blob = json.dumps(meta).lower()

    for volatile in ("timestamp", "createdat", "generatedat", "gitcommit", "git_sha"):
        assert volatile not in blob, f"{volatile} makes the committed snapshot churn"


def test_data_hash_is_unchanged_by_the_richer_metadata():
    """Lineage safety: the model's dataSha256 must still match the snapshot's."""
    X, y, _g, meta = export_dataset.build_snapshot()

    assert meta["dataSha256"] == export_dataset._data_sha256(X, y)
