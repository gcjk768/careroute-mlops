"""Integration tests for the ML CLI entrypoints (the pieces the CI pipeline runs
as jobs rather than via pytest): train, export_dataset, validate_data, monitor,
and the inference log. These are smoke/behaviour tests that also lift coverage of
the modules that are otherwise only exercised in-pipeline.
"""
from __future__ import annotations

import glob
import json

import numpy as np
import pytest

from app.ml import export_dataset, inference_log, monitor, train, validate_data


# --------------------------------------------------------------------------
# export_dataset  (data versioning — Pillar 2)
# --------------------------------------------------------------------------
def test_export_dataset_writes_and_verifies(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREROUTE_DATA_DIR", str(tmp_path))
    meta = export_dataset.write_snapshot()
    assert (tmp_path / "triage_dataset.npz").exists()
    assert (tmp_path / "triage_dataset.meta.json").exists()
    assert len(meta["dataSha256"]) == 64
    # Re-derive: the reproducibility check must pass on the just-written snapshot.
    assert export_dataset.check_snapshot() == 0


def test_snapshot_bytes_do_not_depend_on_zlib(tmp_path, monkeypatch):
    """The DVC pointer pins the file's md5, so the bytes must be a function of the data
    alone — not of the zlib build, nor of the OS that wrote the archive. DEFLATE output varies with the zlib build: a python:3.12-slim image update
    changed the compressed file (+144 bytes, same data hash) and staled the pointer
    (data:version, 2026-09-26). Members are STORED, and two exports are identical."""
    import hashlib
    import zipfile

    monkeypatch.setenv("CAREROUTE_DATA_DIR", str(tmp_path))
    path = tmp_path / "triage_dataset.npz"
    export_dataset.write_snapshot()
    first = hashlib.md5(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as zf:
        assert {i.compress_type for i in zf.infolist()} == {zipfile.ZIP_STORED}
        # zipfile stamps the writing OS into each entry (0 on Windows, 3 on Unix): same size,
        # different md5 between a laptop and CI (pointer gate, 2026-09-27). Pinned to Unix.
        assert {i.create_system for i in zf.infolist()} == {3}
    export_dataset.write_snapshot()
    assert hashlib.md5(path.read_bytes()).hexdigest() == first


def test_export_dataset_main(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREROUTE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["export_dataset"])
    assert export_dataset.main() == 0


# --------------------------------------------------------------------------
# validate_data  (data-validation gate)
# --------------------------------------------------------------------------
def test_validate_data_passes_on_valid_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREROUTE_DATA_DIR", str(tmp_path))
    export_dataset.write_snapshot()
    assert validate_data.main() == 0


def test_validate_data_fails_on_missing_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREROUTE_DATA_DIR", str(tmp_path))  # empty dir
    with pytest.raises(SystemExit) as exc:
        validate_data.main()
    assert exc.value.code == 1


def test_validate_data_fails_on_tampered_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREROUTE_DATA_DIR", str(tmp_path))
    export_dataset.write_snapshot()
    meta_path = tmp_path / "triage_dataset.meta.json"
    meta = json.loads(meta_path.read_text())
    meta["dataSha256"] = "0" * 64  # corrupt the recorded hash
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(SystemExit) as exc:
        validate_data.main()
    assert exc.value.code == 1


# --------------------------------------------------------------------------
# train  (build + release gate + persist + register)
# --------------------------------------------------------------------------
def test_train_main_builds_gated_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREROUTE_MODEL_DIR", str(tmp_path))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    rc = train.main()
    assert rc == 0  # gate passed (acc >= 0.75 AND red-flag recall >= 0.95)
    assert glob.glob(str(tmp_path / "*.joblib"))
    assert glob.glob(str(tmp_path / "*.sha256"))


def _passing_audit(**overrides):
    """A minimal audit that clears every release gate; tests override one field."""
    audit = {"overallAccuracy": 0.9, "redFlagRecall": 0.99,
             "fairnessGapBefore": 0.5, "fairnessGapAfter": 0.1,
             "calibration": {"method": "isotonic", "ece": 0.02, "brier": 0.3},
             # [XRAI] explanation-faithfulness gate (2026-09-16)
             "explanationAgreement": {"reportedTop1Agreement": 0.95, "top3Overlap": 0.35,
                                      "spearman": 0.7},
             # [Responsible-AI] post-processing Equal Opportunity gate (2026-09-16)
             "postProcessing": {"after": {"equalOpportunityGap": 0.05}},
             # [Responsible-AI] per-subgroup red-flag recall gate (2026-09-24)
             "subgroupRedFlagRecall": {"raw": {"65+ · Male": {"recall": 1.0, "nSevere": 30}},
                                       "served": {"65+ · Male": {"recall": 1.0, "nSevere": 30}}}}
    audit.update(overrides)
    return audit


def test_train_validate_accepts_a_healthy_audit():
    train.validate(_passing_audit())  # must not raise


def test_train_validate_rejects_low_recall():
    with pytest.raises(train.ReleaseGateError):
        train.validate(_passing_audit(redFlagRecall=0.5, fairnessGapBefore=0.2))


def test_train_validate_rejects_widened_fairness_gap():
    with pytest.raises(train.ReleaseGateError):
        train.validate(_passing_audit(fairnessGapBefore=0.1, fairnessGapAfter=0.3))


def test_train_validate_rejects_gap_over_the_absolute_ceiling():
    """The relative check alone let a real regression through: measured against
    the age-blind baseline's ~0.549 gap, a jump from 0.17 to 0.45 "improved"."""
    with pytest.raises(train.ReleaseGateError, match="exceeds release-gate maximum"):
        train.validate(_passing_audit(fairnessGapBefore=0.549, fairnessGapAfter=0.45))


def test_train_validate_rejects_poor_calibration():
    with pytest.raises(train.ReleaseGateError, match="calibration ECE"):
        train.validate(_passing_audit(
            calibration={"method": "isotonic", "ece": 0.2, "brier": 0.3}))


def test_train_validate_rejects_a_missing_calibration_block():
    with pytest.raises(train.ReleaseGateError, match="calibration"):
        train.validate(_passing_audit(calibration={}))


# --------------------------------------------------------------------------
# monitor  (drift report + drift gate + retrain trigger guards)
# --------------------------------------------------------------------------
def test_monitor_main_writes_report(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREROUTE_MONITOR_DIR", str(tmp_path))
    monkeypatch.delenv("CAREROUTE_MONITOR_SOURCE", raising=False)
    rc = monitor.main()
    assert rc == 0
    report = json.loads((tmp_path / "drift_report.json").read_text())
    assert "driftGate" in report and "retrain" in report
    assert (tmp_path / "drift_report.html").exists() or report["backend"] == "evidently"


def test_drift_gate_psi_threshold():
    breached = monitor._drift_gate({"backend": "builtin-psi", "dataDriftPSI": 0.30})
    assert breached["breached"] is True
    ok = monitor._drift_gate({"backend": "builtin-psi", "dataDriftPSI": 0.10})
    assert ok["breached"] is False


def test_drift_gate_treats_a_missing_metric_as_a_breach():
    """A metric the backend never produced is UNMEASURED drift, not absent
    drift. `value is not None and value >= threshold` scored it as a clean
    pass — the quietest possible way to switch drift detection off."""
    gate = monitor._drift_gate({"backend": "builtin-psi"})  # no dataDriftPSI key
    assert gate["breached"] is True
    assert gate["reason"] == "drift metric unavailable"
    assert gate["value"] is None


def test_drift_gate_treats_a_missing_evidently_share_as_a_breach():
    gate = monitor._drift_gate({"backend": "evidently"})  # DatasetDriftMetric absent
    assert gate["breached"] is True and gate["reason"] == "drift metric unavailable"


def test_monitor_main_exits_non_zero_when_drift_is_unmeasured(tmp_path, monkeypatch):
    """A monitor that produced no drift number must FAIL the job, not print a
    green 'ok'. (A breach on a REAL number still exits 0 — the shifted sample is
    expected to drift and the CT loop handles it; see test above.)"""
    monkeypatch.setenv("CAREROUTE_MONITOR_DIR", str(tmp_path))
    monkeypatch.delenv("CAREROUTE_MONITOR_SOURCE", raising=False)
    monkeypatch.setattr(monitor, "_live_report", lambda dst: None)
    monkeypatch.setattr(monitor, "_evidently_report", lambda dst: (None, "stubbed"))
    monkeypatch.setattr(monitor, "_psi_fallback", lambda dst: {"backend": "builtin-psi"})
    assert monitor.main() == 1
    report = json.loads((tmp_path / "drift_report.json").read_text())
    assert report["driftGate"]["reason"] == "drift metric unavailable"
    # ...and it must NOT have fired a retrain off a broken measurement.
    assert "not attempted" in report["retrain"]


def test_monitor_records_why_the_evidently_backend_did_not_run(tmp_path, monkeypatch):
    """The report has to say WHICH backend ran and why. `evidently` is the named
    primary tool; in this venv importing it raises TypeError (a pydantic/py3.13
    layout conflict), which the old code logged at INFO as 'not available' —
    indistinguishable from the library simply being absent."""
    monkeypatch.setenv("CAREROUTE_MONITOR_DIR", str(tmp_path))
    monkeypatch.delenv("CAREROUTE_MONITOR_SOURCE", raising=False)
    monkeypatch.setattr(monitor, "_live_report", lambda dst: None)
    monkeypatch.setattr(
        monitor, "_evidently_report",
        lambda dst: (None, "evidently is installed but failed to import (TypeError: x)"),
    )
    monitor.main()
    report = json.loads((tmp_path / "drift_report.json").read_text())
    assert report["backend"] == "builtin-psi"
    assert "failed to import" in report["backendStatus"]


def test_evidently_import_distinguishes_absent_from_broken(monkeypatch):
    """ModuleNotFoundError -> 'not installed' (a non-event). Anything else ->
    'installed but failed to import' (a real, actionable problem)."""
    import builtins

    real_import = builtins.__import__

    def _raise(exc):
        def fake(name, *a, **kw):
            if name.startswith("evidently"):
                raise exc
            return real_import(name, *a, **kw)
        return fake

    monkeypatch.setattr(builtins, "__import__", _raise(ModuleNotFoundError(name="evidently")))
    mods, status = monitor._import_evidently()
    assert mods is None and "not installed" in status

    monkeypatch.setattr(builtins, "__import__", _raise(TypeError("pydantic layout conflict")))
    mods, status = monitor._import_evidently()
    assert mods is None
    assert "installed but failed to import" in status and "TypeError" in status


def test_monitor_reference_sample_is_not_a_prefix_of_the_training_set():
    """[MLOps] The reference ('signed-off') distribution must be data the model
    has never seen. It used to be generate_dataset(n=2000, seed=42) — literally
    the first 2000 rows of the seed-42 training set, ~76% of which land in the
    train split — so referenceAccuracy was measured largely on memorised rows."""
    from app.ml import data as ml_data

    X_ref, _, _ = ml_data.generate_dataset(n=2000, seed=monitor._REFERENCE_SEED)
    X_train, _, _ = ml_data.generate_dataset(n=6000, seed=42)  # model.build_artifact
    assert monitor._REFERENCE_SEED not in (42, 7, 123, 99)
    assert not np.array_equal(X_ref, X_train[: len(X_ref)])


def test_retrain_trigger_skips_without_config(monkeypatch):
    for k in ("CAREROUTE_PIPELINE_TRIGGER_TOKEN", "CI_API_V4_URL", "CI_PROJECT_ID", "CI_PIPELINE_SOURCE"):
        monkeypatch.delenv(k, raising=False)
    msg = monitor._trigger_retrain({"metric": "dataDriftPSI", "value": 0.9})
    assert "skipped" in msg


def test_retrain_trigger_loop_guard(monkeypatch):
    monkeypatch.setenv("CI_PIPELINE_SOURCE", "trigger")
    msg = monitor._trigger_retrain({"metric": "dataDriftPSI", "value": 0.9})
    assert "loop guard" in msg


def test_monitor_psi_fallback_and_html(tmp_path):
    """The built-in PSI report path (what CI runs without Evidently)."""
    m = monitor._psi_fallback(str(tmp_path))
    assert m["backend"] == "builtin-psi"
    assert "dataDriftPSI" in m and "performance" in m
    monitor._write_html(str(tmp_path), m)
    assert (tmp_path / "drift_report.html").exists()


def test_monitor_live_report(tmp_path, monkeypatch):
    """Live-traffic monitoring against a populated inference log."""
    from app.ml.features import FEATURE_NAMES
    monkeypatch.setenv("CAREROUTE_INFERENCE_LOG", str(tmp_path / "inf.jsonl"))
    monkeypatch.setenv("CAREROUTE_MONITOR_SOURCE", "live")
    monkeypatch.setenv("CAREROUTE_MONITOR_MIN_ROWS", "10")
    feats = np.zeros(len(FEATURE_NAMES), dtype=float)
    for i in range(20):
        inference_log.log_prediction(feats, "P3_URGENT", 0.6, "m", f"2026-07-18T00:00:{i:02d}+00:00")
    rep = monitor._live_report(str(tmp_path))
    assert rep is not None
    assert "live-inference-log" in rep["currentSource"]
    assert "dataDriftPSI" in rep


def test_monitor_live_report_insufficient_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREROUTE_INFERENCE_LOG", str(tmp_path / "empty.jsonl"))
    monkeypatch.setenv("CAREROUTE_MONITOR_SOURCE", "live")
    monkeypatch.setenv("CAREROUTE_MONITOR_MIN_ROWS", "100")
    assert monitor._live_report(str(tmp_path)) is None


def test_monitor_live_report_disabled_when_not_live(tmp_path, monkeypatch):
    monkeypatch.delenv("CAREROUTE_MONITOR_SOURCE", raising=False)
    assert monitor._live_report(str(tmp_path)) is None


# --------------------------------------------------------------------------
# inference_log  (Pillar 3 live-traffic feed)
# --------------------------------------------------------------------------
def test_inference_log_roundtrip(tmp_path, monkeypatch):
    path = tmp_path / "inf.jsonl"
    monkeypatch.setenv("CAREROUTE_INFERENCE_LOG", str(path))
    # Derived from the code's feature layout rather than hardcoded, so adding a
    # symptom category (as E5 required) does not break an unrelated test.
    from app.ml.features import FEATURE_NAMES

    n_features = len(FEATURE_NAMES)
    feats = np.arange(n_features, dtype=float) / n_features
    for i in range(5):
        inference_log.log_prediction(feats, "P3_URGENT", 0.7, "model-x", f"2026-07-18T00:00:0{i}+00:00")
    loaded = inference_log.load_recent()
    assert loaded is not None
    X, codes = loaded
    assert X.shape == (5, n_features)
    assert codes == ["P3_URGENT"] * 5


def test_inference_log_missing_file_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREROUTE_INFERENCE_LOG", str(tmp_path / "does-not-exist.jsonl"))
    assert inference_log.load_recent() is None


# --------------------------------------------------------------------------
# Retrain triggers. The CI/CD deck names four reasons to retrain — dips in
# accuracy, data drift, TARGET drift, and CONCEPT drift. Only data drift was
# gated; the other two were computed, reported, and then ignored, so a pipeline
# whose label mix had shifted or whose accuracy had collapsed still scored a
# clean gate as long as the input features looked familiar.
# --------------------------------------------------------------------------

def test_target_drift_alone_breaches_the_gate():
    gate = monitor._drift_gate({
        "backend": "builtin-psi",
        "dataDriftPSI": 0.01,          # inputs look completely normal
        "targetDriftPSI": 0.80,        # the label mix has moved hard
    })

    assert gate["breached"] is True
    assert any(s["metric"] == "targetDriftPSI" and s["breached"] for s in gate["signals"])


def test_accuracy_collapse_breaches_the_gate_as_concept_drift():
    gate = monitor._drift_gate({
        "backend": "builtin-psi",
        "dataDriftPSI": 0.01,
        "performance": {"referenceAccuracy": 0.91, "currentAccuracy": 0.55},
    })

    assert gate["breached"] is True
    assert any(s["metric"] == "accuracyDrop" and s["breached"] for s in gate["signals"])


def test_a_quiet_pipeline_still_passes():
    gate = monitor._drift_gate({
        "backend": "builtin-psi",
        "dataDriftPSI": 0.07,
        "targetDriftPSI": 0.11,
        "performance": {"referenceAccuracy": 0.9145, "currentAccuracy": 0.8855},
    })

    assert gate["breached"] is False


def test_data_drift_remains_the_headline_metric():
    """Backward compatibility: the retrain reason and the printed line still
    lead with the data-drift number."""
    gate = monitor._drift_gate({"backend": "builtin-psi", "dataDriftPSI": 0.30})

    assert gate["metric"] == "dataDriftPSI"
    assert gate["value"] == 0.30
    assert gate["breached"] is True


def test_absent_secondary_signals_are_not_breaches():
    """Only the HEADLINE metric fails closed when missing. A monitor that never
    computed target drift must not be reported as drifting on it."""
    gate = monitor._drift_gate({"backend": "builtin-psi", "dataDriftPSI": 0.05})

    assert gate["breached"] is False
    assert all(s["value"] is not None for s in gate["signals"] if s["breached"])


def test_retrain_reason_names_the_signal_that_actually_breached():
    """With three triggers feeding one gate, a retrain fired by target drift
    must not be labelled with the (healthy) data-drift number."""
    gate = monitor._drift_gate({
        "backend": "builtin-psi", "dataDriftPSI": 0.01, "targetDriftPSI": 0.80,
    })

    reason = monitor._retrain_reason(gate)

    assert "targetDriftPSI" in reason
    assert "dataDriftPSI" not in reason


def test_retrain_reason_falls_back_to_the_headline_when_no_signal_is_marked():
    gate = {"metric": "dataDriftPSI", "value": 0.9, "signals": []}

    assert monitor._retrain_reason(gate) == "drift:dataDriftPSI=0.9"
