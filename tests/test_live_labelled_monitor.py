"""[MLOps][XRAI] LIVE LABELLED metrics — the HITL ground-truth loop, closed.

XRAI Day 1 (deployment review) and the DOAIS logging-and-monitoring deck both
expect production performance to be measured on REAL labelled traffic, not a
synthetic sample. The HITL agent has written `ground_truth.jsonl` since the
label loop was built, but nothing ever read it back. These tests pin the join:
inference-log rows carry a `caseId`, the clinician's final acuity is joined on
it, and the live monitor reports accuracy, red-flag recall, per-subgroup
accuracy and the Equal Opportunity gap on those labelled cases — plus the share
of predictions made with a DEFAULTED age band, which is the serving-time
condition under which the fairness mitigation cannot fire.
"""
from __future__ import annotations

import json

import numpy as np

from app.ml import inference_log, monitor
from app.ml.features import AGE_BANDS, FEATURE_NAMES, extract_features

_IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _write_gt(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_log_prediction_records_case_id_and_age_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREROUTE_INFERENCE_LOG", str(tmp_path / "inf.jsonl"))
    x = extract_features("chest pain", "65+", "Female")
    inference_log.log_prediction(x, "P2_EMERGENT", 0.8, "m", "2026-09-16T00:00:00+00:00",
                                 case_id="case-1", age_provided=True)
    inference_log.log_prediction(x, "P2_EMERGENT", 0.8, "m", "2026-09-16T00:00:01+00:00")
    rows = [json.loads(l) for l in (tmp_path / "inf.jsonl").read_text().splitlines()]
    assert rows[0]["caseId"] == "case-1" and rows[0]["ageProvided"] is True
    assert rows[1]["caseId"] is None and rows[1]["ageProvided"] is False


def test_load_labelled_joins_ground_truth_by_case_id(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREROUTE_INFERENCE_LOG", str(tmp_path / "inf.jsonl"))
    monkeypatch.setenv("CAREROUTE_GROUND_TRUTH_LOG", str(tmp_path / "gt.jsonl"))
    x_old = extract_features("chest pain", "65+", "Female")
    x_young = extract_features("sore throat", "18-39", "Male")
    inference_log.log_prediction(x_old, "P2_EMERGENT", 0.8, "m", "t0", case_id="a", age_provided=True)
    inference_log.log_prediction(x_young, "P4_NON_URGENT", 0.7, "m", "t1", case_id="b", age_provided=True)
    inference_log.log_prediction(x_young, "P4_NON_URGENT", 0.7, "m", "t2", case_id="unlabelled")
    _write_gt(tmp_path / "gt.jsonl", [
        {"caseId": "a", "modelAcuity": "P2_EMERGENT", "clinicianAcuity": "P2_EMERGENT"},
        {"caseId": "b", "modelAcuity": "P4_NON_URGENT", "clinicianAcuity": "P3_URGENT"},
        {"caseId": "zzz", "modelAcuity": "P3_URGENT", "clinicianAcuity": "P3_URGENT"},  # no inference row
        {"caseId": "a", "modelAcuity": "P2_EMERGENT", "clinicianAcuity": None},          # unlabelled row
    ])
    joined = inference_log.load_labelled()
    assert joined is not None
    X, y_true, y_pred, groups, age_provided = joined
    assert X.shape == (2, len(FEATURE_NAMES))
    assert list(y_pred) == [1, 3]            # P2 -> index 1, P4 -> index 3
    assert list(y_true) == [1, 2]            # clinician said P2, P3
    assert groups == ["65+ · Female", "18-39 · Male"]
    assert list(age_provided) == [True, True]


def test_load_labelled_returns_none_without_labels(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREROUTE_INFERENCE_LOG", str(tmp_path / "inf.jsonl"))
    monkeypatch.setenv("CAREROUTE_GROUND_TRUTH_LOG", str(tmp_path / "missing.jsonl"))
    inference_log.log_prediction(np.zeros(len(FEATURE_NAMES)), "P3_URGENT", 0.5, "m", "t", case_id="a")
    assert inference_log.load_labelled() is None


def test_live_report_carries_labelled_performance(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREROUTE_INFERENCE_LOG", str(tmp_path / "inf.jsonl"))
    monkeypatch.setenv("CAREROUTE_GROUND_TRUTH_LOG", str(tmp_path / "gt.jsonl"))
    monkeypatch.setenv("CAREROUTE_MONITOR_SOURCE", "live")
    monkeypatch.setenv("CAREROUTE_MONITOR_MIN_ROWS", "10")
    monkeypatch.setenv("CAREROUTE_MONITOR_MIN_LABELLED", "4")
    gt = []
    for i in range(24):
        band = AGE_BANDS[i % len(AGE_BANDS)]
        sex = "Female" if i % 2 else "Male"
        severe = i % 3 == 0
        x = extract_features("crushing chest pain" if severe else "sore throat", band, sex)
        pred = "P1_RESUSCITATION" if severe else "P4_NON_URGENT"
        inference_log.log_prediction(x, pred, 0.8, "m", f"t{i}", case_id=f"c{i}",
                                     age_provided=(i % 4 != 0))
        # Clinician agrees except on two severe cases, which are labelled P3.
        label = "P3_URGENT" if (severe and i in (0, 3)) else pred
        gt.append({"caseId": f"c{i}", "modelAcuity": pred, "clinicianAcuity": label})
    _write_gt(tmp_path / "gt.jsonl", gt)

    rep = monitor._live_report(str(tmp_path))
    assert rep is not None
    lab = rep["performance"]["labelled"]
    assert lab["n"] == 24
    assert 0.9 <= lab["accuracy"] < 1.0                     # 22 / 24
    assert lab["redFlagRecall"] == 1.0                      # every clinician-P1/P2 was predicted severe
    assert set(lab["subgroups"]) and all(0.0 <= v["accuracy"] <= 1.0 for v in lab["subgroups"].values())
    assert "fairnessGap" in lab and "equalOpportunityGap" in lab
    assert abs(lab["ageDefaultedShare"] - 6 / 24) < 1e-9
    assert rep["performance"]["note"].startswith("labelled")


def test_live_report_notes_when_labels_are_too_few(tmp_path, monkeypatch):
    monkeypatch.setenv("CAREROUTE_INFERENCE_LOG", str(tmp_path / "inf.jsonl"))
    monkeypatch.setenv("CAREROUTE_GROUND_TRUTH_LOG", str(tmp_path / "gt.jsonl"))
    monkeypatch.setenv("CAREROUTE_MONITOR_SOURCE", "live")
    monkeypatch.setenv("CAREROUTE_MONITOR_MIN_ROWS", "10")
    x = extract_features("sore throat", "18-39", "Male")
    for i in range(12):
        inference_log.log_prediction(x, "P4_NON_URGENT", 0.7, "m", f"t{i}", case_id=f"c{i}")
    _write_gt(tmp_path / "gt.jsonl", [{"caseId": "c0", "modelAcuity": "P4_NON_URGENT", "clinicianAcuity": "P4_NON_URGENT"}])
    rep = monitor._live_report(str(tmp_path))
    assert rep["performance"]["labelled"] is None
    assert "labelled" in rep["performance"]["note"]
