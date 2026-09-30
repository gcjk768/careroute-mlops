"""[MLOps] Pillar 3 — the INFERENCE LOG that turns monitoring real.

Every served prediction is appended (best-effort, never fatal) as one JSONL
row: the de-identified feature vector, the predicted acuity, the calibrated
confidence, and the model version. NO free text is ever logged — the feature
vector is the binary symptom/demographic encoding from `features.py`, so the
log is PII-safe by construction (redaction happens upstream of triage anyway).

This is what lets `app/ml/monitor.py` compare the REFERENCE distribution
against REAL production traffic (`CAREROUTE_MONITOR_SOURCE=live`) instead of a
simulated shift, and it is the raw material for the drift -> retrain loop.

Location: $CAREROUTE_INFERENCE_LOG (file path) or
`backend/monitoring/inference_log.jsonl` by default. Size-capped: once the file
exceeds ~10 MB new rows are dropped (monitoring telemetry must never fill the
disk of the serving host).
"""
from __future__ import annotations

import json
import logging
import os
import threading

import numpy as np

logger = logging.getLogger("careroute.ml.inference_log")

_MAX_BYTES = 10 * 1024 * 1024  # stop appending past ~10 MB — telemetry, not a DB
_DEFAULT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "monitoring", "inference_log.jsonl")
)
_DEFAULT_GROUND_TRUTH_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "monitoring", "ground_truth.jsonl")
)
_lock = threading.Lock()


def log_path() -> str:
    return os.environ.get("CAREROUTE_INFERENCE_LOG", _DEFAULT_PATH)


def ground_truth_path() -> str:
    """Where the HITL agent appends clinician labels (see store._record_ground_truth).
    Same env var + default as the writer, so reader and writer cannot drift."""
    return os.environ.get("CAREROUTE_GROUND_TRUTH_LOG", _DEFAULT_GROUND_TRUTH_PATH)


def log_prediction(features: np.ndarray, acuity_code: str, confidence: float,
                   model_version: str, timestamp: str, *,
                   case_id: str | None = None, age_provided: bool = False) -> None:
    """Append one served prediction. Strictly best-effort: any failure is
    logged (the first failure at WARNING) and swallowed — telemetry must never
    break serving.

    `case_id` is the join key for the HITL ground-truth loop: a clinician's
    later final acuity for this case is matched back to THIS feature vector
    and prediction. `age_provided` records whether the age band was supplied
    or defaulted — the age-aware mitigation cannot fire on a defaulted band,
    so the monitor reports how often serving ran blind to age."""
    try:
        path = log_path()
        row = json.dumps({
            "ts": timestamp,
            "caseId": case_id,
            "features": [round(float(v), 4) for v in np.asarray(features).ravel()],
            "acuity": acuity_code,
            "confidence": round(float(confidence), 4),
            "modelVersion": model_version,
            "ageProvided": bool(age_provided),
        })
        with _lock:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if os.path.exists(path) and os.path.getsize(path) > _MAX_BYTES:
                return  # size cap reached — drop silently
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(row + "\n")
    except Exception as exc:  # noqa: BLE001 - telemetry IO is best-effort; serving must not fail on it
        # WARN once, then stay quiet: on AWS the shared volume was root-owned and
        # this non-root process could not write, which at DEBUG left drift
        # monitoring with no data and no trace of why (24 Sep 2026).
        global _warned
        if not _warned:
            _warned = True
            logger.warning("inference-log append failing at %s (%s); drift monitoring will have no data",
                           log_path(), type(exc).__name__)
        else:
            logger.debug("inference-log append skipped", exc_info=False)


_warned = False


def load_recent(limit: int = 5000) -> tuple[np.ndarray, list[str]] | None:
    """Return (X, predicted_acuity_codes) for up to the most recent `limit`
    logged predictions, or None if the log is missing/unreadable/empty."""
    try:
        path = log_path()
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()[-limit:]
        feats, acuities = [], []
        for line in lines:
            try:
                row = json.loads(line)
                feats.append(row["features"])
                acuities.append(row["acuity"])
            except Exception:
                # A torn row is expected when reading a log being appended to
                # concurrently; debug-level so it is discoverable but not noisy.
                logger.debug("skipping unparseable inference-log row", exc_info=True)
                continue
        if not feats:
            return None
        return np.asarray(feats, dtype=float), acuities
    except Exception:  # noqa: BLE001 - telemetry IO is best-effort; serving must not fail on it
        logger.debug("inference-log read skipped", exc_info=False)
        return None


def load_labelled(limit: int = 5000):
    """[MLOps][XRAI] JOIN the inference log to the HITL ground-truth log on
    `caseId` and return the labelled production sample:

        (X, y_true_idx, y_pred_idx, subgroups, age_provided)

    `y_true` is the clinician's final acuity (index), `y_pred` the served
    prediction, `subgroups` the "<age band> · <sex>" label reconstructed from
    the logged one-hot demographics (the same format data.generate_dataset
    uses, so fairness.* functions apply unchanged). Rows without a clinician
    label, or labels without a matching inference row, are skipped. Returns
    None when nothing joins. The ground-truth log is read in order, so the
    LATEST label for a case wins if a clinician revised a decision.
    """
    try:
        from .data import ACUITY_INDEX_TO_CODE
        from .features import AGE_BANDS, FEATURE_NAMES

        gt_path = ground_truth_path()
        if not os.path.exists(gt_path):
            return None
        code_to_idx = {c: i for i, c in enumerate(ACUITY_INDEX_TO_CODE)}
        labels: dict[str, int] = {}
        with open(gt_path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("skipping unparseable ground-truth row", exc_info=True)
                    continue
                cid, code = row.get("caseId"), row.get("clinicianAcuity")
                if cid and code in code_to_idx:
                    labels[str(cid)] = code_to_idx[code]
        if not labels:
            return None

        path = log_path()
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()[-limit:]
        age_cols = [
            FEATURE_NAMES.index("age_" + b.replace("-", "_").replace("+", "p")) for b in AGE_BANDS
        ]
        sex_col = FEATURE_NAMES.index("sex_female")
        feats, y_true, y_pred, groups, provided = [], [], [], [], []
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("skipping unparseable inference-log row", exc_info=True)
                continue
            cid = row.get("caseId")
            if not cid or str(cid) not in labels or row.get("acuity") not in code_to_idx:
                continue
            x = [float(v) for v in row["features"]]
            band = next((b for b, j in zip(AGE_BANDS, age_cols, strict=True) if x[j] > 0.5), AGE_BANDS[2])
            sex = "Female" if x[sex_col] > 0.5 else "Male"
            feats.append(x)
            y_true.append(labels[str(cid)])
            y_pred.append(code_to_idx[row["acuity"]])
            groups.append(f"{band} · {sex}")
            provided.append(bool(row.get("ageProvided", False)))
        if not feats:
            return None
        return (
            np.asarray(feats, dtype=float), np.asarray(y_true), np.asarray(y_pred),
            groups, np.asarray(provided, dtype=bool),
        )
    except Exception:  # noqa: BLE001 - telemetry IO is best-effort; monitoring must degrade, not crash
        logger.debug("labelled join skipped", exc_info=False)
        return None
