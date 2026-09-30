"""[MLOps] BATCH serving — the second serving mode deck 08 names.

Deck 08 treats real-time and batch as the two ways a model is served, and
CareRoute was real-time only: every prediction happened inside one patient's
request. The natural batch job here is a **nightly re-score of the escalation
queue** — the cases sitting in front of a clinician, scored by whichever model
version was live when they arrived.

Why it is worth running rather than being a checkbox: the queue and the model
both move. A case escalated at 09:00 is re-scored at 02:00 against the model
that was promoted at 18:00, and the interesting output is not the new acuity
but the DIRECTION of the change. A case the current model now considers MORE
urgent than the model that triaged it is a case whose place in the queue is
wrong, and nobody would otherwise find out until a clinician got to it.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not mutate escalations. A batch job that silently re-prioritises a
clinical queue overnight is an autonomous clinical decision, which is precisely
what the HITL design says this system does not make (`AGENT` vs policy node,
see the capability audit). The output is a report; acting on it is a person's
job.

INPUT MODES, and the honest limit
---------------------------------
  --from-store   score the in-process `store` (what an in-app scheduler, or a
                 test, would call). The Store is IN-MEMORY, so an out-of-process
                 nightly cron sees an empty one — it is not a scheduling
                 oversight, it is the missing durable store recorded in
                 [[Infra-Dependent Work 2026-09-02]]. Batch serving needs
                 somewhere to read from, and that is the piece not built.
  --input FILE   score a JSONL export: one object per line with `caseId`,
                 `text`, and optionally `acuity`, `ageBand`, `sex`. This is the
                 mode CI and a real scheduler can both use, because a file is a
                 durable store.

Run it:
    python -m app.ml.batch_score --from-store --output rescored.json
    python -m app.ml.batch_score --input queue.jsonl --output rescored.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

logger = logging.getLogger("careroute.ml.batch")

#: Directions a re-score can move a case, most actionable first.
MORE_URGENT, UNCHANGED, LESS_URGENT = "more_urgent", "unchanged", "less_urgent"


def _direction(previous: str | None, rescored: str) -> str:
    from ..models import acuity_rank

    if not previous:
        return UNCHANGED
    before, after = acuity_rank(previous), acuity_rank(rescored)
    if after < before:
        return MORE_URGENT          # lower rank == more severe
    return LESS_URGENT if after > before else UNCHANGED


def rows_from_store(store) -> list[dict]:
    """PENDING escalations only: a decided case has a clinician's answer on it,
    and re-scoring that is second-guessing a human, not serving a model."""
    return [
        {
            "caseId": escalation.caseId,
            "escalationId": escalation.id,
            "text": escalation.normalisedSymptoms or escalation.patientSummary or "",
            "acuity": getattr(escalation.acuity, "code", None) or escalation.acuity,
            "createdAt": escalation.createdAt,
        }
        for escalation in store.list_escalations()
        if escalation.status == "pending"
    ]


def rows_from_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # One malformed line must not cost the whole batch. A batch job
                # that aborts on row 4000 of 5000 has served nothing.
                logger.warning("skipping unparseable line %d of %s", n, path)
    return rows


def score(rows: list[dict], model=None) -> dict:
    """Re-score every row with the CURRENTLY served model."""
    from .model import get_model

    model = model if model is not None else get_model()
    # The version that DID the scoring, recorded on the report: a re-score
    # without it cannot be acted on a week later, because nobody can say which
    # model disagreed with which.
    version = (getattr(model, "_audit", None) or {}).get("modelVersion") or "unknown"

    scored, failures = [], 0
    for row in rows:
        text = (row.get("text") or "").strip()
        if not text:
            failures += 1
            continue
        try:
            prediction = model.predict(text, row.get("ageBand"), row.get("sex"))
        except Exception:  # noqa: BLE001 - one bad row must not end the batch
            logger.warning("re-score failed for case %s", row.get("caseId"), exc_info=False)
            failures += 1
            continue
        previous = row.get("acuity")
        scored.append({
            "caseId": row.get("caseId"),
            "escalationId": row.get("escalationId"),
            "previousAcuity": previous,
            "rescoredAcuity": prediction["acuity_code"],
            "confidence": round(float(prediction.get("confidence", 0.0)), 4),
            "direction": _direction(previous, prediction["acuity_code"]),
        })

    counts = {d: sum(1 for s in scored if s["direction"] == d)
              for d in (MORE_URGENT, UNCHANGED, LESS_URGENT)}
    return {
        "modelVersion": version,
        "scored": len(scored),
        "failed": failures,
        "counts": counts,
        # The cases a clinician would actually want surfaced, most severe first.
        "needsReview": sorted(
            [s for s in scored if s["direction"] == MORE_URGENT],
            key=lambda s: s["rescoredAcuity"],
        ),
        "rows": scored,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch re-score the open escalation queue.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-store", action="store_true", help="score the in-process store")
    source.add_argument("--input", help="JSONL export to score")
    parser.add_argument("--output", help="write the report here (default: stdout)")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.from_store:
        from ..store import store

        rows = rows_from_store(store)
    else:
        rows = rows_from_jsonl(args.input)

    report = score(rows)
    body = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(body + "\n")
    else:
        sys.stdout.write(body + "\n")

    logger.info(
        "batch re-score: %d scored, %d failed, %d now more urgent (model %s)",
        report["scored"], report["failed"], report["counts"][MORE_URGENT], report["modelVersion"],
    )
    # Exit 0 even when cases need review: "the queue has drifted" is the job's
    # OUTPUT, not a failure of the job. A non-zero exit here would train whoever
    # reads the schedule to ignore it.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
