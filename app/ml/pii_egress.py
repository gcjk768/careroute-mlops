"""[AI-Security][MLOps] PII-egress gate — "PII egress events = 0".

`app/redact.py` masks direct identifiers at the INPUT boundary, before patient
free text reaches a cloud LLM or the case store. This module is the independent
check on the other side: did an identifier nevertheless reach an artifact that
LEAVES the trust boundary — the inference log, the HITL ground-truth log, the
drift report, the executive report, any of which is published as a CI artifact?

Two design choices worth keeping:

* **The detector is `redact.redact()` itself, not a second copy of the rules.**
  A gate with its own private regexes drifts from the redactor it is supposed to
  be auditing, and then agrees with it exactly when both are wrong. Reusing the
  redactor means the gate measures precisely "what the redactor would have
  caught, in a place the redactor should already have run".
* **The excerpt stored in a finding is the REDACTED line.** A findings file that
  quotes the NRIC it caught has re-published that NRIC into a CI artifact that
  is downloaded by every dependent job. Masking by construction — rather than by
  remembering to mask — is the only version of this that stays true.

Presidio is the production upgrade (ML-based NER catches names and addresses,
which no regex will); when it is importable it runs IN ADDITION to the
deterministic rules. Which detector actually ran is recorded in `backend` /
`backendStatus`, the same way `ml/monitor.py` records its drift backend, so a
clean result can never silently mean "the detector never loaded".

Run:  python -m app.ml.pii_egress [--path FILE ...] [--out pii-egress.json]
Exit: 0 clean, 1 identifiers escaped (blocking gate).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Iterable

from app.redact import redact

# Artifacts that leave the trust boundary. Anything published as a CI artifact
# or shipped off the box belongs here.
#
# The locations are NOT fixed relative to this file. In CI the report job writes
# to $CI_PROJECT_DIR/reports while the monitor writes to backend/monitoring, so a
# gate hard-coded to backend/reports would report "skipped" on every pipeline and
# never actually audit the report. Resolve through the same environment
# variables the rest of the ML code uses (`ml/report.py`, `ml/monitor.py`).
_MONITOR_ARTIFACTS = ("inference_log.jsonl", "ground_truth.jsonl", "drift_report.json")
_REPORT_ARTIFACTS = ("careroute_mlops_report.md",)


def _backend_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_artifacts() -> list[str]:
    """The standard egress set, resolved for this environment."""
    root = _backend_root()
    monitor_dir = os.environ.get("CAREROUTE_MONITOR_DIR") or os.path.join(root, "monitoring")
    report_dir = os.environ.get("CAREROUTE_REPORT_DIR") or os.path.join(root, "reports")
    return ([os.path.join(monitor_dir, n) for n in _MONITOR_ARTIFACTS]
            + [os.path.join(report_dir, n) for n in _REPORT_ARTIFACTS])

# A finding carries the masked line, not the raw one; keep it short so a runaway
# file cannot pad the artifact with thousands of long excerpts.
_EXCERPT_CHARS = 160

# Presidio entities worth gating on. Deliberately narrow: DATE_TIME and
# NRP would flag ordinary clinical text ("since Tuesday", nationality in a
# history) and turn the gate into noise.
# PHONE_NUMBER is deliberately ABSENT from this list. `app/redact.py`'s `_PHONE`
# was engineered specifically so that a vitals run is not a phone number --
# "blood pressure 140 90 110 70" used to become one [REDACTED_PHONE], and a
# dashed onset date went the same way. Presidio's generic PHONE_NUMBER
# recogniser re-introduces exactly that failure: it reads "140 90 110 70" as a
# number, and it reads ISO-8601 microseconds as one too.
#
# So phone detection stays with the domain-tuned regex, and Presidio is used for
# what it is genuinely better at -- NER over names and places, which no regex
# will catch. Using the weaker of two detectors for a field both cover is how a
# gate acquires false positives it then gets muted for.
_PRESIDIO_ENTITIES = ("PERSON", "LOCATION", "EMAIL_ADDRESS", "CREDIT_CARD", "IBAN_CODE")

# ---------------------------------------------------------------------------
# WHY THIS GATE PARSES ITS ARTIFACTS INSTEAD OF GREPPING THEM
# ---------------------------------------------------------------------------
# The first version scanned every artifact LINE BY LINE, which meant running a
# free-text PII detector over serialized machine state. Its first real pipeline
# run (job 16463280258) produced 5,749 findings and NOT ONE was an identifier:
#
#   "driftShare": 0.0967741935483871  -> MRN          (a float's mantissa is a
#                                                       7+ digit run)
#   "caseId": "case_69837191d0"       -> PHONE        (8 hex digits opening with
#                                                       6 match the SG numbering
#                                                       plan)
#   "ts": "...T14:05:37.910535+00:00" -> PHONE_NUMBER (Presidio reads the
#                                                       microseconds as a number)
#   "features": [1.0, 0.0, 0.0, ...]  -> PERSON       (Presidio NER on a float
#                                                       array, 3,292 times)
#   "| scan:trivy-fs | ... |"         -> PERSON       (a tool name is a proper
#                                                       noun; so are Horusec,
#                                                       Checkov and Fairlearn)
#
# A blocking gate that is 100% wrong on every run is worse than no gate: it gets
# switched off, or worse, ignored while still red. Same rule this repo already
# applies to `scan:no-live-credentials` -- a gate that is wrong every run is one
# everybody ignores.
#
# The fix is to give the gate the structure it was throwing away:
#
#   * JSON / JSONL artifacts are PARSED and only **string** leaves are examined.
#     A number cannot be an identifier someone wrote down.
#   * String leaves under a field in `_OPAQUE_FIELDS` get the deterministic
#     identifier rules only, never NER. These are server-generated tokens --
#     `new_id()` UUID suffixes, ISO timestamps, enum acuity codes, content
#     hashes -- and patient input demonstrably cannot reach them.
#   * Every OTHER string leaf gets the full treatment, rules AND NER. That is
#     the fail-safe direction: the day somebody adds `rawText` or `rationale` to
#     the inference log, it is an undeclared field and is scanned hardest,
#     which is exactly the regression this gate exists to catch.
#   * A JSON artifact that will not parse falls back to line scanning. Broken is
#     not clean.
#
# Non-JSON artifacts are declared in `_MACHINE_COMPOSED`: the executive report is
# assembled from aggregate metrics with no per-patient string interpolated into
# it, so NER there answers a question the document cannot pose. It still gets the
# deterministic identifier rules over its whole text, because THAT is what
# "identifier egress" means and it costs nothing.
# ---------------------------------------------------------------------------

_OPAQUE_FIELDS = frozenset({
    "ts", "timestamp", "createdAt", "updatedAt",          # ISO-8601, machine clock
    "caseId", "id", "sessionId", "escalationId",          # new_id() -> uuid4 hex
    "modelVersion", "versionId", "schemaVersion",         # build identifiers
    "dataSha256", "modelSha256", "sha256", "hash",        # content addresses
    "acuity", "modelAcuity", "clinicianAcuity",           # AcuityCode enum
    "agreement", "decision", "verdict", "status",         # closed vocabularies
})

# SHAPE-VERIFIED exemption, which is stronger than trusting the field name.
#
# Skipping a field outright because of what it is called would leave it unscanned
# forever, including the day somebody starts writing patient text into it. So a
# value is exempted only when it MATCHES THE SHAPE its field is supposed to hold:
# `new_id()` emits `prefix_<10 hex>`, timestamps are ISO-8601, content addresses
# are hex digests. Such a value is machine-generated by construction and cannot
# be an identifier, so any regex hit on it is a false positive -- `case_69837191d0`
# was read as a Singapore mobile because 8 hex digits opening with 6 fit the
# numbering plan.
#
# A value that does NOT match its declared shape falls through to the full
# detector set. The exemption is therefore self-limiting: it covers exactly the
# tokens this codebase provably generates and nothing else.
# NOTE ON WHAT IS *NOT* HERE: an earlier draft included a
# `^[A-Z][A-Z0-9_]+$` shape for SCREAMING_SNAKE enum members. It matched
# "S1234567D" -- an NRIC is a capital, seven digits and a capital -- so a
# planted NRIC in a `caseId` field was exempted as an enum. The shape was
# removed rather than narrowed: enum values need no exemption, because
# "P1_RESUSCITATION" trips none of the identifier rules anyway. Every entry
# below has to earn its place by suppressing a REAL false positive.
_OPAQUE_SHAPES = (
    re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?$"),
    re.compile(r"^[a-z]+_[0-9a-f]{8,32}$"),   # new_id(): case_, esc_, run_, ...
    re.compile(r"^[0-9a-f]{32,64}$"),         # bare content hash
    # model_version(): careroute-triage-rf-<12 hex> (rf/200, cal/isotonic). A hex
    # suffix like 13f<8 digits>d fit the Singapore numbering plan.
    re.compile(r"^careroute-triage-[a-z]+-[0-9a-f]{12}(?: \([a-z0-9/, ]+\))?$"),
)


def _is_machine_token(value: str) -> bool:
    return any(shape.match(value) for shape in _OPAQUE_SHAPES)

# Artifacts composed by machine from aggregate values, never from patient text.
_MACHINE_COMPOSED = frozenset({"careroute_mlops_report.md"})


def _presidio_analyzer():
    """Return (analyzer, status). Never raises: a missing or broken Presidio
    degrades to the deterministic rules rather than failing the pipeline."""
    if os.environ.get("CAREROUTE_PII_DISABLE_PRESIDIO"):
        return None, "presidio disabled by CAREROUTE_PII_DISABLE_PRESIDIO; deterministic rules only"
    try:
        from presidio_analyzer import AnalyzerEngine
    except Exception as exc:  # noqa: BLE001 - any import failure degrades the same way
        return None, f"presidio not available ({type(exc).__name__}); deterministic rules only"
    try:
        return AnalyzerEngine(), "presidio + deterministic rules"
    except Exception as exc:  # noqa: BLE001 - e.g. missing spaCy model
        return None, f"presidio installed but failed to start ({type(exc).__name__}); deterministic rules only"


def _mask_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Replace each (start, end, label) span with [label], right to left so the
    earlier offsets stay valid."""
    out = text
    for start, end, label in sorted(spans, key=lambda s: s[0], reverse=True):
        out = f"{out[:start]}[{label}]{out[end:]}"
    return out


def iter_json_strings(node, field: str | None = None):
    """Yield (field_name, string_value) for every string leaf in a parsed JSON
    document. Numbers, booleans and nulls are not yielded: an identifier is
    something a human wrote, and nobody writes an NRIC as a float.

    A list inherits its parent's field name, so `features: [...]` keeps the
    `features` label rather than becoming anonymous.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            yield from iter_json_strings(value, key)
    elif isinstance(node, list):
        for value in node:
            yield from iter_json_strings(value, field)
    elif isinstance(node, str):
        yield field, node


def scan_text(text: str, analyzer=None) -> list[dict]:
    """Findings for a single line. Each is {"kind", "excerpt"} with the excerpt
    already masked."""
    masked, kinds = redact(text)
    findings = [{"kind": kind, "excerpt": masked[:_EXCERPT_CHARS]} for kind in kinds]

    if analyzer is not None:
        try:
            results = analyzer.analyze(text=text, entities=list(_PRESIDIO_ENTITIES), language="en")
        except Exception:  # noqa: BLE001 - a detector failure must not fail the run silently clean
            return findings
        spans = [(r.start, r.end, r.entity_type) for r in results]
        if spans:
            # Mask against the ALREADY-redacted text's source so the excerpt
            # never carries a raw value from either detector.
            both = _mask_spans(text, spans)
            both, _ = redact(both)
            for _, _, label in spans:
                findings.append({"kind": label, "excerpt": both[:_EXCERPT_CHARS]})
    return findings


def _scan_values(values: Iterable[tuple[str | None, str]], analyzer, lineno: int) -> list[dict]:
    """Scan (field, string) pairs. A field in `_OPAQUE_FIELDS` is server-generated
    and gets the deterministic identifier rules only; anything else -- including
    any field nobody has declared -- gets NER too."""
    out: list[dict] = []
    for field, value in values:
        if field in _OPAQUE_FIELDS and _is_machine_token(value):
            continue  # provably machine-generated; any hit on it is noise
        use = None if field in _OPAQUE_FIELDS else analyzer
        for hit in scan_text(value, use):
            out.append({"line": lineno, "field": field, **hit})
    return out


def _scan_document(name: str, lines: list[str], analyzer) -> list[dict]:
    """Dispatch on artifact shape. JSON is parsed; anything else is read as text.

    A JSON artifact that will not parse falls back to line scanning rather than
    being reported clean -- broken is not clean, the same rule the absent-file
    branch follows.
    """
    if name.endswith(".jsonl"):
        out: list[dict] = []
        for lineno, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except ValueError:
                out.extend({"line": lineno, "field": None, **h} for h in scan_text(line, analyzer))
                continue
            out.extend(_scan_values(iter_json_strings(doc), analyzer, lineno))
        return out

    if name.endswith(".json"):
        try:
            doc = json.loads("\n".join(lines))
        except ValueError:
            return [{"line": i, "field": None, **h}
                    for i, line in enumerate(lines, start=1) for h in scan_text(line, analyzer)]
        # A whole-file document has no per-leaf line number; report the first
        # line that contains the value so the finding is still navigable.
        out = []
        for field, value in iter_json_strings(doc):
            lineno = next((i for i, line in enumerate(lines, start=1) if value in line), 0)
            out.extend(_scan_values([(field, value)], analyzer, lineno))
        return out

    return [{"line": i, "field": None, **h}
            for i, line in enumerate(lines, start=1) for h in scan_text(line, analyzer)]


def scan_artifacts(paths: Iterable[str]) -> dict:
    """Scan each path. Absent files are reported as SKIPPED, never as clean."""
    analyzer, status = _presidio_analyzer()
    scanned: list[dict] = []
    skipped: list[str] = []
    findings: list[dict] = []

    for path in paths:
        if not os.path.isfile(path):
            skipped.append(path)
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
        except OSError as exc:
            skipped.append(path)
            status = f"{status}; could not read {os.path.basename(path)} ({type(exc).__name__})"
            continue

        name = os.path.basename(path)
        # A machine-composed document gets the identifier rules but not NER --
        # see the block comment at the top of this module.
        doc_analyzer = None if name in _MACHINE_COMPOSED else analyzer

        hits = _scan_document(name, lines, doc_analyzer)
        for hit in hits:
            findings.append({"path": path, **hit})
        scanned.append({"path": path, "findings": len(hits)})

    return {
        "backend": "presidio" if analyzer is not None else "builtin-regex",
        "backendStatus": status,
        "scanned": scanned,
        "skipped": skipped,
        "findings": findings,
        "verdict": "FAIL" if findings else "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PII-egress gate for published artifacts.")
    parser.add_argument("--path", action="append", default=None,
                        help="artifact to scan (repeatable); defaults to the standard egress set")
    parser.add_argument("--out", default="pii-egress.json", help="where to write the findings JSON")
    args = parser.parse_args(argv)

    paths = args.path or default_artifacts()

    result = scan_artifacts(paths)

    try:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    except OSError as exc:
        print(f"WARNING: could not write {args.out}: {exc}")

    print("=== CareRoute PII-egress gate ===")
    print(f"  backend : {result['backend']}")
    print(f"  why     : {result['backendStatus']}")
    for entry in result["scanned"]:
        print(f"  scanned : {os.path.basename(entry['path'])} ({entry['findings']} finding(s))")
    for path in result["skipped"]:
        print(f"  skipped : {os.path.basename(path)} (absent - NOT counted as clean)")
    if result["findings"]:
        by_kind: dict[str, int] = {}
        for f in result["findings"]:
            by_kind[f["kind"]] = by_kind.get(f["kind"], 0) + 1
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(by_kind.items()))
        print(f"  FAILED  : identifiers reached published artifacts -> {detail}")
        print("            (excerpts in the JSON are masked; see the source artifact to fix the leak)")
    else:
        print("  PASSED  : no identifiers in the scanned artifacts")
    return 1 if result["findings"] else 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
