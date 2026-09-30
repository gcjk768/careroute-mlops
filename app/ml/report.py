"""[MLOps] Executive PDF report generator - a management-facing summary of the
model + pipeline, suitable to hand to non-technical stakeholders.

    python -m app.ml.report

Reads the model audit (from CAREROUTE_MODEL_DIR/model_audit.json, or computes it
fresh via get_model().fairness()) and the drift report (CAREROUTE_MONITOR_DIR/
drift_report.json if present) and renders a polished PDF:

    reports/careroute_mlops_report.pdf   (override dir with CAREROUTE_REPORT_DIR)

Pure-Python (reportlab) - no system dependencies, runs locally and in CI.
"""
from __future__ import annotations

import json
import os
import subprocess  # nosec B404  # only fixed-argv `git rev-parse`, no shell
from datetime import UTC, datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_BRAND = colors.HexColor("#1f6feb")
_DARK = colors.HexColor("#0b1f3a")
_MUTED = colors.HexColor("#5b6b7b")
_OK = colors.HexColor("#1a7f37")


def _load_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001 - report generation is best-effort and must not fail the pipeline
        return None


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            # build-time metadata only; `git` resolves from the trusted CI image PATH
            ["git", "rev-parse", "--short", "HEAD"], text=True, timeout=10,  # noqa: S607  # nosec B603 B607
        ).strip()
    except Exception:  # noqa: BLE001 - report generation is best-effort and must not fail the pipeline
        return os.environ.get("CI_COMMIT_SHORT_SHA", "local")


def _audit() -> dict:
    md = os.environ.get("CAREROUTE_MODEL_DIR", "models")
    audit = _load_json(os.path.join(md, "model_audit.json"))
    if audit:
        return audit
    from .model import get_model  # compute fresh if the CI artifact isn't present
    return get_model().fairness()


def _drift() -> dict | None:
    mon = os.environ.get("CAREROUTE_MONITOR_DIR", "monitoring")
    return _load_json(os.path.join(mon, "drift_report.json"))


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H1c", parent=ss["Title"], textColor=_DARK, fontSize=22, spaceAfter=6))
    ss.add(ParagraphStyle("Sub", parent=ss["Normal"], textColor=_MUTED, fontSize=11, alignment=TA_CENTER))
    ss.add(ParagraphStyle("H2", parent=ss["Heading2"], textColor=_BRAND, fontSize=13, spaceBefore=14, spaceAfter=4))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=9.5, leading=13, textColor=_DARK))
    ss.add(ParagraphStyle("Small", parent=ss["Normal"], fontSize=8, textColor=_MUTED))
    return ss


def _kv_table(rows: list[list], col_widths=None) -> Table:
    t = Table(rows, colWidths=col_widths or [6 * cm, 6 * cm, 4.5 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cdd7e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fb")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def _status(ok: bool) -> str:
    return "PASS" if ok else "REVIEW"


def _worst_subgroup_recall(audit: dict) -> tuple[str, bool]:
    """("raw / served" worst per-subgroup red-flag recall, all gated subgroups
    >= floor). ("n/a", False) for an audit that predates the block."""
    block = audit.get("subgroupRedFlagRecall") or {}
    if not block:
        return "n/a", False
    floor, support = block.get("floor", 0.95), block.get("minSevereSupport", 20)
    worst, ok = [], True
    for key in ("raw", "served"):
        gated = [v["recall"] for v in (block.get(key) or {}).values()
                 if v.get("recall") is not None and v.get("nSevere", 0) >= support]
        worst.append(min(gated, default=1.0))
        ok = ok and all(r >= floor for r in gated)
    return f"{worst[0]:.3f} / {worst[1]:.3f}", ok


def _spread(d) -> float:
    """Scalar parity spread (max-min across subgroups) from a {'byGroup': {...}}."""
    try:
        vals = list((d or {}).get("byGroup", {}).values())
        return round(max(vals) - min(vals), 3) if vals else 0.0
    except Exception:  # noqa: BLE001 - report generation is best-effort and must not fail the pipeline
        return 0.0


# One-line purpose per pipeline job (used by the security sections below).
_JOB_PURPOSE = {
    "lint:backend": "Python static lint (ruff)",
    "lint:frontend": "JS/JSX lint (eslint)",
    "train:model": "Build + release-gate + MLflow-register the model",
    "data:version": "Snapshot + hash the training dataset (DVC)",
    "publish:model": "Publish model to GitLab Package Registry",
    "test:backend": "Unit + integration tests (pytest)",
    "test:pipeline": "End-to-end agent-pipeline test",
    "test:model-gate": "BLOCKING gate: accuracy + red-flag recall",
    "test:data-lineage": "BLOCKING gate: dataset hash == model training data",
    "test:triage-eval": "Offline triage benchmark on gold vignettes",
    "test:robustness": "Adversarial robustness (IBM ART)",
    "monitor:evidently": "Data-drift + performance report (Evidently/PSI)",
    "report:pdf": "This executive PDF report",
    "ai-security:guardrail-regression": "Prompt-injection / safety-override regression",
    "ai-security:fairness-gate": "Subgroup accuracy parity (Fairlearn)",
    "ai-security:promptfoo": "LLM red-team regression (Promptfoo)",
    "ai-security:garak": "LLM vulnerability scan (Garak)",
    "ai-security:deepteam": "OWASP LLM Top 10 red-team (DeepTeam)",
    "ai-security:pyrit": "Multi-turn adversarial orchestration (PyRIT)",
    "scan:secrets-gitleaks": "Secret scanning (Gitleaks)",
    "scan:secrets-trufflehog": "Secret scanning with live credential verification (TruffleHog)",
    "scan:secrets-detect-secrets": "Secret scanning, baseline-diff (detect-secrets)",
    "secret_detection": "Secret scanning (GitLab native template)",  # nosec B105  # a job label, not a secret
    "sast": "SAST (GitLab native template)",
    "scan:sast-semgrep": "SAST (Semgrep)",
    "scan:sast-bandit": "Python SAST (Bandit)",
    "scan:sast-ruff-security": "Python SAST over scripts/tests (Ruff flake8-bandit rules)",
    "scan:sast-njsscan": "Node / Next.js SAST (njsscan)",
    "scan:sast-eslint-security": "Frontend SAST (ESLint security plugin)",
    "scan:sast-horusec": "Multi-language SAST aggregator (Horusec)",
    "scan:sast-sonarqube": "Code quality + security hotspots (SonarQube)",
    "scan:trivy-fs": "SCA / secret / IaC scan (Trivy)",
    "scan:deps-audit": "Dependency CVE audit (pip-audit / Safety / npm)",
    "scan:deps-osv": "Dependency CVE audit, OSV database (OSV-Scanner)",
    "scan:deps-grype": "Dependency CVE audit, Anchore database (Grype)",
    "scan:deps-dependency-check": "Dependency CVE audit, NVD (OWASP Dependency-Check)",
    "scan:deps-retirejs": "Known-vulnerable JS libraries (Retire.js)",
    "scan:iac-checkov": "IaC / compose / CI-config scan (Checkov)",
    "scan:iac-kics": "IaC / compose / CI-config scan (KICS)",
    "scan:licenses": "Dependency licence compliance (pip-licenses / license-checker)",
    "scan:modelscan": "Malicious-model scan (modelscan)",
    "scan:model-fickling": "Pickle-level model artifact analysis (Fickling)",
    "scan:sbom-cyclonedx": "SBOM + AI-BOM (CycloneDX)",
    "scan:dockerfile-hadolint": "Dockerfile lint (Hadolint)",
    "test:load-locust": "Load test against a booted backend (Locust)",
    "test:api-fuzz-schemathesis": "OpenAPI property-based fuzzing (Schemathesis)",
    "build:frontend": "Build the Next.js production bundle",
    "build:images": "Build backend + frontend container images",
    "scan:container-image": "Container image CVE scan (Trivy)",
    "scan:container-dockle": "Container CIS / hardening lint (Dockle)",
    "deploy:push-images": "Push the scanned images to ECR (commit-SHA tag)",
    "deploy:trigger-infra": "Trigger the infra pipeline with IMAGE_TAG",
    "deploy:shadow-model": "Shadow-deploy candidate to staging (manual)",
    "deploy:promote-production": "Promote model to Production (manual)",
    "rollback:production": "Rollback: infra re-applies the previous release + registry re-pin",
    "dast:owasp-zap": "DAST against the running app (OWASP ZAP)",
    "dast:zap-api": "API-aware DAST driven by the OpenAPI schema (OWASP ZAP)",
    "dast:zap-full": "Active/full DAST attack scan (OWASP ZAP)",
    "dast:nikto": "Web-server misconfiguration scan (Nikto)",
    "dast:nuclei": "Template-based vulnerability scan (Nuclei)",
    "loadtest:staging": "Load test against the deployed target (Locust)",
}

_STATUS_LABEL = {
    "success": "PASS", "failed": "FAIL", "manual": "manual",
    "skipped": "skipped", "canceled": "canceled", "running": "running",
}

# The two SECURITY stages get dedicated report sections (job -> purpose -> status):
# ai-security (Responsible-AI + LLM red-team) and security-scan (supply chain /
# AppSec), plus the scan jobs that live in build/post-deploy (container + DAST).
_SECURITY_SECTIONS = [
    ("AI security - Responsible-AI &amp; LLM red-team gates",
     ("Guardrail and fairness regressions run on every pipeline; the LLM red-team scanners "
     "(OWASP LLM Top 10) run against a configured target model on release/manual cadence."),
     [("ai-security", None)]),
    ("Security scanning - supply chain &amp; AppSec",
     ("Static analysis, secrets, dependency CVEs, model-artifact and container scanning, plus an "
     "SBOM &amp; AI-BOM for traceability. Gitleaks and CRITICAL CVEs (Trivy) block the release; "
     "container scans run against the exact built images; DAST probes the running app."),
     [("security-scan", None),
      ("build", ("scan:container-image", "scan:container-dockle")),
      ("post-deploy", ("dast:owasp-zap",))]),
    ("Performance &amp; API robustness",
     ("Load testing and OpenAPI property-based fuzzing run against a backend booted inside the "
      "job, so both produce evidence on every pipeline rather than waiting on a deployed target. "
      "The load thresholds are ADVISORY until a baseline exists (see README)."),
     [("test", ("test:load-locust", "test:api-fuzz-schemathesis")),
      ("post-deploy", ("loadtest:staging",))]),
]


# ---------------------------------------------------------------------------
# Security-scan RESULTS: parse the scanners' machine-readable artifacts (when
# present) into per-scanner finding counts. In CI the report job runs after the
# security-scan stage and `needs:` these artifacts; locally, missing files are
# simply skipped.
# ---------------------------------------------------------------------------
def _artifact(name: str) -> dict | list | None:
    """Find a scanner artifact by name in the likely roots (CI project dir,
    repo root relative to backend/, or CWD)."""
    for root in (os.environ.get("CI_PROJECT_DIR", ""), "..", "."):
        if root == "":
            continue
        data = _load_json(os.path.join(root, name))
        if data is not None:
            return data
    return None


def _artifact_jsonl(name: str) -> list[dict] | None:
    """Read a JSON-LINES artifact (one JSON object per line) — the native output
    of TruffleHog and Nuclei. Returns None when the file is absent, so a missing
    scanner is skipped rather than reported as zero findings. Malformed lines are
    ignored: a truncated log must not break report generation."""
    for root in (os.environ.get("CI_PROJECT_DIR", ""), "..", "."):
        if root == "":
            continue
        path = os.path.join(root, name)
        if not os.path.exists(path):
            continue
        out: list[dict] = []
        try:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(obj, dict):
                        out.append(obj)
        except OSError:
            return None
        return out
    return None


_SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "MODERATE": 2, "LOW": 3, "WARNING": 4}

# SARIF is the one format most of these scanners agree on, so rather than a
# bespoke parser per tool we read SARIF generically and register the tools that
# emit it. Adding a SARIF-capable scanner is then one line here plus its CI job.
_SARIF_LEVEL_SEV = {"error": "HIGH", "warning": "MEDIUM", "note": "LOW", "none": "LOW"}

_SARIF_SCANNERS: list[tuple[str, str]] = [
    ("Checkov (IaC/config)", "checkov.sarif"),
    ("KICS (IaC/config)", "kics.sarif"),
    ("njsscan (Node SAST)", "njsscan.sarif"),
    ("ESLint security (frontend)", "eslint.sarif"),
    ("OSV-Scanner (deps)", "osv.sarif"),
    ("Grype (SCA)", "grype.sarif"),
    ("OWASP Dependency-Check", "dependency-check.sarif"),
]


def _sarif_results(name: str) -> list[dict] | None:
    """Flatten a SARIF document's runs[].results[]. None when absent/unparseable,
    which is how a scanner that did not run gets skipped instead of counted as
    clean."""
    doc = _artifact(name)
    if not isinstance(doc, dict):
        return None
    out: list[dict] = []
    for run in doc.get("runs") or []:
        if isinstance(run, dict):
            out.extend(r for r in (run.get("results") or []) if isinstance(r, dict))
    return out


def _sarif_finding(r: dict) -> tuple[str, str]:
    """(severity, human text) for one SARIF result.

    `level` is the SARIF-standard field, but most security scanners also set
    properties.security-severity (a CVSS-style 0-10 score) which is strictly more
    informative — so that wins when present."""
    sev = _SARIF_LEVEL_SEV.get(str(r.get("level") or "warning").lower(), "MEDIUM")
    try:
        score = float((r.get("properties") or {}).get("security-severity"))
    except (TypeError, ValueError):
        score = None
    if score is not None:
        sev = ("CRITICAL" if score >= 9.0 else "HIGH" if score >= 7.0
               else "MEDIUM" if score >= 4.0 else "LOW")
    msg = ((r.get("message") or {}).get("text") or "").strip()
    loc = ""
    locations = r.get("locations") or []
    if locations and isinstance(locations[0], dict):
        phys = locations[0].get("physicalLocation") or {}
        uri = (phys.get("artifactLocation") or {}).get("uri") or ""
        line = (phys.get("region") or {}).get("startLine")
        if uri:
            loc = f" ({os.path.basename(uri)}{':' + str(line) if line else ''})"
    return sev, f"{r.get('ruleId') or 'rule'}: {msg}{loc}"


def _gitlab_vulns(name: str) -> list[dict] | None:
    """GitLab's native SAST / Secret-Detection templates emit their own schema
    (gl-*-report.json), not SARIF. The jobs run on the Free tier even though the
    MR security widget does not, so the artifacts are worth parsing here."""
    doc = _artifact(name)
    if not isinstance(doc, dict):
        return None
    return [v for v in (doc.get("vulnerabilities") or []) if isinstance(v, dict)]


def _scan_results() -> list[list]:
    """Rows of [scanner, findings-count, severity breakdown / verdict] for each
    scan artifact that exists. An empty list means no artifacts were found."""
    rows: list[list] = []

    def sev_detail(by_sev: dict[str, int], empty: str) -> str:
        return ", ".join(f"{k}: {v}" for k, v in
                         sorted(by_sev.items(), key=lambda kv: _SEV_ORDER.get(kv[0], 5))) or empty

    leaks = _artifact("gitleaks.json")
    if isinstance(leaks, list):
        n = len(leaks)
        rows.append(["Gitleaks (secrets)", str(n),
                     "PASS - no committed secrets" if n == 0 else f"REVIEW - {n} potential secret(s)"])

    sarif = _artifact("semgrep.sarif")
    if isinstance(sarif, dict):
        results = (sarif.get("runs") or [{}])[0].get("results") or []
        by_level: dict[str, int] = {}
        for r in results:
            by_level[r.get("level", "warning")] = by_level.get(r.get("level", "warning"), 0) + 1
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(by_level.items())) or "no findings"
        rows.append(["Semgrep (SAST)", str(len(results)),
                     "PASS - " + detail if not results else "REVIEW - " + detail])

    bandit = _artifact("bandit.json")
    if isinstance(bandit, dict):
        results = bandit.get("results") or []
        by_sev: dict[str, int] = {}
        for r in results:
            sev = r.get("issue_severity", "UNKNOWN")
            by_sev[sev] = by_sev.get(sev, 0) + 1
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(by_sev.items())) or "no findings"
        rows.append(["Bandit (Python SAST)", str(len(results)),
                     "PASS - " + detail if not results else "REVIEW - " + detail])

    trivy = _artifact("trivy-fs.json")
    if isinstance(trivy, dict):
        by_sev = {}
        for res in trivy.get("Results") or []:
            for v in res.get("Vulnerabilities") or []:
                sev = v.get("Severity", "UNKNOWN")
                by_sev[sev] = by_sev.get(sev, 0) + 1
        total = sum(by_sev.values())
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(by_sev.items())) or "no CVEs"
        crit = by_sev.get("CRITICAL", 0)
        rows.append(["Trivy (SCA/CVE)", str(total),
                     ("PASS - " if crit == 0 else "BLOCKING - ") + detail])

    pa = _artifact("pip-audit.json")
    if isinstance(pa, dict):
        vulns = [v for dep in pa.get("dependencies") or [] for v in dep.get("vulns") or []]
        affected = sum(1 for dep in pa.get("dependencies") or [] if dep.get("vulns"))
        rows.append(["pip-audit (Python deps)", str(len(vulns)),
                     "PASS - no known CVEs" if not vulns
                     else f"REVIEW - {len(vulns)} CVE(s) across {affected} package(s)"])

    npm = _artifact("npm-audit.json")
    if isinstance(npm, dict):
        sev = (npm.get("metadata") or {}).get("vulnerabilities") or {}
        total = sum(v for k, v in sev.items() if k != "total" and isinstance(v, int))
        detail = ", ".join(f"{k}: {v}" for k, v in sev.items()
                           if k != "total" and isinstance(v, int) and v) or "no findings"
        rows.append(["npm audit (frontend deps)", str(total),
                     ("PASS - " if total == 0 else "REVIEW - ") + detail])

    ms = _artifact("modelscan.json")
    if isinstance(ms, dict):
        issues = (ms.get("issues") or [])
        by_sev = {}
        for i in issues:
            s = i.get("severity", "UNKNOWN")
            by_sev[s] = by_sev.get(s, 0) + 1
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(by_sev.items())) or "no unsafe operators"
        rows.append(["modelscan (model artifacts)", str(len(issues)),
                     ("PASS - " if not issues else "REVIEW - ") + detail])

    # hadolint emits one JSON array per Dockerfile (see the CI job) — read both.
    hl_all: list = []
    hl_seen = False
    for fname in ("hadolint-backend.json", "hadolint-frontend.json", "hadolint.json"):
        part = _artifact(fname)
        if isinstance(part, list):
            hl_seen = True
            hl_all.extend(part)
    if hl_seen:
        by_level = {}
        for f in hl_all:
            lvl = f.get("level", "info")
            by_level[lvl] = by_level.get(lvl, 0) + 1
        detail = ", ".join(f"{k}: {v}" for k, v in sorted(by_level.items())) or "no findings"
        rows.append(["Hadolint (Dockerfiles)", str(len(hl_all)),
                     ("PASS - " if not hl_all else "REVIEW - ") + detail])

    # Every SARIF-emitting scanner, read through one generic parser.
    for label, fname in _SARIF_SCANNERS:
        res = _sarif_results(fname)
        if res is None:
            continue
        by_sev = {}
        for r in res:
            s, _ = _sarif_finding(r)
            by_sev[s] = by_sev.get(s, 0) + 1
        rows.append([label, str(len(res)),
                     ("PASS - " if not res else "REVIEW - ") + sev_detail(by_sev, "no findings")])

    # GitLab's native SAST / Secret-Detection templates (their own schema).
    for label, fname in (("GitLab SAST (native)", "gl-sast-report.json"),
                         ("GitLab Secret Detection (native)", "gl-secret-detection-report.json")):
        vulns = _gitlab_vulns(fname)
        if vulns is None:
            continue
        by_sev = {}
        for v in vulns:
            s = str(v.get("severity", "Unknown")).upper()
            by_sev[s] = by_sev.get(s, 0) + 1
        rows.append([label, str(len(vulns)),
                     ("PASS - " if not vulns else "REVIEW - ") + sev_detail(by_sev, "no findings")])

    # TruffleHog emits JSON-LINES. Its distinguishing feature is `Verified`: a
    # credential it actually authenticated against the live provider. An
    # unverified hit is a candidate; a VERIFIED hit is an active leaked secret.
    th = _artifact_jsonl("trufflehog.json")
    if th is not None:
        verified = sum(1 for f in th if f.get("Verified") is True)
        rows.append(["TruffleHog (secrets)", str(len(th)),
                     "PASS - no secrets found" if not th
                     else (f"BLOCKING - {verified} VERIFIED live credential(s)" if verified
                           else f"REVIEW - {len(th)} unverified candidate(s)")])

    ds = _artifact("detect-secrets.json")
    if isinstance(ds, dict):
        found = ds.get("results") or {}
        n = sum(len(v) for v in found.values() if isinstance(v, list))
        rows.append(["detect-secrets", str(n),
                     "PASS - no secrets found" if n == 0
                     else f"REVIEW - {n} candidate(s) across {len(found)} file(s)"])

    rj = _artifact("retire.json")
    if isinstance(rj, list):
        items = [v for entry in rj for comp in (entry.get("results") or [])
                 for v in (comp.get("vulnerabilities") or [])]
        by_sev = {}
        for v in items:
            s = str(v.get("severity", "unknown")).upper()
            by_sev[s] = by_sev.get(s, 0) + 1
        rows.append(["Retire.js (JS libs)", str(len(items)),
                     ("PASS - " if not items else "REVIEW - ") + sev_detail(by_sev, "no findings")])

    hz = _artifact("horusec.json")
    if isinstance(hz, dict):
        vulns = [av.get("vulnerabilities") or {} for av in (hz.get("analysisVulnerabilities") or [])]
        by_sev = {}
        for v in vulns:
            s = str(v.get("severity", "UNKNOWN")).upper()
            by_sev[s] = by_sev.get(s, 0) + 1
        rows.append(["Horusec (multi-language SAST)", str(len(vulns)),
                     ("PASS - " if not vulns else "REVIEW - ") + sev_detail(by_sev, "no findings")])

    nuclei = _artifact_jsonl("nuclei.json")
    if nuclei is not None:
        by_sev = {}
        for f in nuclei:
            s = str((f.get("info") or {}).get("severity", "info")).upper()
            by_sev[s] = by_sev.get(s, 0) + 1
        rows.append(["Nuclei (DAST templates)", str(len(nuclei)),
                     ("PASS - " if not nuclei else "REVIEW - ") + sev_detail(by_sev, "no findings")])

    for label, fname in (("OWASP ZAP (API scan)", "zap-api.json"),
                         ("OWASP ZAP (full scan)", "zap-full.json")):
        zap = _artifact(fname)
        if not isinstance(zap, dict):
            continue
        alerts = [a for site in (zap.get("site") or []) for a in (site.get("alerts") or [])]
        by_sev = {}
        for a in alerts:
            # riskdesc looks like "Medium (High)" — the word before the paren.
            s = str(a.get("riskdesc", "")).split("(")[0].strip().upper() or "UNKNOWN"
            by_sev[s] = by_sev.get(s, 0) + 1
        rows.append([label, str(len(alerts)),
                     ("PASS - " if not alerts else "REVIEW - ") + sev_detail(by_sev, "no alerts")])

    nikto = _artifact("nikto.json")
    if isinstance(nikto, dict):
        vulns = nikto.get("vulnerabilities") or []
        rows.append(["Nikto (web server)", str(len(vulns)),
                     "PASS - no findings" if not vulns else f"REVIEW - {len(vulns)} finding(s)"])

    # Not a scanner, but the same shape of machine-checked evidence: the load
    # test's own SLO verdict (see backend/tests/load/locustfile.py).
    load = _artifact("locust-summary.json")
    if isinstance(load, dict):
        breaches = load.get("breaches") or []
        detail = (f"p95 {load.get('p95Ms', '?')}ms, {load.get('rps', '?')} rps, "
                  f"{load.get('failRatio', 0):.2%} failed")
        rows.append(["Locust (load test)", str(load.get("requests", 0)),
                     ("PASS - " + detail) if not breaches
                     else ("REVIEW - " + detail + " | " + "; ".join(breaches))])

    return rows


def _agentic_gates() -> list[list]:
    """Rows for the scored AGENT-level gates, as opposed to the model-level ones.

    Same rule the scanner table follows: an artifact that is ABSENT is reported
    as "not run", never as a pass. A gate nobody ran and a gate that passed are
    different facts, and only one of them is evidence.
    """
    rows: list[list] = []

    score = _artifact("guardrail-score.json")
    if isinstance(score, dict):
        counts = score.get("counts") or {}
        detail = (f"recall {score.get('recall', 0):.0%} / "
                  f"false-pos {score.get('falsePositiveRate', 0):.0%} / "
                  f"injection bypass {score.get('bypassRate', 0):.0%} "
                  f"over {counts.get('cases', 0)} labelled cases")
        verdict = str(score.get("verdict", "?"))
        if score.get("breaches"):
            verdict += " - " + "; ".join(str(b) for b in score["breaches"])
        rows.append(["Guardrail effectiveness (E9)", detail, verdict])
    else:
        rows.append(["Guardrail effectiveness (E9)", "artifact absent", "not run"])

    creds = _artifact("no-live-credentials.json")
    if isinstance(creds, dict):
        live = creds.get("live") or []
        if not creds.get("applicable"):
            detail, verdict = f"not applicable in {creds.get('environment', '?')}", "not run"
        elif live:
            detail = "live credentials present: " + ", ".join(str(x) for x in live)
            verdict = "FAIL"
        else:
            detail, verdict = f"{len(creds.get('checked') or [])} agent-reachable credentials checked", "PASS"
        rows.append(["No live credentials below production", detail, verdict])
    else:
        rows.append(["No live credentials below production", "artifact absent", "not run"])

    return rows


def _notable_findings(limit_per_scanner: int | None = None) -> list[list]:
    """Enumerate EVERY finding (not just counts) from each scanner's artifact, as
    rows of [scanner, severity, finding]. Findings are DE-DUPLICATED (scanners
    often report the same advisory across many files) and, within a scanner,
    sorted most-severe first. Pass `limit_per_scanner` to cap the list; the
    default (None) lists them all. Empty when nothing was found."""
    out: list[list] = []

    def add(scanner, items):
        seen: set = set()
        rows: list[list] = []
        for sev, text in items:
            sev_u, txt = str(sev).upper(), str(text)[:130]
            key = (sev_u, txt)
            if key in seen:
                continue
            seen.add(key)
            rows.append([scanner, sev_u, txt])
        rows.sort(key=lambda r: _SEV_ORDER.get(r[1], 5))
        shown = rows if limit_per_scanner is None else rows[:limit_per_scanner]
        out.extend(shown)
        if limit_per_scanner is not None and len(rows) > limit_per_scanner:
            out.append([scanner, "", f"... and {len(rows) - limit_per_scanner} more"])

    leaks = _artifact("gitleaks.json")
    if isinstance(leaks, list) and leaks:
        add("Gitleaks", [(x.get("RuleID", "secret"),
                          f"{x.get('Description', 'secret')} in {x.get('File', '?')}") for x in leaks])

    sarif = _artifact("semgrep.sarif")
    if isinstance(sarif, dict):
        res = (sarif.get("runs") or [{}])[0].get("results") or []
        add("Semgrep", [(r.get("level", "warning"),
                         f"{r.get('ruleId', 'rule')}: {((r.get('message') or {}).get('text') or '')}") for r in res])

    bandit = _artifact("bandit.json")
    if isinstance(bandit, dict):
        add("Bandit", [(r.get("issue_severity", "?"),
                        f"{r.get('test_id', '')} {r.get('issue_text', '')} ({os.path.basename(r.get('filename', ''))}:{r.get('line_number', '')})")
                       for r in bandit.get("results") or []])

    trivy = _artifact("trivy-fs.json")
    if isinstance(trivy, dict):
        vs = [(v.get("Severity", "?"), f"{v.get('VulnerabilityID', '')} in {v.get('PkgName', '')} {v.get('InstalledVersion', '')}")
              for res in trivy.get("Results") or [] for v in res.get("Vulnerabilities") or []]
        # Show CRITICAL/HIGH first.
        vs.sort(key=lambda x: {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}.get(x[0], 3))
        add("Trivy", vs)

    pa = _artifact("pip-audit.json")
    if isinstance(pa, dict):
        add("pip-audit", [((v.get("fix_versions") and "fixable") or "open",
                           f"{dep.get('name', '')} {dep.get('version', '')}: {v.get('id', '')}")
                          for dep in pa.get("dependencies") or [] for v in dep.get("vulns") or []])

    npm = _artifact("npm-audit.json")
    if isinstance(npm, dict):
        items = []
        for pkg, info in (npm.get("vulnerabilities") or {}).items():
            for via in info.get("via") or []:
                if isinstance(via, dict):
                    items.append((via.get("severity", info.get("severity", "?")),
                                  f"{pkg}: {via.get('title', '')}"))
        add("npm audit", items)

    ms = _artifact("modelscan.json")
    if isinstance(ms, dict):
        add("modelscan", [(i.get("severity", "?"), i.get("description", i.get("operator", "unsafe operator")))
                          for i in ms.get("issues") or []])

    hl_all = []
    for fname in ("hadolint-backend.json", "hadolint-frontend.json", "hadolint.json"):
        part = _artifact(fname)
        if isinstance(part, list):
            hl_all.extend(part)
    if hl_all:
        add("Hadolint", [(f.get("level", "info"), f"{f.get('code', '')} {f.get('message', '')}") for f in hl_all])

    for label, fname in _SARIF_SCANNERS:
        res = _sarif_results(fname)
        if res:
            add(label.split(" (")[0], [_sarif_finding(r) for r in res])

    for label, fname in (("GitLab SAST", "gl-sast-report.json"),
                         ("GitLab Secret Detection", "gl-secret-detection-report.json")):
        vulns = _gitlab_vulns(fname)
        if vulns:
            add(label, [(v.get("severity", "?"),
                         f"{v.get('name', '')}: {(v.get('location') or {}).get('file', '')}")
                        for v in vulns])

    th = _artifact_jsonl("trufflehog.json")
    if th:
        add("TruffleHog", [("CRITICAL" if f.get("Verified") else "MEDIUM",
                            f"{'VERIFIED ' if f.get('Verified') else ''}"
                            f"{f.get('DetectorName', 'secret')} in "
                            f"{((f.get('SourceMetadata') or {}).get('Data') or {})}"[:130])
                           for f in th])

    ds = _artifact("detect-secrets.json")
    if isinstance(ds, dict) and (ds.get("results") or {}):
        add("detect-secrets", [(s.get("type", "secret"), f"{s.get('type', '')} in {path}:{s.get('line_number', '')}")
                               for path, hits in (ds.get("results") or {}).items()
                               for s in hits if isinstance(s, dict)])

    rj = _artifact("retire.json")
    if isinstance(rj, list) and rj:
        add("Retire.js", [(v.get("severity", "?"),
                           (f"{comp.get('component', '')} {comp.get('version', '')}: "
                            f"{', '.join((v.get('identifiers') or {}).get('CVE') or []) or (v.get('identifiers') or {}).get('summary', '')}"))
                          for entry in rj for comp in (entry.get("results") or [])
                          for v in (comp.get("vulnerabilities") or [])])

    hz = _artifact("horusec.json")
    if isinstance(hz, dict) and (hz.get("analysisVulnerabilities") or []):
        add("Horusec", [((av.get("vulnerabilities") or {}).get("severity", "?"),
                         (f"{(av.get('vulnerabilities') or {}).get('details', '')} "
                          f"({os.path.basename((av.get('vulnerabilities') or {}).get('file', ''))})"))
                        for av in hz.get("analysisVulnerabilities") or []])

    nuclei = _artifact_jsonl("nuclei.json")
    if nuclei:
        add("Nuclei", [((f.get("info") or {}).get("severity", "info"),
                        f"{(f.get('info') or {}).get('name', '')} at {f.get('matched-at', '')}")
                       for f in nuclei])

    for label, fname in (("ZAP (API)", "zap-api.json"), ("ZAP (full)", "zap-full.json")):
        zap = _artifact(fname)
        if not isinstance(zap, dict):
            continue
        alerts = [a for site in (zap.get("site") or []) for a in (site.get("alerts") or [])]
        if alerts:
            add(label, [(str(a.get("riskdesc", "")).split("(")[0].strip() or "?",
                         f"{a.get('alert', '')} ({len(a.get('instances') or [])} instance(s))")
                        for a in alerts])

    nikto = _artifact("nikto.json")
    if isinstance(nikto, dict) and (nikto.get("vulnerabilities") or []):
        add("Nikto", [("MEDIUM", f"{v.get('id', '')} {v.get('msg', '')}")
                      for v in nikto.get("vulnerabilities") or []])

    return out


def _find_ci_yaml() -> str | None:
    for c in (".gitlab-ci.yml", "../.gitlab-ci.yml",
              os.path.join(os.environ.get("CI_PROJECT_DIR", ""), ".gitlab-ci.yml")):
        if c and os.path.exists(c):
            return c
    return None


def _pipeline_model() -> tuple[list, dict]:
    """Return (stages, {stage: [job, ...]}) parsed from .gitlab-ci.yml, or ([],{})."""
    path = _find_ci_yaml()
    if not path:
        return [], {}
    try:
        import yaml
        with open(path, encoding="utf-8") as fh:
            d = yaml.safe_load(fh)
    except Exception:  # noqa: BLE001 - report generation is best-effort and must not fail the pipeline
        return [], {}
    stages = d.get("stages", [])
    by_stage: dict = {s: [] for s in stages}
    for name, cfg in d.items():
        if isinstance(cfg, dict) and "stage" in cfg:
            by_stage.setdefault(cfg["stage"], []).append(name)
    return stages, by_stage


def _pipeline_status() -> dict:
    """Optional {jobName: status} to annotate the pipeline table (from CI/API)."""
    return _load_json(os.environ.get("CAREROUTE_PIPELINE_STATUS", "")) or {}


def build(out_path: str) -> str:
    audit = _audit()
    drift = _drift()
    ss = _styles()
    story: list = []

    # ---- Title ----
    story += [
        Spacer(1, 1.5 * cm),
        Paragraph("CareRoute AI", ss["H1c"]),
        Paragraph("MLOps &amp; Responsible-AI Pipeline Report", ss["Sub"]),
        Spacer(1, 0.3 * cm),
        Paragraph(
            f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} &nbsp;·&nbsp; "
            f"commit {_git_sha()} &nbsp;·&nbsp; model {audit.get('modelVersion', 'n/a')}",
            ss["Sub"],
        ),
        Spacer(1, 0.4 * cm),
        HRFlowable(width="100%", color=_BRAND, thickness=2),
        Spacer(1, 0.4 * cm),
    ]

    # ---- Executive summary ----
    acc = audit.get("overallAccuracy", 0.0)
    rec = audit.get("redFlagRecall", 0.0)
    gate_ok = acc >= 0.75 and rec >= 0.95
    story += [
        Paragraph("Executive summary", ss["H2"]),
        Paragraph(
            "CareRoute AI is a multi-agent clinical-triage assistant with a deterministic safety layer "
            "and human-in-the-loop review. This report summarises the automated quality, fairness, "
            "data-lineage, drift and supply-chain checks the CI/CD pipeline enforces on every release. "
            f"The current model <b>{'clears' if gate_ok else 'requires review against'}</b> the release "
            f"gate (accuracy >= 0.75 and safety-critical red-flag recall >= 0.95).",
            ss["Body"],
        ),
        Spacer(1, 0.2 * cm),
    ]

    # ---- Model quality & release gates ----
    story += [Paragraph("Model quality &amp; release gates", ss["H2"])]
    story += [_kv_table([
        ["Metric", "Value", "Gate / status"],
        ["Overall accuracy", f"{acc:.3f}", f">= 0.75 - {_status(acc >= 0.75)}"],
        ["Red-flag recall (safety-critical)", f"{rec:.3f}", f">= 0.95 - {_status(rec >= 0.95)}"],
        ["Worst-subgroup red-flag recall (raw / served)", _worst_subgroup_recall(audit)[0],
         f">= 0.95 per subgroup - {_status(_worst_subgroup_recall(audit)[1])}"],
        ["Model version", audit.get("modelVersion", "n/a").split(" (")[0], "content-addressed"],
    ])]

    # ---- Responsible-AI / fairness ----
    cf = audit.get("counterfactual", {}) or {}
    story += [Spacer(1, 0.3 * cm), Paragraph("Responsible-AI &amp; fairness", ss["H2"])]
    story += [_kv_table([
        ["Metric", "Value", "Interpretation"],
        ["Fairness gap (before -> after)",
         f"{audit.get('fairnessGapBefore', 0):.3f} -> {audit.get('fairnessGapAfter', 0):.3f}",
         _status(audit.get("fairnessGapAfter", 1) <= audit.get("fairnessGapBefore", 0))],
        ["Demographic parity spread (max-min)", f"{_spread(audit.get('demographicParity')):.3f}", "lower is fairer"],
        ["Equal-opportunity spread (TPR max-min)", f"{_spread(audit.get('equalOpportunity')):.3f}", "lower is fairer"],
        ["Counterfactual sex-flip rate", f"{cf.get('sexFlipRate', 0):.3f}", "should be ~0"],
        ["Counterfactual sex-flip rate (served pipeline, all bands)",
         f"{(audit.get('counterfactualServed') or {}).get('sexFlipRate', 0):.3f}", "should be ~0"],
    ])]

    # ---- Calibration + integrity ----
    cal = audit.get("calibration", {}) or {}
    integ = audit.get("integrity", {}) or {}
    story += [Spacer(1, 0.3 * cm), Paragraph("Calibration &amp; data lineage / integrity", ss["H2"])]
    story += [_kv_table([
        ["Item", "Value", "Note"],
        ["Calibration method", str(cal.get("method", "n/a")), f"ECE {cal.get('ece', 0):.3f} / Brier {cal.get('brier', 0):.3f}"],
        ["Training-data SHA-256", (integ.get("dataSha256", "n/a") or "n/a")[:24] + "...", "data lineage anchor"],
        ["Serialized-model SHA-256", (integ.get("modelSha256", "n/a") or "n/a")[:24] + "...", "tamper-evidence"],
    ])]

    # ---- Drift monitoring ----
    story += [Spacer(1, 0.3 * cm), Paragraph("Drift monitoring", ss["H2"])]
    if drift:
        perf = drift.get("performance", {}) or {}
        story += [_kv_table([
            ["Metric", "Value", "Note"],
            ["Monitoring backend", str(drift.get("backend", "n/a")), "Evidently / PSI"],
            ["Data drift (PSI)", str(drift.get("dataDriftPSI", "n/a")), "vs shifted sample"],
            ["Target drift (PSI)", str(drift.get("targetDriftPSI", "n/a")), "label shift"],
            ["Accuracy (ref -> current)",
             f"{perf.get('referenceAccuracy', 'n/a')} -> {perf.get('currentAccuracy', 'n/a')}", "on shifted sample"],
        ])]
    else:
        d = audit.get("drift", {}) or {}
        story += [Paragraph(
            f"Offline drift audit (from model): data={d.get('data', 'n/a')}, "
            f"target={d.get('target', 'n/a')}, concept={d.get('concept', 'n/a')}. "
            "A shifted production sample is expected to drift; in production this signal triggers "
            "evaluation/retraining (the monitor -> continuous-training loop).", ss["Body"])]

    # ---- Pipeline + security (narrative) ----
    story += [Spacer(1, 0.3 * cm), Paragraph("CI/CD pipeline &amp; supply chain", ss["H2"])]
    story += [Paragraph(
        "The GitLab pipeline runs, on every push and on a schedule: <b>train</b> (build + release-gate + "
        "MLflow-register the model, version the dataset), <b>test</b> (unit/e2e + hard model-quality gate + "
        "data-lineage gate), <b>monitor</b> (Evidently drift), <b>ai-security</b> (Fairlearn parity + LLM "
        "red-teaming), <b>security-scan</b> (SAST/SCA/secret/container + SBOM &amp; AI-BOM), <b>build</b>, "
        "and manual-gated <b>deploy</b> (shadow -> blue-green/canary with rollback). All gates are "
        "deterministic pass/fail checks.", ss["Body"])]

    # ---- Security sections: ai-security + security-scan (from .gitlab-ci.yml) ----
    stages, by_stage = _pipeline_model()
    if stages:
        status = _pipeline_status()
        show_status = bool(status)
        for title, blurb, groups in _SECURITY_SECTIONS:
            # Collect this section's jobs: whole stages, or a named subset of one.
            jobs: list = []
            for stage_name, only in groups:
                for name in by_stage.get(stage_name, []):
                    if only is None or name in only:
                        jobs.append(name)
            if not jobs:
                continue
            story += [Spacer(1, 0.3 * cm), Paragraph(title, ss["H2"]),
                      Paragraph(blurb, ss["Body"]), Spacer(1, 0.15 * cm)]
            header = ["Job", "Purpose"] + (["Status"] if show_status else [])
            rows = [header]
            for name in jobs:
                row = [name, _JOB_PURPOSE.get(name, "")]
                if show_status:
                    row.append(_STATUS_LABEL.get(status.get(name, ""), status.get(name) or "-"))
                rows.append(row)
            widths = ([5.5 * cm, 9 * cm, 2 * cm] if show_status else [6 * cm, 10.5 * cm])
            t = Table(rows, colWidths=widths, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), _DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cdd7e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story += [t]

        # Actual scan RESULTS from the scanners' machine-readable artifacts
        # (gitleaks.json / semgrep.sarif / bandit.json / trivy-fs.json). In CI
        # the report job runs after security-scan and `needs:` these artifacts.
        agentic_rows = _agentic_gates()
        story += [Spacer(1, 0.3 * cm), Paragraph("Agentic gate results", ss["H2"])]
        story += [_kv_table([["Gate", "Measured", "Verdict"], *agentic_rows],
                            col_widths=[5.2 * cm, 8.4 * cm, 3.4 * cm])]

        scan_rows = _scan_results()
        story += [Spacer(1, 0.3 * cm), Paragraph("Security scan results", ss["H2"])]
        if scan_rows:
            t = Table([["Scanner", "Findings", "Severity breakdown / verdict"], *scan_rows],
                      colWidths=[4.5 * cm, 2 * cm, 10 * cm], repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), _DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cdd7e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story += [t]
        else:
            story += [Paragraph(
                "No scan artifacts found in this run - the scanners publish their results "
                "(gitleaks.json, semgrep.sarif, bandit.json, trivy-fs.json, pip-audit.json, "
                "npm-audit.json, modelscan.json, hadolint.json) when the report is generated "
                "in the CI pipeline after the security-scan stage.", ss["Body"])]

        # ---- Findings summary: the ACTUAL findings, listed (not just counts) ----
        findings = _notable_findings()
        story += [Spacer(1, 0.3 * cm), Paragraph("Findings summary", ss["H2"])]
        if findings:
            blocking = sum(1 for f in findings if f[1] == "CRITICAL")
            story += [Paragraph(
                f"{len([f for f in findings if not str(f[2]).startswith('...')])} notable finding(s) "
                f"listed below across {len({f[0] for f in findings})} scanner(s)"
                + (f"; <b>{blocking} CRITICAL (release-blocking)</b>." if blocking else
                   "; none CRITICAL — all advisory (surfaced for review, non-blocking)."),
                ss["Body"]), Spacer(1, 0.15 * cm)]
            cell = ParagraphStyle("FCell", parent=ss["Body"], fontSize=8, leading=10)
            frows = [["Scanner", "Severity", "Finding"]]
            frows += [[Paragraph(c, cell) for c in r] for r in findings]
            ft = Table(frows, colWidths=[2.6 * cm, 2.2 * cm, 11.7 * cm], repeatRows=1)
            ft.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), _DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cdd7e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fb")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story += [ft]
        else:
            story += [Paragraph(
                "No findings to list — every scanner that ran reported a clean result "
                "(no secrets, no CRITICAL CVEs, no unsafe model operators).", ss["Body"])]

    # ---- Footer / disclaimer ----
    story += [
        Spacer(1, 0.5 * cm),
        HRFlowable(width="100%", color=colors.HexColor("#cdd7e1"), thickness=0.5),
        Spacer(1, 0.1 * cm),
        Paragraph(
            "Prototype for the NUS-ISS “Architecting AI Systems” Practice Module (Team 3). "
            "AI assists; the clinician decides. Not a certified medical device.", ss["Small"]),
    ]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm,
        title="CareRoute AI - MLOps & Responsible-AI Pipeline Report",
    ).build(story)
    return out_path


def _md_table(header: list, rows: list) -> str:
    """Render a GitHub-flavoured markdown table."""
    def esc(x):
        return str(x).replace("|", "\\|")
    out = ["| " + " | ".join(esc(h) for h in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(out)


def build_markdown(out_path: str) -> str:
    """[MLOps] Plain-text/markdown TWIN of the PDF — readable in any editor or on
    GitLab when downloaded as an artifact (no PDF viewer needed). Same data, same
    sections. This is the 'readable when downloaded' companion to the PDF."""
    audit = _audit()
    drift = _drift()
    acc = audit.get("overallAccuracy", 0.0)
    rec = audit.get("redFlagRecall", 0.0)
    gate_ok = acc >= 0.75 and rec >= 0.95
    cf = audit.get("counterfactual", {}) or {}
    cal = audit.get("calibration", {}) or {}
    integ = audit.get("integrity", {}) or {}
    L = []
    L.append("# CareRoute AI — MLOps & Responsible-AI Pipeline Report\n")
    L.append(f"_Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} · "
             f"commit {_git_sha()} · model {audit.get('modelVersion', 'n/a')}_\n")

    L.append("## Executive summary\n")
    L.append(f"CareRoute AI is a multi-agent clinical-triage assistant with a deterministic safety "
             f"layer and human-in-the-loop review. The current model "
             f"**{'clears' if gate_ok else 'requires review against'}** the release gate "
             f"(accuracy ≥ 0.75 and safety-critical red-flag recall ≥ 0.95).\n")

    L.append("## Model quality & release gates\n")
    L.append(_md_table(["Metric", "Value", "Gate / status"], [
        ["Overall accuracy", f"{acc:.3f}", f"≥ 0.75 — {_status(acc >= 0.75)}"],
        ["Red-flag recall (safety-critical)", f"{rec:.3f}", f"≥ 0.95 — {_status(rec >= 0.95)}"],
        ["Worst-subgroup red-flag recall (raw / served)", _worst_subgroup_recall(audit)[0],
         f"≥ 0.95 per subgroup — {_status(_worst_subgroup_recall(audit)[1])}"],
        ["Model version", audit.get("modelVersion", "n/a").split(" (")[0], "content-addressed"],
    ]) + "\n")

    L.append("## Responsible-AI & fairness\n")
    L.append(_md_table(["Metric", "Value", "Interpretation"], [
        ["Fairness gap (before → after)",
         f"{audit.get('fairnessGapBefore', 0):.3f} → {audit.get('fairnessGapAfter', 0):.3f}",
         _status(audit.get("fairnessGapAfter", 1) <= audit.get("fairnessGapBefore", 0))],
        ["Demographic parity spread", f"{_spread(audit.get('demographicParity')):.3f}", "lower is fairer"],
        ["Equal-opportunity spread", f"{_spread(audit.get('equalOpportunity')):.3f}", "lower is fairer"],
        ["Counterfactual sex-flip rate", f"{cf.get('sexFlipRate', 0):.3f}", "should be ~0"],
        ["Counterfactual sex-flip rate (served pipeline, all bands)",
         f"{(audit.get('counterfactualServed') or {}).get('sexFlipRate', 0):.3f}", "should be ~0"],
    ]) + "\n")

    L.append("## Calibration & data lineage / integrity\n")
    L.append(_md_table(["Item", "Value", "Note"], [
        ["Calibration method", str(cal.get("method", "n/a")),
         f"ECE {cal.get('ece', 0):.3f} / Brier {cal.get('brier', 0):.3f}"],
        ["Training-data SHA-256", (integ.get("dataSha256", "n/a") or "n/a")[:24] + "…", "data lineage anchor"],
        ["Serialized-model SHA-256", (integ.get("modelSha256", "n/a") or "n/a")[:24] + "…", "tamper-evidence"],
    ]) + "\n")

    L.append("## Drift monitoring\n")
    if drift:
        perf = drift.get("performance", {}) or {}
        L.append(_md_table(["Metric", "Value", "Note"], [
            ["Monitoring backend", str(drift.get("backend", "n/a")), "Evidently / PSI"],
            ["Data drift (PSI)", str(drift.get("dataDriftPSI", "n/a")), "vs shifted sample"],
            ["Target drift (PSI)", str(drift.get("targetDriftPSI", "n/a")), "label shift"],
            ["Accuracy (ref → current)",
             f"{perf.get('referenceAccuracy', 'n/a')} → {perf.get('currentAccuracy', 'n/a')}", "on shifted sample"],
        ]) + "\n")
    else:
        d = audit.get("drift", {}) or {}
        L.append(f"Offline drift audit (from model): data={d.get('data', 'n/a')}, "
                 f"target={d.get('target', 'n/a')}, concept={d.get('concept', 'n/a')}.\n")

    # Security job tables + actual scan results
    stages, by_stage = _pipeline_model()
    if stages:
        for title, blurb, groups in _SECURITY_SECTIONS:
            jobs = [name for stage_name, only in groups
                    for name in by_stage.get(stage_name, []) if only is None or name in only]
            if not jobs:
                continue
            L.append(f"## {title.replace('&amp;', '&')}\n")
            L.append(blurb.replace("&amp;", "&") + "\n")
            L.append(_md_table(["Job", "Purpose"],
                               [[n, _JOB_PURPOSE.get(n, "")] for n in jobs]) + "\n")

    L.append("## Agentic gate results\n")
    L.append(_md_table(["Gate", "Measured", "Verdict"], _agentic_gates()))
    L.append("")
    L.append("## Security scan results\n")
    scan_rows = _scan_results()
    if scan_rows:
        L.append(_md_table(["Scanner", "Findings", "Severity breakdown / verdict"], scan_rows) + "\n")
    else:
        L.append("_No scan artifacts found in this run — the scanners publish their results "
                 "when the report is generated in CI after the security-scan stage._\n")

    L.append("## Findings summary\n")
    findings = _notable_findings()
    if findings:
        blocking = sum(1 for f in findings if f[1] == "CRITICAL")
        listed = len([f for f in findings if not str(f[2]).startswith("...")])
        L.append(f"{listed} notable finding(s) across {len({f[0] for f in findings})} scanner(s)"
                 + (f"; **{blocking} CRITICAL (release-blocking)**.\n" if blocking
                    else "; none CRITICAL — all advisory (non-blocking).\n"))
        L.append(_md_table(["Scanner", "Severity", "Finding"], findings) + "\n")
    else:
        L.append("_No findings — every scanner that ran reported clean "
                 "(no secrets, no CRITICAL CVEs, no unsafe model operators)._\n")

    L.append("---\n")
    L.append("_Prototype for the NUS-ISS “Architecting AI Systems” Practice Module (Team 3). "
             "AI assists; the clinician decides. Not a certified medical device._\n")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return out_path


def main() -> int:
    out_dir = os.environ.get("CAREROUTE_REPORT_DIR", "reports")
    pdf = build(os.path.join(out_dir, "careroute_mlops_report.pdf"))
    print(f"PDF report written:      {pdf}")
    md = build_markdown(os.path.join(out_dir, "careroute_mlops_report.md"))
    print(f"Markdown report written: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
