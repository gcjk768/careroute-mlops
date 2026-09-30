"""[MLOps] Pipeline-outcome notification — closing the feedback half of CI/CD.

The CI/CD deck gives two slides to this ("Notification of Build Outcome",
"Notification of Test outcome ... via email, chat, or dashboards") for a reason:
feedback is what makes a pipeline a loop rather than a log. CareRoute had none.
That matters most for the job nobody is watching — `train:model` runs on a
SCHEDULED pipeline, so a retrain that fails its release gate at 02:00 stays
failed and silent until somebody happens to open the pipelines page.

TWO RULES SHAPE THE DIGEST.

* **Failures lead.** A message that opens with 66 green jobs and buries the one
  red one has optimised for looking good instead of being read.
* **An advisory failure is not a broken pipeline.** `test:e2e` and the red-team
  scanners carry `allow_failure: true` on purpose; folding them into the headline
  would cry wolf until the headline stopped being read at all. They are reported,
  separately, under the verdict rather than in it.

An empty job list is UNKNOWN, never PASSED: no jobs means the API call failed,
and "I could not see the pipeline" is not "the pipeline was green".

Run in CI as the last job, `when: always`:
    python -m app.ml.notify
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

_TERMINAL_STATUSES = ("success", "failed", "canceled", "skipped", "manual", "running", "created")

# Both URLs below arrive from the environment. `urlopen` honours `file:` and
# other schemes, so an operator typo — or a poisoned CI variable — could turn a
# notification into a local file read. Only http(s) is ever a notification.
_ALLOWED_SCHEMES = ("http://", "https://")  # NOSONAR: an allowlist that exists to reject file:// etc.; internal webhooks may be http


def _is_http_url(url: str) -> bool:
    return isinstance(url, str) and url.startswith(_ALLOWED_SCHEMES)


def build_digest(jobs: list[dict], pipeline_id: str, ref: str) -> dict:
    """Summarise a pipeline's jobs into a report-ready digest."""
    counts: dict[str, int] = {}
    blocking: list[str] = []
    allowed: list[str] = []

    for job in jobs:
        status = str(job.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        if status == "failed":
            (allowed if job.get("allow_failure") else blocking).append(str(job.get("name")))

    if not jobs:
        verdict = "UNKNOWN"
    elif blocking:
        verdict = "FAILED"
    else:
        verdict = "PASSED"

    return {
        "pipelineId": str(pipeline_id),
        "ref": str(ref),
        "verdict": verdict,
        "counts": counts,
        "blockingFailures": blocking,
        "allowedFailures": allowed,
        "total": len(jobs),
    }


def format_message(digest: dict) -> str:
    """A plain-ASCII message. Webhook targets and CI consoles mangle non-ASCII
    differently — the locustfile already learned that on a cp1252 console."""
    lines = [f"CareRoute pipeline #{digest['pipelineId']} ({digest['ref']}): {digest['verdict']}"]

    if digest["blockingFailures"]:
        lines.append(f"  BROKEN: {', '.join(digest['blockingFailures'])}")
    if digest["verdict"] == "UNKNOWN":
        lines.append("  could not read the pipeline's jobs - treat as unverified, not green")

    counts = digest["counts"]
    summary = ", ".join(f"{k}: {v}" for k, v in sorted(counts.items()) if k in _TERMINAL_STATUSES)
    lines.append(f"  {digest['total']} jobs - {summary}")

    if digest["allowedFailures"]:
        lines.append(f"  advisory (non-blocking) failures: {', '.join(digest['allowedFailures'])}")
    return "\n".join(lines)


def fetch_jobs(api_url: str, project_id: str, pipeline_id: str, token: str | None) -> list[dict]:
    """Read a pipeline's jobs from the GitLab API. Returns [] on any failure —
    the caller reports that as UNKNOWN rather than as a green pipeline."""
    url = f"{api_url}/projects/{project_id}/pipelines/{pipeline_id}/jobs?per_page=100"
    if not _is_http_url(url):
        print("WARNING: refusing non-http API URL scheme; reporting UNKNOWN")
        return []
    req = urllib.request.Request(url)  # noqa: S310 - scheme checked above
    if token:
        req.add_header("JOB-TOKEN", token)
    try:
        # Scheme checked above (http/https only), so not an SSRF / file:// sink.
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected, bandit.B310-1
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310  # nosec B310
            return json.load(resp)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print(f"WARNING: could not read pipeline jobs ({type(exc).__name__}); reporting UNKNOWN")
        return []


def send(digest: dict, webhook: str | None) -> bool:
    """Post the digest to a Slack-compatible webhook. Always prints it too, so
    the outcome is in the job log whether or not a webhook exists."""
    message = format_message(digest)
    print(message)

    if not webhook:
        print("(no CAREROUTE_NOTIFY_WEBHOOK configured - printed above only)")
        return False
    if not _is_http_url(webhook):
        print("WARNING: CAREROUTE_NOTIFY_WEBHOOK is not an http(s) URL - not sending")
        return False

    payload = json.dumps({"text": message}).encode("utf-8")
    req = urllib.request.Request(  # noqa: S310 - scheme checked above
        webhook, data=payload, headers={"Content-Type": "application/json"},
    )
    try:
        # Scheme checked above (http/https only), so not an SSRF / file:// sink.
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected, bandit.B310-1
        with urllib.request.urlopen(req, timeout=20):  # noqa: S310  # nosec B310
            return True
    except (urllib.error.URLError, OSError) as exc:
        # Never fail the pipeline because the chat server was down.
        print(f"WARNING: webhook post failed ({type(exc).__name__})")
        return False


def main(argv: list[str] | None = None) -> int:
    _ = argv
    api = os.environ.get("CI_API_V4_URL", "")
    project = os.environ.get("CI_PROJECT_ID", "")
    pipeline = os.environ.get("CI_PIPELINE_ID", "")
    ref = os.environ.get("CI_COMMIT_REF_NAME", "unknown")

    jobs = fetch_jobs(api, project, pipeline, os.environ.get("CI_JOB_TOKEN")) if api and project else []
    digest = build_digest(jobs, pipeline_id=pipeline or "local", ref=ref)
    send(digest, os.environ.get("CAREROUTE_NOTIFY_WEBHOOK"))

    # Reporting the outcome must not itself change the outcome.
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
