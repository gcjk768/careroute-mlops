"""[AI-Security] OWASP LLM02 — Sensitive Information Disclosure.

Masks personally identifiable / protected health information (PII/PHI) out of
free-text patient input BEFORE it is (a) sent to a cloud LLM provider and (b)
persisted in the case store. Clinical symptom words are never touched — only
direct identifiers (NRIC/FIN, phone, email, long digit runs like MRNs) are
masked — so triage classification is unaffected while the identity leak is
closed at the trust boundary.

Microsoft Presidio is the production-grade drop-in (ML-based NER for names/
addresses); these deterministic regexes cover the high-risk structured
identifiers with zero dependencies and are fully unit-testable.
"""
from __future__ import annotations

import re

# Singapore NRIC/FIN: S/T/F/G/M + 7 digits + checksum letter.
_NRIC = re.compile(r"\b[STFGM]\d{7}[A-Z]\b", re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Phone. The previous pattern (`\+?\d[\d\s-]{7,}\d`) was really "any long
# numeric sequence": the inner character class allowed an unlimited run of
# spaces and dashes, so it swallowed clinical numerics that a triage decision
# depends on — "blood pressure 140 90 110 70" became one [REDACTED_PHONE], and
# so did the dashed onset date in "chest pain since 12-05-2026". It also needed
# 9+ characters, so a bare 8-digit Singapore mobile (91234567) fell through to
# the MRN rule and was mislabelled in the audit trail.
#
# So match the shapes a phone number actually has, not "lots of digits":
#   * the Singapore numbering plan — mobile/landline prefixes 3/6/8/9 followed
#     by 8 digits in total, with an optional +65 country code and AT MOST ONE
#     separator between the two 4-digit halves; and
#   * an international number, which must carry an explicit `+` country prefix
#     and again allows at most one separator between digit groups.
# A vitals run has no 4-digit group starting 3/6/8/9 and no `+`; a dd-mm-yyyy
# date's year does not start with 3/6/8/9 either. Both are therefore untouched.
_SG_NUMBER = r"(?:\+?65[\s-]?)?[3689]\d{3}[\s-]?\d{4}"
_INTL_NUMBER = r"\+\d{1,3}[\s-]?\d{2,4}(?:[\s-]?\d{2,4}){1,3}"
# The lookarounds stop a match being carved out of the middle of a longer digit
# run (e.g. a 12-digit MRN), which must stay an MRN.
_PHONE = re.compile(rf"(?<![\d+-])(?:{_INTL_NUMBER}|{_SG_NUMBER})(?![\d-])")
# Long bare digit runs (MRN / account numbers), 7+ digits.
_LONGNUM = re.compile(r"\b\d{7,}\b")

_PATTERNS = [("NRIC", _NRIC), ("EMAIL", _EMAIL), ("PHONE", _PHONE), ("MRN", _LONGNUM)]


def redact(text: str) -> tuple[str, list[str]]:
    """Return (masked_text, kinds_found). `kinds_found` lists the identifier
    types that were masked (e.g. ["NRIC", "EMAIL"]) for the audit trail."""
    if not text:
        return text, []
    found: list[str] = []
    masked = text
    for kind, pattern in _PATTERNS:
        if pattern.search(masked):
            found.append(kind)
            masked = pattern.sub(f"[REDACTED_{kind}]", masked)
    return masked, found
