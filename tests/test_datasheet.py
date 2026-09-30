"""[Responsible-AI] The datasheet must agree with the dataset it describes.

*Datasheets for Datasets* (Gebru et al.; XRAI Day 1-2) is a governance artefact, and
a governance artefact that silently goes stale is worse than none — it is a
confident, wrong answer to "what was this model trained on?".

`DATASHEET.md` is prose, so it cannot be generated without losing the judgement
that makes it worth reading. Instead every FALSIFIABLE number in it is checked
against `data/triage_dataset.meta.json` here, so changing the dataset without
updating the datasheet fails the build.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
DATASHEET = _BACKEND / "DATASHEET.md"
META = _BACKEND / "data" / "triage_dataset.meta.json"

ACUITY_ROW_ORDER = ["P1", "P2", "P3", "P4", "P5"]


@pytest.fixture(scope="module")
def sheet() -> str:
    return DATASHEET.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def meta() -> dict:
    return json.loads(META.read_text(encoding="utf-8"))


def test_headline_facts_match(sheet, meta):
    assert f"| **Rows** | {meta['rows']} |" in sheet
    assert f"| **Features** | {meta['featureCount']} |" in sheet
    assert meta["dataSha256"] in sheet
    assert f"seed {meta['provenance']['seed']}" in sheet


def test_label_distribution_matches(sheet, meta):
    balance = meta["statistics"]["labelBalance"]
    total = meta["rows"]
    for index, code in enumerate(ACUITY_ROW_ORDER):
        count = balance[str(index)] if str(index) in balance else balance[index]
        share = f"{count / total * 100:.1f}%"
        pattern = rf"\|\s*{code}[^|]*\|\s*{count}\s*\|\s*{re.escape(share)}\s*\|"
        assert re.search(pattern, sheet), f"{code} row ({count}, {share}) not found in DATASHEET.md"


def test_subgroup_counts_match(sheet, meta):
    # The prose uses a typographic en-dash in "0–17"; the metadata uses a plain
    # hyphen. Normalise rather than forcing the document to spell ranges badly.
    prose = sheet.replace("–", "-")
    for group, count in meta["statistics"]["subgroupBalance"].items():
        band = group.split()[0]  # "65+ · Female" -> "65+"
        row = re.search(rf"\|\s*\**{re.escape(band)}\**\s*\|([^|]*)\|([^|]*)\|", prose)
        assert row, f"no datasheet row for age band {band}"
        assert str(count) in row.group(0), f"{group} count {count} missing from its row"


def test_the_under_representation_is_stated_not_buried(meta):
    """The 65+ imbalance is the datasheet's most important claim; pin the fact
    itself so a future dataset change cannot quietly make the prose wrong."""
    balance = meta["statistics"]["subgroupBalance"]
    elderly = sum(v for k, v in balance.items() if k.startswith("65+"))
    others = sum(v for k, v in balance.items() if not k.startswith("65+"))

    assert elderly / meta["rows"] < 0.20
    assert elderly < others / 3


def test_split_parameters_match(sheet, meta):
    split = meta["split"]
    assert split["stratified"] is True
    assert f"{int((1 - split['testSize']) * 100)}/{int(split['testSize'] * 100)}" in sheet
    assert f"random_state={split['randomState']}" in sheet


def test_datasheet_declares_the_data_synthetic(sheet, meta):
    assert meta["provenance"]["synthetic"] is True
    assert "| **Synthetic** | Yes" in sheet
