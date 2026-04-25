"""Каталог офіційних бенчмарків і їхніх джерел."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Опис офіційного benchmark-джерела."""

    name: str
    family: str
    source_name: str
    source_url: str
    citation_hint: str


OFFICIAL_BENCHMARKS: Dict[str, BenchmarkSpec] = {
    "openml_cc18": BenchmarkSpec(
        name="openml_cc18",
        family="batch_suite",
        source_name="OpenML-CC18",
        source_url="https://docs.openml.org/benchmark/",
        citation_hint="OpenML benchmark suites documentation and OpenML-CC18 study.",
    ),
    "uci_adult": BenchmarkSpec(
        name="uci_adult",
        family="batch",
        source_name="UCI Adult",
        source_url="https://archive.ics.uci.edu/dataset/2/adult",
        citation_hint="Becker & Kohavi (1996), UCI Adult dataset.",
    ),
    "uci_bank_marketing": BenchmarkSpec(
        name="uci_bank_marketing",
        family="batch",
        source_name="UCI Bank Marketing",
        source_url="https://archive.ics.uci.edu/dataset/222",
        citation_hint="Moro, Rita, Cortez (2014), UCI Bank Marketing dataset.",
    ),
    "uci_covertype": BenchmarkSpec(
        name="uci_covertype",
        family="batch",
        source_name="UCI Covertype",
        source_url="https://archive.ics.uci.edu/dataset/31/covertype%29",
        citation_hint="Blackard (1998), UCI Covertype dataset.",
    ),
    "assistments": BenchmarkSpec(
        name="assistments",
        family="stream",
        source_name="ASSISTments Skill Builder",
        source_url="https://www.assistments.org/individual-resource/downloading-the-assignment-report",
        citation_hint="ASSISTments official CSV export / Skill Builder logs.",
    ),
    "elec2": BenchmarkSpec(
        name="elec2",
        family="stream",
        source_name="River Elec2",
        source_url="https://riverml.xyz/0.7.0/api/datasets/Elec2/",
        citation_hint="Australian NSW Electricity Market benchmark as wrapped by River.",
    ),
}


def list_benchmarks() -> List[BenchmarkSpec]:
    """Повертає впорядкований перелік офіційних benchmark-специфікацій."""

    return [OFFICIAL_BENCHMARKS[key] for key in sorted(OFFICIAL_BENCHMARKS)]
