"""Smoke-тести benchmark-suite."""

from __future__ import annotations

import pandas as pd

from src.benchmark_catalog import list_benchmarks
from src.downloader import BatchDataset, StreamDataset
from src.evaluators import summarize_results


def test_catalog_contains_real_benchmarks() -> None:
    names = {item.name for item in list_benchmarks()}
    assert "openml_cc18" in names
    assert "uci_adult" in names
    assert "assistments" in names
    assert "elec2" in names


def test_batch_dataset_structure() -> None:
    dataset = BatchDataset(
        name="demo",
        X=pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}),
        y=pd.Series([0, 1]),
        source_url="https://example.com",
    )
    assert len(dataset.X) == len(dataset.y)


def test_stream_dataset_structure() -> None:
    dataset = StreamDataset(
        name="stream_demo",
        rows=[({"f1": 0.1}, 1), ({"f1": 0.2}, 0)],
        source_url="https://example.com",
    )
    assert len(dataset.rows) == 2


def test_summarize_results_empty() -> None:
    frame = summarize_results([])
    assert frame.empty
