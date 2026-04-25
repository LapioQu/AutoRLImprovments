"""Оцінювання batch і stream систем на реальних benchmark-ах."""

from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from river import metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src.downloader import BatchDataset, StreamDataset


@dataclass(slots=True)
class BenchmarkResult:
    """Один рядок benchmark-результату."""

    benchmark_name: str
    benchmark_family: str
    system_name: str
    source_url: str
    n_samples: int
    fit_time_sec: float
    score_primary: float
    score_secondary: float
    metric_primary: str
    metric_secondary: str

    def to_dict(self) -> Dict[str, Any]:
        """Повертає серіалізоване представлення рядка."""

        return asdict(self)


def evaluate_batch_system(dataset: BatchDataset, estimator: Any, system_name: str, random_seed: int) -> BenchmarkResult:
    """Оцінює batch-систему на реальному датасеті."""

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=0.2,
        random_state=random_seed,
        stratify=dataset.y,
    )
    started = time.perf_counter()
    estimator.fit(X_train, y_train)
    fit_time = time.perf_counter() - started
    predictions = estimator.predict(X_test)
    return BenchmarkResult(
        benchmark_name=dataset.name,
        benchmark_family="batch",
        system_name=system_name,
        source_url=dataset.source_url,
        n_samples=len(dataset.X),
        fit_time_sec=float(fit_time),
        score_primary=float(accuracy_score(y_test, predictions)),
        score_secondary=float(balanced_accuracy_score(y_test, predictions)),
        metric_primary="accuracy",
        metric_secondary="balanced_accuracy",
    )


def evaluate_stream_system(dataset: StreamDataset, estimator: Any, system_name: str) -> BenchmarkResult:
    """Оцінює streaming-систему у prequential-режимі."""

    metric_accuracy = metrics.Accuracy()
    metric_f1 = metrics.F1()
    started = time.perf_counter()
    first = True
    for features, target in dataset.rows:
        if not first:
            prediction = estimator.predict_one(features)
            if prediction is not None:
                metric_accuracy.update(target, prediction)
                metric_f1.update(target, prediction)
        estimator.learn_one(features, target)
        first = False
    elapsed = time.perf_counter() - started
    return BenchmarkResult(
        benchmark_name=dataset.name,
        benchmark_family="stream",
        system_name=system_name,
        source_url=dataset.source_url,
        n_samples=len(dataset.rows),
        fit_time_sec=float(elapsed),
        score_primary=float(metric_accuracy.get()),
        score_secondary=float(metric_f1.get()),
        metric_primary="prequential_accuracy",
        metric_secondary="prequential_f1",
    )


def summarize_results(results: List[BenchmarkResult]) -> pd.DataFrame:
    """Агрегує результати у табличний вигляд."""

    frame = pd.DataFrame([result.to_dict() for result in results])
    if frame.empty:
        return frame
    return frame.sort_values(["benchmark_family", "benchmark_name", "system_name"]).reset_index(drop=True)
