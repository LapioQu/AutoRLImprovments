"""CLI для запуску професійного benchmark-suite на реальних системах."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.benchmark_catalog import OFFICIAL_BENCHMARKS
from src.downloader import OfficialBenchmarkDownloader
from src.evaluators import BenchmarkResult, evaluate_batch_system, evaluate_stream_system, summarize_results
from src.model_registry import build_batch_model, build_stream_model

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Фіксує основні генератори випадковості."""

    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str) -> Dict[str, Any]:
    """Завантажує YAML-конфігурацію benchmark-suite."""

    with Path(path).open("r", encoding="utf-8") as file_pointer:
        return yaml.safe_load(file_pointer)


def _split_feature_types(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Ділить ознаки на числові й категоріальні."""

    numeric_features = frame.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [column for column in frame.columns if column not in numeric_features]
    return numeric_features, categorical_features


def _cap_batch_rows(dataset: Any, max_rows: int | None, random_seed: int) -> Any:
    """Детерміновано обмежує розмір batch-датасету для практичного benchmark-run."""

    if max_rows is None or len(dataset.X) <= max_rows:
        return dataset
    target_fraction = max_rows / float(len(dataset.X))
    sampled_X, _, sampled_y, _ = train_test_split(
        dataset.X,
        dataset.y,
        train_size=target_fraction,
        random_state=random_seed,
        stratify=dataset.y,
    )
    dataset.X = sampled_X.reset_index(drop=True)
    dataset.y = sampled_y.reset_index(drop=True)
    return dataset


def run_benchmarks(config: Dict[str, Any]) -> List[BenchmarkResult]:
    """Виконує повний набір benchmark-запусків."""

    set_global_seed(int(config.get("random_seed", 42)))
    downloader = OfficialBenchmarkDownloader(data_dir=str(config.get("data_dir", "data")))
    results: List[BenchmarkResult] = []

    for benchmark_item in config["benchmarks"]["batch"]:
        if not benchmark_item.get("enabled", True):
            continue
        benchmark_name = str(benchmark_item["name"])
        logger.info("Запуск batch benchmark=%s", benchmark_name)
        if benchmark_name == "openml_cc18":
            datasets = downloader.load_openml_cc18(
                max_tasks=int(benchmark_item.get("max_tasks", 2)),
                max_rows_per_task=int(benchmark_item.get("max_rows_per_task", 15000)),
            )
        elif benchmark_name == "uci_adult":
            datasets = [downloader.load_uci_dataset(2, "uci_adult", OFFICIAL_BENCHMARKS["uci_adult"].source_url)]
        elif benchmark_name == "uci_bank_marketing":
            datasets = [downloader.load_uci_dataset(222, "uci_bank_marketing", OFFICIAL_BENCHMARKS["uci_bank_marketing"].source_url)]
        elif benchmark_name == "uci_covertype":
            datasets = [downloader.load_uci_dataset(31, "uci_covertype", OFFICIAL_BENCHMARKS["uci_covertype"].source_url)]
        else:
            raise ValueError(f"Невідомий batch benchmark: {benchmark_name}")

        for dataset in datasets:
            dataset = _cap_batch_rows(
                dataset=dataset,
                max_rows=int(benchmark_item["max_rows"]) if "max_rows" in benchmark_item else None,
                random_seed=int(config.get("random_seed", 42)),
            )
            numeric_features, categorical_features = _split_feature_types(dataset.X)
            for system_name in config["systems"]["batch"]:
                model_spec = build_batch_model(system_name, numeric_features, categorical_features)
                result = evaluate_batch_system(
                    dataset=dataset,
                    estimator=model_spec.estimator,
                    system_name=model_spec.name,
                    random_seed=int(config.get("random_seed", 42)),
                )
                results.append(result)
                logger.info(
                    "Готово batch benchmark=%s system=%s primary=%.4f",
                    dataset.name,
                    system_name,
                    result.score_primary,
                )

    for benchmark_item in config["benchmarks"]["stream"]:
        if not benchmark_item.get("enabled", True):
            continue
        benchmark_name = str(benchmark_item["name"])
        logger.info("Запуск stream benchmark=%s", benchmark_name)
        if benchmark_name == "elec2":
            dataset = downloader.load_elec2(max_samples=int(benchmark_item.get("max_samples", 12000)))
        elif benchmark_name == "assistments":
            dataset = downloader.load_assistments(
                url=str(benchmark_item["assistments_url"]),
                max_samples=int(benchmark_item.get("max_samples", 60000)),
            )
        else:
            raise ValueError(f"Невідомий stream benchmark: {benchmark_name}")

        for system_name in config["systems"]["stream"]:
            model_spec = build_stream_model(system_name)
            result = evaluate_stream_system(dataset=dataset, estimator=model_spec.estimator, system_name=model_spec.name)
            results.append(result)
            logger.info(
                "Готово stream benchmark=%s system=%s primary=%.4f",
                dataset.name,
                system_name,
                result.score_primary,
            )

    return results


def persist_results(results: List[BenchmarkResult], results_dir: str) -> None:
    """Зберігає benchmark-артефакти у CSV і JSON."""

    target_dir = Path(results_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    frame = summarize_results(results)
    frame.to_csv(target_dir / "benchmark_results.csv", index=False)
    summary = (
        frame.groupby(["benchmark_family", "benchmark_name", "system_name"])[["score_primary", "score_secondary", "fit_time_sec", "n_samples"]]
        .mean()
        .reset_index()
    )
    summary.to_csv(target_dir / "benchmark_summary.csv", index=False)
    (target_dir / "benchmark_summary.json").write_text(summary.to_json(orient="records", indent=2), encoding="utf-8")


def main() -> None:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Professional ML benchmark runner")
    parser.add_argument("--config", type=str, default="configs/benchmark_suite.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config = load_config(args.config)
    results = run_benchmarks(config)
    persist_results(results=results, results_dir=str(config.get("results_dir", "results")))


if __name__ == "__main__":
    main()
