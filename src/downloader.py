"""Завантаження й стандартизація офіційних benchmark-датасетів."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import openml
import pandas as pd
import requests
from river import datasets as river_datasets
from ucimlrepo import fetch_ucirepo


@dataclass(slots=True)
class BatchDataset:
    """Стандартизоване batch-представлення датасету."""

    name: str
    X: pd.DataFrame
    y: pd.Series
    source_url: str


@dataclass(slots=True)
class StreamDataset:
    """Стандартизоване stream-представлення датасету."""

    name: str
    rows: List[Tuple[Dict[str, float], int]]
    source_url: str


def _sha256_of_file(path: Path) -> str:
    """Обчислює SHA-256 локального файла."""

    digest = hashlib.sha256()
    with path.open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_target(y: Iterable[object]) -> pd.Series:
    """Нормалізує target до одномірної серії."""

    target = pd.Series(y).copy()
    if target.dtype == object:
        target = target.astype(str).str.strip().str.rstrip(".")
    return target.reset_index(drop=True)


class OfficialBenchmarkDownloader:
    """Завантажувач реальних benchmark-датасетів із офіційних джерел."""

    def __init__(self, data_dir: str = "data") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def load_uci_dataset(self, dataset_id: int, dataset_name: str, source_url: str) -> BatchDataset:
        """Завантажує офіційний датасет UCI через `ucimlrepo`."""

        payload = fetch_ucirepo(id=dataset_id)
        features = payload.data.features.copy()
        targets = payload.data.targets.copy()
        if isinstance(targets, pd.DataFrame) and targets.shape[1] == 1:
            targets = targets.iloc[:, 0]
        target_series = _normalize_target(targets)
        assert len(features) == len(target_series)
        return BatchDataset(
            name=dataset_name,
            X=features.reset_index(drop=True),
            y=target_series,
            source_url=source_url,
        )

    def load_openml_cc18(self, max_tasks: int = 2, max_rows_per_task: int = 15000) -> List[BatchDataset]:
        """Завантажує підмножину офіційного benchmark-suite OpenML-CC18."""

        suite = openml.study.get_suite("OpenML-CC18")
        datasets: List[BatchDataset] = []
        for task_id in suite.tasks[:max_tasks]:
            task = openml.tasks.get_task(task_id)
            X, y = task.get_X_and_y(dataset_format="dataframe")
            frame = pd.DataFrame(X).copy()
            target = _normalize_target(y)
            if len(frame) > max_rows_per_task:
                frame = frame.iloc[:max_rows_per_task].reset_index(drop=True)
                target = target.iloc[:max_rows_per_task].reset_index(drop=True)
            datasets.append(
                BatchDataset(
                    name=f"openml_cc18_task_{task_id}_{task.get_dataset().name}",
                    X=frame.reset_index(drop=True),
                    y=target.reset_index(drop=True),
                    source_url="https://docs.openml.org/benchmark/",
                )
            )
        return datasets

    def load_elec2(self, max_samples: int = 12000) -> StreamDataset:
        """Завантажує реальний потоковий benchmark Elec2 через River."""

        dataset = river_datasets.Elec2()
        rows: List[Tuple[Dict[str, float], int]] = []
        for index, (features, target) in enumerate(dataset):
            numeric_features = {str(key): float(value) for key, value in features.items()}
            rows.append((numeric_features, int(bool(target))))
            if index + 1 >= max_samples:
                break
        return StreamDataset(
            name="elec2",
            rows=rows,
            source_url="https://riverml.xyz/0.7.0/api/datasets/Elec2/",
        )

    def _download_assistments_csv(self, url: str) -> Path:
        """Завантажує офіційний ASSISTments CSV у локальний кеш."""

        target_path = self.data_dir / "assistments_skill_builder.csv"
        if target_path.exists():
            return target_path
        with requests.get(url, stream=True, timeout=180) as response:
            response.raise_for_status()
            with target_path.open("wb") as file_pointer:
                for chunk in response.iter_content(chunk_size=1_048_576):
                    if chunk:
                        file_pointer.write(chunk)
        return target_path

    def load_assistments(self, url: str, max_samples: int = 60000) -> StreamDataset:
        """Завантажує й нормалізує реальний ASSISTments у streaming-форму."""

        csv_path = self._download_assistments_csv(url)
        frame = pd.read_csv(csv_path)
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        rename_map = {
            "sequence_id": "problem_id",
            "problemid": "problem_id",
            "userid": "user_id",
            "logid": "log_id",
        }
        frame = frame.rename(columns=rename_map)
        required = {"user_id", "problem_id", "correct"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"ASSISTments CSV не містить обов'язкові колонки: {sorted(missing)}")

        if "log_id" in frame.columns:
            frame["log_id"] = pd.to_numeric(frame["log_id"], errors="coerce")
        frame["correct"] = pd.to_numeric(frame["correct"], errors="coerce")
        frame = frame.dropna(subset=["user_id", "problem_id", "correct"]).copy()
        frame = frame[(frame["correct"] >= 0.0) & (frame["correct"] <= 1.0)].copy()
        sort_columns = [column for column in ["user_id", "problem_id", "log_id"] if column in frame.columns]
        frame = frame.sort_values(sort_columns, kind="mergesort")
        frame = frame.groupby(["user_id", "problem_id"], as_index=False).tail(1).copy()
        frame["reward"] = frame["correct"].astype(float)
        mean_reward = float(frame["reward"].mean())
        if not (0.85 < mean_reward < 0.95):
            raise ValueError(
                f"ASSISTments reward distribution не збігається з очікуваною: mean_reward={mean_reward:.6f}"
            )

        rows: List[Tuple[Dict[str, float], int]] = []
        for row in frame.head(max_samples).itertuples(index=False):
            user_hash = float(abs(hash(str(row.user_id))) % 10_000) / 10_000.0
            problem_hash = float(abs(hash(str(row.problem_id))) % 10_000) / 10_000.0
            features = {
                "user_hash": user_hash,
                "problem_hash": problem_hash,
                "interaction": user_hash * problem_hash,
                "distance": abs(user_hash - problem_hash),
            }
            rows.append((features, int(float(row.correct) >= 0.5)))

        return StreamDataset(
            name=f"assistments_sha256_{_sha256_of_file(csv_path)[:12]}",
            rows=rows,
            source_url=url,
        )
