"""Реєстр реальних ML-систем для benchmark-тестування."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from river import linear_model, naive_bayes, preprocessing, tree
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True, slots=True)
class BatchModelSpec:
    """Специфікація batch-моделі."""

    name: str
    estimator: Any


@dataclass(frozen=True, slots=True)
class StreamModelSpec:
    """Специфікація streaming-моделі."""

    name: str
    estimator: Any


def build_batch_model(name: str, numeric_features: List[str], categorical_features: List[str]) -> BatchModelSpec:
    """Створює реальну scikit-learn систему для табличної задачі."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    categorical_pipeline_dense = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ]
    )

    if name == "logreg":
        estimator = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000, n_jobs=None)),
            ]
        )
    elif name == "random_forest":
        estimator = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
            ]
        )
    elif name == "hist_gb":
        dense_preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_features),
                ("cat", categorical_pipeline_dense, categorical_features),
            ]
        )
        estimator = Pipeline(
            steps=[
                ("preprocessor", dense_preprocessor),
                ("model", HistGradientBoostingClassifier(random_state=42)),
            ]
        )
    else:
        raise ValueError(f"Невідома batch-система: {name}")
    return BatchModelSpec(name=name, estimator=estimator)


def build_stream_model(name: str) -> StreamModelSpec:
    """Створює реальну river-систему для streaming benchmark-ів."""

    if name == "river_logreg":
        estimator = preprocessing.StandardScaler() | linear_model.LogisticRegression()
    elif name == "river_nb":
        estimator = naive_bayes.GaussianNB()
    elif name == "river_hoeffding_tree":
        estimator = tree.HoeffdingTreeClassifier()
    else:
        raise ValueError(f"Невідома stream-система: {name}")
    return StreamModelSpec(name=name, estimator=estimator)
