"""Feature preprocessing. Imputer and scaler live inside the sklearn Pipeline,
so they are fit on training folds only (no test-set statistics leak)."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def preprocessing_steps(scale: bool = True) -> list[tuple[str, object]]:
    steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return steps


def make_pipeline(estimator, scale: bool = True) -> Pipeline:
    return Pipeline(preprocessing_steps(scale) + [("clf", estimator)])
