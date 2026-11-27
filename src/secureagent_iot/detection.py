"""IDS baselines: single-feature threshold rule, Random Forest, optional XGBoost.
The detector is supporting infrastructure; its output is advisory only."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from .data import GROUP_COL
from .features import make_pipeline

try:  # optional
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:  # pragma: no cover
    HAVE_XGB = False


def build_models(cfg: dict, seed: int, y_train: np.ndarray) -> dict:
    d = cfg["detection"]
    models = {
        # "rule": one feature, one threshold, both chosen on the training set only
        "rule_threshold": make_pipeline(DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=seed), scale=False),
        "random_forest": make_pipeline(RandomForestClassifier(random_state=seed, **d["rf"]), scale=False),
    }
    if HAVE_XGB:
        pos, neg = max(int((y_train == 1).sum()), 1), max(int((y_train == 0).sum()), 1)
        models["xgboost"] = make_pipeline(
            XGBClassifier(random_state=seed, scale_pos_weight=neg / pos, eval_metric="logloss", tree_method="hist", **d["xgb"]),
            scale=False)
    return models


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision_attack": precision_score(y_true, y_pred, zero_division=0),
        "recall_attack": recall_score(y_true, y_pred, zero_division=0),
        "f1_attack": f1_score(y_true, y_pred, zero_division=0),
        "precision_benign": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_benign": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_benign": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score) if y_score is not None else float("nan"),
        "fpr": fp / (fp + tn) if (fp + tn) else float("nan"),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def cross_validate(model, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray | None, folds: int, seed: int) -> list[dict]:
    if groups is not None and len(np.unique(groups)) >= folds:
        splitter, kw = GroupKFold(n_splits=folds), {"groups": groups}
    else:
        splitter, kw = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed), {}
    out = []
    from sklearn.base import clone
    for i, (tr, va) in enumerate(splitter.split(X, y, **kw)):
        m = clone(model).fit(X.iloc[tr], y[tr])
        score = m.predict_proba(X.iloc[va])[:, 1]
        out.append({"fold": i, **binary_metrics(y[va], m.predict(X.iloc[va]), score)})
    return out


def fit_and_evaluate(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], cfg: dict, seed: int):
    Xtr, ytr, Xte, yte = train[feats], train["y"].to_numpy(), test[feats], test["y"].to_numpy()
    groups = train[GROUP_COL].to_numpy() if GROUP_COL in train else None
    rows, cv_rows, curves, fitted = [], [], {}, {}
    for name, model in build_models(cfg, seed, ytr).items():
        for r in cross_validate(model, Xtr, ytr, groups, int(cfg["detection"]["cv_folds"]), seed):
            cv_rows.append({"model": name, **r})
        model.fit(Xtr, ytr)
        score = model.predict_proba(Xte)[:, 1]
        rows.append({"model": name, "split": "holdout_test", **binary_metrics(yte, model.predict(Xte), score)})
        p, r, _ = precision_recall_curve(yte, score)
        curves[name] = pd.DataFrame({"precision": p, "recall": r})
        fitted[name] = model
    return pd.DataFrame(rows), pd.DataFrame(cv_rows), curves, fitted


def describe_rule(model, feats: list[str]) -> str:
    tree = model.named_steps["clf"].tree_
    if tree.feature[0] < 0:
        return "constant prediction (no split found)"
    return f"attack if {feats[tree.feature[0]]} {'>' if tree.value[2][0][1] > tree.value[1][0][1] else '<='} {tree.threshold[0]:.6g}"
