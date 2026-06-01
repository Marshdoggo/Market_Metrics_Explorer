from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import numpy as np
import pandas as pd

from .evaluation import classification_summary, time_train_test_split
from .features import build_supervised_frame, latest_feature_rows


MODEL_OPTIONS = ["Logistic Regression baseline", "Random Forest", "Simple Neural Net"]


@dataclass
class ForecastResult:
    validation: dict
    board: pd.DataFrame
    drilldown: pd.DataFrame
    feature_columns: list[str]
    target_description: str


def sklearn_available() -> tuple[bool, str | None]:
    if importlib.util.find_spec("sklearn") is None:
        return False, "scikit-learn is not installed"
    return True, None


def run_forecast(
    history: pd.DataFrame,
    *,
    universe: str,
    metric: str,
    top_n: int,
    horizon: int,
    model_type: str,
) -> ForecastResult:
    ok, detail = sklearn_available()
    if model_type != "Logistic Regression baseline" and not ok:
        raise RuntimeError(f"{model_type} requires scikit-learn. Install requirements first. Detail: {detail}")

    supervised, feature_cols = build_supervised_frame(
        history,
        universe=universe,
        metric=metric,
        top_n=top_n,
        horizon=horizon,
    )
    if supervised.empty:
        raise ValueError("No supervised rows could be built for this selection.")

    dates = supervised["as_of_date"].dropna().astype(str).nunique()
    if dates <= int(horizon) or dates < 6:
        raise ValueError(
            f"Not enough historical snapshots for validation. Found {dates} labeled dates; "
            f"try a shorter horizon or backfill more history."
        )
    if supervised["target"].nunique() < 2:
        raise ValueError("The target has only one class for this selection; try a larger Top N or more history.")

    split = time_train_test_split(supervised)
    train = supervised[supervised["as_of_date"].isin(split.train_dates)].copy()
    test = supervised[supervised["as_of_date"].isin(split.test_dates)].copy()
    if train["target"].nunique() < 2 or test.empty:
        raise ValueError("The time split does not contain enough class variation for validation.")

    model = _build_model(model_type)
    x_train = train[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_test = test[feature_cols].apply(pd.to_numeric, errors="coerce")
    y_train = train["target"].astype(int)
    y_test = test["target"].astype(int)

    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    proba = _positive_probability(model, x_test)
    metrics = classification_summary(y_test, pred, proba)
    metrics.update(
        {
            "split_method": "Time-aware holdout: earlier snapshot dates train, later snapshot dates validate",
            "split_date": split.split_date,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "sample_count": int(len(supervised)),
            "snapshot_dates": int(dates),
            "positive_class_rate": float(supervised["target"].mean()),
            "train_date_min": min(split.train_dates),
            "train_date_max": max(split.train_dates),
            "test_date_min": min(split.test_dates),
            "test_date_max": max(split.test_dates),
            "model_type": model_type,
        }
    )

    latest, _, latest_date = latest_feature_rows(history, universe=universe, metric=metric)
    if latest.empty:
        raise ValueError("Could not build latest feature rows for scoring.")
    latest_x = latest[feature_cols].apply(pd.to_numeric, errors="coerce")
    latest["forecast_probability"] = _positive_probability(model, latest_x)
    target_description = f"Top {int(top_n)} by {metric} after {int(horizon)} trading snapshots"
    latest["confidence_bucket"] = latest["forecast_probability"].apply(
        lambda p: confidence_bucket(p, metrics.get("roc_auc"))
    )
    latest["latest_as_of_date"] = latest_date
    latest["target_description"] = target_description
    latest["currently_top_n"] = latest["current_rank"] <= int(top_n)

    board_cols = [
        "ticker",
        "name",
        "sector",
        "subindustry",
        "current_rank",
        "selected_metric_value",
        "forecast_probability",
        "confidence_bucket",
        "latest_as_of_date",
        "target_description",
        "currently_top_n",
    ]
    board = latest[[c for c in board_cols if c in latest.columns]].copy()
    board = board.sort_values("forecast_probability", ascending=False)
    return ForecastResult(
        validation=metrics,
        board=board,
        drilldown=supervised,
        feature_columns=feature_cols,
        target_description=target_description,
    )


def confidence_bucket(probability: float, roc_auc: float | None) -> str:
    if probability >= 0.70 and roc_auc is not None and roc_auc >= 0.60:
        return "High"
    if probability >= 0.55:
        return "Medium"
    return "Low"


def _build_model(model_type: str):
    if model_type == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline

        estimator = RandomForestClassifier(
            n_estimators=250,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", estimator)])
    if model_type == "Simple Neural Net":
        from sklearn.impute import SimpleImputer
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            alpha=0.001,
            max_iter=400,
            early_stopping=True,
            random_state=42,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()), ("model", estimator)])

    return NumpyLogisticClassifier()


def _positive_probability(model, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classes = getattr(model.named_steps.get("model"), "classes_", None)
        if classes is not None and 1 in list(classes):
            idx = list(classes).index(1)
        else:
            idx = min(1, proba.shape[1] - 1)
        return proba[:, idx]
    pred = model.predict(x)
    return np.asarray(pred, dtype=float)


class NumpyLogisticClassifier:
    """Small deterministic logistic baseline for environments without sklearn."""

    def __init__(self, *, learning_rate: float = 0.08, max_iter: int = 1200, l2: float = 0.01):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.l2 = l2
        self.classes_ = np.array([0, 1])

    def fit(self, x: pd.DataFrame, y: pd.Series):
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        self.medians_ = np.nanmedian(x_arr, axis=0)
        self.medians_ = np.where(np.isfinite(self.medians_), self.medians_, 0.0)
        x_arr = np.where(np.isnan(x_arr), self.medians_, x_arr)
        self.mean_ = x_arr.mean(axis=0)
        self.scale_ = x_arr.std(axis=0)
        self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        x_scaled = (x_arr - self.mean_) / self.scale_
        x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])

        pos = max(float(y_arr.sum()), 1.0)
        neg = max(float(len(y_arr) - y_arr.sum()), 1.0)
        weights = np.where(y_arr == 1, len(y_arr) / (2 * pos), len(y_arr) / (2 * neg))

        beta = np.zeros(x_design.shape[1], dtype=float)
        for _ in range(int(self.max_iter)):
            logits = np.clip(x_design @ beta, -35, 35)
            probs = 1.0 / (1.0 + np.exp(-logits))
            error = (probs - y_arr) * weights
            grad = (x_design.T @ error) / len(y_arr)
            grad[1:] += self.l2 * beta[1:]
            beta -= self.learning_rate * grad
        self.coef_ = beta
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        x_arr = np.where(np.isnan(x_arr), self.medians_, x_arr)
        x_scaled = (x_arr - self.mean_) / self.scale_
        x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
        logits = np.clip(x_design @ self.coef_, -35, 35)
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - probs, probs])

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)
