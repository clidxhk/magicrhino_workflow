"""Final ROS classifier pipeline: nested internal validation, top-10 full-data refits,
and the configured deployment model (see config/analysis_config.json).

Evaluation is nested internal validation only: each classifier family is
scored separately, the outer test partition is never used for preprocessing,
tuning, or threshold selection, and no pooled outer threshold is optimized.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.utils import ClassifierTags
from lightgbm import LGBMClassifier
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier as NativeXGBClassifier

ROOT = Path(__file__).resolve().parent
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
TOP10 = MODELS / "top10"
METRIC_COLUMNS = ["roc_auc", "average_precision", "mcc", "balanced_accuracy",
                  "sensitivity", "specificity", "accuracy", "f1", "precision"]
FAMILY_FILE_NAMES = {"LogisticRegression": "logistic_regression", "GaussianNB": "gaussian_nb",
                     "KNearestNeighbors": "k_nearest_neighbors", "RandomForest": "random_forest",
                     "LightGBM": "lightgbm", "XGBoost": "xgboost"}


def deployment_stem(config: dict[str, Any]) -> str:
    name = FAMILY_FILE_NAMES.get(config["deployment_model"], config["deployment_model"].lower())
    return f"{name}_seed{config['deployment_seed']}"


class CompatibleXGBClassifier(NativeXGBClassifier):
    """Local tags adapter for XGBoost 2.1.2 with scikit-learn 1.9.

    Only estimator metadata changes; fit, prediction, and booster parameters
    remain the installed XGBoost implementation. Upstream fixed this in
    XGBoost 2.1.4. Loading a saved XGBoost artifact requires this class,
    i.e. run_final_pipeline must be importable at load time.
    """

    def __sklearn_tags__(self):
        tags = BaseEstimator.__sklearn_tags__(self)
        tags.estimator_type = "classifier"
        tags.target_tags.required = True
        tags.classifier_tags = ClassifierTags()
        tags.input_tags.sparse = True
        tags.input_tags.allow_nan = True
        tags.no_validation = True
        return tags


XGBClassifier = (NativeXGBClassifier if is_classifier(NativeXGBClassifier())
                 else CompatibleXGBClassifier)
if __name__ == "__main__":
    # Pickle portability: saved XGBoost artifacts must reference an importable
    # module name, not __main__, so any process importing run_final_pipeline
    # can load them.
    CompatibleXGBClassifier.__module__ = "run_final_pipeline"
    sys.modules.setdefault("run_final_pipeline", sys.modules[__name__])


warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.linear_model")
warnings.filterwarnings("ignore", category=UserWarning, message="Inconsistent values: penalty=.*")
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class ModelSpec:
    name: str
    estimator: Pipeline
    parameters: dict[str, Any]


def to_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def write_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                    default=to_json_value, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def json_params(params: dict[str, Any]) -> str:
    return json.dumps({key: to_json_value(value) for key, value in params.items()}, sort_keys=True)


def base_pipeline(model: Any, k: int = 200, scaler: Any | None = None) -> Pipeline:
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold()),
        ("select", SelectKBest(score_func=f_classif, k=k)),
    ]
    if scaler is not None:
        steps.append(("scale", scaler))
    steps.append(("model", model))
    return Pipeline(steps)


def model_specs(seed: int, scale_pos_weight: float) -> list[ModelSpec]:
    # scale_pos_weight is retained unused, as in the original pipeline.
    common = {"select__k": [20, 50, 100, 200, 400]}
    return [
        ModelSpec(
            "LogisticRegression",
            base_pipeline(LogisticRegression(max_iter=5000, solver="liblinear", random_state=seed), scaler=RobustScaler()),
            common | {"model__C": loguniform(1e-3, 10), "model__penalty": ["l1", "l2"], "model__class_weight": [None, "balanced"]},
        ),
        ModelSpec(
            "GaussianNB",
            base_pipeline(GaussianNB(), scaler=StandardScaler()),
            common | {"model__var_smoothing": loguniform(1e-12, 1e-2)},
        ),
        ModelSpec(
            "KNearestNeighbors",
            base_pipeline(KNeighborsClassifier(n_jobs=1), scaler=StandardScaler()),
            common | {"model__n_neighbors": randint(3, 40), "model__weights": ["uniform", "distance"], "model__p": [1, 2], "model__leaf_size": randint(15, 60)},
        ),
        ModelSpec(
            "RandomForest",
            base_pipeline(RandomForestClassifier(n_estimators=250, n_jobs=1, random_state=seed)),
            common | {"model__max_depth": [None, 3, 5, 8, 12, 18], "model__min_samples_leaf": randint(1, 12), "model__max_features": ["sqrt", "log2", 0.2, 0.4, 0.7], "model__class_weight": [None, "balanced", "balanced_subsample"]},
        ),
        ModelSpec(
            "LightGBM",
            base_pipeline(LGBMClassifier(objective="binary", n_jobs=1, verbosity=-1, random_state=seed)),
            common | {"model__n_estimators": randint(60, 280), "model__learning_rate": loguniform(0.01, 0.2), "model__num_leaves": randint(5, 40), "model__max_depth": [-1, 3, 5, 7, 10], "model__min_child_samples": randint(5, 45), "model__subsample": uniform(0.65, 0.35), "model__colsample_bytree": uniform(0.35, 0.65), "model__reg_alpha": loguniform(1e-4, 10), "model__reg_lambda": loguniform(1e-4, 30), "model__class_weight": [None, "balanced"]},
        ),
        ModelSpec(
            "XGBoost",
            base_pipeline(XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=seed, verbosity=0)),
            common | {"model__n_estimators": randint(60, 260), "model__learning_rate": loguniform(0.01, 0.2), "model__max_depth": randint(2, 8), "model__min_child_weight": loguniform(0.3, 20), "model__subsample": uniform(0.65, 0.35), "model__colsample_bytree": uniform(0.35, 0.65), "model__reg_alpha": loguniform(1e-4, 10), "model__reg_lambda": loguniform(1e-3, 30)},
        ),
    ]


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    """MCC-optimal threshold with balanced-accuracy, centering, and value tie-breaking.

    Vectorized confusion counts; only ever receives labels from an outer
    TRAINING partition.
    """
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    values = np.unique(probability)
    thresholds = np.r_[0.0, (values[:-1] + values[1:]) / 2, 1.0]
    prediction = probability[:, None] >= thresholds[None, :]
    positive = y == 1
    tp = prediction[positive].sum(axis=0).astype(float)
    fp = prediction[~positive].sum(axis=0).astype(float)
    fn = positive.sum() - tp
    tn = (~positive).sum() - fp
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denominator,
                    out=np.zeros_like(tp), where=denominator != 0)
    balanced = (tp / positive.sum() + tn / (~positive).sum()) / 2
    best = max(range(len(thresholds)), key=lambda i: (
        float(mcc[i]), float(balanced[i]), -abs(float(thresholds[i]) - 0.5),
        float(thresholds[i])))
    return float(thresholds[best])


def metrics(y, probability, prediction) -> dict[str, Any]:
    """Actual per-fold decisions; never a pooled outer threshold."""
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def selected_features(pipeline: Pipeline, columns) -> list[str]:
    names = np.asarray(columns, dtype=object)
    for _, transformer in pipeline.steps[:-1]:
        names = transformer.get_feature_names_out(names)
    return names.tolist()


def load_data(config: dict[str, Any]) -> tuple[Path, pd.DataFrame, pd.DataFrame, np.ndarray]:
    path = (ROOT / config["training_file"]).resolve()
    frame = pd.read_csv(path)
    identifier, target = config["index_column"], config["target_column"]
    if not {identifier, target}.issubset(frame.columns):
        raise ValueError(f"Required columns: {identifier}, {target}")
    if frame[identifier].isna().any() or not frame[identifier].is_unique:
        raise ValueError("Sample IDs must be nonmissing and unique")
    if frame[target].isna().any() or set(frame[target].unique()) != {0, 1}:
        raise ValueError("Target must contain only 0 and 1, with both classes present")
    x = frame.drop(columns=[identifier, target])
    if x.empty or len(x.select_dtypes(exclude="number").columns):
        raise ValueError("All features must be numeric")
    if np.isinf(x.to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature values are not supported")
    return path, frame, x, frame[target].to_numpy(dtype=int)


def run_fold(spec: ModelSpec, x_train: pd.DataFrame, y_train: np.ndarray, x_test: pd.DataFrame,
             *, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    """Outer test labels never enter this function."""
    inner_cv = StratifiedKFold(n_splits=config["inner_folds"], shuffle=True, random_state=seed)
    splits = list(inner_cv.split(x_train, y_train))
    if min(np.bincount(y_train)) < config["inner_folds"]:
        raise ValueError("Insufficient class counts for the inner split")
    search = RandomizedSearchCV(
        spec.estimator, spec.parameters, n_iter=config["search_trials"],
        scoring={"roc_auc": "roc_auc", "average_precision": "average_precision",
                 "balanced_accuracy": "balanced_accuracy", "mcc": "matthews_corrcoef"},
        refit="roc_auc", cv=splits, random_state=seed, n_jobs=config["workers"],
        pre_dispatch=config["workers"], return_train_score=False, error_score="raise",
    )
    search.fit(x_train, y_train)
    # Internal OOF of the selected candidate reuses the tuning folds; this can
    # affect the threshold, but outer labels remain unseen.
    best = clone(spec.estimator).set_params(**search.best_params_)
    inner_probability = cross_val_predict(
        best, x_train, y_train, cv=splits, method="predict_proba",
        n_jobs=config["workers"], pre_dispatch=config["workers"])[:, 1]
    threshold = choose_threshold(y_train, inner_probability)
    fitted = search.best_estimator_  # Refitted on the entire outer training set.
    probability = fitted.predict_proba(x_test)[:, 1]
    prediction = (probability >= threshold).astype(int)
    cv = search.cv_results_
    candidates = []
    for i, params in enumerate(cv["params"]):
        candidates.append({"candidate": i + 1, "parameters": json_params(params),
                           "rank_inner_roc_auc": int(cv["rank_test_roc_auc"][i]),
                           "mean_inner_roc_auc": float(cv["mean_test_roc_auc"][i]),
                           "sd_inner_roc_auc": float(cv["std_test_roc_auc"][i]),
                           "mean_inner_average_precision": float(cv["mean_test_average_precision"][i])})
    return {
        "probability": probability, "prediction": prediction,
        "threshold": threshold, "best_parameters": search.best_params_,
        "best_inner_mean_roc_auc": float(search.best_score_),
        "selected_features": selected_features(fitted, x_train.columns),
        "inner_candidates": candidates, "inner_probability": inner_probability,
    }


def fit_full_data(spec: ModelSpec, x: pd.DataFrame, y: np.ndarray,
                  *, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    """Repeat the identical selection procedure on all samples, then refit on all."""
    inner_cv = StratifiedKFold(n_splits=config["inner_folds"], shuffle=True, random_state=seed)
    splits = list(inner_cv.split(x, y))
    with threadpool_limits(limits=1):
        search = RandomizedSearchCV(
            spec.estimator, spec.parameters, n_iter=config["search_trials"],
            scoring={"roc_auc": "roc_auc", "average_precision": "average_precision",
                     "balanced_accuracy": "balanced_accuracy", "mcc": "matthews_corrcoef"},
            refit="roc_auc", cv=splits, random_state=seed, n_jobs=config["workers"],
            pre_dispatch=config["workers"], return_train_score=False, error_score="raise")
        search.fit(x, y)
        best = clone(spec.estimator).set_params(**search.best_params_)
        inner_probability = cross_val_predict(
            best, x, y, cv=splits, method="predict_proba",
            n_jobs=config["workers"], pre_dispatch=config["workers"])[:, 1]
        threshold = choose_threshold(y, inner_probability)
    fitted = search.best_estimator_  # Refitted on the full dataset.
    return {"fitted": fitted, "threshold": threshold,
            "best_parameters": search.best_params_,
            "mean_inner_roc_auc": float(search.best_score_),
            "pooled_inner_oof_roc_auc": float(roc_auc_score(y, inner_probability)),
            "selected_features": selected_features(fitted, x.columns)}


def run_nested_validation(config: dict[str, Any], frame: pd.DataFrame, x: pd.DataFrame,
                          y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (outer predictions, inner threshold predictions, fold metrics, candidates, by-seed scores, summary)."""
    outer_rows, inner_frames, candidate_rows = [], [], []
    prediction_frames = []
    for seed in config["comparison_seeds"]:
        specs = {spec.name: spec for spec in model_specs(seed, 1.0)}
        outer_cv = StratifiedKFold(n_splits=config["outer_folds"], shuffle=True, random_state=seed)
        outer_splits = list(outer_cv.split(x, y))
        for name in config["models"]:
            for fold, (train, test) in enumerate(outer_splits, 1):
                assert not np.intersect1d(train, test).size
                print(f"Searching {name} seed={seed} outer_fold={fold}", flush=True)
                started = time.perf_counter()
                with threadpool_limits(limits=1):
                    result = run_fold(specs[name], x.iloc[train], y[train], x.iloc[test],
                                      seed=seed, config=config)
                outer_metrics = metrics(y[test], result["probability"], result["prediction"])
                prediction_frames.append(pd.DataFrame({
                    "sample_position": test,
                    "Index": frame.iloc[test][config["index_column"]].to_numpy(),
                    "Y": y[test], "seed": seed, "model": name, "outer_fold": fold,
                    "outer_probability": result["probability"],
                    "threshold": result["threshold"],
                    "outer_prediction": result["prediction"]}))
                inner_frames.append(pd.DataFrame({
                    "sample_position": train,
                    "Index": frame.iloc[train][config["index_column"]].to_numpy(),
                    "Y": y[train], "seed": seed, "model": name, "outer_fold": fold,
                    "inner_probability": result["inner_probability"]}))
                outer_rows.append({"seed": seed, "model": name, "outer_fold": fold,
                                   "n_train": len(train), "n_test": len(test),
                                   "threshold": result["threshold"],
                                   "n_selected_features": len(result["selected_features"]),
                                   "best_inner_mean_roc_auc": result["best_inner_mean_roc_auc"],
                                   "best_parameters": json_params(result["best_parameters"]),
                                   **outer_metrics})
                for candidate in result["inner_candidates"]:
                    candidate_rows.append({"seed": seed, "model": name, "outer_fold": fold, **candidate})
                print(f"DONE {name} seed={seed} fold={fold}: AUC={outer_metrics['roc_auc']:.4f}, "
                      f"MCC={outer_metrics['mcc']:.4f}, seconds={time.perf_counter() - started:.1f}", flush=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions.duplicated(["seed", "model", "sample_position"]).any():
        raise AssertionError("Duplicate outer prediction for one model/seed/sample")
    predictions = predictions.sort_values(["seed", "model", "sample_position"], ignore_index=True)
    by_seed = []
    for (seed, name), group in predictions.groupby(["seed", "model"], sort=False):
        if set(group.sample_position) != set(range(len(frame))):
            raise AssertionError("Outer OOF does not cover every sample exactly once")
        by_seed.append({"seed": int(seed), "model": name, "n_samples": len(group),
                        **metrics(group.Y, group.outer_probability, group.outer_prediction)})
    by_seed = pd.DataFrame(by_seed)
    summary = by_seed.groupby("model", as_index=False).agg(
        n_seeds=("seed", "size"), mean_outer_roc_auc=("roc_auc", "mean"),
        sd_outer_roc_auc=("roc_auc", "std"), mean_outer_mcc=("mcc", "mean"),
        sd_outer_mcc=("mcc", "std"), mean_outer_average_precision=("average_precision", "mean"),
        mean_outer_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_outer_sensitivity=("sensitivity", "mean"),
        mean_outer_specificity=("specificity", "mean"),
    ).sort_values("mean_outer_roc_auc", ascending=False)
    inner_predictions = pd.concat(inner_frames, ignore_index=True)
    inner_predictions = inner_predictions.sort_values(["seed", "model", "outer_fold", "sample_position"],
                                                      ignore_index=True)
    return (predictions, inner_predictions, pd.DataFrame(outer_rows),
            pd.DataFrame(candidate_rows), by_seed, summary)


def write_rf_plot_data(rf_oof: pd.DataFrame, stem: str) -> None:
    """Plot data from the deployment combination's own outer OOF probabilities."""
    probability = rf_oof.outer_probability.to_numpy()
    fold_thresholds = rf_oof.threshold.unique()
    fpr, tpr, thresholds = roc_curve(rf_oof.Y, probability, drop_intermediate=False)
    roc = pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr,
                        "threshold": thresholds, "specificity": 1 - fpr})
    roc.loc[~np.isfinite(roc["threshold"]), "threshold"] = np.nan
    roc.to_csv(TABLES / f"{stem}_roc_curve.csv", index=False)
    precision, recall, pr_thresholds = precision_recall_curve(rf_oof.Y, probability)
    pr = pd.DataFrame({"precision": precision, "recall": recall})
    pr["threshold"] = np.r_[pr_thresholds, np.nan]
    pr.to_csv(TABLES / f"{stem}_pr_curve.csv", index=False)
    values = np.unique(probability)
    sweep = np.r_[0.0, (values[:-1] + values[1:]) / 2, 1.0]
    rows = []
    for point in sweep:
        prediction = (probability >= point).astype(int)
        scored = metrics(rf_oof.Y, probability, prediction)
        rows.append({"threshold": float(point), "mcc": scored["mcc"],
                     "sensitivity": scored["sensitivity"], "specificity": scored["specificity"],
                     "balanced_accuracy": scored["balanced_accuracy"],
                     "accuracy": scored["accuracy"],
                     "is_nearest_to_locked_fold_threshold": False})
    # Fold thresholds are chosen on inner OOF probabilities, so they generally do
    # not coincide with any midpoint of the outer OOF values; flag the nearest
    # sweep point of each locked fold threshold instead.
    flags = pd.DataFrame(rows)
    for locked in fold_thresholds:
        flags.loc[int(np.argmin(np.abs(sweep - locked))), "is_nearest_to_locked_fold_threshold"] = True
    flags.to_csv(TABLES / f"{stem}_threshold_metrics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Remove existing outputs before running")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/analysis_config.json").read_text(encoding="utf-8"))
    if (config["outer_folds"] < 2 or config["inner_folds"] < 2
            or config["search_trials"] < 1 or config["workers"] < 1):
        raise ValueError("Invalid folds, trials, or workers")
    for key in ("models", "comparison_seeds"):
        if not config[key] or len(config[key]) != len(set(config[key])):
            raise ValueError(f"{key} must be nonempty and unique")
    if args.clean and OUTPUTS.exists():
        shutil.rmtree(OUTPUTS)
    for directory in [TABLES, TOP10]:
        directory.mkdir(parents=True, exist_ok=True)

    path, frame, x, y = load_data(config)
    if min(np.bincount(y)) < config["outer_folds"]:
        raise ValueError("Insufficient class counts for the outer split")
    if config["deployment_model"] not in config["models"] or config["deployment_seed"] not in config["comparison_seeds"]:
        raise ValueError("Deployment model/seed must be among the compared combinations")

    predictions, inner_predictions, fold_metrics, candidates, by_seed, summary = run_nested_validation(config, frame, x, y)
    predictions.to_csv(TABLES / "outer_oof_predictions.csv", index=False)
    inner_predictions.to_csv(TABLES / "inner_threshold_predictions.csv", index=False)
    fold_metrics.to_csv(TABLES / "outer_fold_metrics.csv", index=False)
    candidates.to_csv(TABLES / "inner_search_candidates.csv", index=False)
    by_seed.to_csv(TABLES / "model_comparison_by_seed.csv", index=False)
    summary.to_csv(TABLES / "model_comparison_summary.csv", index=False)

    ranked = by_seed.sort_values("roc_auc", ascending=False).head(config["n_top_models"]).reset_index(drop=True)
    if len(ranked) < config["n_top_models"]:
        raise RuntimeError("Not enough completed combinations to rank")
    manifest_rows, deployment = [], None
    for rank, row in enumerate(ranked.itertuples(index=False), 1):
        seed = int(row.seed)
        spec = {s.name: s for s in model_specs(seed, 1.0)}[row.model]
        fit = fit_full_data(spec, x, y, seed=seed, config=config)
        stem = f"rank{rank:02d}_{row.model}_seed{seed}"
        joblib.dump(fit["fitted"], TOP10 / f"{stem}.joblib", compress=3)
        write_json(TOP10 / f"{stem}.json", {
            "rank": rank, "model": row.model, "seed": seed,
            "nested_validation_performance": {c: float(getattr(row, c)) for c in METRIC_COLUMNS},
            "selection": (f"Full-data stratified inner {config['inner_folds']}-fold randomized search, "
                          f"{config['search_trials']} candidates, "
                          f"refit on mean inner ROC-AUC, seed {seed}, original parameter distributions"),
            "best_parameters": fit["best_parameters"],
            "mean_inner_roc_auc": fit["mean_inner_roc_auc"],
            "pooled_inner_oof_roc_auc": fit["pooled_inner_oof_roc_auc"],
            "threshold": fit["threshold"],
            "threshold_selection": "MCC on full-data internal OOF probabilities (development quantity)",
            "n_samples": len(frame), "n_features": int(x.shape[1]),
            "selected_features": fit["selected_features"],
            "caveat": ("Nested scores predate this full-data fit; all samples were previously "
                       "used in model development"),
        })
        manifest_rows.append({"rank": rank, "file": f"top10/{stem}.joblib", "model": row.model, "seed": seed,
                              "nested_outer_roc_auc": float(row.roc_auc), "nested_outer_mcc": float(row.mcc),
                              "mean_inner_roc_auc": fit["mean_inner_roc_auc"],
                              "pooled_inner_oof_roc_auc": fit["pooled_inner_oof_roc_auc"],
                              "threshold": fit["threshold"]})
        if row.model == config["deployment_model"] and seed == config["deployment_seed"]:
            deployment = {"rank": rank, **fit}
        print(f"{stem}: nested AUC={row.roc_auc:.4f}, inner OOF AUC={fit['mean_inner_roc_auc']:.4f}, "
              f"threshold={fit['threshold']:.6f}", flush=True)
    if deployment is None:
        # The deployment combination need not be part of the top-N ranking; fit it
        # separately with the identical full-data selection procedure.
        spec = {s.name: s for s in model_specs(config["deployment_seed"], 1.0)}[config["deployment_model"]]
        deployment = {"rank": None,
                      **fit_full_data(spec, x, y, seed=config["deployment_seed"], config=config)}
        print(f"{deployment_stem(config)}: outside the top {config['n_top_models']}; "
              "fitted separately for deployment", flush=True)

    deploy_stem = deployment_stem(config)
    joblib.dump(deployment["fitted"], MODELS / f"{deploy_stem}.joblib", compress=3)
    write_json(MODELS / f"{deploy_stem}.json", {
        "family": config["deployment_model"], "selection_seed": config["deployment_seed"],
        "fit": f"All {len(frame)} samples, pipeline identical to the nested validation",
        "rank_among_top10": deployment["rank"],
        "best_parameters": deployment["best_parameters"],
        "mean_inner_roc_auc": deployment["mean_inner_roc_auc"],
        "pooled_inner_oof_roc_auc": deployment["pooled_inner_oof_roc_auc"],
        "threshold": deployment["threshold"],
        "threshold_selection": "MCC on full-data internal OOF probabilities (development quantity)",
        "selected_features": deployment["selected_features"],
        "input_columns": [c for c in frame.columns if c not in
                          (config["index_column"], config["target_column"])],
    })
    rf_oof = predictions[(predictions.model == config["deployment_model"])
                         & (predictions.seed == config["deployment_seed"])].reset_index(drop=True)
    rf_oof.to_csv(TABLES / f"{deploy_stem}_outer_oof_predictions.csv", index=False)
    write_rf_plot_data(rf_oof, deploy_stem)

    family = summary[summary.model == config["deployment_model"]].iloc[0]
    report = {
        "configuration": config,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_versions": {p: importlib.metadata.version(p) for p in
                             ["numpy", "pandas", "scipy", "scikit-learn", "lightgbm",
                              "xgboost", "joblib", "threadpoolctl"]},
        "python_version": platform.python_version(),
        "n_training_samples": int(len(frame)), "n_features": int(x.shape[1]),
        "class_counts": {str(k): int(v) for k, v in
                         frame[config["target_column"]].value_counts().sort_index().items()},
        "evaluation": {
            "design": (f"Per-classifier nested internal validation ({config['outer_folds']} outer x "
                       f"{config['inner_folds']} inner folds, {config['search_trials']}-candidate randomized "
                       f"search per outer training set, seeds {'/'.join(str(s) for s in config['comparison_seeds'])})"),
            "threshold_selection": "MCC-optimal on selected-parameter inner OOF within each outer training set; "
                                   "no pooled outer threshold",
            "uncertainty": "Seed SD describes split sensitivity, not independent replication or a confidence interval",
            "prior_development": "All samples were previously used in exploratory model development; "
                                 "these are not newly collected external tests",
        },
        "model_family_summary": {row.model: {
            "mean_outer_roc_auc": float(row.mean_outer_roc_auc), "sd_outer_roc_auc": float(row.sd_outer_roc_auc),
            "mean_outer_mcc": float(row.mean_outer_mcc), "sd_outer_mcc": float(row.sd_outer_mcc)}
            for row in summary.itertuples(index=False)},
        "deployment_model": {
            "family": config["deployment_model"], "selection_seed": config["deployment_seed"],
            "artifact": f"models/{deployment_stem(config)}.joblib",
            "threshold": deployment["threshold"],
            "best_parameters": deployment["best_parameters"],
            "mean_inner_roc_auc": deployment["mean_inner_roc_auc"],
            "pooled_inner_oof_roc_auc": deployment["pooled_inner_oof_roc_auc"],
            "n_selected_features": len(deployment["selected_features"]),
        },
        "expected_performance": (
            f"Family-level nested estimate: ROC-AUC {family.mean_outer_roc_auc:.3f} ± {family.sd_outer_roc_auc:.3f}, "
            f"MCC {family.mean_outer_mcc:.3f} ± {family.sd_outer_mcc:.3f}; per-seed scores are not unbiased"),
        "top10_models": manifest_rows,
    }
    write_json(TABLES / "analysis_report.json", report)
    write_json(MODELS / "top10_manifest.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "ranking_source": "outputs/tables/model_comparison_by_seed.csv sorted by outer OOF ROC-AUC",
        "models": manifest_rows,
        "loading": ("joblib.load requires this environment; XGBoost artifacts additionally require "
                    "run_final_pipeline importable for the local tags adapter class"),
    })
    print(f"COMPLETE: outputs written to {OUTPUTS}", flush=True)


if __name__ == "__main__":
    main()
