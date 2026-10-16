"""Publication-style randomized cross-validation parameter search."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

MODEL_NAMES = (
    "LogisticRegression", "SGDLogistic", "ExtraTrees", "RandomForest",
    "HistGradientBoosting", "LightGBM", "XGBoost", "CatBoost",
    "GaussianNB", "KNN",
)
SCORING = {
    "roc_auc": "roc_auc", "average_precision": "average_precision",
    "balanced_accuracy": "balanced_accuracy", "mcc": "matthews_corrcoef",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Pipeline
    parameters: dict


def feature_count_options(n_features):
    if n_features < 1:
        raise ValueError("The dataset must contain at least one feature column.")
    return [k for k in (20, 50, 100, 200, 400) if k <= n_features] + ["all"]


def make_pipeline(model, counts, scaled=False):
    steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold()),
        ("select", SelectKBest(score_func=f_classif, k=counts[0])),
    ]
    if scaled:
        steps.append(("scale", RobustScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def model_specs(n_features, seed, selected_models=None):
    """Build only selected models so optional native libraries load on demand."""
    selected = set(selected_models or MODEL_NAMES)
    counts = feature_count_options(n_features)
    common = {"select__k": counts}
    specs = []
    if "LogisticRegression" in selected:
        specs.append(ModelSpec(
            "LogisticRegression",
            make_pipeline(LogisticRegression(solver="liblinear", max_iter=5000, random_state=seed), counts, True),
            common | {"model__C": loguniform(1e-3, 10), "model__penalty": ["l1", "l2"], "model__class_weight": [None, "balanced"]},
        ))
    if "SGDLogistic" in selected:
        specs.append(ModelSpec(
            "SGDLogistic",
            make_pipeline(SGDClassifier(loss="log_loss", max_iter=5000, tol=1e-4, early_stopping=True, validation_fraction=0.15, random_state=seed), counts, True),
            common | {"model__alpha": loguniform(1e-6, 1e-1), "model__penalty": ["l1", "l2", "elasticnet"], "model__l1_ratio": uniform(0.05, 0.9), "model__class_weight": [None, "balanced"]},
        ))
    if "ExtraTrees" in selected:
        specs.append(ModelSpec(
            "ExtraTrees",
            make_pipeline(ExtraTreesClassifier(n_estimators=250, n_jobs=1, random_state=seed), counts),
            common | {"model__max_depth": [None, 3, 5, 8, 12, 18], "model__min_samples_leaf": randint(1, 12), "model__max_features": ["sqrt", "log2", 0.2, 0.4, 0.7, 1.0], "model__class_weight": [None, "balanced", "balanced_subsample"]},
        ))
    if "RandomForest" in selected:
        specs.append(ModelSpec(
            "RandomForest",
            make_pipeline(RandomForestClassifier(n_estimators=250, n_jobs=1, random_state=seed), counts),
            common | {"model__max_depth": [None, 3, 5, 8, 12, 18], "model__min_samples_leaf": randint(1, 12), "model__max_features": ["sqrt", "log2", 0.2, 0.4, 0.7], "model__class_weight": [None, "balanced", "balanced_subsample"]},
        ))
    if "HistGradientBoosting" in selected:
        specs.append(ModelSpec(
            "HistGradientBoosting",
            make_pipeline(HistGradientBoostingClassifier(early_stopping=True, validation_fraction=0.15, random_state=seed), counts),
            common | {"model__learning_rate": loguniform(0.015, 0.25), "model__max_iter": randint(60, 260), "model__max_leaf_nodes": randint(5, 40), "model__min_samples_leaf": randint(5, 35), "model__l2_regularization": loguniform(1e-3, 30)},
        ))
    if "LightGBM" in selected:
        from lightgbm import LGBMClassifier
        specs.append(ModelSpec(
            "LightGBM",
            make_pipeline(LGBMClassifier(objective="binary", n_jobs=1, verbosity=-1, random_state=seed), counts),
            common | {"model__n_estimators": randint(60, 280), "model__learning_rate": loguniform(0.01, 0.2), "model__num_leaves": randint(5, 40), "model__max_depth": [-1, 3, 5, 7, 10], "model__min_child_samples": randint(5, 45), "model__subsample": uniform(0.65, 0.35), "model__colsample_bytree": uniform(0.35, 0.65), "model__reg_alpha": loguniform(1e-4, 10), "model__reg_lambda": loguniform(1e-4, 30), "model__class_weight": [None, "balanced"]},
        ))
    if "XGBoost" in selected:
        from xgboost import XGBClassifier
        specs.append(ModelSpec(
            "XGBoost",
            make_pipeline(XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist", n_jobs=1, random_state=seed), counts),
            common | {"model__n_estimators": randint(60, 260), "model__learning_rate": loguniform(0.01, 0.2), "model__max_depth": randint(2, 8), "model__min_child_weight": loguniform(0.3, 20), "model__subsample": uniform(0.65, 0.35), "model__colsample_bytree": uniform(0.35, 0.65), "model__reg_alpha": loguniform(1e-4, 10), "model__reg_lambda": loguniform(1e-3, 30)},
        ))
    if "CatBoost" in selected:
        from catboost import CatBoostClassifier
        specs.append(ModelSpec(
            "CatBoost",
            make_pipeline(CatBoostClassifier(verbose=False, allow_writing_files=False, thread_count=1, random_seed=seed, loss_function="Logloss"), counts),
            common | {"model__iterations": randint(80, 260), "model__learning_rate": loguniform(0.01, 0.2), "model__depth": randint(3, 8), "model__l2_leaf_reg": loguniform(0.2, 30), "model__random_strength": loguniform(1e-3, 5), "model__auto_class_weights": ["None", "Balanced"]},
        ))
    if "GaussianNB" in selected:
        specs.append(ModelSpec(
            "GaussianNB",
            make_pipeline(GaussianNB(), counts, True),
            common | {"model__var_smoothing": loguniform(1e-12, 1e-2)},
        ))
    if "KNN" in selected:
        specs.append(ModelSpec(
            "KNN",
            make_pipeline(KNeighborsClassifier(n_jobs=1), counts, True),
            common | {
                "model__n_neighbors": randint(3, 40),
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],
                "model__leaf_size": randint(15, 60),
            },
        ))
    return specs


SPACE_TEXT = {
    "LogisticRegression": {"model__C": "log-uniform [0.001, 10]", "model__penalty": "[l1, l2]", "model__class_weight": "[None, balanced]"},
    "SGDLogistic": {"model__alpha": "log-uniform [1e-6, 0.1]", "model__penalty": "[l1, l2, elasticnet]", "model__l1_ratio": "uniform [0.05, 0.95]", "model__class_weight": "[None, balanced]"},
    "ExtraTrees": {"model__max_depth": "[None, 3, 5, 8, 12, 18]", "model__min_samples_leaf": "integer [1, 11]", "model__max_features": "[sqrt, log2, 0.2, 0.4, 0.7, 1.0]", "model__class_weight": "[None, balanced, balanced_subsample]"},
    "RandomForest": {"model__max_depth": "[None, 3, 5, 8, 12, 18]", "model__min_samples_leaf": "integer [1, 11]", "model__max_features": "[sqrt, log2, 0.2, 0.4, 0.7]", "model__class_weight": "[None, balanced, balanced_subsample]"},
    "HistGradientBoosting": {"model__learning_rate": "log-uniform [0.015, 0.25]", "model__max_iter": "integer [60, 259]", "model__max_leaf_nodes": "integer [5, 39]", "model__min_samples_leaf": "integer [5, 34]", "model__l2_regularization": "log-uniform [0.001, 30]"},
    "LightGBM": {"model__n_estimators": "integer [60, 279]", "model__learning_rate": "log-uniform [0.01, 0.2]", "model__num_leaves": "integer [5, 39]", "model__max_depth": "[-1, 3, 5, 7, 10]", "model__min_child_samples": "integer [5, 44]", "model__subsample": "uniform [0.65, 1.0]", "model__colsample_bytree": "uniform [0.35, 1.0]", "model__reg_alpha": "log-uniform [1e-4, 10]", "model__reg_lambda": "log-uniform [1e-4, 30]", "model__class_weight": "[None, balanced]"},
    "XGBoost": {"model__n_estimators": "integer [60, 259]", "model__learning_rate": "log-uniform [0.01, 0.2]", "model__max_depth": "integer [2, 7]", "model__min_child_weight": "log-uniform [0.3, 20]", "model__subsample": "uniform [0.65, 1.0]", "model__colsample_bytree": "uniform [0.35, 1.0]", "model__reg_alpha": "log-uniform [1e-4, 10]", "model__reg_lambda": "log-uniform [0.001, 30]"},
    "CatBoost": {"model__iterations": "integer [80, 259]", "model__learning_rate": "log-uniform [0.01, 0.2]", "model__depth": "integer [3, 7]", "model__l2_leaf_reg": "log-uniform [0.2, 30]", "model__random_strength": "log-uniform [0.001, 5]", "model__auto_class_weights": "[None, Balanced]"},
    "GaussianNB": {"model__var_smoothing": "log-uniform [1e-12, 0.01]"},
    "KNN": {"model__n_neighbors": "integer [3, 39]", "model__weights": "[uniform, distance]", "model__p": "[1, 2]", "model__leaf_size": "integer [15, 59]"},
}


def search_space_records(n_features):
    spaces = {"All models": {"select__k": str(feature_count_options(n_features))}} | SPACE_TEXT
    return [{"model": model, "parameter": parameter, "search_distribution": definition} for model, parameters in spaces.items() for parameter, definition in parameters.items()]


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    return None if pd.isna(value) else value


def _validated_data(X, y, cv_folds):
    features = pd.DataFrame(X).copy()
    target = np.asarray(y).reshape(-1)
    if len(features) != len(target):
        raise ValueError("Feature and target row counts do not match.")
    if features.empty:
        raise ValueError("The dataset contains no rows or feature columns.")
    invalid = features.select_dtypes(exclude=[np.number]).columns.tolist()
    if invalid:
        raise ValueError("All features must be numeric. Invalid columns: " + ", ".join(map(str, invalid[:5])))
    if np.isinf(features.to_numpy(dtype=float)).any():
        raise ValueError("Feature data contains infinite values.")
    classes, counts = np.unique(target, return_counts=True)
    if len(classes) != 2:
        raise ValueError("This workflow requires a binary target.")
    if counts.min() < cv_folds:
        raise ValueError(f"Each class needs at least {cv_folds} rows; the smallest has {counts.min()}.")
    return features, target


def run_parameter_search(X, y, selected_models, n_trials=20, cv_folds=5, seed=20260725, workers=1, progress_callback=None):
    if n_trials < 1 or workers < 1 or cv_folds < 2:
        raise ValueError("Trials/workers must be >= 1 and CV folds must be >= 2.")
    features, target = _validated_data(X, y, cv_folds)
    requested = list(dict.fromkeys(selected_models))
    if not requested:
        raise ValueError("Select at least one model family.")
    unknown = sorted(set(requested) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model families: {unknown}")
    specs = model_specs(features.shape[1], seed, requested)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    all_candidates, best_rows = [], []
    for number, spec in enumerate(specs, 1):
        started = time.perf_counter()
        search = RandomizedSearchCV(
            spec.estimator, spec.parameters, n_iter=n_trials, scoring=SCORING,
            refit="roc_auc", cv=cv, random_state=seed, n_jobs=workers,
            pre_dispatch=workers, error_score=np.nan, return_train_score=False,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message="Features .* are constant")
            search.fit(features, target)
        result = pd.DataFrame(search.cv_results_)
        if result["mean_test_roc_auc"].isna().all():
            raise RuntimeError(f"Every {spec.name} candidate failed.")
        for index, row in result.iterrows():
            params = {key: _json_value(value) for key, value in row["params"].items()}
            record = {
                "model": spec.name, "candidate": index + 1,
                "rank_roc_auc": _json_value(row["rank_test_roc_auc"]),
                "mean_cv_roc_auc": row["mean_test_roc_auc"], "sd_cv_roc_auc": row["std_test_roc_auc"],
                "mean_cv_pr_auc": row["mean_test_average_precision"], "sd_cv_pr_auc": row["std_test_average_precision"],
                "mean_cv_balanced_accuracy": row["mean_test_balanced_accuracy"], "sd_cv_balanced_accuracy": row["std_test_balanced_accuracy"],
                "mean_cv_mcc": row["mean_test_mcc"], "sd_cv_mcc": row["std_test_mcc"],
                "mean_fit_time_seconds": row["mean_fit_time"],
                "parameters": json.dumps(params, ensure_ascii=False, sort_keys=True),
            }
            record.update({f"param_{key}": value for key, value in params.items()})
            all_candidates.append(record)
        best_index = search.best_index_
        best_params = {key: _json_value(value) for key, value in search.best_params_.items()}
        best_rows.append({
            "model": spec.name, "mean_cv_roc_auc": result.loc[best_index, "mean_test_roc_auc"],
            "sd_cv_roc_auc": result.loc[best_index, "std_test_roc_auc"],
            "mean_cv_pr_auc": result.loc[best_index, "mean_test_average_precision"],
            "mean_cv_balanced_accuracy": result.loc[best_index, "mean_test_balanced_accuracy"],
            "mean_cv_mcc": result.loc[best_index, "mean_test_mcc"],
            "best_parameters": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
            "search_trials": n_trials, "cv_folds": cv_folds, "seed": seed,
            "runtime_seconds": time.perf_counter() - started,
        })
        if progress_callback:
            progress_callback(number, len(specs), spec.name)
    candidates = pd.DataFrame(all_candidates).sort_values(["model", "rank_roc_auc", "candidate"], na_position="last")
    best = pd.DataFrame(best_rows).sort_values("mean_cv_roc_auc", ascending=False)
    return candidates.reset_index(drop=True), best.reset_index(drop=True), pd.DataFrame(search_space_records(features.shape[1]))


def save_search_results(output_dir, candidates, best, spaces, metadata):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / f"parameter_search_{time.strftime('%Y%m%d-%H%M%S')}"
    paths = [prefix.with_name(prefix.name + suffix) for suffix in ("_all_candidates.csv", "_best_by_model.csv", "_search_space.csv", "_metadata.json")]
    candidates.to_csv(paths[0], index=False)
    best.to_csv(paths[1], index=False)
    spaces.to_csv(paths[2], index=False)
    paths[3].write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths
