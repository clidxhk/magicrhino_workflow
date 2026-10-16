"""Adapters that expose fitted pipeline estimators and transformed SHAP data."""

from __future__ import annotations

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "CatBoost": "CatBoost",
    "ExtraTrees": "Random Forest",
    "HistGradientBoosting": "Random Forest",
    "LightGBM": "LightGBM",
    "LogisticRegression": "MLPClassifier",
    "RandomForest": "Random Forest",
    "SGDLogistic": "MLPClassifier",
    "XGBoost": "XGBoost",
}

TREE_PROBABILITY_ESTIMATORS = {
    "RandomForestClassifier",
    "ExtraTreesClassifier",
}


class TreeProbabilityExplainer:
    """Expose TreeExplainer through the interface used by this SHAP page."""

    def __init__(self, model, background):
        import shap

        self._explainer = shap.TreeExplainer(model, background)

    def __call__(self, X, **kwargs):
        kwargs.setdefault("check_additivity", False)
        return self._explainer(X, **kwargs)

    def shap_values(self, X, **kwargs):
        kwargs.setdefault("check_additivity", False)
        return self._explainer.shap_values(X, **kwargs)


class PermutationProbabilityExplainer:
    """Permutation explainer with an evaluation budget sized to the feature count.

    shap's default max_evals=500 cannot explain pipelines that keep hundreds of
    selected features (Permutation needs at least 2 * n_features + 1 evals).
    """

    def __init__(self, model, background):
        import shap

        self._explainer = shap.Explainer(model.predict_proba, background)

    def __call__(self, X, **kwargs):
        n_features = np.shape(X)[1]
        kwargs.setdefault("max_evals", max(500, 2 * n_features + 1))
        return self._explainer(X, **kwargs)

    def shap_values(self, X, **kwargs):
        # shap's legacy shap_values() accepts no max_evals budget, so route
        # through __call__ and return an Explanation; positive_class_array()
        # already understands Explanation objects.
        return self(X, **kwargs)


def make_probability_explainer(model, background):
    """Prefer the efficient exact tree explainer for supported classifiers."""
    if type(model).__name__ in TREE_PROBABILITY_ESTIMATORS:
        return TreeProbabilityExplainer(model, background)

    return PermutationProbabilityExplainer(model, background)


def positive_class_array(values):
    """Return a two-dimensional positive-class SHAP value array."""
    if hasattr(values, "values"):
        values = values.values
    if isinstance(values, list):
        return np.asarray(values[-1])
    values = np.asarray(values)
    if values.ndim == 3:
        class_index = 1 if values.shape[2] > 1 else 0
        return values[:, :, class_index]
    return values


def positive_class_base_value(base_values):
    """Return a scalar base value for the positive class of one explanation."""
    values = np.asarray(base_values)
    if values.ndim == 0:
        return float(values)
    if values.ndim > 1:
        values = values[0]
    values = values.reshape(-1)
    if not len(values):
        raise ValueError("SHAP returned an empty base value.")
    class_index = 1 if len(values) > 1 else 0
    return float(values[class_index])


def selected_feature_contribution(values, feature_names, selected_features):
    """Calculate selected features' share of total mean absolute SHAP importance."""
    shap_array = positive_class_array(values)
    if shap_array.ndim != 2:
        raise ValueError(f"Expected a 2D SHAP array, received shape {shap_array.shape}")

    feature_names = list(feature_names)
    selected_features = list(selected_features)
    if shap_array.shape[1] != len(feature_names):
        raise ValueError(
            "SHAP feature count does not match the number of feature names: "
            f"{shap_array.shape[1]} != {len(feature_names)}"
        )

    missing = [name for name in selected_features if name not in feature_names]
    if missing:
        raise ValueError(f"Selected SHAP features were not found: {', '.join(map(str, missing[:5]))}")

    mean_absolute = np.mean(np.abs(shap_array), axis=0)
    total_importance = float(mean_absolute.sum())
    selected_indices = [feature_names.index(name) for name in selected_features]
    selected_importance = mean_absolute[selected_indices]
    selected_total = float(selected_importance.sum())

    overall_percent = (
        selected_importance / total_importance * 100 if total_importance else np.zeros_like(selected_importance)
    )
    within_selected_percent = (
        selected_importance / selected_total * 100 if selected_total else np.zeros_like(selected_importance)
    )
    detail = pd.DataFrame({
        "Feature": selected_features,
        "Mean |SHAP|": selected_importance,
        "Overall contribution (%)": overall_percent,
        "Within selection (%)": within_selected_percent,
    }).sort_values("Mean |SHAP|", ascending=False, ignore_index=True)

    contribution_percent = selected_total / total_importance * 100 if total_importance else 0.0
    return contribution_percent, detail


def infer_shap_model_label(artifact):
    """Map model-package family names to the existing SHAP page branches."""
    name = artifact.get("model_name", type(artifact["model"]).__name__)
    return MODEL_LABELS.get(name, name)


def prepare_shap_inputs(artifact, X):
    """Return the final estimator and fold-safe transformed feature frame."""
    model = artifact["model"]
    if not hasattr(model, "named_steps") or "model" not in model.named_steps:
        return model, pd.DataFrame(X).copy()

    expected = artifact.get("feature_columns")
    frame = pd.DataFrame(X).copy()
    if expected:
        missing = [column for column in expected if column not in frame.columns]
        if missing:
            preview = ", ".join(map(str, missing[:5]))
            raise ValueError(f"Dataset is missing model features: {preview}")
        frame = frame.loc[:, expected]

    names = np.asarray(frame.columns, dtype=object)
    values = frame
    for step_name in ("impute", "variance", "select", "scale"):
        transformer = model.named_steps.get(step_name)
        if transformer is None:
            continue
        values = transformer.transform(values)
        if step_name in {"variance", "select"} and hasattr(transformer, "get_support"):
            names = names[transformer.get_support()]

    transformed = pd.DataFrame(values, index=frame.index, columns=names)
    return model.named_steps["model"], transformed

