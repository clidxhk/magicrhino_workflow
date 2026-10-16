# MagicRhino project reference

## Files and runtime

- Work from the repository root with `conda activate magicrhino`.
- Start the optional UI with `python app_nicegui.py`.
- Core helpers live in `utils/model_utils.py`, `utils/data_utils.py`, and `utils/shap_model_utils.py`.
- NiceGUI page implementations are in `nicegui_app/pages/`.

## Dataset contracts

| Operation | Required columns |
| --- | --- |
| Train / evaluate | `Index`, `Y`, and feature columns; `Y` must be binary |
| Predict | `Index` is recommended; every model `feature_columns` entry is required |

Use `utils.data_utils.load_data(df)` or `utils.model_utils.load_data(df)` for labelled data. These set `Index` as the frame index and separate `Y` from the feature matrix.

Check duplicated or missing `Index` values, missing features, target distribution, non-numeric/unexpected feature values, and enough samples per class for the requested CV folds. Never reorder model features manually: select them in the artifact's stored order.

## Artifact format

`saved_model/*.joblib` is normally a dictionary produced by `train_model_artifact`:

- `model`: fitted scikit-learn pipeline
- `model_name`, `seed`, `cv_folds`, `best_parameters`
- `threshold`: MCC-selected decision threshold
- `feature_columns`, `label_column` (`Y`), `index_column` (`Index`)
- `oof_metrics`, `oof_probability`, `oof_prediction`, `oof_target`, `sample_ids`
- `created_at`, `artifact_version`

Use `load_model_artifact(path)` and `save_model_artifact(artifact, path)`. Legacy `.joblib`, `.sav`, and `.h5` formats may expose only a fitted model and partial metadata; report those limitations.

## Training and evaluation

```python
from utils.model_utils import (
    train_model_artifact, save_model_artifact,
    evaluate_artifact_on_data, load_model_artifact,
)

artifact = train_model_artifact(
    model_name, best_parameters, X, y,
    sample_ids=X.index, cv_folds=5, seed=20250725, workers=1,
)
save_model_artifact(artifact, output_path)
metrics, probability = evaluate_artifact_on_data(artifact, X_eval, y_eval)
```

Training requires both classes and at least `cv_folds` samples in the smaller class. The stored `oof_metrics` are cross-validated estimates; `evaluate_artifact_on_data` is an external/hold-out evaluation and applies the artifact's `threshold`. Use `feature_importance_table(artifact)` only when it returns non-empty rows.

For predictions, select `X_new.loc[:, artifact["feature_columns"]]`, obtain `model.predict_proba(...)[:, 1]`, and compute `prediction = probability >= artifact.get("threshold", 0.5)`. Export an output table containing `Index`, `probability`, `prediction`, and the threshold used.

## SHAP

```python
from utils.shap_model_utils import (
    prepare_shap_inputs, make_probability_explainer,
    positive_class_array, positive_class_base_value,
)

estimator, X_shap = prepare_shap_inputs(artifact, X)
background = X_shap.sample(n=min(100, len(X_shap)), random_state=artifact.get("seed", 20250725))
explainer = make_probability_explainer(estimator, background)
values = explainer(X_shap_sample)
positive_values = positive_class_array(values)
```

Use a bounded sample and report it. Show only user-selected plot outputs. `max_display` controls the number of displayed features: always state its value. Use the positive class for binary-classification plots. For a single-observation force plot, use `positive_class_base_value`; do not attempt to render a static multi-observation force plot as an equivalent overall figure. For cluster colouring, allow either a selected feature's SHAP value or total `|SHAP|` across all features.
