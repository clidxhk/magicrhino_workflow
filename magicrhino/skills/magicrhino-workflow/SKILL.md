---
name: magicrhino-workflow
description: Operate the MagicRhino binary-classification project directly from natural-language requests. Use for inspecting CSV/XLSX data, checking saved .joblib model artifacts, training or evaluating a MagicRhino model, predicting a new dataset, producing feature importance or SHAP explanations, or managing project model/data outputs. Trigger when the user mentions MagicRhino, its saved models, `Index`/`Y` datasets, model training/evaluation, prediction, or SHAP visualization.
---

# MagicRhino Workflow

Operate this repository's ML workflow without requiring the NiceGUI interface. Work from the project root and use the `magicrhino` conda environment. Read [project-workflow.md](references/project-workflow.md) before a task that needs file formats, helper signatures, or output conventions.

## Choose the operation

1. **Inspect data or model** — safely report schema, class balance, missing values, duplicate IDs, artifact metadata, CV metrics, decision threshold, and feature importance when available.
2. **Train** — validate the labelled dataset, inspect available model families/parameter results, then confirm material choices before creating or replacing an artifact.
3. **Evaluate** — evaluate a saved artifact against a labelled compatible dataset using its stored threshold; report ROC-AUC, PR-AUC, MCC, balanced accuracy, confusion-matrix counts, sensitivity, and specificity.
4. **Predict** — verify feature compatibility before producing probabilities and thresholded labels. Preserve the input `Index` in exported results.
5. **Explain** — generate selected SHAP plots or tables on a bounded representative sample. Reuse the artifact's stored `seed`; do not invent a second seed unless the user asks for one.

## Safety and data integrity

- Treat input data as read-only. Write derived outputs to a clearly named new file unless the user explicitly authorizes overwriting an exact path.
- Before training, ask for confirmation when model family, parameter file/search outcome, CV folds, seed, output name, or overwrite target is not already specified. State the proposed choices succinctly.
- Before deletion, overwrite, model replacement, external data collection, or external upload/send, identify the exact target and ask for confirmation.
- Validate all schemas before a model operation. A labelled training/evaluation dataset requires `Index` and binary `Y`; prediction data must contain all artifact `feature_columns` with compatible names.
- Do not claim a result was produced without running the relevant operation and checking the generated file or returned metrics.

## Use project utilities

Prefer `utils/model_utils.py`, `utils/data_utils.py`, and `utils/shap_model_utils.py` over reimplementing training, persistence, thresholding, or pipeline transformation. Load artifacts with `load_model_artifact`; preserve their dictionary format and metadata. Use `train_model_artifact` for training, `evaluate_artifact_on_data` for evaluation, and `prepare_shap_inputs` plus the probability explainer helpers for SHAP.

## Explain results clearly

State the input/model paths, sample counts, and whether metrics are out-of-fold or hold-out evaluation. For SHAP, name the chosen plot types, number of displayed features, sample size, and positive-class interpretation. Note constraints rather than silently falling back: e.g. a multi-sample force plot requires an interactive JavaScript renderer and is not a static image output.

## UI relationship

The NiceGUI application remains available through `app_nicegui.py`; its pages are a reference for the same workflows, not a prerequisite. Use direct utility calls when they make the request faster, reproducible, and verifiable.
