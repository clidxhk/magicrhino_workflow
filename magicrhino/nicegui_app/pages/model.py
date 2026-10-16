"""Model page: train, evaluate, and manage saved model artifacts."""
from __future__ import annotations

import io
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve
from sklearn.model_selection import StratifiedKFold, learning_curve

from nicegui import ui

from ..state import page_header, panel, get_data_file, show_figure, DATA_DIR, MODEL_DIR
from utils.data_utils import load_data
from utils.model_utils import (
    evaluate_artifact_on_data,
    feature_importance_table,
    load_model_artifact,
    save_model_artifact,
    train_model_artifact,
    train_model_artifact_nested,
)
from utils.params_utils import MODEL_NAMES


METRIC_LABELS = [
    ("ROC AUC", "roc_auc"),
    ("PR AUC", "average_precision"),
    ("Balanced acc", "balanced_accuracy"),
    ("MCC", "mcc"),
    ("Accuracy", "accuracy"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("F1", "f1"),
]


def build_page():
    page_header(
        "🧠 Model Training & Evaluation",
        subtitle="Train complete preprocessing-and-model pipelines, evaluate saved artifacts, and manage model files.",
        quote="玉不琢，不成器；人不學，不知道。 ——《禮記·學記》",
    )

    if not get_data_file().exists():
        ui.label("Upload labeled data on the Data page first.").classes("text-negative")
        return

    data = pd.read_csv(get_data_file())
    try:
        X, y = load_data(data.copy())
    except Exception as exc:
        ui.notify(f"Could not prepare data: {exc}", type="negative")
        return

    target = np.asarray(y).ravel()
    positives = int((target == 1).sum()) if target.size else 0
    negatives = int((target == 0).sum()) if target.size else 0
    model_files = [p for p in MODEL_DIR.iterdir() if p.suffix.lower() in {".joblib", ".sav", ".h5"}]

    with ui.row().classes("w-full gap-3 flex-wrap"):
        _metric_card("Samples", f"{len(X):,}")
        _metric_card("Features", f"{X.shape[1]:,}")
        _metric_card("Positive / Negative", f"{positives:,} / {negatives:,}")
        _metric_card("Saved models", f"{len(model_files):,}")

    with panel("Workspace", width_full=True):
        ui.label("Train from parameter-search results, evaluate existing artifacts, or delete obsolete model files.").classes("text-grey-6")
        mode = ui.toggle(["Train", "Evaluate", "Delete"], value="Train").classes("w-full mt-2")

    container = ui.column().classes("w-full gap-4 mt-3")

    def on_mode(_=None):
        container.clear()
        if mode.value == "Train":
            _train_view(X, y, container)
        elif mode.value == "Evaluate":
            _evaluate_view(X, y, container)
        else:
            _delete_view(container)

    mode.on("update:model-value", on_mode)
    on_mode()


def _metric_card(label: str, value: str):
    with ui.element("div").classes("metric-card min-w-[180px] flex-1"):
        ui.html(f'<div class="label">{label}</div><div class="value">{value}</div>')


def _metric_cards(metrics, prefix=""):
    with ui.row().classes("gap-3 flex-wrap w-full"):
        for label, key in METRIC_LABELS:
            value = metrics.get(key)
            text = "—" if value is None else f"{prefix}{float(value):.3f}"
            _metric_card(label, text)


# ---- Train ---------------------------------------------------------------
def _train_view(X, y, container):
    search_dir = DATA_DIR / "parameter_search"
    saved_results = sorted(search_dir.glob("*_best_by_model.csv"), reverse=True)
    records_holder = {"records": None, "label": None}

    with container:
        with ui.row().classes("w-full gap-4 items-start"):
            with ui.element("div").classes("panel w-[420px] max-w-full"):
                ui.html('<div class="panel-title">Parameter source</div>')
                source = ui.toggle(["Saved search result", "Upload file"], value="Saved search result").classes("w-full")
                src_area = ui.column().classes("w-full gap-3 mt-3")

            model_area = ui.column().classes("flex-1 gap-4")

        def render_source(_=None):
            src_area.clear()
            model_area.clear()
            records_holder["records"] = None
            records_holder["label"] = None
            with src_area:
                if source.value == "Saved search result":
                    if not saved_results:
                        ui.label("No saved best-by-model file. Save one on the Params page first.").classes("text-grey-6")
                        return
                    options = [p.name for p in saved_results]
                    sel = ui.select(options, value=options[0], label="Search result").classes("w-full")

                    def load_selected(_=None):
                        selected_path = next(p for p in saved_results if p.name == sel.value)
                        records_holder["records"] = _parse_parameter_file(selected_path.read_bytes(), selected_path.name)
                        records_holder["label"] = selected_path.stem
                        _render_training_config(records_holder, model_area, X, y)

                    load_selected()
                    sel.on("update:model-value", load_selected)
                else:
                    ui.label("Upload a best-by-model CSV or a JSON parameter file.").classes("text-grey-6")
                    upload = ui.upload(label="Upload CSV/JSON", auto_upload=True, multiple=False).props("accept=.csv,.json").classes("w-full")

                    async def on_upload(e):
                        raw_bytes = await e.file.read()
                        records_holder["records"] = _parse_parameter_file(raw_bytes, e.file.name)
                        records_holder["label"] = Path(e.file.name).stem
                        _render_training_config(records_holder, model_area, X, y)

                    upload.on_upload(on_upload)

        source.on("update:model-value", render_source)
        render_source()


def _parse_parameter_file(raw_bytes, name):
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(io.BytesIO(raw_bytes))
        records = frame.to_dict("records")
        for record in records:
            if isinstance(record.get("best_parameters"), str):
                record["best_parameters"] = json.loads(record["best_parameters"])
        return records
    if suffix == ".json":
        payload = json.loads(raw_bytes.decode("utf-8-sig"))
        params = payload.get("best_parameters", payload)
        if isinstance(params, str):
            params = json.loads(params)
        return [{"model": payload.get("model"), "best_parameters": params, **payload}]
    raise ValueError("Only CSV and JSON parameter files are supported.")


def _render_training_config(holder, model_area, X, y):
    model_area.clear()
    records = holder.get("records") or []
    if not records:
        with model_area:
            ui.label("No parameter records loaded.").classes("text-grey-6")
        return

    available = [r.get("model") for r in records if r.get("model")]
    initial_model = available[0] if available else list(MODEL_NAMES)[0]

    with model_area:
        with ui.element("div").classes("panel w-full"):
            ui.html('<div class="panel-title">Training configuration</div>')
            with ui.row().classes("w-full gap-3 items-end"):
                model_name = ui.select(
                    available or list(MODEL_NAMES),
                    value=initial_model,
                    label="Model family",
                    with_input=True,
                ).classes("flex-1 min-w-[260px]")
                cv = ui.number("OOF CV folds", min=2, max=10, value=5).classes("w-36")
                seed = ui.number("Random seed", min=0, value=20260725).classes("w-44")
                workers = ui.number("Workers", min=1, max=16, value=1).classes("w-32")

            with ui.row().classes("w-full gap-3 items-end"):
                eval_mode = ui.toggle(
                    {"dev": "Development (fixed parameters)", "nested": "Nested validation"},
                    value="dev",
                    label="Evaluation mode",
                ).classes("flex-1 min-w-[300px]")
                inner = ui.number("Inner CV folds", min=2, max=10, value=5).classes("w-36").bind_visibility_from(
                    eval_mode, "value", value="nested")
                trials = ui.number("Inner search trials", min=1, max=100, value=6).classes("w-44").bind_visibility_from(
                    eval_mode, "value", value="nested")

            def auto_name():
                prefix = "nested_" if eval_mode.value == "nested" else ""
                return f"{prefix}{model_name.value}_{_safe_name(holder.get('label') or 'search')}_{datetime.now():%Y%m%d}"

            name_in = ui.input("Model artifact name", value=auto_name()).classes("w-full mt-3")
            holder["last_auto"] = name_in.value

            params_area = ui.column().classes("w-full mt-3")
            result_area = ui.column().classes("w-full gap-3 mt-4")

            def selected_record():
                return next((r for r in records if r.get("model") == model_name.value), records[0])

            def refresh_params(_=None):
                record = selected_record()
                params_area.clear()
                with params_area:
                    if eval_mode.value == "nested":
                        ui.label("Nested validation").classes("text-weight-medium")
                        ui.label("Hyperparameters and the MCC threshold are re-selected inside each outer training fold; "
                                 "pooled outer-test scores are the honest estimate. The loaded parameter file is not used.").classes("text-grey-6")
                    else:
                        ui.label("Best parameters").classes("text-weight-medium")
                        ui.code(json.dumps(record.get("best_parameters", {}), indent=2), language="json").classes("w-full")
                if not name_in.value or name_in.value == holder.get("last_auto"):
                    name_in.value = auto_name()
                    holder["last_auto"] = name_in.value

            def train():
                safe_name = _safe_name(name_in.value)
                if not safe_name:
                    ui.notify("Enter a valid artifact name", type="warning")
                    return
                record = selected_record()
                result_area.clear()
                with result_area:
                    with panel("Training", width_full=True):
                        ui.spinner("dots", size="lg", color="primary")
                        if eval_mode.value == "nested":
                            status = ui.label("Nested validation running...").classes("text-grey-6")

                            def report(done, total, message):
                                status.text = "Full-data selection..." if message == "full-data selection" else f"Nested validation: {message}"
                        else:
                            ui.label("Training model package and generating out-of-fold predictions...").classes("text-grey-6")
                ui.notify("Training…", type="ongoing")
                try:
                    if eval_mode.value == "nested":
                        artifact = train_model_artifact_nested(
                            model_name=model_name.value,
                            X=X,
                            y=y,
                            sample_ids=X.index,
                            outer_folds=int(cv.value),
                            inner_folds=int(inner.value),
                            search_trials=int(trials.value),
                            seed=int(seed.value),
                            workers=int(workers.value),
                            progress_callback=report,
                        )
                    else:
                        artifact = train_model_artifact(
                            model_name=model_name.value,
                            best_parameters=record.get("best_parameters", {}),
                            X=X,
                            y=y,
                            sample_ids=X.index,
                            cv_folds=int(cv.value),
                            seed=int(seed.value),
                            workers=int(workers.value),
                        )
                    path = save_model_artifact(artifact, MODEL_DIR / f"{safe_name}.joblib")
                except Exception as exc:
                    result_area.clear()
                    ui.notify(f"Training failed: {exc}", type="negative")
                    return
                result_area.clear()
                ui.notify(f"Saved: {path.name}", type="positive")
                with result_area:
                    with panel("Training completed", width_full=True):
                        ui.label(f"Saved artifact: {path.name}").classes("text-positive")
                        if artifact.get("evaluation") == "nested":
                            detail = artifact.get("nested", {})
                            ui.label(f"Deployment threshold (full-data inner OOF): {float(artifact['threshold']):.4f}").classes("text-grey-6")
                            ui.label("Outer-fold thresholds: " + ", ".join(f"{t:.4f}" for t in detail.get("fold_thresholds", []))).classes("text-caption text-slate-500")
                            ui.label(f"Full-data mean inner ROC-AUC: {float(detail.get('full_data_mean_inner_roc_auc', float('nan'))):.4f}").classes("text-caption text-slate-500")
                            _metric_cards(artifact.get("oof_metrics", {}), "Nested OOF ")
                        else:
                            ui.label(f"Decision threshold: {float(artifact.get('threshold', 0.5)):.4f}").classes("text-grey-6")
                            _metric_cards(artifact.get("oof_metrics", {}), "OOF ")

            model_name.on("update:model-value", refresh_params)
            eval_mode.on("update:model-value", refresh_params)
            refresh_params()
            ui.button("Train and save model package", on_click=train).props("color=primary icon=play_arrow").classes("mt-3 px-6")


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("._")


# ---- Evaluate ------------------------------------------------------------
def _evaluate_view(X, y, container):
    files = sorted(
        [p for p in MODEL_DIR.iterdir() if p.suffix.lower() in {".joblib", ".sav", ".h5"}],
        key=lambda p: p.name,
    )
    with container:
        if not files:
            with panel("Evaluate model", width_full=True):
                ui.label("No saved model files found.").classes("text-grey-6")
            return

        with ui.row().classes("w-full gap-4 items-start"):
            with ui.element("div").classes("panel w-[420px] max-w-full"):
                ui.html('<div class="panel-title">Evaluation controls</div>')
                sel = ui.select([p.name for p in files], value=files[0].name, label="Model package").classes("w-full")
                ui.label("Outputs to display").classes("text-caption text-slate-500 mt-3")
                checks = {}
                with ui.grid(columns=2).classes("w-full gap-1"):
                    for name in ("Metrics", "ROC & PR", "Learning curve", "Confusion", "Feature importance", "Predictions", "Metadata"):
                        checks[name] = ui.checkbox(name, value=(name in {"Metrics", "ROC & PR"}))
                start_btn = ui.button("Evaluate", on_click=None).props("color=primary icon=analytics").classes("w-full mt-3")

            area = ui.column().classes("flex-1 gap-4")

        def run(_=None):
            area.clear()
            try:
                artifact = load_model_artifact(next(p for p in files if p.name == sel.value))
            except Exception as exc:
                with area:
                    ui.notify(f"Could not load model: {exc}", type="negative")
                return
            _render_eval(artifact, X, y, area, checks)

        start_btn.on("click", run)
        sel.on("update:model-value", lambda _: area.clear())


def _render_eval(artifact, X, y, area, checks):
    is_oof = bool(artifact.get("oof_probability"))
    if is_oof:
        target = np.asarray(artifact["oof_target"], dtype=int)
        prob = np.asarray(artifact["oof_probability"], dtype=float)
        pred = np.asarray(artifact["oof_prediction"], dtype=int)
        metrics = artifact.get("oof_metrics", {})
    else:
        try:
            metrics, prob = evaluate_artifact_on_data(artifact, X, y)
        except Exception as exc:
            with area:
                ui.notify(f"Could not evaluate: {exc}", type="negative")
            return
        target = np.asarray(y).reshape(-1)
        threshold = float(artifact.get("threshold", 0.5))
        pred = (prob >= threshold).astype(int)

    with area:
        with panel(f"{artifact.get('model_name', 'Model')} evaluation", width_full=True):
            if artifact.get("evaluation") == "nested":
                ui.label("Nested validation: pooled outer-test predictions with per-fold thresholds.").classes("text-grey-6")
            else:
                ui.label("Using stored OOF predictions." if is_oof else "Evaluated on current labeled dataset.").classes("text-grey-6")
            if checks["Metrics"].value:
                _metric_cards(metrics, "OOF " if is_oof else "")
                ui.label(f"Decision threshold: {float(artifact.get('threshold', 0.5)):.4f}").classes("text-caption text-slate-500")
                if "threshold" not in artifact and not is_oof:
                    ui.label("This artifact stores no threshold; decisions fall back to 0.5. Package it with its real MCC threshold before trusting class decisions.").classes("text-warning")

        tabs = []
        if checks["ROC & PR"].value:
            tabs.append(("ROC & PR", lambda: _roc_pr(target, prob)))
        if checks["Learning curve"].value:
            tabs.append(("Learning curve", lambda: _learning_curve(artifact, X, y)))
        if checks["Confusion"].value:
            tabs.append(("Confusion", lambda: _confusion(target, pred)))
        if checks["Feature importance"].value:
            tabs.append(("Feature importance", lambda: _feat_importance(artifact)))
        if checks["Predictions"].value:
            tabs.append(("Predictions", lambda: _predictions(artifact, target, prob, pred)))
        if checks["Metadata"].value:
            tabs.append(("Metadata", lambda: _render_meta(artifact)))

        if not tabs:
            ui.label("Select at least one output to display.").classes("text-warning")
            return

        with panel("Evaluation outputs", width_full=True):
            with ui.tabs().classes("w-full") as tab_bar:
                tab_objs = [ui.tab(name) for name, _ in tabs]
            with ui.tab_panels(tab_bar, value=tab_objs[0]).classes("w-full"):
                for (_, render_fn), tab in zip(tabs, tab_objs):
                    with ui.tab_panel(tab):
                        render_fn()


def _roc_pr(target, probability):
    target = np.asarray(target).reshape(-1)
    probability = np.asarray(probability, dtype=float).reshape(-1)
    fpr, tpr, _ = roc_curve(target, probability)
    precision, recall, _ = precision_recall_curve(target, probability)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#1f77b4", linewidth=2)
    axes[0].plot([0, 1], [0, 1], "--", color="#94a3b8")
    axes[0].set_title("ROC curve")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].grid(alpha=0.2)

    axes[1].plot(recall, precision, color="#008BFB", linewidth=2)
    axes[1].set_title("Precision-Recall curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    show_figure(fig)


def _learning_curve(artifact, X, y):
    features = pd.DataFrame(X).loc[:, artifact.get("feature_columns", list(X.columns))]
    target = np.asarray(y).reshape(-1).astype(int)
    _, counts = np.unique(target, return_counts=True)
    folds = min(int(artifact.get("cv_folds", 5)), int(counts.min())) if len(counts) == 2 else 0
    if folds < 2:
        ui.label("Learning curve requires at least two samples in each class.").classes("text-grey-6")
        return

    try:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(artifact.get("seed", 20260725)))
        sizes, train_scores, validation_scores = learning_curve(
            artifact["model"],
            features,
            target,
            cv=cv,
            scoring="roc_auc",
            train_sizes=np.linspace(0.4, 1.0, 5),
            n_jobs=1,
        )
    except Exception as exc:
        ui.label(f"Could not calculate learning curve: {exc}").classes("text-negative")
        return

    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    validation_mean, validation_std = validation_scores.mean(axis=1), validation_scores.std(axis=1)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(sizes, train_mean, marker="o", linewidth=2, color="#1f77b4", label="Training ROC-AUC")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, color="#1f77b4", alpha=0.15)
    ax.plot(sizes, validation_mean, marker="o", linewidth=2, color="#008BFB", label="Validation ROC-AUC")
    ax.fill_between(sizes, validation_mean - validation_std, validation_mean + validation_std, color="#008BFB", alpha=0.15)
    ax.set_title(f"Learning curve ({folds}-fold stratified CV)")
    ax.set_xlabel("Training samples")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    show_figure(fig)


def _confusion(target, prediction):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ConfusionMatrixDisplay.from_predictions(target, prediction, labels=[0, 1], cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    show_figure(fig)


def _feat_importance(artifact):
    table = feature_importance_table(artifact)
    if table.empty:
        ui.label("Feature importance is unavailable for this model.").classes("text-grey-6")
        return
    top = table.head(30).copy()
    fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.25)))
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#1f77b4")
    ax.set_title("Top feature importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    show_figure(fig)
    ui.table.from_pandas(table.head(100).round(6)).props("flat dense").classes("w-full").style("max-height:420px; overflow:auto;")


def _predictions(artifact, target, probability, prediction):
    sample_ids = artifact.get("sample_ids") or list(range(len(target)))
    df = pd.DataFrame({
        "Sample": sample_ids[:len(target)],
        "Target": target,
        "Probability": probability,
        "Prediction": prediction,
    })
    df["Result"] = np.where(df["Target"] == df["Prediction"], "Correct", "Incorrect")

    result_options = {
        "All predictions": df,
        "Correct predictions": df[df["Result"] == "Correct"],
        "Incorrect predictions": df[df["Result"] == "Incorrect"],
    }
    filter_toggle = ui.toggle(list(result_options), value="All predictions").classes("w-full")
    table_container = ui.column().classes("w-full")

    def render_table():
        table_container.clear()
        filtered = result_options[filter_toggle.value]
        with table_container:
            ui.label(f"{len(filtered)} predictions").classes("text-caption text-slate-500")
            ui.table.from_pandas(filtered.round(6)).props("flat dense").classes("w-full").style(
                "max-height:620px; overflow:auto;"
            )

    filter_toggle.on("update:model-value", lambda _: render_table())
    render_table()
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    ui.button("Download predictions", on_click=lambda: ui.download(buf.getvalue(), filename="model_predictions.csv")).props("flat color=primary icon=download")


def _render_meta(artifact):
    meta = {k: v for k, v in artifact.items()
            if k not in {"model", "oof_probability", "oof_prediction", "oof_target", "sample_ids"}}
    ui.code(json.dumps(meta, indent=2, default=str), language="json").classes("w-full")


# ---- Delete --------------------------------------------------------------
def _delete_view(container):
    files = sorted(
        [p for p in MODEL_DIR.iterdir() if p.suffix.lower() in {".joblib", ".sav", ".h5"}],
        key=lambda p: p.name,
    )
    with container:
        if not files:
            with panel("Delete saved models", width_full=True):
                ui.label("No saved model files.").classes("text-grey-6")
            return
        with panel("Delete saved models", width_full=True):
            ui.label("Select obsolete model artifacts to remove.").classes("text-grey-6")
            sel = ui.select({p.name: p.name for p in files}, multiple=True, label="Select files").classes("w-full mt-2")
            confirm = ui.checkbox("I understand these files will be deleted").classes("mt-2")

            def do_delete():
                selected = list(sel.value or [])
                if not selected:
                    ui.notify("Select at least one file", type="warning")
                    return
                for name in selected:
                    (MODEL_DIR / name).unlink()
                ui.notify(f"Deleted {len(selected)} file(s)", type="positive")
                container.clear()
                _delete_view(container)

            ui.button("Delete selected", on_click=do_delete).props("color=negative icon=delete").classes("mt-3").bind_visibility_from(confirm, "value")
