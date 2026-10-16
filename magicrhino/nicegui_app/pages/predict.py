"""Predict page: upload external CSV, predict with a saved model, optional SHAP."""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from nicegui import ui

from ..state import page_header, panel, get_data_file, show_figure, MODEL_DIR
from utils.model_utils import load_data, load_model_artifact
from utils.shap_model_utils import (
    make_probability_explainer, positive_class_array,
    positive_class_base_value, prepare_shap_inputs,
)
from utils.shap_waterfall import waterfall_selected_other


def build_page():
    page_header(
        "🚀 Model Prediction",
        subtitle="Upload an external dataset to predict with the saved model and decision threshold.",
        quote="前事不忘，後事之師。 ——《戰國策·趙策一》",
    )

    if not get_data_file().exists():
        ui.label("Reference dataset not found. Upload data on the Data page first.").classes("text-negative")
        return

    X_reference = load_data(pd.read_csv(get_data_file()))[0]
    model_files = sorted(
        [p.name for p in MODEL_DIR.iterdir() if p.suffix.lower() in {".joblib", ".sav", ".h5"}],
    )
    if not model_files:
        ui.label("No saved models found.").classes("text-negative")
        return

    with ui.row().classes("w-full gap-3 flex-wrap"):
        _metric_card("Reference samples", f"{len(X_reference):,}")
        _metric_card("Reference features", f"{X_reference.shape[1]:,}")
        _metric_card("Saved models", f"{len(model_files):,}")
        _metric_card("Required index", "Index")

    ctx_holder = {"ctx": None}

    with ui.row().classes("w-full gap-4 items-stretch mt-3"):
        with ui.element("div").classes("panel w-[520px] max-w-full self-stretch"):
            ui.html('<div class="panel-title">Prediction workspace</div>')
            sel = ui.select(model_files, value=model_files[0], label="Model package").classes("w-full")
            ui.label("Choose a trained model package. Its threshold and feature list are loaded automatically.").classes("text-grey-6 mt-2")
            ui.html('<div class="quote" style="margin:8px 0 12px;">前事不忘，後事之師。</div>')

        with ui.element("div").classes("panel flex-1 min-w-[420px] self-stretch"):
            ui.html('<div class="panel-title">Upload prediction data</div>')
            ui.label("CSV must contain an Index column and all model feature columns.").classes("text-grey-6")
            up = ui.upload(label="Choose a CSV file", auto_upload=True, multiple=False).props("accept=.csv").classes("w-full mt-3")
            ui.label("Prediction results and optional SHAP explanation will appear below after upload.").classes("text-caption text-slate-500 mt-2")

    # These are created after the controls so uploaded results stay below the options.
    cfg_area = ui.column().classes("w-full gap-2 mt-4")
    out = ui.column().classes("w-full gap-4 mt-4")

    def load_model(_=None):
        cfg_area.clear()
        out.clear()
        try:
            artifact = load_model_artifact(sel.value, is_neural_network=sel.value.endswith(".h5"))
        except Exception as exc:
            with cfg_area:
                ui.notify(f"Could not load model: {exc}", type="negative")
            ctx_holder["ctx"] = None
            return
        model = artifact["model"]
        model_name = artifact.get("model_name", type(model).__name__)
        threshold = float(artifact.get("threshold", 0.5))
        if "threshold" not in artifact:
            ui.notify("Model has no stored threshold; decisions use the 0.5 default.", type="warning")
        feat_names = list(artifact.get("feature_columns") or X_reference.columns)
        ctx_holder["ctx"] = (artifact, model, threshold, feat_names)
        with cfg_area:
            with panel("Loaded model", width_full=True):
                with ui.row().classes("w-full gap-3 flex-wrap"):
                    _metric_card("Model", model_name)
                    _metric_card("Threshold", f"{threshold:.4f}")
                    _metric_card("Features", f"{len(feat_names):,}")
                    _metric_card("Artifact", sel.value)
                with ui.expansion("Required feature columns", icon="view_column").classes("w-full mt-2"):
                    ui.label(", ".join(map(str, feat_names))).classes("text-grey-7").style("word-break:break-all;")

    async def on_upload(e):
        try:
            await _handle_upload(e, ctx_holder["ctx"], X_reference, out)
        except Exception as exc:
            with out:
                ui.notify(f"Upload handler error: {exc}", type="negative")

    sel.on("update:model-value", load_model)
    up.on_upload(on_upload)
    load_model()


async def _handle_upload(e, ctx, X_reference, out):
    out.clear()
    if ctx is None:
        with out:
            ui.notify("Load a model first", type="warning")
        return
    artifact, model, threshold, feat_names = ctx
    try:
        raw = await e.file.read()
        data = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        with out:
            ui.notify(f"Could not read CSV: {exc}", type="negative")
        return
    if "Index" not in data.columns:
        with out:
            ui.notify("CSV must contain an Index column", type="negative")
        return

    data = data.set_index("Index")
    missing = [feature for feature in feat_names if feature not in data.columns]
    if missing:
        with out:
            with panel("Missing features", width_full=True):
                ui.notify(f"Missing {len(missing)} features", type="negative")
                ui.label(", ".join(missing[:30])).classes("text-negative").style("word-break:break-all;")
        return

    filtered = data.loc[:, feat_names].copy()
    is_nn = sel_value_endswith_h5(artifact)

    try:
        if is_nn:
            probs = np.asarray(model.predict(filtered)).reshape(-1)
        else:
            probs = np.asarray(model.predict_proba(filtered))[:, 1]
        preds = (probs >= threshold).astype(int)
    except Exception as exc:
        with out:
            ui.notify(f"Prediction failed: {exc}", type="negative")
        return

    results = pd.DataFrame({
        "Name": filtered.index,
        "Positive Probability": probs,
        "Decision Threshold": threshold,
        "Prediction": preds,
    })
    positives = int(preds.sum())
    negatives = int(len(preds) - positives)

    with out:
        with ui.row().classes("w-full gap-3 flex-wrap"):
            _metric_card("Uploaded samples", f"{len(filtered):,}")
            _metric_card("Features used", f"{len(feat_names):,}")
            _metric_card("Predicted positive", f"{positives:,}")
            _metric_card("Predicted negative", f"{negatives:,}")
            _metric_card("Mean probability", f"{float(np.mean(probs)):.4f}")

        with panel("Prediction results", width_full=True):
            ui.table.from_pandas(results.round(6)).props("flat dense").classes("w-full").style("max-height:520px; overflow:auto;")
            buf = io.BytesIO()
            results.to_csv(buf, index=False)
            ui.button("Download CSV", on_click=lambda: ui.download(buf.getvalue(), filename="prediction_results.csv")).props("flat color=primary icon=download")

        with panel("SHAP analysis", width_full=True):
            ui.label("Explain individual uploaded samples using the selected model.").classes("text-grey-6")
            show_shap = ui.switch("Show SHAP analysis", value=False)
            shap_area = ui.column().classes("w-full gap-3 mt-2")

            def on_shap_toggle(_=None):
                shap_area.clear()
                if not show_shap.value:
                    return
                try:
                    _shap_single(model, artifact, filtered, X_reference, feat_names, is_nn, shap_area)
                except Exception as exc:
                    with shap_area:
                        ui.notify(f"SHAP error: {exc}", type="negative")

            show_shap.on_value_change(on_shap_toggle)


def _shap_single(model, artifact, filtered, X_reference, feat_names, is_nn, container):
    with container:
        sample_options = {i: str(filtered.index[i]) for i in range(len(filtered))}
        with ui.row().classes("w-full gap-3 items-end"):
            idx_sel = ui.select(sample_options, value=0, label="Select sample").classes("min-w-[260px] flex-1")
            ui.label("Choose plots to display below.").classes("text-grey-6")
        plot_checks = {}
        with ui.row().classes("gap-5 flex-wrap"):
            for name in ("Decision", "Force", "Waterfall", "Bar"):
                plot_checks[name] = ui.checkbox(name, value=True)
        if is_nn:
            waterfall_feature_names = list(feat_names)
        else:
            reference = X_reference.loc[:, feat_names]
            _, shap_reference = prepare_shap_inputs(artifact, reference)
            waterfall_feature_names = list(shap_reference.columns)

        waterfall_use_all = ui.switch("Use all features", value=True)
        waterfall_features = ui.select(
            waterfall_feature_names,
            multiple=True,
            value=waterfall_feature_names[:min(10, len(waterfall_feature_names))],
            label="Waterfall features",
        ).classes("w-full")
        waterfall_hint = ui.label().classes("text-caption text-slate-500")

        def sync_waterfall_features(_=None):
            use_all = waterfall_use_all.value
            waterfall_features.set_visibility(not use_all)
            if use_all:
                waterfall_hint.set_text("The Waterfall displays the 30 largest contributions.")
            else:
                selected_count = len(waterfall_features.value or [])
                waterfall_hint.set_text(
                    f"Selected: {selected_count} features. Remaining features are combined "
                    "into one Other features row."
                )

        waterfall_use_all.on("update:model-value", sync_waterfall_features)
        waterfall_features.on("update:model-value", sync_waterfall_features)
        sync_waterfall_features()
        draw_btn = ui.button("Run SHAP for selected sample").props("color=primary icon=analytics").classes("mt-2")
        plot_area = ui.column().classes("w-full gap-3")

        def draw(_=None):
            plot_area.clear()
            i = int(idx_sel.value)
            sample = filtered.iloc[i:i + 1]
            try:
                if is_nn:
                    prob = float(np.asarray(model.predict(sample)).reshape(-1)[0])
                else:
                    prob = float(model.predict_proba(sample)[0, 1])
                pred = int(prob >= float(artifact.get("threshold", 0.5)))
            except Exception as exc:
                with plot_area:
                    ui.notify(f"Predict failed: {exc}", type="negative")
                return

            with plot_area:
                _wide_metric_card("Sample", str(filtered.index[i]))
                with ui.row().classes("gap-3 flex-wrap w-full"):
                    _compact_metric_card("Prediction", str(pred))
                    _compact_metric_card("Probability", f"{prob:.4f}")
                    _compact_metric_card("Threshold", f"{float(artifact.get('threshold', 0.5)):.4f}")
                try:
                    if is_nn:
                        shap_input = sample
                        shap_bg = X_reference.loc[:, feat_names].sample(n=min(100, len(X_reference)), random_state=2025)
                        explainer = shap.Explainer(model.predict, shap_bg)
                    else:
                        ref = X_reference.loc[:, feat_names]
                        shap_model, shap_bg = prepare_shap_inputs(artifact, ref)
                        _, shap_input = prepare_shap_inputs(artifact, sample)
                        shap_bg = shap_bg.sample(n=min(100, len(shap_bg)), random_state=2025)
                        explainer = make_probability_explainer(shap_model, shap_bg)
                    explanation = explainer(shap_input)
                    sv = positive_class_array(explanation)
                    ev = positive_class_base_value(explanation.base_values)
                except Exception as exc:
                    ui.notify(f"SHAP failed: {exc}", type="negative")
                    return

                selected = [name for name in ("Decision", "Force", "Waterfall", "Bar") if plot_checks[name].value]
                if not selected:
                    ui.label("Select at least one plot to display.").classes("text-warning")
                    return

                waterfall_selected_features = waterfall_features.value or []
                if (
                    ("Waterfall" in selected or "Bar" in selected)
                    and not waterfall_use_all.value
                    and not waterfall_selected_features
                ):
                    ui.label("Select at least one Waterfall or Bar feature.").classes("text-warning")
                    return
                waterfall_indices = [
                    list(shap_input.columns).index(feature)
                    for feature in waterfall_selected_features
                ]

                for name in selected:
                    with panel(name, width_full=True):
                        if name == "Decision":
                            plt.figure(figsize=(10, 7))
                            shap.decision_plot(ev, sv[0], shap_input.iloc[0], show=False)
                            show_figure(plt.gcf())
                        elif name == "Force":
                            plt.figure(figsize=(12, 3))
                            shap.force_plot(ev, sv[0], shap_input.iloc[0], matplotlib=True, show=False)
                            show_figure(plt.gcf())
                        elif name == "Waterfall":
                            plt.figure(figsize=(10, 7))
                            exp = shap.Explanation(values=sv[0], base_values=ev,
                                                   data=shap_input.iloc[0].values, feature_names=list(shap_input.columns))
                            if waterfall_use_all.value:
                                shap.plots.waterfall(exp, max_display=30, show=False)
                            else:
                                waterfall_selected_other(exp, waterfall_indices, show=False)
                            show_figure(plt.gcf())
                        elif name == "Bar":
                            plt.figure(figsize=(10, 7))
                            if waterfall_use_all.value:
                                exp = shap.Explanation(
                                    values=sv[0], base_values=ev,
                                    data=shap_input.iloc[0].values,
                                    feature_names=list(shap_input.columns),
                                )
                                max_display = 20
                            else:
                                exp = shap.Explanation(
                                    values=sv[0, waterfall_indices], base_values=ev,
                                    data=shap_input.iloc[0, waterfall_indices].values,
                                    feature_names=waterfall_selected_features,
                                )
                                max_display = len(waterfall_selected_features)
                            shap.plots.bar(exp, max_display=max_display, show=False)
                            show_figure(plt.gcf())
        draw_btn.on("click", draw)


def _wide_metric_card(label: str, value: str):
    with ui.element("div").classes("metric-card w-full"):
        ui.html(
            f'<div class="label">{label}</div>'
            f'<div class="value" style="font-size:1rem;line-height:1.45;word-break:break-all;white-space:normal;">{value}</div>'
        )


def _compact_metric_card(label: str, value: str):
    with ui.element("div").classes("metric-card min-w-[180px] flex-1").style("min-height:88px;display:flex;flex-direction:column;justify-content:flex-end;"):
        ui.html(f'<div class="label">{label}</div><div class="value">{value}</div>')


def _metric_card(label: str, value: str):
    with ui.element("div").classes("metric-card min-w-[160px] flex-1"):
        ui.html(f'<div class="label">{label}</div><div class="value" style="font-size:1rem;word-break:break-all;">{value}</div>')


def sel_value_endswith_h5(artifact):
    """Check if artifact wraps a neural network (.h5)."""
    return type(artifact.get("model")).__name__ in {"Sequential", "Functional", "KerasModel"}