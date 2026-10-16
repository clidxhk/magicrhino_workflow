"""SHAP page — 6 analysis modes: Model, Summary, Multi, Single, Interaction, Cluster."""
from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
import pandas as pd
import shap

from nicegui import ui, app
from sklearn.model_selection import train_test_split

from ..state import page_header, panel, get_data_file, show_figure, show_svg, MODEL_DIR
from plot.molecules import shap_draw_molecules
from utils.model_utils import load_data, load_model_artifact
from utils.shap_utils import num_samples, shap_cluster_color
from utils.shap_model_utils import (
    infer_shap_model_label, make_probability_explainer,
    positive_class_array, selected_feature_contribution,
    prepare_shap_inputs,
)
from utils.shap_waterfall import waterfall_selected_other



def _metric_card(label: str, value: str):
    with ui.element("div").classes("metric-card min-w-[180px] flex-1"):
        ui.html(f'<div class="label">{label}</div>')
        value_el = ui.html(f'<div class="value">{value}</div>')
    return value_el


def _pos_neg_text(target) -> str:
    positives = int((target == 1).sum()) if target.size else 0
    negatives = int((target == 0).sum()) if target.size else 0
    return f"{positives:,} / {negatives:,}"

def build_page():
    page_header(
        "📊 SHAP Visualization",
        subtitle="Understand the significance of each feature in the model's predictions.",
        quote="萬物靜觀皆自得。 ——程顥《秋日偶成》",
    )

    if not get_data_file().exists():
        ui.label("Upload labeled data on the Data page first.").classes("text-negative")
        return

    data = pd.read_csv(get_data_file())
    X, y = load_data(data.copy())
    all_sample_positions = np.arange(len(X))
    X_train_raw, _, y_train, _, train_sample_positions, _ = train_test_split(
        X, y, all_sample_positions, test_size=0.3, random_state=42
    )

    def current_X_raw():
        """Return the raw feature frame for the selected sample source."""
        if sample_source.value == "Train set (70%)":
            return X_train_raw.copy()
        return X.copy()

    def current_target():
        if sample_source.value == "Train set (70%)":
            return np.asarray(y_train).ravel()
        return np.asarray(y).ravel()

    def current_sample_positions():
        """Return original whole-dataset row numbers for rows in the selected sample source."""
        if sample_source.value == "Train set (70%)":
            return np.asarray(train_sample_positions, dtype=int)
        return all_sample_positions

    model_files = sorted(
        [p.name for p in MODEL_DIR.iterdir() if p.suffix.lower() in {".joblib", ".sav", ".h5"}],
    )
    if not model_files:
        ui.label("No saved models found. Train one on the Model page first.").classes("text-negative")
        return

    with ui.row().classes("w-full gap-3 flex-wrap"):
        samples_value = _metric_card("SHAP samples", f"{len(X):,}")
        _metric_card("Features", f"{X.shape[1]:,}")
        pos_neg_value = _metric_card("Positive / Negative", _pos_neg_text(np.asarray(y).ravel()))
        _metric_card("Available models", f"{len(model_files):,}")

    controls = {}
    with ui.row().classes("w-full gap-4 items-stretch mt-3"):
        with ui.element("div").classes("panel w-[640px] max-w-full self-stretch"):
            ui.html('<div class="panel-title">SHAP workspace</div>')
            sel = ui.select(model_files, value=model_files[0], label="Model package").classes("w-full")
            ui.label("Samples").classes("text-caption text-slate-500 mt-2")
            sample_source = ui.toggle(
                ["All samples", "Train set (70%)"], value="All samples"
            ).classes("w-full flex-wrap")
            seed_label = ui.label("Seed: read from model artifact when started").classes(
                "text-caption text-slate-500"
            )
            ui.label("Analysis mode").classes("text-caption text-slate-500 mt-3")
            option = ui.toggle(
                ["Model", "Summary", "Multi", "Single", "Interaction", "Cluster Analysis"],
                value="Model",
            ).classes("w-full flex-wrap")
            start_button = ui.button("Start analysis").props("color=primary icon=play_arrow").classes("w-full mt-3")

        with ui.element("div").classes("panel flex-1 min-w-[360px] min-h-[282px] self-stretch"):
            ui.html('<div class="panel-title">Mode options</div>')
            suboptions = ui.column().classes("w-full gap-3")

    out = ui.column().classes("w-full gap-4 mt-4")
    feature_names = list(X.columns)

    def render_model_details():
        area = controls.get("model_info_area")
        if area is None:
            return
        area.clear()
        try:
            artifact = load_model_artifact(sel.value, is_neural_network=sel.value.endswith(".h5"))
            if sel.value.endswith(".joblib"):
                model, _ = prepare_shap_inputs(artifact, current_X_raw())
                a_option = infer_shap_model_label(artifact)
            else:
                model = artifact["model"]
                a_option = "Neural Network"
            artifact_seed = artifact.get("seed") if isinstance(artifact, dict) else None
            if artifact_seed is None:
                seed_label.set_text("Seed: 42 (model artifact has no saved seed)")
            else:
                seed_label.set_text(f"Seed: {int(artifact_seed)} (from model artifact)")
            with area:
                _mode_model(model, a_option)
        except Exception as exc:
            with area:
                ui.notify(f"Could not load model: {exc}", type="negative")
    def render_suboptions():
        controls.clear()
        suboptions.clear()
        start_button.set_visibility(option.value != "Model")
        with suboptions:
            if option.value == "Model":
                ui.label("Model parameters are loaded automatically for the selected package.").classes("text-grey-6")
                controls["model_info_area"] = ui.column().classes("w-full gap-2 mt-2")
                render_model_details()
            elif option.value == "Summary":
                with panel("Summary options", width_full=True):
                    ui.label("Global feature importance overview: summarize how each descriptor contributes across samples.").classes("text-grey-6")
                    ui.html('<div class="quote" style="margin:6px 0 12px;">觀其大略，而知其要。 ——《史記》</div>')
                    controls["summary_use_all"] = ui.switch("Use all features", value=True)
                    controls["summary_features"] = ui.select(
                        feature_names, multiple=True, value=feature_names[:5], label="Selected features"
                    ).classes("w-full")
                    summary_default = min(20, len(feature_names))
                    summary_max = min(20, len(feature_names))
                    summary_label = ui.label(
                        f"Max display: {summary_default} / {summary_max} features"
                    ).classes("text-caption text-slate-500")
                    controls["summary_max_display"] = ui.slider(
                        min=1, max=summary_max, value=summary_default
                    ).props("label label-always markers").classes("w-full")
                    controls["summary_max_display"].on(
                        "update:model-value",
                        lambda event: summary_label.set_text(
                            f"Max display: {int(event.args)} / {summary_max} features"
                        ),
                    )
                    def _sync_summary(e=None):
                        use_all = controls["summary_use_all"].value
                        controls["summary_features"].set_visibility(not use_all)
                        controls["summary_max_display"].set_visibility(use_all)
                        if use_all:
                            val = int(controls["summary_max_display"].value)
                            summary_label.set_text(f"Max display: {val} / {summary_max} features")
                        else:
                            sel = controls["summary_features"].value or []
                            summary_label.set_text(f"Selected: {len(sel)} features")
                    controls["summary_use_all"].on("update:model-value", _sync_summary)
                    controls["summary_features"].on("update:model-value", _sync_summary)
                    _sync_summary()
            elif option.value == "Multi":
                with panel("Multi-sample options", width_full=True):
                    ui.label("Compare SHAP patterns across multiple samples with complementary visual views.").classes("text-grey-6")
                    ui.html('<div class="quote" style="margin:6px 0 12px;">兼聽則明，偏信則暗。 ——《資治通鑑》</div>')
                    ui.label("Choose the plots to display after starting.").classes("text-grey-6")
                    controls["multi_plots"] = {}
                    with ui.row().classes("gap-5 flex-wrap"):
                        for plot_name in ("Decision", "Heatmap", "Bar"): 
                            controls["multi_plots"][plot_name] = ui.checkbox(
                                plot_name, value=True
                            )
                    controls["multi_use_all"] = ui.switch("Use all features", value=True)
                    controls["multi_features"] = ui.select(
                        feature_names, multiple=True, value=feature_names[:10], label="Selected features"
                    ).classes("w-full")
                    multi_min = min(1, len(feature_names))
                    multi_max = min(20, len(feature_names))
                    multi_default = multi_max
                    multi_label = ui.label(
                        f"Max display: {multi_default} / {multi_max} features"
                    ).classes("text-caption text-slate-500")
                    controls["multi_max_display"] = ui.slider(
                        min=multi_min, max=multi_max, value=multi_default
                    ).props("label label-always markers").classes("w-full")
                    controls["multi_max_display"].on(
                        "update:model-value",
                        lambda event: multi_label.set_text(
                            f"Max display: {int(event.args)} / {multi_max} features"
                        ),
                    )
                    def _sync_multi(e=None):
                        use_all = controls["multi_use_all"].value
                        controls["multi_features"].set_visibility(not use_all)
                        controls["multi_max_display"].set_visibility(use_all)
                        if use_all:
                            multi_label.set_text(
                                f"Max display: {int(controls['multi_max_display'].value)} / {multi_max} features"
                            )
                        else:
                            sel = controls["multi_features"].value or []
                            multi_label.set_text(f"Selected: {len(sel)} features")
                    controls["multi_use_all"].on("update:model-value", _sync_multi)
                    controls["multi_features"].on("update:model-value", _sync_multi)
                    _sync_multi()

            elif option.value == "Single":
                with panel("Single-sample options", width_full=True):
                    ui.label("Inspect one prediction in depth: decompose how features push a single sample toward its output.").classes("text-grey-6")
                    ui.html('<div class="quote" style="margin:6px 0 12px;">察己則可以知人，察今則可以知古。 ——《呂氏春秋》</div>')
                    controls["single_index"] = ui.number(
                        "Sample index (original dataset)", min=0, max=len(X) - 1, value=0
                    )
                    ui.label(
                        "When using Train set, enter the original full-dataset row number; "
                        "the sample must be part of the selected train split."
                    ).classes("text-caption text-slate-500")
                    ui.label("Plots to display").classes("text-caption text-slate-500")
                    controls["single_outputs"] = {}
                    with ui.row().classes("gap-5 flex-wrap"):
                        for output_name in (
                            "Decision", "Force", "Waterfall", "Bar", "Molecule"
                        ):
                            controls["single_outputs"][output_name] = ui.checkbox(
                                output_name, value=True
                            )
                    controls["single_use_all"] = ui.switch("Use all features", value=True)
                    controls["single_features"] = ui.select(
                        feature_names, multiple=True, value=feature_names[:10], label="Selected features"
                    ).classes("w-full")
                    single_min = min(1, len(feature_names))
                    single_max = min(20, len(feature_names))
                    single_default = single_max
                    single_label = ui.label(
                        f"Max display: {single_default} / {single_max} features"
                    ).classes("text-caption text-slate-500")
                    controls["single_max_display"] = ui.slider(
                        min=single_min, max=single_max, value=single_default
                    ).props("label label-always markers").classes("w-full")
                    controls["single_max_display"].on(
                        "update:model-value",
                        lambda event: single_label.set_text(
                            f"Max display: {int(event.args)} / {single_max} features"
                        ),
                    )
                    def _sync_single(e=None):
                        use_all = controls["single_use_all"].value
                        controls["single_features"].set_visibility(not use_all)
                        controls["single_max_display"].set_visibility(use_all)
                        if use_all:
                            single_label.set_text(
                                f"Max display: {int(controls['single_max_display'].value)} / {single_max} features"
                            )
                        else:
                            sel = controls["single_features"].value or []
                            single_label.set_text(f"Selected: {len(sel)} features")
                    controls["single_use_all"].on("update:model-value", _sync_single)
                    controls["single_features"].on("update:model-value", _sync_single)
                    _sync_single()
            elif option.value == "Interaction":
                with panel("Interaction options", width_full=True):
                    ui.label("Available for XGBoost, LightGBM, and Random Forest models.").classes("text-grey-6")
                    controls["interaction_outputs"] = {}
                    controls["interaction_outputs"]["Interaction"] = ui.checkbox("Interaction", value=True)
                    # Interaction options: only visible when Interaction is checked
                    interaction_opts = ui.column().classes("w-full gap-2")
                    with interaction_opts:
                        controls["interaction_use_all"] = ui.switch("Use all features", value=True)
                        controls["interaction_features"] = ui.select(
                            feature_names, multiple=True, value=feature_names[:5], label="Features"
                        ).classes("w-full")
                        interaction_default = min(5, len(feature_names))
                        interaction_max = min(20, len(feature_names))
                        interaction_label = ui.label(
                            f"Max display: {interaction_default} / {interaction_max} features"
                        ).classes("text-caption text-slate-500")
                        controls["interaction_max_display"] = ui.slider(
                            min=1, max=interaction_max, value=interaction_default
                        ).props("label label-always markers").classes("w-full")
                        controls["interaction_max_display"].on(
                            "update:model-value",
                            lambda event: interaction_label.set_text(
                                f"Max display: {int(event.args)} / {interaction_max} features"
                            ),
                        )
                        def _sync_interaction(e=None):
                            use_all = controls["interaction_use_all"].value
                            controls["interaction_features"].set_visibility(not use_all)
                            controls["interaction_max_display"].set_visibility(use_all)
                            if use_all:
                                val = int(controls["interaction_max_display"].value)
                                interaction_label.set_text(f"Max display: {val} / {interaction_max} features")
                            else:
                                sel = controls["interaction_features"].value or []
                                interaction_label.set_text(f"Selected: {len(sel)} features")
                        controls["interaction_use_all"].on("update:model-value", _sync_interaction)
                        controls["interaction_features"].on("update:model-value", _sync_interaction)
                        _sync_interaction()
                    interaction_opts.bind_visibility_from(
                        controls["interaction_outputs"]["Interaction"], "value"
                    )
                    # Dependence checkbox + options together
                    controls["interaction_outputs"]["Dependence"] = ui.checkbox("Dependence", value=False)
                    dependence_opts = ui.column().classes("w-full gap-2")
                    with dependence_opts:
                        ui.label("Dependence options").classes("text-caption text-slate-500")
                        controls["interaction_mode"] = ui.toggle(["Single", "Two"], value="Single")
                        controls["interaction_feature_a"] = ui.select(
                            feature_names, value=feature_names[0], label="Feature A"
                        ).classes("w-full")
                        controls["interaction_feature_b"] = ui.select(
                            feature_names, value=feature_names[min(1, len(feature_names) - 1)], label="Feature B"
                        ).classes("w-full")
                        controls["interaction_feature_b"].bind_visibility_from(
                            controls["interaction_mode"], "value", lambda v: v == "Two"
                        )
                    dependence_opts.bind_visibility_from(
                        controls["interaction_outputs"]["Dependence"], "value"
                    )

            elif option.value == "Cluster Analysis":
                with panel("Cluster options", width_full=True):
                    controls["cluster_method"] = ui.toggle(["PCA", "UMAP", "TSNE"], value="PCA")
                    controls["cluster_components"] = ui.number("n_components", min=2, value=2)
                    controls["cluster_perplexity"] = ui.number(
                        "Perplexity (TSNE only)", min=2, value=min(50, max(2, len(X) - 1))
                    )
                    controls["cluster_title"] = ui.input("Title")
                    controls["cluster_show_true_labels"] = ui.switch("Show true labels", value=False)
                    color_options = {
                        "__all_features__": "All features (total |SHAP|)"
                    }
                    color_options.update({feature: feature for feature in feature_names})
                    controls["cluster_feature"] = ui.select(
                        color_options, value="__all_features__", label="Color by feature"
                    ).classes("w-full")

    def run():
        out.clear()
        try:
            artifact = load_model_artifact(sel.value, is_neural_network=sel.value.endswith(".h5"))
        except Exception as exc:
            with out:
                ui.notify(f"Could not load model: {exc}", type="negative")
            return
        if sel.value.endswith(".joblib"):
            model, X_labeled = prepare_shap_inputs(artifact, current_X_raw())
            a_option = infer_shap_model_label(artifact)
        else:
            model = artifact["model"]
            X_labeled = current_X_raw()
            a_option = "Neural Network"
        artifact_seed = artifact.get("seed") if isinstance(artifact, dict) else None
        if artifact_seed is None:
            artifact_seed = 42
            seed_label.set_text("Seed: 42 (model artifact has no saved seed)")
        else:
            artifact_seed = int(artifact_seed)
            seed_label.set_text(f"Seed: {artifact_seed} (from model artifact)")
        np.random.seed(artifact_seed)
        if option.value == "Model" and "model_info_area" in controls:
            controls["model_info_area"].clear()
            with controls["model_info_area"]:
                _mode_model(model, a_option)
        else:
            with out:
                _dispatch(
                    option.value, model, X_labeled, a_option, data,
                    current_X_raw(), current_sample_positions(), current_target(), controls,
                )
    def on_option_change():
        out.clear()
        render_suboptions()

    def on_sample_source_change():
        samples_value.set_content(f'<div class="value">{len(current_X_raw()):,}</div>')
        pos_neg_value.set_content(f'<div class="value">{_pos_neg_text(current_target())}</div>')
        out.clear()
        render_suboptions()

    render_suboptions()
    start_button.on("click", run)
    sample_source.on("update:model-value", on_sample_source_change)
    def on_model_change():
        out.clear()
        seed_label.set_text("Seed: read from model artifact when started")
        if option.value == "Model":
            render_suboptions()

    sel.on("update:model-value", on_model_change)
    option.on("update:model-value", on_option_change)


def _dispatch(option, model, X, a_option, data, X_raw, sample_positions, target, controls):
    if option == "Model":
        _mode_model(model, a_option)
    elif option == "Summary":
        _mode_summary(model, X, a_option, controls)
    elif option == "Multi":
        _mode_multi(model, X, a_option, data, controls)
    elif option == "Single":
        _mode_single(model, X, a_option, X_raw, sample_positions, controls)
    elif option == "Interaction":
        _mode_interaction(model, X, a_option, controls)
    elif option == "Cluster Analysis":
        _mode_cluster(model, X, a_option, data, target, controls)


def _mode_model(model, a_option):
    if a_option == "Neural Network":
        lines = []
        model.summary(print_fn=lines.append)
        ui.label("Neural-network model loaded.").classes("text-grey-6")
        with ui.expansion("Model summary", icon="article", value=False).classes("w-full"):
            ui.code("\n".join(lines), language="text").classes("w-full").style(
                "max-height:420px; overflow:auto; white-space:pre-wrap; word-break:break-word;"
            )
        with ui.expansion("Weights", icon="storage").classes("w-full"):
            ui.label(f"Weights: {len(model.get_weights())} arrays").classes("text-grey-6")
        return

    params = {k: v for k, v in model.get_params().items() if v is not None}
    records = [
        {"Parameter": str(key), "Value": _format_model_param(value)}
        for key, value in sorted(params.items())
    ]
    ui.label(f"{a_option} model loaded. Expand sections below to inspect details.").classes("text-grey-6")
    with ui.expansion("Model parameters", icon="tune", value=False).classes("w-full"):
        ui.table(
            columns=[
                {"name": "Parameter", "label": "Parameter", "field": "Parameter", "align": "left", "sortable": True},
                {"name": "Value", "label": "Value", "field": "Value", "align": "left"},
            ],
            rows=records,
            row_key="Parameter",
        ).props("flat dense wrap-cells").classes("w-full").style(
            "max-height:420px; overflow:auto; word-break:break-word;"
        )
    with ui.expansion("Parameter JSON", icon="data_object").classes("w-full"):
        ui.code(json.dumps(params, indent=2, default=str, ensure_ascii=False), language="json").classes("w-full").style(
            "max-height:360px; overflow:auto; white-space:pre-wrap; word-break:break-word;"
        )


def _format_model_param(value, max_length: int = 140):
    """Return a compact, table-safe representation of a model parameter."""
    text = json.dumps(value, default=str, ensure_ascii=False)
    if len(text) > max_length:
        return text[:max_length - 1] + "…"
    return text

def _mode_summary(model, X, a_option, settings):
    feature_names = list(X.columns)
    use_all = bool(settings["summary_use_all"].value)
    selected_features = feature_names if use_all else (settings["summary_features"].value or [feature_names[0]])
    indices = [feature_names.index(feature) for feature in selected_features]
    max_display = int(settings["summary_max_display"].value)
    try:
        if a_option in ("LightGBM", "XGBoost"):
            explainer = shap.TreeExplainer(model, X)
            shap_values = explainer(X, check_additivity=False)
            values = shap_values.values if use_all else shap_values.values[:, indices]
            display_data = X if use_all else X[selected_features]
        elif a_option == "Neural Network":
            background = X.iloc[np.random.choice(X.shape[0], min(100, X.shape[0]), replace=False)]
            explainer = shap.Explainer(model.predict, background)
            shap_values = explainer.shap_values(X)
            shap_values = np.reshape(shap_values, (shap_values.shape[0], -1))
            values = shap_values if use_all else shap_values[:, indices]
            display_data = X if use_all else X[selected_features]
        else:
            explainer = make_probability_explainer(model, X)
            shap_values = explainer(X)
            values = shap_values[:, :, 1] if use_all else shap_values[:, indices, 1]
            display_data = X if use_all else X[selected_features]
        with panel("Summary", width_full=True):
            plt.figure(figsize=(10, 7))
            shap.summary_plot(values, display_data, max_display=max_display, show=False)
            show_figure(plt.gcf())
        if not use_all:
            _show_contribution(shap_values, feature_names, selected_features)
    except Exception as exc:
        ui.notify(f"Error: {exc}", type="negative")

def _show_contribution(shap_values, feature_names, selected_features):
    try:
        contribution, detail = selected_feature_contribution(shap_values, feature_names, selected_features)
    except Exception as exc:
        ui.notify(f"Contribution calc failed: {exc}", type="warning")
        return
    with panel("Selected feature contribution"):
        with ui.row().classes("gap-2"):
            with ui.element("div").classes("metric-card"):
                ui.html(f'<div class="label">Selected</div><div class="value">{len(selected_features)}</div>')
            with ui.element("div").classes("metric-card"):
                ui.html(f'<div class="label">Overall SHAP %</div><div class="value">{contribution:.2f}%</div>')
        ui.table.from_pandas(detail.round(4)).props("flat dense").classes("w-full")


def _selected_feature_view(settings, prefix, feature_names):
    """Return selected feature names and column indices for a SHAP mode."""
    feature_names = list(feature_names)
    use_all = bool(settings[f"{prefix}_use_all"].value)
    if use_all:
        selected_features = feature_names
    else:
        selected_features = settings[f"{prefix}_features"].value or [feature_names[0]]
    indices = [feature_names.index(feature) for feature in selected_features]
    return use_all, selected_features, indices


def _explanation_from_values(values, base_values, X_display):
    """Build a SHAP Explanation for the selected feature view."""
    values = np.asarray(values)
    if values.ndim == 1:
        data = X_display.iloc[0].values if len(X_display) == 1 else X_display.values
    else:
        data = X_display.values
    return shap.Explanation(
        values=values,
        base_values=base_values,
        data=data,
        feature_names=list(X_display.columns),
    )


def _add_other_feature(values, data, feature_names, selected_indices, other_name="Other features"):
    """Keep selected features and aggregate all remaining SHAP values into one additive Other item.

    This is required for additive plots such as decision, force, and waterfall: dropping
    unselected features breaks base_value + sum(SHAP) = prediction.
    """
    values = np.asarray(values)
    feature_names = list(feature_names)
    selected_indices = list(selected_indices)
    all_indices = np.arange(len(feature_names))
    other_indices = [idx for idx in all_indices if idx not in set(selected_indices)]

    data_frame = pd.DataFrame(data, columns=feature_names)
    selected_values = values[..., selected_indices]
    selected_data = data_frame.iloc[:, selected_indices].to_numpy()
    selected_names = [feature_names[idx] for idx in selected_indices]

    if not other_indices:
        return selected_values, selected_data, selected_names

    other_values = values[..., other_indices].sum(axis=-1, keepdims=True)
    combined_values = np.concatenate([selected_values, other_values], axis=-1)
    # Use an object array so additive plots can show the aggregate count instead
    # of a meaningless numeric feature value or nan.
    combined_data = np.empty(combined_values.shape, dtype=object)
    combined_data[..., :len(selected_indices)] = selected_data
    combined_data[..., len(selected_indices)] = ""
    other_label = f"{len(other_indices)} other features"
    return combined_values, combined_data, selected_names + [other_label]


def _mode_multi(model, X, a_option, data, settings):
    plot_order = ("Decision", "Heatmap", "Bar")
    selected_plots = [
        name for name in plot_order if settings["multi_plots"][name].value
    ]
    if not selected_plots:
        ui.label("Select at least one plot before starting.").classes("text-warning")
        return
    use_all, selected_features, indices = _selected_feature_view(settings, "multi", X.columns)
    max_display = int(settings["multi_max_display"].value) if use_all else len(selected_features)
    X_display = X if use_all else X[selected_features]
    with panel("Multi-sample analysis", width_full=True):
        shown_plots = ", ".join(selected_plots)
        feature_text = (
            f"all features, max {max_display}"
            if use_all else f"selected {len(selected_features)} features + Other features"
        )
        ui.label(f"Showing: {shown_plots}; features: {feature_text}").classes("text-grey-6")

    if a_option in ("LightGBM", "XGBoost"):
        expected_value = model.predict(X).mean(0)
        explainer = shap.TreeExplainer(model, X)
        shap_values = explainer.shap_values(X, check_additivity=False)
    elif a_option == "Neural Network":
        expected_value = model.predict(X).mean(0)
        background = X.iloc[np.random.choice(X.shape[0], min(100, X.shape[0]), replace=False)]
        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer.shap_values(X)
        shap_values = np.reshape(shap_values, (shap_values.shape[0], -1))
    else:
        expected_value = model.predict_proba(X)[:, 1].mean(0)
        explainer = make_probability_explainer(model, X)
        shap_values = explainer.shap_values(X)

    with ui.column().classes("w-full gap-4"):
        if "Decision" in selected_plots:
            with panel("Decision", width_full=True):
                if not use_all:
                    ui.label(
                        "Decision plot is unavailable in selected-feature mode. "
                        "Use all features to view the original decision plot, or use Waterfall for selected-feature inspection."
                    ).classes("text-warning")
                else:
                    plt.figure(figsize=(10, 7))
                    decision_values = (
                        positive_class_array(shap_values)
                        if a_option != "Neural Network" and a_option not in ("LightGBM", "XGBoost")
                        else np.asarray(shap_values)
                    )
                    shap.decision_plot(
                        expected_value, decision_values, X,
                        feature_display_range=slice(None, -max_display - 1, -1), show=False,
                    )
                    show_figure(plt.gcf())
                    ui.label("How features incrementally contribute to predictions across samples.").classes("text-grey-6")

        if "Heatmap" in selected_plots:
            with panel("Heatmap", width_full=True):
                plt.figure(figsize=(10, 7))
                if a_option in ("LightGBM", "XGBoost"):
                    heat_values = positive_class_array(explainer(X, check_additivity=False))
                elif a_option == "Neural Network":
                    heat_values = np.reshape(explainer.shap_values(X), (X.shape[0], -1))
                else:
                    heat_values = positive_class_array(explainer(X))
                if not use_all:
                    heat_values = heat_values[:, indices]
                shap.plots.heatmap(
                    _explanation_from_values(heat_values, np.full(len(X_display), expected_value), X_display),
                    max_display=max_display, show=False,
                )
                show_figure(plt.gcf())


        if "Bar" in selected_plots:
            with panel("Bar", width_full=True):
                values = (
                    shap_values
                    if a_option in ("LightGBM", "XGBoost")
                    else positive_class_array(shap_values)
                )
                values = positive_class_array(values)
                if not use_all:
                    values = values[:, indices]
                explanation = _explanation_from_values(
                    values, np.full(values.shape[0], expected_value), X_display
                )
                plt.figure(figsize=(10, 7))
                shap.plots.bar(explanation, max_display=max_display, show=False)
                show_figure(plt.gcf())

def _mode_single(model, X, a_option, X_raw, sample_positions, settings):
    output_order = ("Decision", "Force", "Waterfall", "Bar", "Molecule")
    selected_outputs = [
        name for name in output_order if settings["single_outputs"][name].value
    ]
    if not selected_outputs:
        ui.label("Select at least one output before starting.").classes("text-warning")
        return
    original_i = max(0, int(settings["single_index"].value))
    matches = np.flatnonzero(np.asarray(sample_positions, dtype=int) == original_i)
    if not len(matches):
        ui.label(
            f"Original sample index {original_i} is not in the selected sample source. "
            "Switch to All samples or choose an index that belongs to the train split."
        ).classes("text-warning")
        return
    i = int(matches[0])
    use_all, selected_features, indices = _selected_feature_view(settings, "single", X.columns)
    max_display = int(settings["single_max_display"].value) if use_all else len(selected_features)
    X_display = X if use_all else X[selected_features]
    needs_shap = any(
        name in selected_outputs for name in ("Decision", "Force", "Waterfall", "Bar")
    )

    if needs_shap:
        if a_option in ("LightGBM", "XGBoost"):
            explainer = shap.TreeExplainer(model, X)
            shap_values = explainer.shap_values(X, check_additivity=False)
            expected_value = explainer.expected_value
            single_values = shap_values[i, :]
        elif a_option == "Neural Network":
            expected_value = model.predict(X).mean(0)
            background = X.iloc[np.random.choice(X.shape[0], min(100, X.shape[0]), replace=False)]
            explainer = shap.Explainer(model.predict, background)
            shap_values = explainer.shap_values(X)
            shap_values = np.reshape(shap_values, (shap_values.shape[0], -1))
            single_values = shap_values[i, :]
        else:
            expected_value = model.predict_proba(X)[:, 1].mean(0)
            explainer = make_probability_explainer(model, X)
            shap_values = explainer.shap_values(X)
            single_values = shap_values[:, :, 1][i, :]

    if needs_shap:
        single_values = np.asarray(single_values)
        if use_all:
            additive_values = single_values
            additive_data = X.iloc[i].values
            additive_feature_names = list(X.columns)
            additive_max_display = max_display
            selected_single_values = single_values
        else:
            additive_values, additive_data, additive_feature_names = _add_other_feature(
                single_values.reshape(1, -1), X.iloc[[i]], X.columns, indices
            )
            additive_values = additive_values[0]
            additive_data = additive_data[0]
            additive_max_display = len(selected_features) + 1
            selected_single_values = single_values[indices]

    with ui.column().classes("w-full gap-4"):
        if "Decision" in selected_outputs:
            with panel("Decision", width_full=True):
                if not use_all:
                    ui.label(
                        "Decision plot is unavailable in selected-feature mode. "
                        "Use all features to view the original decision plot, or use Waterfall for selected-feature inspection."
                    ).classes("text-warning")
                else:
                    plt.figure(figsize=(10, 7))
                    shap.decision_plot(
                        expected_value, single_values, X,
                        feature_display_range=slice(None, -max_display - 1, -1), show=False,
                    )
                    show_figure(plt.gcf())

        if "Force" in selected_outputs:
            with panel("Force", width_full=True):
                if not use_all:
                    ui.label(
                        "Force plot is unavailable in selected-feature mode. "
                        "Use all features to view the original force plot, or use Waterfall for selected-feature inspection."
                    ).classes("text-warning")
                else:
                    plt.figure(figsize=(12, 3))
                    shap.force_plot(
                        expected_value, additive_values, additive_data,
                        feature_names=additive_feature_names, matplotlib=True, show=False
                    )
                    show_figure(plt.gcf())

        if "Waterfall" in selected_outputs or "Bar" in selected_outputs:
            additive_explanation = shap.Explanation(
                values=additive_values,
                base_values=expected_value,
                data=additive_data,
                feature_names=additive_feature_names,
            )
            selected_explanation = shap.Explanation(
                values=selected_single_values,
                base_values=expected_value,
                data=X_display.iloc[i].values,
                feature_names=list(X_display.columns),
            )

        if "Waterfall" in selected_outputs:
            with panel("Waterfall", width_full=True):
                plt.figure(figsize=(10, 7))
                if use_all:
                    shap.plots.waterfall(additive_explanation, max_display=additive_max_display, show=False)
                else:
                    # Selected features individually + one pinned bottom aggregate row,
                    # rendered exactly like SHAP's native "N other features" style.
                    full_explanation = shap.Explanation(
                        values=single_values,
                        base_values=expected_value,
                        data=X.iloc[i].values,
                        feature_names=list(X.columns),
                    )
                    waterfall_selected_other(full_explanation, indices, show=False)
                show_figure(plt.gcf())

        if "Bar" in selected_outputs:
            with panel("Bar", width_full=True):
                plt.figure(figsize=(10, 7))
                shap.plots.bar(selected_explanation, max_display=max_display, show=False)
                show_figure(plt.gcf())

        if "Molecule" in selected_outputs:
            with panel("Molecule", width_full=True):
                try:
                    svg = shap_draw_molecules(X_raw, i)
                    if svg is None:
                        ui.label("Invalid molecule").classes("text-grey-6")
                    else:
                        show_svg(
                            svg,
                            filename=f"shap-molecule-original-{original_i}.svg",
                            style="width:200px;border:1px solid #e6eaf0;border-radius:10px;",
                        )
                except Exception as exc:
                    ui.label(f"Molecule viz unavailable: {exc}").classes("text-grey-6")

def _mode_interaction(model, X, a_option, settings):
    if a_option not in ("LightGBM", "XGBoost", "Random Forest"):
        ui.label("Interaction plot is available for XGBoost, LightGBM, and Random Forest models.").classes("text-negative")
        return
    # tree_path_dependent (no background) is required for shap_interaction_values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)
    shap_values = positive_class_array(shap_values)
    interaction_values = explainer.shap_interaction_values(X)
    interaction_values = np.asarray(interaction_values)
    if interaction_values.ndim == 4:
        # Binary classification returns (N, M, M, 2); select positive class
        interaction_values = interaction_values[:, :, :, 1]
    feature_names = list(X.columns)
    use_all = bool(settings["interaction_use_all"].value)
    selected_features = feature_names if use_all else (settings["interaction_features"].value or [feature_names[0]])
    indices = [feature_names.index(feature) for feature in selected_features]
    max_display = int(settings["interaction_max_display"].value)
    feature_a = settings["interaction_feature_a"].value or feature_names[0]
    feature_b = settings["interaction_feature_b"].value or feature_a
    interaction_mode = settings["interaction_mode"].value

    selected_outputs = [
        name for name in ("Interaction", "Dependence") if settings["interaction_outputs"][name].value
    ]
    if not selected_outputs:
        ui.label("Select at least one output before starting.").classes("text-warning")
        return

    if "Interaction" in selected_outputs:
        with panel("Interaction summary", width_full=True):
            plt.figure(figsize=(10, 7))
            if use_all:
                shap.summary_plot(interaction_values, X, max_display=max_display, show=False)
            else:
                filtered_data = X[selected_features]
                filtered_values = interaction_values[:, indices, :][:, :, indices]
                shap.summary_plot(filtered_values, filtered_data, max_display=len(selected_features), show=False)
            show_figure(plt.gcf())

    if "Dependence" in selected_outputs:
        with panel("Dependence", width_full=True):
            if interaction_mode == "Two" and feature_a != feature_b:
                # Two-feature mode: interaction effect + colored dependence
                # 1. Interaction effect between feature_a and feature_b
                plt.figure()
                shap.dependence_plot((feature_a, feature_b), interaction_values, X, show=False)
                show_figure(plt.gcf())
                # 2. Dependence of feature_a colored by feature_b
                plt.figure()
                shap.dependence_plot(feature_a, shap_values, X, interaction_index=feature_b, show=False)
                show_figure(plt.gcf())
            else:
                # Single-feature mode: just the dependence of feature_a
                plt.figure()
                shap.dependence_plot(feature_a, shap_values, X, show=False, interaction_index=None)
                show_figure(plt.gcf())

def _mode_cluster(model, X, a_option, data, target, settings):

    method = settings["cluster_method"].value
    components = int(settings["cluster_components"].value)
    title = settings["cluster_title"].value or "SHAP"
    color_feature = settings["cluster_feature"].value or "__all_features__"
    show_true_labels = bool(settings["cluster_show_true_labels"].value)
    if a_option in ("LightGBM", "XGBoost"):
        explainer = shap.TreeExplainer(model, X)
        shap_values = explainer.shap_values(X, check_additivity=False)
    elif a_option == "Neural Network":
        background = X.iloc[np.random.choice(X.shape[0], 1, replace=False)]
        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer.shap_values(X)
        shap_values = np.reshape(shap_values, (shap_values.shape[0], -1))
    else:
        explainer = make_probability_explainer(model, X)
        shap_values = positive_class_array(explainer.shap_values(X))

    sample_count = min(num_samples(data), len(X), len(shap_values))
    if method == "PCA":
        from sklearn.decomposition import PCA
        embedding = PCA(n_components=components).fit_transform(shap_values[:sample_count, :])
    elif method == "UMAP":
        from utils.data_utils import patch_sklearn_check_array_for_umap
        patch_sklearn_check_array_for_umap()
        from umap import UMAP
        import umap.umap_ as umap_module
        from sklearn.utils import validation as sklearn_validation
        umap_module.check_array = sklearn_validation.check_array
        embedding = UMAP(n_components=components, n_jobs=1).fit_transform(shap_values[:sample_count, :])
    else:
        from sklearn.manifold import TSNE
        perplexity = min(int(settings["cluster_perplexity"].value), max(1, sample_count - 1))
        embedding = TSNE(
            n_components=components, perplexity=perplexity, random_state=42
        ).fit_transform(shap_values[:sample_count, :])

    red_blue = shap_cluster_color()
    if color_feature == "__all_features__":
        color_values = np.abs(shap_values[:sample_count]).sum(axis=1)
        color_label = "All features (total |SHAP|)"
    else:
        color_values = X[color_feature].values[:sample_count].astype(np.float64)
        color_label = color_feature
    panel_title = f"Cluster colored by {color_label}"
    if show_true_labels:
        panel_title += " with true labels"
    with panel(panel_title, width_full=True):
        plt.figure(figsize=(7, 6))
        shap_points = pl.scatter(
            embedding[:, 0], embedding[:, 1],
            c=color_values,
            linewidth=0, alpha=1.0, cmap=red_blue, s=100,
        )
        if show_true_labels:
            labels = np.asarray(target).ravel()[:sample_count]
            marker_cycle = ["o", "^", "s", "D", "P", "X", "v", "*"]
            for marker_index, label in enumerate(pd.unique(labels)):
                mask = labels == label
                pl.scatter(
                    embedding[mask, 0], embedding[mask, 1],
                    facecolors="none", edgecolors="black", linewidths=1.3,
                    marker=marker_cycle[marker_index % len(marker_cycle)],
                    s=145, label=f"True label {label}",
                )
            pl.legend(loc="best", frameon=False, fontsize=10)
        pl.title(title, fontsize=14)
        colorbar = pl.colorbar(shap_points, label=color_label, aspect=40, orientation="horizontal")
        colorbar.set_alpha(1)
        colorbar.outline.set_linewidth(0)
        colorbar.set_label(color_label, fontsize=14)
        colorbar.ax.tick_params("x", length=0, labelsize=14)
        colorbar.ax.xaxis.set_label_position("top")
        pl.gca().axis("off")
        show_figure(plt.gcf())