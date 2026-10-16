"""Params page: randomized hyperparameter search with stratified CV."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from nicegui import ui, app

from ..state import page_header, panel, get_data_file, SEARCH_DIR
from utils.data_utils import load_data
from utils.params_utils import (
    MODEL_NAMES, run_parameter_search, save_search_results,
    search_space_records,
)


def build_page():
    page_header(
        "🔎 Parameter Search",
        subtitle="Stratified cross-validation with fold-safe preprocessing and randomized hyperparameter search.",
        quote="工欲善其事，必先利其器。 ——《論語·衛靈公》",
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

    with ui.row().classes("w-full gap-3 flex-wrap"):
        _metric_card("Samples", f"{len(X):,}")
        _metric_card("Features", f"{X.shape[1]:,}")
        _metric_card("Positive / Negative", f"{positives:,} / {negatives:,}")
        _metric_card("Saved runs", f"{len(list(SEARCH_DIR.glob('*'))):,}")

    with panel("Workspace", width_full=True):
        ui.label("Choose an action below. Search runs in memory first; use Save complete run to persist CSV files.").classes("text-grey-6")
        mode = ui.toggle(["Search", "Results", "Delete"], value="Search").classes("w-full mt-2")

    container = ui.column().classes("w-full gap-4 mt-3")

    def on_mode(_=None):
        container.clear()
        if mode.value == "Search":
            _search_view(X, y, container)
        elif mode.value == "Results":
            _results_view(container)
        else:
            _delete_view(container)

    mode.on("update:model-value", on_mode)
    on_mode()


def _metric_card(label: str, value: str):
    with ui.element("div").classes("metric-card min-w-[180px] flex-1"):
        ui.html(f'<div class="label">{label}</div><div class="value">{value}</div>')


def _search_view(X, y, container):
    controls = {}
    with container:
        with ui.row().classes("w-full gap-4 items-start"):
            with ui.element("div").classes("panel flex-1 min-w-[420px]"):
                ui.html('<div class="panel-title">Search configuration</div>')
                controls["models"] = ui.select(
                    list(MODEL_NAMES),
                    multiple=True,
                    value=["RandomForest"],
                    label="Model families",
                    with_input=True,
                ).classes("w-full")

                with ui.grid(columns=4).classes("w-full gap-3 mt-3"):
                    controls["n_trials"] = ui.number("Candidates / model", min=1, max=1000, value=20).classes("w-full")
                    controls["cv_folds"] = ui.number("CV folds", min=2, max=10, value=5).classes("w-full")
                    controls["seed"] = ui.number("Random seed", min=0, value=20260725).classes("w-full")
                    controls["workers"] = ui.number("Parallel workers", min=1, max=16, value=1).classes("w-full")

                ui.separator().classes("my-4")
                with ui.row().classes("w-full items-center gap-3"):
                    run_btn = ui.button("Start parameter search").props("color=primary icon=play_arrow").classes("px-6")
                    ui.label("Recommended: start with 10–20 candidates/model, then increase after screening.").classes("text-grey-6")

            with ui.element("div").classes("panel w-[420px] max-w-full"):
                ui.html('<div class="panel-title">Methodology</div>')
                ui.markdown(
                    """
- Fold-safe median imputation, variance filtering, feature selection, optional robust scaling.
- Ranking metric: mean validation ROC AUC.
- Also records PR AUC, balanced accuracy, MCC, SD, and runtime.
- Estimators use one internal worker; outer parallelism is controlled here.
                    """.strip()
                ).classes("text-grey-7")
                with ui.expansion("View effective search space", icon="tune").classes("w-full mt-2"):
                    ui.table.from_pandas(
                        pd.DataFrame(search_space_records(X.shape[1]))
                    ).props("flat dense").classes("w-full").style("max-height:360px; overflow:auto;")

        result_area = ui.column().classes("w-full gap-4")

        def run():
            _run_search(
                X, y,
                controls["models"].value,
                int(controls["n_trials"].value),
                int(controls["cv_folds"].value),
                int(controls["seed"].value),
                int(controls["workers"].value),
                result_area,
            )

        run_btn.on("click", run)

        stored = app.storage.client.get("parameter_search_results")
        if stored:
            _show_search_results(stored, result_area)


def _run_search(X, y, models, n_trials, cv_folds, seed, workers, container):
    if not models:
        ui.notify("Select at least one model family", type="warning")
        return
    container.clear()
    with container:
        with panel("Running search", width_full=True):
            ui.spinner("dots", size="lg", color="primary")
            ui.label("Searching hyperparameters... This can take a while.").classes("text-grey-6")
    ui.notify("Running search…", type="ongoing")
    try:
        candidates, best, spaces = run_parameter_search(
            X, y, models, n_trials, cv_folds, seed, workers,
            progress_cb=None,
        )
    except TypeError:
        # Older utility signature uses progress_callback instead of progress_cb.
        try:
            candidates, best, spaces = run_parameter_search(
                X, y, models, n_trials, cv_folds, seed, workers,
                progress_callback=None,
            )
        except Exception as exc:
            container.clear()
            ui.notify(f"Search failed: {exc}", type="negative")
            return
    except Exception as exc:
        container.clear()
        ui.notify(f"Search failed: {exc}", type="negative")
        return

    app.storage.client["parameter_search_results"] = {
        "candidates": candidates, "best": best, "spaces": spaces,
        "metadata": {
            "models": list(models), "trials_per_model": n_trials,
            "cv_folds": cv_folds, "seed": seed, "workers": workers,
            "rows": len(X), "features": X.shape[1],
        },
    }
    ui.notify("Search completed", type="positive")
    container.clear()
    _show_search_results(app.storage.client["parameter_search_results"], container)


def _show_search_results(stored, container):
    candidates, best, spaces = stored["candidates"], stored["best"], stored["spaces"]
    with container:
        top = best.iloc[0]
        with ui.row().classes("w-full gap-3 flex-wrap"):
            _metric_card("Best model", str(top["model"]))
            _metric_card("ROC AUC", f'{float(top["mean_cv_roc_auc"]):.4f}')
            _metric_card("PR AUC", f'{float(top["mean_cv_pr_auc"]):.4f}')
            _metric_card("Runtime", f'{float(top["runtime_seconds"]):.1f}s')

        with panel("Best configuration by model", width_full=True):
            cols = ["model", "mean_cv_roc_auc", "sd_cv_roc_auc", "mean_cv_pr_auc",
                    "mean_cv_balanced_accuracy", "mean_cv_mcc", "runtime_seconds"]
            ui.table.from_pandas(best[cols].round(4)).props("flat dense").classes("w-full")
            _bar_chart(best, ["mean_cv_roc_auc", "mean_cv_pr_auc"])

            sel = ui.select(best["model"].tolist(), value=best["model"].iloc[0], label="Inspect model").classes("w-full mt-3")
            param_area = ui.column().classes("w-full mt-2")
            candidate_area = ui.column().classes("w-full mt-4")

            def show_model_details(_=None):
                param_area.clear()
                candidate_area.clear()
                row = best.loc[best["model"] == sel.value].iloc[0]
                with param_area:
                    ui.label("Best parameters").classes("text-weight-medium")
                    ui.code(json.dumps(json.loads(row["best_parameters"]), indent=2), language="json").classes("w-full")
                with candidate_area:
                    ui.label(f"Candidates for {sel.value}").classes("text-weight-medium")
                    ui.table.from_pandas(
                        candidates.loc[candidates["model"] == sel.value].round(4)
                    ).props("flat dense").classes("w-full").style("max-height:520px; overflow:auto;")

            show_model_details()
            sel.on("update:model-value", show_model_details)

        with panel("Effective search space", width_full=True):
            ui.table.from_pandas(spaces).props("flat dense").classes("w-full").style("max-height:420px; overflow:auto;")

        with panel("Export", width_full=True):
            with ui.row().classes("gap-2 flex-wrap"):
                ui.button("Download all candidates", on_click=lambda: _download(candidates, "all_candidates.csv")).props("flat color=primary icon=download")
                ui.button("Download best by model", on_click=lambda: _download(best, "best_by_model.csv")).props("flat color=primary icon=download")
                ui.button("Download search space", on_click=lambda: _download(spaces, "search_space.csv")).props("flat color=primary icon=download")
                ui.button("Save complete run", on_click=lambda: _save_run(stored, container)).props("color=primary icon=save")


def _bar_chart(best, metrics):
    models = best["model"].tolist()
    series = []
    for metric in metrics:
        series.append({
            "name": metric,
            "type": "bar",
            "data": [float(best.loc[best["model"] == model, metric].iloc[0]) for model in models],
        })
    ui.echart({
        "tooltip": {"trigger": "axis"},
        "legend": {"data": metrics},
        "grid": {"left": 50, "right": 20, "top": 45, "bottom": 80},
        "xAxis": {"type": "category", "data": models, "axisLabel": {"rotate": 25}},
        "yAxis": {"type": "value"},
        "series": series,
    }).classes("w-full").style("height:320px;")


def _download(df, name):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    ui.download(buf.getvalue(), filename=name)


def _save_run(stored, container):
    paths = save_search_results(
        SEARCH_DIR, stored["candidates"], stored["best"],
        stored["spaces"], stored["metadata"],
    )
    ui.notify("Saved: " + ", ".join(p.name for p in paths), type="positive")


def _results_view(container):
    with container:
        csv_files = sorted(SEARCH_DIR.glob("*.csv"), reverse=True)
        if not csv_files:
            with panel("Saved results", width_full=True):
                ui.label("No saved parameter-search CSV files found.").classes("text-grey-6")
            return
        with panel("Saved results", width_full=True):
            sel = ui.select([p.name for p in csv_files], value=csv_files[0].name, label="Saved result").classes("w-full")
            area = ui.column().classes("w-full mt-3")

            def show(_=None):
                area.clear()
                df = pd.read_csv(SEARCH_DIR / sel.value)
                with area:
                    ui.table.from_pandas(df).props("flat dense").classes("w-full").style("max-height:650px; overflow:auto;")
            show()
            sel.on("update:model-value", show)


def _delete_view(container):
    with container:
        files = sorted([p for p in SEARCH_DIR.iterdir() if p.is_file()], key=lambda p: p.name)
        if not files:
            with panel("Delete saved files", width_full=True):
                ui.label("Nothing to delete.").classes("text-grey-6")
            return
        with panel("Delete saved files", width_full=True):
            ui.label("Select files to remove from saved parameter-search results.").classes("text-grey-6")
            sel = ui.select({p.name: p.name for p in files}, multiple=True, label="Select files").classes("w-full mt-2")
            confirm = ui.checkbox("I understand these files will be deleted").classes("mt-2")

            def do_delete():
                selected = list(sel.value or [])
                if not selected:
                    ui.notify("Select at least one file", type="warning")
                    return
                for name in selected:
                    (SEARCH_DIR / name).unlink()
                ui.notify(f"Deleted {len(selected)} file(s)", type="positive")
                container.clear()
                _delete_view(container)

            ui.button("Delete selected", on_click=do_delete).props("color=negative icon=delete").classes("mt-3").bind_visibility_from(confirm, "value")
