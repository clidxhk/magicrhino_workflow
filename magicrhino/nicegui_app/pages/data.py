"""Data page: load CSV, visualize distributions, correlation, clusters, scatter."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nicegui import ui, app

from ..state import page_header, panel, get_data_file, show_figure, show_svg, DATA_DIR
from plot.scatter import plot_scatter
from plot.molecules import draw_molecule
from plot.correlation import plot_correlation_matrix
from utils.data_utils import load_data, cluster


plt.rcParams["font.family"] = "Arial"


def build_page():
    page_header(
        "📈 Data Preparation",
        subtitle="Load, visualize, and explore feature relationships in your dataset.",
        quote="致知在格物，物格而後知至。 ——《禮記·大學》",
    )

    with ui.row().classes("w-full gap-4 items-stretch"):
        with ui.column().classes("w-64 flex-shrink-0 self-stretch"):
            async def on_upload(e):
                data = pd.read_csv(io.BytesIO(await e.file.read()))
                save_path = get_data_file()
                data.to_csv(save_path, index=False)
                ui.notify(f"Saved {len(data):,} rows to {save_path.name}", type="positive")
                _render(data, controls, output)

            ui.upload(
                label="Choose a CSV file",
                auto_upload=True,
                on_upload=on_upload,
                multiple=False,
            ).props("accept=.csv").classes("w-full h-full").style("height:100%; min-height:100%;")

        controls = ui.column().classes("flex-1 gap-2")

    # Output must be created after the top row so figures/tables stay below controls.
    output = ui.column().classes("w-full gap-2 mt-4")

    # Auto-load if data already exists
    data_file = get_data_file()
    if data_file.exists():
        data = pd.read_csv(data_file)
        _render(data, controls, output)
    else:
        with output:
            ui.label("Upload a CSV file to begin.").classes("text-grey-6")

def _render(data: pd.DataFrame, controls, output):
    controls.clear()
    output.clear()
    widgets: dict = {}
    with controls:
        with panel("Visualization option"):
            option = ui.toggle(
                ["Data Table", "Molecular Image", "Correlation Matrix", "Cluster", "Scatter Plot"],
                value="Data Table",
            ).classes("w-full")
            params = ui.column().classes("w-full gap-2 mt-3")
            draw_btn = ui.button("Draw").props("color=primary").classes("mt-4")
    with output:
        ui.markdown(f"**{len(data):,} rows** · {len(data.columns)} columns")

    def render_params():
        params.clear()
        widgets.clear()
        with params:
            if option.value == "Data Table":
                _build_data_table_params(data, widgets)
            elif option.value == "Molecular Image":
                idx = ui.number("Sample Index", min=0, max=len(data) - 1, value=0)
                widgets["idx"] = idx
            elif option.value == "Correlation Matrix":
                method = ui.toggle(["pearson", "spearman", "kendall"], value="pearson")
                widgets["method"] = method
            elif option.value == "Cluster":
                widgets["red"] = ui.toggle(["PCA", "UMAP", "t-SNE"], value="PCA")
                widgets["n"] = ui.number("n_components", min=2, max=10, value=2)
                widgets["perp"] = ui.number("Perplexity (t-SNE only)", min=2, value=30)
                ui.label("Adjust parameters and click Draw to update.").classes("text-grey-6")
            elif option.value == "Scatter Plot":
                cols = list(data.columns)
                widgets["x_sel"] = ui.select(cols, value="Index", label="X-axis")
                y_opts = [c for c in cols if c != "Index"]
                widgets["y_sel"] = ui.select(
                    y_opts or cols,
                    value=y_opts[0] if y_opts else cols[0],
                    label="Y-axis",
                )

    def draw_output(_=None):
        output.clear()
        with output:
            ui.markdown(f"**{len(data):,} rows** · {len(data.columns)} columns")
            try:
                if option.value == "Data Table":
                    _draw_data_table(data, widgets)
                elif option.value == "Molecular Image":
                    _show_molecule(data, int(widgets["idx"].value))
                elif option.value == "Correlation Matrix":
                    fig = plot_correlation_matrix(data, widgets["method"].value)
                    show_figure(fig)
                elif option.value == "Cluster":
                    X, y = load_data(data.copy())
                    fig, score = cluster(
                        widgets["red"].value,
                        int(widgets["n"].value),
                        X,
                        np.ravel(y),
                        perplexity=int(widgets["perp"].value),
                    )
                    show_figure(fig)
                    ui.label(f"Adjusted Rand Score: {score:.4f}")
                elif option.value == "Scatter Plot":
                    _draw_scatter(data, widgets)
            except Exception as exc:
                ui.notify(f"Error: {exc}", type="negative")

    def on_option_change(_=None):
        render_params()
        output.clear()
        with output:
            ui.markdown(f"**{len(data):,} rows** · {len(data.columns)} columns")
            ui.label("Configure the options above, then click Draw.").classes("text-grey-6")

    option.on("update:model-value", on_option_change)
    draw_btn.on("click", draw_output)

    # Show sub-options immediately. Plot/table output is generated only by Draw.
    render_params()
    output.clear()
    with output:
        ui.markdown(f"**{len(data):,} rows** · {len(data.columns)} columns")
        ui.label("Configure the options above, then click Draw.").classes("text-grey-6")

def _build_data_table_params(data, widgets):
    """Column picker for the data table. Updates the grid in place."""
    # The dataset can have thousands of columns (molecular descriptors).
    # Sending all of them at once overloads the websocket; a searchable
    # column picker keeps the AG Grid payload small.
    cols = list(data.columns)
    default_cols = [c for c in ("Index", "Y") if c in cols]
    default_cols += [c for c in cols if c not in default_cols][:6]
    col_sel = ui.select(
        cols, multiple=True, value=default_cols,
        label=f"Columns ({len(cols)} total)", with_input=True,
    ).classes("w-full")
    widgets["col_sel"] = col_sel

    def _update_columns(_e=None):
        grid = widgets.get("grid")
        if grid is None:
            return
        selected = list(col_sel.value or [])
        if not selected:
            return
        sub = data[selected].copy()
        obj_cols = sub.select_dtypes(include="object").columns
        if len(obj_cols):
            sub[obj_cols] = sub[obj_cols].astype(str)
        grid.options["columnDefs"] = [{"field": str(c)} for c in selected]
        grid.options["rowData"] = sub.to_dict("records")
        grid.update()

    col_sel.on_value_change(_update_columns)


def _draw_data_table(data, widgets):
    col_sel = widgets["col_sel"]
    selected = list(col_sel.value or [])
    if not selected:
        cols = list(data.columns)
        selected = [c for c in ("Index", "Y") if c in cols]
        selected += [c for c in cols if c not in selected][:6]
    sub = data[selected].copy()
    obj_cols = sub.select_dtypes(include="object").columns
    if len(obj_cols):
        sub[obj_cols] = sub[obj_cols].astype(str)
    ui.label(
        f"{len(data):,} rows x {len(data.columns):,} columns"
        f" - select columns above to browse"
    ).classes("text-grey-7")
    grid = ui.aggrid.from_pandas(
        sub, auto_size_columns=False,
    ).classes("w-full").style("height: 65vh;")
    widgets["grid"] = grid


def _draw_scatter(data, widgets):
    x = widgets["x_sel"].value
    y = widgets["y_sel"].value
    fig = plot_scatter(data, x, y)
    show_figure(fig)
    stats = {
        "Mean": data[y].mean(),
        "Median": data[y].median(),
        "Range": data[y].max() - data[y].min(),
        "Max": data[y].max(),
        "Min": data[y].min(),
    }
    with ui.row().classes("gap-2 flex-wrap"):
        for k, v in stats.items():
            with ui.element("div").classes("metric-card"):
                ui.html(f'<div class="label">{k}</div><div class="value">{v:.3f}</div>')


def _show_molecule(data, index):
    try:
        svg = draw_molecule(data, index)
        if svg is None:
            ui.label("Invalid molecule at this index").classes("text-negative")
            return
        show_svg(
            svg,
            filename=f"magicrhino-molecule-{index}.svg",
            style="width:240px;border:1px solid #e6eaf0;border-radius:10px;",
        )
    except Exception as exc:
        ui.notify(f"Could not draw molecule: {exc}", type="negative")
