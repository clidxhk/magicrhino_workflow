"""Atom Map page: per-atom descriptor contribution mapping."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from nicegui import ui

from ..state import page_header, panel, get_data_file, show_figure, show_svg, MODEL_DIR
from utils.model_utils import load_model_artifact
from utils.atom_map_utils import (
    compute_atom_contributions,
    get_selected_features,
    get_available_descriptor_names,
    draw_atom_heatmap,
    contribution_table,
)


def build_page():
    page_header(
        "🧬 Atom Map",
        subtitle="Map descriptor contributions to individual atoms using RDKit and Mordred descriptors.",
        quote="致廣大而盡精微。 ——《禮記·中庸》",
    )

    if not get_data_file().exists():
        ui.label("Upload labeled data on the Data page first.").classes("text-negative")
        return

    data = pd.read_csv(get_data_file())
    smiles_col = "Index"
    model_files = sorted(
        [p.name for p in MODEL_DIR.iterdir() if p.suffix.lower() in {".joblib", ".sav", ".h5"}],
    )
    if not model_files:
        ui.label("No saved models found. Train one on the Model page first.").classes("text-negative")
        return

    with ui.row().classes("w-full gap-3 flex-wrap"):
        _metric_card("Samples", f"{len(data):,}")
        _metric_card("Columns", f"{len(data.columns):,}")
        _metric_card("Saved models", f"{len(model_files):,}")
        _metric_card("SMILES column", smiles_col)

    source_holder: dict = {"value": {}, "artifact": None}

    with ui.row().classes("w-full gap-4 items-stretch mt-3"):
        with ui.element("div").classes("panel w-[520px] max-w-full self-stretch"):
            ui.html('<div class="panel-title">Atom workspace</div>')
            sel = ui.select(model_files, value=model_files[0], label="Model package").classes("w-full")
            sample_input = ui.number("Sample index", min=0, max=len(data) - 1, value=0).classes("w-full mt-3")
            ui.label("Select a model and sample, then choose an atom-mappable descriptor.").classes("text-grey-6 mt-2")
            ui.html('<div class="quote" style="margin:8px 0 12px;">析毫剖釐，以求其理。</div>')
            start_button = ui.button("Draw atom map").props("color=primary icon=hub").classes("w-full mt-3")

        with ui.element("div").classes("panel flex-1 min-w-[420px] min-h-[260px] self-stretch"):
            ui.html('<div class="panel-title">Descriptor options</div>')
            feature_select = ui.select([], label="Feature", multiple=False, with_input=True).classes("w-full")
            status_label = ui.label("Loading atom-mappable descriptors...").classes("text-grey-6 mt-2")
            with ui.expansion("Methodology", icon="science").classes("w-full mt-3"):
                ui.markdown(
                    """
- Only RDKit-native and resolvable Mordred descriptors can be mapped to atoms.
- For each atom, the descriptor is recomputed after perturbing that atom.
- Raw absolute descriptor changes are normalized by the sum of all atom-level changes.
- Percentages are atom contribution shares for the selected descriptor and sum to about 100%.
                    """.strip()
                ).classes("text-grey-7")

    # Output is created after controls so figures/tables stay below the options.
    out = ui.column().classes("w-full gap-4 mt-4")

    def load_features(_=None):
        out.clear()
        feature_select.options = []
        feature_select.value = None
        feature_select.update()
        try:
            artifact = load_model_artifact(sel.value, is_neural_network=sel.value.endswith(".h5"))
        except Exception as exc:
            status_label.set_text(f"Could not load model: {exc}")
            ui.notify(f"Could not load model: {exc}", type="negative")
            return

        if isinstance(artifact, dict) and "model" in artifact:
            feature_cols = get_selected_features(artifact)
        else:
            feature_cols = list(data.drop(columns=[smiles_col, "Y"], errors="ignore").columns)

        available = get_available_descriptor_names(feature_cols)
        source_holder["value"] = available
        source_holder["artifact"] = artifact
        if not available:
            status_label.set_text("No atom-mappable descriptors found in this model.")
            ui.notify("No atom-mappable descriptors found in this model.", type="warning")
            return

        options = {name: f"{name}  ({available[name]})" for name in sorted(available.keys())}
        feature_select.options = options
        feature_select.value = list(options.keys())[0]
        feature_select.update()
        rdkit_count = sum(1 for src in available.values() if src == "RDKit")
        mordred_count = sum(1 for src in available.values() if src == "Mordred")
        status_label.set_text(
            f"Loaded {len(available):,} atom-mappable descriptors: "
            f"{rdkit_count:,} RDKit · {mordred_count:,} Mordred"
        )

    def run(_=None):
        out.clear()
        feature_name = feature_select.value
        if not feature_name:
            ui.notify("Load a model and select a feature first.", type="warning")
            return
        source = source_holder["value"].get(feature_name, "RDKit")
        idx = max(0, min(int(sample_input.value), len(data) - 1))
        smiles = str(data.iloc[idx][smiles_col])
        try:
            contributions, mol = compute_atom_contributions(smiles, feature_name, source)
        except Exception as exc:
            with out:
                ui.notify(f"Error: {exc}", type="negative")
            return

        df = contribution_table(mol, contributions)
        df_sorted = df.sort_values("Contribution (%)", ascending=False).reset_index(drop=True)
        max_contrib = float(np.nanmax(contributions)) if len(contributions) else 0.0
        active_atoms = int((np.asarray(contributions, dtype=float) > 0).sum())

        with out:
            with ui.row().classes("w-full gap-3 flex-wrap"):
                _metric_card("Descriptor", feature_name)
                _metric_card("Source", source)
                _metric_card("Atoms", f"{mol.GetNumAtoms():,}")
                _metric_card("Active atoms", f"{active_atoms:,}")
                _metric_card("Max contribution", f"{max_contrib:.2f}%")

            result_card_height = "460px"
            molecule_svg_height = "315px"
            with ui.row().classes("w-full gap-4 items-stretch"):
                with ui.element("div").classes("panel flex-1 min-w-[520px] self-stretch flex flex-col").style(
                    f"height:{result_card_height}; overflow:hidden;"
                ):
                    ui.html(f'<div class="panel-title">Molecule #{idx} · {feature_name} ({source})</div>')
                    ui.label(f"SMILES: {smiles}").classes("text-grey-6").style(
                        "font-family:monospace;font-size:0.85rem;word-break:break-all;"
                        "display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;"
                        "overflow:hidden;"
                    )
                    svg = draw_atom_heatmap(mol, contributions, size=(560, 400))
                    with ui.element("div").classes("w-full flex-1 min-h-0 overflow-hidden"):
                        show_svg(
                            svg,
                            filename=f"atom-map-{idx}-{feature_name}.svg",
                            style=f"height:{molecule_svg_height}; max-height:{molecule_svg_height}; object-fit:contain;",
                        )

                with ui.element("div").classes("panel w-[420px] max-w-full self-stretch flex flex-col").style(
                    f"height:{result_card_height}; overflow:hidden;"
                ):
                    ui.html('<div class="panel-title">Atom contribution percentages</div>')
                    df_display = df_sorted.copy()
                    df_display["Contribution (%)"] = df_display["Contribution (%)"].round(2)
                    with ui.element("div").classes("w-full flex-1 min-h-0 overflow-auto"):
                        ui.table.from_pandas(df_display).props("flat dense").classes("w-full").style(
                            "height:auto;"
                        )

            with panel("Contribution percentage bar chart", width_full=True):
                _draw_contribution_bar(df, contributions, feature_name)

    load_features()
    start_button.on("click", run)
    sel.on("update:model-value", load_features)


def _metric_card(label: str, value: str):
    with ui.element("div").classes("metric-card min-w-[160px] flex-1"):
        ui.html(f'<div class="label">{label}</div><div class="value">{value}</div>')


def _draw_contribution_bar(df: pd.DataFrame, contributions, feature_name: str):
    plt.figure(figsize=(9, 4.2))
    labels = [f"{row.Element}{row.Atom}" for row in df.itertuples()]
    _cdict = {
        "red": ((0.0, 0.117, 0.117), (1.0, 0.961, 0.961)),
        "green": ((0.0, 0.533, 0.533), (1.0, 0.153, 0.153)),
        "blue": ((0.0, 0.898, 0.898), (1.0, 0.341, 0.341)),
        "alpha": ((0.0, 1, 1), (0.5, 1, 1), (1.0, 1, 1)),
    }
    red_blue = LinearSegmentedColormap("RedBlue", _cdict)
    contrib_arr = np.nan_to_num(np.asarray(contributions, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    denom = max(float(contrib_arr.max()), 1e-10)
    colors = red_blue(contrib_arr / denom)
    plt.bar(labels, contrib_arr, color=colors)
    plt.xlabel("Atom")
    plt.ylabel(f"Contribution to {feature_name} (%)")
    plt.title("Per-atom percentage share of descriptor perturbation")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    show_figure(plt.gcf())