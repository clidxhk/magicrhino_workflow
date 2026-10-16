"""Hello / overview page — MagicRhino landing."""
from __future__ import annotations

from nicegui import ui

from ..state import page_header


CARDS = [
    ("📈", "Data", "Load, visualize, and analyze your dataset. Correlation, scatter, clustering, molecular images.", "/data"),
    ("🛒", "Collector", "Collect molecular data into SQLite databases with custom fields and duplicate detection.", "/collector"),
    ("🔎", "Params", "Stratified CV with fold-safe preprocessing and randomized hyperparameter search.", "/params"),
    ("🧠", "Model", "Train a publication-style pipeline and evaluate with ROC, PR, confusion matrix.", "/model"),
    ("📊", "SHAP", "Interpret predictions with SHAP summary, decision, force, waterfall, interaction.", "/shap"),
    ("🧬", "Atom", "Map RDKit and Mordred descriptor contributions to individual atoms in the molecule.", "/atom_map"),
    ("🚀", "Predict", "Upload an external dataset and predict with a saved model + SHAP analysis.", "/predict"),
    ("🤖", "Expert", "AI-assisted interpretation of model metrics and SHAP plots.", "/expert"),
]


def build_page():
    page_header(
        "👋 灵犀 MagicRhino",
        subtitle="Interactive Machine Learning Visualization Platform",
        quote="天地與我並生，而萬物與我爲一。 ——《莊子·齊物論》",
    )

    ui.markdown(
        "Apply machine learning to your research effortlessly — without "
        "writing any code. Click through the modules on the left to "
        "collect data, search parameters, train models, and explain "
        "predictions."
    )

    # Feature card grid (responsive)
    with ui.row().classes("w-full flex-wrap gap-4"):
        for icon, title, desc, link in CARDS:
            with ui.element("div").classes("panel").style(
                "width:300px;min-height:184px;cursor:pointer;display:flex;"
                "flex-direction:column;gap:6px;"
            ):
                with ui.row().classes("items-center gap-2"):
                    ui.html(f'<span style="font-size:1.4rem">{icon}</span>')
                    ui.html(f'<div class="panel-title" style="margin:0">{title}</div>')
                ui.html(f'<div style="color:#64748b;font-size:0.88rem">{desc}</div>')
                ui.button("Open", on_click=lambda l=link: ui.navigate.to(l)).props(
                    "flat dense color=primary"
                ).style("margin-top:auto;align-self:flex-start;")
