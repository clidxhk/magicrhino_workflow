"""Shared NiceGUI layout: left navigation rail + top header."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from nicegui import ui

from .state import install_branding, LOGO_PATH, ACCENT


# Each entry: (key, label, icon, route, group)
PAGES: list[tuple[str, str, str, str, str]] = [
    ("hello",    "Hello",     "👋", "/hello",    "Overview"),
    ("collector","Collector", "🛒", "/collector","Overview"),
    ("data",     "Data",      "📈", "/data",     "Overview"),
    ("params",   "Params",    "🔎", "/params",   "Model"),
    ("model",    "Model",     "🧠", "/model",    "Model"),
    ("shap",     "SHAP",      "📊", "/shap",    "Explain & Predict"),
    ("atom_map","Atom",      "🧬", "/atom_map","Explain & Predict"),
    ("predict",  "Predict",   "🚀", "/predict", "Explain & Predict"),
    ("expert",   "Expert",    "🤖", "/expert",  "Explain & Predict"),
]


def build_shell() -> None:
    """Install global branding + a fixed left drawer (nav rail)."""
    install_branding()
    # Left navigation rail
    with ui.left_drawer(top_corner=True, bottom_corner=True).props("width=230 bordered"):
        with ui.column().classes("w-full gap-0 p-0"):
            # Brand block
            with ui.row().classes("items-center gap-3 px-4 py-5"):
                if LOGO_PATH.exists():
                    ui.image(str(LOGO_PATH)).classes("w-9 h-9").style(
                        "object-fit:contain;border-radius:8px;"
                    )
                with ui.column().classes("gap-0"):
                    ui.html('<div class="brand-title">MagicRhino</div>')
                    ui.html('<div class="brand-sub">ML Visualization</div>')
            ui.separator().props("color=white").style("opacity:0.12")

            groups: dict[str, list[tuple[str, str, str, str]]] = {}
            for key, label, icon, target, group in PAGES:
                groups.setdefault(group, []).append((key, label, icon, target))

            for group_name, items in groups.items():
                ui.html(
                    f'<div style="padding:16px 16px 6px;color:#94a3b8;'
                    f'font-size:0.65rem;letter-spacing:0.12em;text-transform:uppercase;font-weight:600;">'
                    f'{group_name}</div>'
                )
                for key, label, icon, target in items:
                    ui.link(label, target).classes("q-item q-item--dark row items-center gap-2") \
                        .props(f"icon={icon}")
