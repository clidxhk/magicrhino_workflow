"""MagicRhino — NiceGUI entry point.

Run with:
    python app_nicegui.py
or:
    uvicorn app_nicegui:app --reload

This file wires the shared shell (branding + left nav) and registers
every page under its NiceGUI client-side route.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from nicegui import ui, app

from nicegui_app.layout import build_shell
from nicegui_app.state import PRIMARY

# Register all pages (importing them registers @ui.page handlers).
from nicegui_app.pages import (
    hello,
    collector,
    data,
    params,
    model,
    shap,
    predict,
    expert,
    atom_map,
)


@ui.page("/")
def index():
    """Root redirects to the Hello page."""
    ui.navigate.to("/hello")


def build(page_builder):
    """Wrap a page builder so every page shares the branding shell + header."""
    def wrapper():
        ui.colors(primary=PRIMARY)
        build_shell()
        # Top header
        with ui.header().classes("items-center justify-between"):
            with ui.row().classes("items-center gap-3"):
                ui.html('<span class="brand-title">MagicRhino</span>')
                ui.html('<span class="brand-sub" style="opacity:0.6;">·</span>')
                ui.html('<span class="brand-sub">Interactive ML Visualization</span>')
            with ui.row().classes("items-center gap-2"):
                ui.link("Overview", "/hello").classes("text-primary")
        page_builder()
    return wrapper


# Wire every page through the shared shell.
for path, module, fn_name in [
    ("/hello",     hello,     "build_page"),
    ("/collector", collector, "build_page"),
    ("/data",      data,      "build_page"),
    ("/params",    params,    "build_page"),
    ("/model",     model,     "build_page"),
    ("/shap",      shap,     "build_page"),
    ("/predict",   predict,  "build_page"),
    ("/expert",    expert,   "build_page"),
    ("/atom_map", atom_map, "build_page"),
]:
    ui.page(path)(build(getattr(module, fn_name)))


# Start the server when executed directly.
if __name__ in {"__main__", "__mp_main__"}:
    # storage_secret enables app.storage.client (used by some pages).
    ui.run(
        title="MagicRhino — ML Visualization",
        port=8080,
        reload=True,
        show=True,
        favicon=str(__import__("pathlib").Path(__file__).resolve().parent / "logo.png"),
        dark=False,
        storage_secret="magicrhino-nicegui-session-v1",
    )
