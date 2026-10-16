"""Shared state, styles, and layout for the MagicRhino NiceGUI app."""
from __future__ import annotations

import base64
import html
import io
import re
import uuid

from pathlib import Path
from typing import Callable

from nicegui import app, ui

# ---- Project paths -------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "saved_model"
SAVED_DB_DIR = PROJECT_DIR / "saved_db"
SEARCH_DIR = DATA_DIR / "parameter_search"
LOGO_PATH = PROJECT_DIR / "logo.png"

for directory in (DATA_DIR, MODEL_DIR, SAVED_DB_DIR, SEARCH_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# ---- Brand colors --------------------------------------------------------
PRIMARY = "#1f77b4"
ACCENT = "#008BFB"
WARN = "#FF004F"
INK = "#0f172a"
MUTED = "#64748b"
SURFACE = "#ffffff"
PAGE_BG = "#f5f7fb"
RAIL_BG = "#0f172a"

# Inject a coherent visual identity once at startup.
CSS = f"""
:root {{
  --q-color-primary: {PRIMARY};
  --q-color-secondary: {ACCENT};
  --q-color-negative: {WARN};
}}
body {{
  background: {PAGE_BG};
  color: {INK};
  font-family: 'Inter', 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
}}
.q-drawer {{
  background: {RAIL_BG} !important;
  color: #e2e8f0;
}}
.q-drawer .q-item {{
  color: #cbd5e1;
  border-radius: 10px;
  margin: 4px 10px;
  text-decoration: none;
}}
.q-drawer .q-item.q-router-link--active,
.q-drawer .q-item.active {{
  background: rgba(31,119,180,0.18);
  color: #fff;
  box-shadow: inset 3px 0 0 {ACCENT};
}}
.q-drawer .q-item:hover {{
  background: rgba(255,255,255,0.06);
}}
.q-drawer .brand-title {{
  color: #ffffff;
  text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}}
.q-drawer .brand-sub {{
  color: #94a3b8;
}}
.q-header {{
  background: {SURFACE} !important;
  color: {INK} !important;
  border-bottom: 1px solid #e2e8f0;
  box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}}
.brand-title {{
  font-weight: 700;
  font-size: 1.15rem;
  letter-spacing: 0.2px;
  color: {INK};
}}
.brand-sub {{
  font-size: 0.72rem;
  color: {MUTED};
  letter-spacing: 0.08em;
  text-transform: uppercase;
}}
.page-title {{
  font-size: 1.55rem;
  font-weight: 700;
  color: {INK};
  margin: 0;
}}
.page-subtitle {{
  color: {MUTED};
  font-size: 0.92rem;
}}
.metric-card {{
  background: {SURFACE};
  border: 1px solid #e6eaf0;
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}}
.metric-card .label {{
  color: {MUTED};
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}}
.metric-card .value {{
  font-size: 1.55rem;
  font-weight: 700;
  color: {INK};
  margin-top: 4px;
}}
.panel {{
  background: {SURFACE};
  border: 1px solid #e6eaf0;
  border-radius: 14px;
  padding: 18px 20px;
  box-shadow: 0 1px 2px rgba(15,23,42,0.03);
}}
.panel-title {{
  font-size: 1.02rem;
  font-weight: 600;
  color: {INK};
  margin-bottom: 8px;
}}
.quote {{
  color: {MUTED};
  font-style: italic;
  border-left: 3px solid {ACCENT};
  padding-left: 12px;
  margin: 8px 0;
}}
.nicegui-content {{
  padding: 20px 28px 60px;
}}
.svg-download {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  width: fit-content;
  color: {ACCENT};
  font-size: 0.78rem;
  font-weight: 600;
  text-decoration: none;
  margin-top: 4px;
}}
.svg-download:hover {{
  text-decoration: underline;
}}
.svg-download::before {{
  content: "↧";
  font-size: 0.8rem;
}}
"""


def install_branding() -> None:
    """Inject global CSS and the Inter font."""
    ui.add_head_html(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">'
    )
    ui.add_css(CSS)


# ---- Shared in-memory session state --------------------------------------
def get_data_file() -> Path:
    return DATA_DIR / "data_labeled.csv"


def data_is_loaded() -> bool:
    return get_data_file().exists()


def list_model_files() -> list[Path]:
    if not MODEL_DIR.exists():
        return []
    return sorted(
        [p for p in MODEL_DIR.iterdir() if p.suffix.lower() in {".joblib", ".sav", ".h5"}],
        key=lambda p: p.name,
    )


def metric_card(label: str, value, prefix: str = "") -> None:
    """Render a compact metric card."""
    text = "—" if value is None else (f"{prefix}{value:.3f}" if isinstance(value, (int, float)) else str(value))
    with ui.element("div").classes("metric-card"):
        ui.html(f'<div class="label">{label}</div><div class="value">{text}</div>')


def _svg_data_uri(svg: str | bytes) -> str:
    """Return a base64 SVG data URI."""
    svg_text = svg.decode("utf-8", errors="replace") if isinstance(svg, bytes) else str(svg)
    source = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return "data:image/svg+xml;base64," + source


def _safe_svg_filename(filename: str | None) -> str:
    """Create a conservative SVG download filename."""
    if not filename:
        filename = f"magicrhino-figure-{uuid.uuid4().hex[:8]}.svg"
    filename = Path(filename).name
    if not filename.lower().endswith(".svg"):
        filename += ".svg"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", filename)


def show_svg(svg: str | bytes, *, filename: str | None = None, style: str = "") -> None:
    """Render an SVG image and expose a real backend-triggered SVG download button."""
    svg_text = svg.decode("utf-8", errors="replace") if isinstance(svg, bytes) else str(svg)
    uri = _svg_data_uri(svg_text)
    download_name = _safe_svg_filename(filename)

    def _download_svg() -> None:
        ui.download(svg_text.encode("utf-8"), filename=download_name)

    with ui.column().classes("w-full gap-1"):
        image = ui.image(uri).classes("w-full").props("fit=contain")
        if style:
            image.style(style)
        ui.button("Download SVG", icon="download", on_click=_download_svg)             .props("flat dense color=primary no-caps")             .classes("self-start")


def show_figure(fig, *, dpi: int = 150, filename: str | None = None) -> None:
    """Render a Matplotlib figure as editable, browser-safe SVG."""
    buffer = io.StringIO()
    try:
        import matplotlib as mpl

        # Keep text as <text> nodes instead of converting it to paths, so the
        # downloaded SVG remains editable in Illustrator/Inkscape/Figma.
        with mpl.rc_context({"svg.fonttype": "none"}):
            fig.savefig(buffer, format="svg", dpi=dpi, bbox_inches="tight")
        show_svg(buffer.getvalue(), filename=filename)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)

def page_header(title: str, subtitle: str = "", quote: str = "") -> None:
    """Standard page header with optional quote."""
    ui.html(f'<h1 class="page-title">{title}</h1>')
    if subtitle:
        ui.html(f'<div class="page-subtitle">{subtitle}</div>')
    if quote:
        ui.html(f'<div class="quote">{quote}</div>')
    ui.space().style("height: 8px")


import contextlib


@contextlib.contextmanager
def panel(title: str | None = None, width_full: bool = False):
    """A rounded surface card. Use as a context manager."""
    cls = "panel" + (" w-full" if width_full else "")
    with ui.element("div").classes(cls) as card:
        if title:
            ui.html(f'<div class="panel-title">{title}</div>')
        yield card
