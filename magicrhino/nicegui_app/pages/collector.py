"""Collector page — molecular data collection with SQLite storage."""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path

import pandas as pd
from nicegui import ui, app

from ..state import page_header, SAVED_DB_DIR, panel, show_svg
from utils.collector_utils import (
    create_database, load_field_config, save_field_config,
    load_molecules_from_db, check_and_add_molecule, export_data,
    mol_to_image, DEFAULT_FIELDS,
)


# ---- per-client session state -------------------------------------------
def _state():
    """Lazily initialize per-client collector state."""
    if "collector_db_path" not in app.storage.client:
        default_db = str(SAVED_DB_DIR / "molecules.db")
        if not Path(default_db).exists():
            create_database(default_db)
        app.storage.client["collector_db_path"] = default_db
        app.storage.client["molecules_db"] = load_molecules_from_db(default_db)
        app.storage.client["field_config"] = load_field_config(default_db)
    return app.storage.client


def _available_dbs() -> list[Path]:
    if not SAVED_DB_DIR.exists():
        return []
    return sorted(SAVED_DB_DIR.glob("*.db"))


def _switch_db(path: str):
    state = _state()
    state["collector_db_path"] = path
    state["molecules_db"] = load_molecules_from_db(path)
    state["field_config"] = load_field_config(path)
    ui.navigate.to("/collector", reload=True)


def build_page():
    page_header(
        "🛒 Molecular Data Collection",
        subtitle="Collect, deduplicate, and export molecular data into SQLite databases.",
        quote="積土成山，風雨興焉；積水成淵，蛟龍生焉。 ——《荀子·勸學》",
    )
    state = _state()

    # ---- left column: DB management + stats ----
    with ui.row().classes("w-full gap-4 items-start"):
        with ui.column().classes("gap-2").style("width:300px;flex:0 0 300px;"):
            with panel("Database", width_full=True):
                dbs = _available_dbs()
                current = state["collector_db_path"]
                db_names = [p.name for p in dbs] or ["(none)"]
                cur_name = Path(current).name if current else db_names[0]
                ui.select(
                    db_names, value=cur_name,
                    label="Select Database",
                    on_change=lambda e: _switch_db(str(SAVED_DB_DIR / e.value)),
                ).classes("w-full")

                ui.separator().classes("my-2")
                ui.html('<div class="panel-title">New database</div>')
                new_db = ui.input("Database name", placeholder="e.g., activity_data").classes("w-full")
                ui.button("Create", on_click=lambda: _create_db(new_db.value)).classes("w-full").props("color=primary dense")

            with panel("Statistics", width_full=True):
                mdb = state["molecules_db"]
                total_records = sum(1 + len(i["duplicate_smiles"]) for i in mdb.values())
                _stat("Unique molecules", len(mdb))
                _stat("Total records", total_records)
                _stat("Duplicate detections", sum(i["count"] - 1 for i in mdb.values()))

        # ---- right column: form + table ----
        with ui.column().classes("gap-3").style("flex:1;min-width:0;"):
            _build_form()
            _build_table()


def _stat(label: str, value: int):
    ui.html(f'<div style="display:flex;justify-content:space-between;padding:4px 0;">'
            f'<span style="color:#64748b;font-size:0.85rem">{label}</span>'
            f'<b>{value}</b></div>')


def _create_db(name: str):
    if not name:
        ui.notify("Enter a database name", type="warning")
        return
    if not name.endswith(".db"):
        name += ".db"
    path = str(SAVED_DB_DIR / name)
    ok, msg = create_database(path)
    if ok:
        _switch_db(path)
        ui.notify(f"Database {name} created", type="positive")
    else:
        ui.notify(msg, type="negative")


def _build_form():
    state = _state()
    cfg = state["field_config"]

    with panel("Add molecule"):
        smiles_input = ui.input("SMILES *", placeholder="e.g., CCO, c1ccccc1").classes("w-full")
        # Dynamic custom-field inputs
        field_inputs: dict = {}
        custom_container = ui.column().classes("w-full gap-2")

        def render_fields():
            field_inputs.clear()
            custom_container.clear()
            with custom_container:
                for key, info in cfg.items():
                    if key == "smiles" or not info["enabled"]:
                        continue
                    label = info["name"] + (" *" if info["required"] else "")
                    if info["type"] == "textarea":
                        field_inputs[key] = ui.textarea(label).classes("w-full")
                    elif info["type"] == "number":
                        field_inputs[key] = ui.number(label).classes("w-full")
                    elif info["type"] == "select":
                        opts = info.get("options", [])
                        field_inputs[key] = ui.select(opts, label=label).classes("w-full")
                    elif info["type"] == "checkbox":
                        field_inputs[key] = ui.switch(label)
                    elif info["type"] == "date":
                        field_inputs[key] = ui.date_input(label).classes("w-full")
                    else:
                        field_inputs[key] = ui.input(label).classes("w-full")
        render_fields()

        result_area = ui.column().classes("w-full gap-2")
        image_area = ui.column().classes("w-full gap-2")

        def submit():
            if not smiles_input.value:
                ui.notify("Enter a SMILES", type="warning")
                return
            custom = {}
            for k, inp in field_inputs.items():
                info = cfg[k]
                v = inp.value
                if info["type"] == "checkbox":
                    custom[k] = bool(v)
                elif info["type"] == "date" and v:
                    custom[k] = str(v)
                else:
                    custom[k] = v if v is not None else ""
            # Required validation
            for k, info in cfg.items():
                if k != "smiles" and info["enabled"] and info["required"] and not custom.get(k):
                    ui.notify(f"{info['name']} is required", type="warning")
                    return
            status, msg, canon, mol = check_and_add_molecule(
                smiles_input.value, custom, state["molecules_db"], state["collector_db_path"]
            )
            result_area.clear()
            image_area.clear()
            with result_area:
                if status == "duplicate":
                    ui.notify(msg, type="warning")
                elif status == "added":
                    ui.notify(msg, type="positive")
                else:
                    ui.notify(msg, type="negative")
            if mol is not None:
                with image_area:
                    ui.html('<div class="panel-title">Molecular structure</div>')
                    svg = mol_to_image(mol, size=(240, 240))
                    if svg:
                        show_svg(
                            svg,
                            filename="magicrhino-collected-molecule.svg",
                            style="width:240px;border:1px solid #e6eaf0;border-radius:10px;",
                        )
            smiles_input.value = ""
            # Refresh table
            _refresh_table()

        ui.button("Check and Submit", on_click=submit).props("color=primary").classes("mt-2")
        # Re-render custom fields when config changes externally
        app.storage.client["collector_render_fields"] = render_fields


def _build_table():
    state = _state()
    state["collector_table_holder"] = ui.column().classes("w-full gap-2")
    _refresh_table()


def _refresh_table():
    state = _state()
    holder = state.get("collector_table_holder")
    if not holder:
        return
    holder.clear()
    with holder:
        mdb = state["molecules_db"]
        if not mdb:
            ui.label("No molecules yet. Enter a SMILES to start.").classes("text-grey-6")
            return
        cfg = state["field_config"]
        rows = []
        for idx, (canon, info) in enumerate(mdb.items(), 1):
            row = {
                "No.": idx,
                "Original SMILES": info["original_smiles"],
                "Canonical SMILES": canon,
                "Input Time": info["timestamp"],
                "Duplicate Count": info["count"],
                "Status": "🔄 Duplicate" if info["count"] > 1 else "✅ Unique",
            }
            for key, fi in cfg.items():
                if key == "smiles" or not fi["enabled"]:
                    continue
                v = info.get("custom_fields", {}).get(key, "")
                if fi["type"] == "checkbox":
                    v = "Yes" if v else "No"
                row[fi["name"]] = v
            rows.append(row)
        df = pd.DataFrame(rows)
        ui.table.from_pandas(df).props("flat dense").classes("w-full")
        ui.button("📥 Export CSV", on_click=lambda: _export(df)).props("flat color=primary dense")


def _export(df: pd.DataFrame):
    if df.empty:
        return
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    name = f"molecules_{datetime.now():%Y%m%d_%H%M%S}.csv"
    ui.download(buf.getvalue(), filename=name)
