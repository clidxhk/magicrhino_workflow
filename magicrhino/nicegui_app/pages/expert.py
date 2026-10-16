"""Expert page: AI-assisted analysis of metrics and SHAP visualizations."""
from __future__ import annotations

import base64
import os
import io

from nicegui import ui

from ..state import page_header, panel
from utils.expert_utils import ExpertChat, get_default_image_prompt


def build_page():
    page_header(
        "🤖 Expert",
        subtitle="AI-assisted interpretation of model metrics and SHAP visualizations.",
        quote="敏而好學，不恥下問。 ——《論語·公冶長》",
    )

    with ui.row().classes("w-full gap-3 flex-wrap"):
        _metric_card("Input modes", "Text / Image")
        _metric_card("Output", "Markdown")
        _metric_card("Default model", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        _metric_card("Provider", "OpenAI-compatible")

    body = ui.column().classes("w-full gap-4 mt-4")

    with ui.row().classes("w-full gap-4 items-stretch mt-3"):
        with ui.element("div").classes("panel w-[520px] max-w-full self-stretch"):
            ui.html('<div class="panel-title">API configuration</div>')
            api_key = ui.input("API Key", password=True, value=os.getenv("OPENAI_API_KEY", "")).classes("w-full")
            base_url = ui.input("Base URL", value=os.getenv("OPENAI_BASE_URL", "https://aihubmix.com/v1")).classes("w-full mt-2")
            model_in = ui.input("Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini")).classes("w-full mt-2")
            temp_label = ui.label("Temperature: 0.50").classes("text-caption text-slate-500 mt-3")
            temp = ui.slider(min=0.0, max=1.0, value=0.5, step=0.05).props("label label-always").classes("w-full")
            temp.on("update:model-value", lambda e: temp_label.set_text(f"Temperature: {float(e.args):.2f}"))

        with ui.element("div").classes("panel flex-1 min-w-[420px] self-stretch"):
            ui.html('<div class="panel-title">Expert workspace</div>')
            ui.label("Choose whether to analyze numerical metrics or interpret a model/SHAP image.").classes("text-grey-6")
            ui.html('<div class="quote" style="margin:8px 0 12px;">三人行，必有我師焉。 ——《論語·述而》</div>')
            chat_type = ui.toggle(["Text", "Image"], value="Text").classes("w-full")
            with ui.expansion("Prompt behavior", icon="psychology").classes("w-full mt-3"):
                ui.markdown(
                    """
- **Text** mode summarizes standard model metrics and suggests optimization actions.
- **Image** mode interprets SHAP or model-visualization figures.
- Responses are generated in Markdown and are intended as expert commentary, not absolute proof.
                    """.strip()
                ).classes("text-grey-7")

    def on_type(_=None):
        body.clear()
        if chat_type.value == "Text":
            _text_view(api_key, base_url, model_in, temp, body)
        else:
            _image_view(api_key, base_url, model_in, temp, body)

    chat_type.on("update:model-value", on_type)
    on_type()


def _text_view(api_key, base_url, model_in, temp, container):
    with container:
        with panel("Metric analysis", width_full=True):
            ui.label("Enter model metrics, then generate a concise expert assessment.").classes("text-grey-6")
            ui.html('<div class="quote" style="margin:6px 0 12px;">知不足而奮進，望遠山而前行。</div>')
            with ui.grid(columns=4).classes("w-full gap-3"):
                acc = ui.number("Accuracy", value=0.0, format="%.3f").classes("w-full")
                prec = ui.number("Precision", value=0.0, format="%.3f").classes("w-full")
                rec = ui.number("Recall", value=0.0, format="%.3f").classes("w-full")
                f1 = ui.number("F1", value=0.0, format="%.3f").classes("w-full")
                auc_tr = ui.number("AUC-Train", value=0.0, format="%.3f").classes("w-full")
                auc_te = ui.number("AUC-Test", value=0.0, format="%.3f").classes("w-full")
            run_btn = ui.button("Generate analysis", on_click=None).props("color=primary icon=auto_awesome").classes("mt-3")

        out = ui.column().classes("w-full gap-3")

        def run():
            out.clear()
            if not api_key.value.strip():
                ui.notify("Enter an API Key", type="warning")
                return
            metrics = {
                "Accuracy": acc.value,
                "Precision": prec.value,
                "Recall": rec.value,
                "F1 Score": f1.value,
                "AUC-Train": auc_tr.value,
                "AUC-Test": auc_te.value,
            }
            chat = ExpertChat(api_key.value.strip(), base_url.value.strip(), model_in.value.strip(), float(temp.value))
            try:
                with out:
                    with panel("Generating", width_full=True):
                        ui.spinner("dots", size="lg", color="primary")
                        ui.label("Consulting expert model...").classes("text-grey-6")
                result = chat.generate_text_analysis(metrics)
                out.clear()
                with out:
                    with panel("Expert analysis", width_full=True):
                        ui.markdown(result)
            except Exception as exc:
                out.clear()
                ui.notify(f"Error: {exc}", type="negative")
            finally:
                chat.close()

        run_btn.on("click", run)


def _image_view(api_key, base_url, model_in, temp, container):
    b64_holder = {"value": None}
    with container:
        with ui.row().classes("w-full gap-4 items-stretch"):
            with ui.element("div").classes("panel w-[460px] max-w-full self-stretch"):
                ui.html('<div class="panel-title">Upload image</div>')
                up = ui.upload(label="Choose an image", auto_upload=True, multiple=False).props(
                    "accept=.jpg,.jpeg,.png,.gif,.bmp,.webp"
                ).classes("w-full")
                preview = ui.column().classes("w-full gap-2 mt-3")

            with ui.element("div").classes("panel flex-1 min-w-[420px] self-stretch"):
                ui.html('<div class="panel-title">Image prompt</div>')
                user_text = ui.textarea(
                    "Prompt",
                    value="Explain the contribution of features to model predictions by image",
                ).classes("w-full").props("autogrow")
                with ui.expansion("Default image-analysis instruction", icon="article").classes("w-full mt-2"):
                    ui.label(get_default_image_prompt()).classes("text-grey-6").style("font-size:0.82rem;white-space:pre-wrap;")
                run_btn = ui.button("Generate image analysis", on_click=None).props("color=primary icon=auto_awesome").classes("mt-3")

        result_area = ui.column().classes("w-full gap-3")

        async def on_upload(e):
            preview.clear()
            buf = await e.file.read()
            b64 = base64.b64encode(buf).decode("utf-8")
            b64_holder["value"] = b64
            with preview:
                ui.image("data:image/png;base64," + b64).classes("w-full").style(
                    "max-width:420px;border-radius:10px;border:1px solid #e6eaf0;"
                )
                ui.label(f"Image loaded: {len(buf) / 1024:.1f} KB").classes("text-caption text-slate-500")

        def run():
            result_area.clear()
            if not api_key.value.strip():
                ui.notify("Enter an API Key", type="warning")
                return
            if b64_holder["value"] is None:
                ui.notify("Upload an image first", type="warning")
                return
            chat = ExpertChat(api_key.value.strip(), base_url.value.strip(), model_in.value.strip(), float(temp.value))
            try:
                with result_area:
                    with panel("Generating", width_full=True):
                        ui.spinner("dots", size="lg", color="primary")
                        ui.label("Analyzing image...").classes("text-grey-6")
                result = chat.generate_image_analysis(b64_holder["value"], user_text.value)
                result_area.clear()
                with result_area:
                    with panel("Expert image analysis", width_full=True):
                        ui.markdown(result)
            except Exception as exc:
                result_area.clear()
                ui.notify(f"Error: {exc}", type="negative")
            finally:
                chat.close()

        up.on_upload(on_upload)
        run_btn.on("click", run)


def _metric_card(label: str, value: str):
    with ui.element("div").classes("metric-card min-w-[180px] flex-1"):
        ui.html(f'<div class="label">{label}</div><div class="value" style="font-size:1rem;word-break:break-all;">{value}</div>')
