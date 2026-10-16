# -*- coding: utf-8 -*-
"""
Photosensitizer & ROS Literature Analyzer

Analyze scientific PDFs to extract:
  1. Photosensitizer (PS) identification
  2. Reactive oxygen species (ROS) types
  3. ROS detection methods (probes, EPR, etc.)

Usage: Run python analyze_photosensitizers.py
"""

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber
import openai

# Fix Python 3.14 stdout encoding on Windows
sys.stdout.reconfigure(encoding="utf-8")


# ===========================================================================

def _load_env(env_path: str | None = None) -> dict[str, str]:
    """Read KEY=VALUE pairs from .env file. Simple, no dependencies."""
    if env_path is None:
        env_path = str(Path(__file__).parent / ".env")
    vals: dict[str, str] = {}
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    vals[k.strip()] = v.strip().strip('\'"')
    except FileNotFoundError:
        print(f"[WARN] .env not found at {env_path}, using defaults")
    return vals

_env = _load_env()

# ==== CONFIG (edit these values as needed) =================================
# ===========================================================================

_PDF_DIR    = r"F:\Github\Thesis\Project_1\literature_analysis\input"
_OUTPUT_DIR = r"F:\Github\Thesis\Project_1\literature_analysis\output"
_API_KEY    = _env.get("OPENAI_API_KEY", "sk-YOUR_API_KEY_HERE")
_API_BASE   = _env.get("OPENAI_API_BASE", "https://api.openai.com/v1")
_MODEL      = _env.get("OPENAI_MODEL", "gpt-4o")

_TEMPERATURE          = 0.1
_MAX_RETRIES          = 3
_DELAY_BETWEEN_CALLS  = 1.0

# ===========================================================================
# PDF PROCESSOR
# ===========================================================================

def extract_text_from_pdf(pdf_path: Path) -> str:
    pages: list[str] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    pages.append(f"\n--- Page {i} ---\n{text.strip()}")
    except Exception as e:
        print(f"  [WARN] Failed to extract text from {pdf_path.name}: {e}")
        return ""
    return "\n".join(pages)


def clean_text(raw: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", raw)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
    text = re.sub(r"[\u2000-\u200a\u202f\u205f\u3000]", " ", text)
    return text.strip()

def strip_references(text: str) -> str:
    """Remove everything from References / Acknowledgements onward."""
    for marker in ["\nReferences\n", "\nREFERENCES\n", "\nAcknowledgements\n", "\nACKNOWLEDGMENTS\n"]:
        idx = text.find(marker)
        if idx > len(text) * 0.5:
            return text[:idx]
    return text



# ===========================================================================
# LLM PROMPTS — Two-pass: (1) PS scan, (2) PS-centric deep extraction
# ===========================================================================

SYSTEM_PROMPT = (
    "You are a senior researcher specializing in photodynamic therapy (PDT) and "
    "photosensitizer chemistry. Your task is to carefully read scientific paper "
    "text and extract structured information about photosensitizers (PS), the "
    "reactive oxygen species (ROS) they generate, and the methods used to detect "
    "those ROS.\n\n"
    "IMPORTANT RULES:\n"
    "1. Only report what is explicitly stated in the provided text. Do not infer.\n"
    "2. If information is not found, state \"Not mentioned\" — never guess.\n"
    "3. Output ONLY valid JSON as specified — no extra commentary.\n"
    "4. Be precise with chemical names, probe names, and numerical values."
)

# --- Pass 1: PS scan (lightweight, names only) ---

PASS1_PROMPT = (
    "Identify ONLY the primary photosensitizer(s) that are designed, synthesized, "
    "formulated, experimentally evaluated, or otherwise treated as main study "
    "objects in this paper.\n\n"
    "Do NOT include reference standards, commercial benchmark PS, quantum-yield "
    "standards, positive controls, negative controls, comparison controls, probe "
    "reagents, quenchers, scavengers, or literature-only examples. For example, "
    "Ce6, Rose Bengal/RB, ICG, methylene blue, or porphyrin standards should be "
    "excluded when they are only used as controls, standards, comparators, or "
    "calibration references rather than as the paper's target PS.\n\n"
    "If a compound is only mentioned in Introduction, References, methods for "
    "calibration, or as a standard for singlet oxygen quantum yield, do not list it.\n"
    "When uncertain whether an item is a main PS or merely a reference/control, "
    "exclude it.\n\n"
    "Return ONLY the following JSON — no extra text:\n"
    "{\n"
    "  \"has_photosensitizer\": true/false,\n"
    '  "photosensitizers": [\n'
    '    {"name": "full chemical or material name", "abbreviation": "short code used in paper"},\n'
    '    ...\n'
    "  ]\n"
    "}\n\n"
    "Paper text:\n---\n{text}\n---\n"
)

# --- Pass 2: one-PS-at-a-time deep extraction ---

PASS2_PROMPT = (
    "Analyze ONLY this target photosensitizer (PS): {target_ps}\n\n"
    "Extract the following information for this target PS from the paper text. "
    "Do not analyze other photosensitizers, controls, reference standards, "
    "commercial benchmarks, quantum-yield standards, probes, quenchers, or "
    "scavengers except when they are needed as evidence or controls for this "
    "target PS.\n\n"
    "If the paper compares this target PS with Ce6, Rose Bengal/RB, ICG, "
    "methylene blue, or another benchmark/reference PS, keep those comparators "
    "inside evidence/control fields only. Do not create separate entries for them.\n\n"
    "1. BASIC PROPERTIES:\n"
    "   - Full chemical name and abbreviation\n"
    "   - Molecular type (small_molecule / nanomaterial / polymer / MOF / other)\n"
    "   - Core structure description (scaffold, donors, acceptors, substituents, metal center)\n"
    "   - Excitation wavelength (nm)\n"
    "   - Type I, Type II, or both? Cite the specific evidence from the paper.\n"
    "2. LIGHT CONDITIONS:\n"
    "   - Light source, power density (mW/cm2), irradiation time\n"
    "3. ROS GENERATED (per PS):\n"
    "   For each ROS type generated by this specific PS, list:\n"
    "   - ROS type (singlet_oxygen / superoxide / hydroxyl_radical / hydrogen_peroxide / peroxynitrite / other)\n"
    "   - ROS label (1O2 / O2- / .OH / H2O2 / ...)\n"
    "   - What evidence supports this ROS generation?\n"
    "   - Quantum yield if reported\n"
    "   - Which detection methods confirmed this specific ROS?\n"
    "4. DETECTION METHODS (per PS, with per-method detail):\n"
    "   For each method used to detect ROS from this PS:\n"
    "   - Category (fluorescent_probe / epr / absorbance / fluorescence / chemiluminescence / biological_assay / other)\n"
    "   - Probe/reagent name and full chemical name\n"
    "   - Which ROS does it target?\n"
    "   - Detection details (how it works, what was measured)\n"
    "   - Excitation/emission wavelength if applicable\n"
    "   - Concentration used\n"
    "   - Incubation time\n"
    "   - Positive/negative controls\n"
    "5. DISTINGUISH target-PS evidence from comparator/control evidence. If a ROS "
    "signal belongs only to another PS, control, standard, probe, quencher, or "
    "scavenger, do not report it as generated by this target PS.\n"
    "6. If a detection method was used for this target PS and also for comparators, "
    "list it only as this target PS's method, with comparator details in controls.\n\n"
    "Return ONLY the following JSON — no extra text:\n"
    "{\n"
    '  "photosensitizer": {\n'
    '      "name": "full name",\n'
    '      "abbreviation": "abbrev",\n'
    '      "type": "small_molecule | nanomaterial | polymer | MOF | other",\n'
    '      "structure_summary": "...",\n'
    '      "excitation_wavelength_nm": null or number,\n'
    '      "ps_classification": "type_I | type_II | both | not_specified",\n'
    '      "classification_evidence": "...",\n'
    '      "light_source": "...",\n'
    '      "power_density": "...",\n'
    '      "irradiation_time": "...",\n'
    '      "ros_assessment": [\n'
    "        {\n"
    '          "ros_name": "singlet_oxygen | superoxide | hydroxyl_radical | hydrogen_peroxide | peroxynitrite | other",\n'
    '          "ros_label": "1O2 | O2- | .OH | H2O2 | ...",\n'
    '          "evidence": "specific experimental evidence cited in the paper",\n'
    '          "quantum_yield": null or number,\n'
    '          "quantum_yield_standard": "standard used if applicable",\n'
    '          "detection_methods_used": ["probe_or_method_name", ...]\n'
    "        }\n"
    "      ],\n"
    '      "detection_methods": [\n'
    "        {\n"
    '          "category": "fluorescent_probe | epr | absorbance | fluorescence | chemiluminescence | biological_assay | other",\n'
    '          "probe_name": "...",\n'
    '          "full_name": "...",\n'
    '          "target_ros": ["..."],\n'
    '          "detection_details": "...",\n'
    '          "excitation_emission_nm": "...",\n'
    '          "concentration": "...",\n'
    '          "incubation_time": "...",\n'
    '          "controls": "..."\n'
    "        }\n"
    "      ]\n"
    "  },\n"
    '  "notes": "..."\n'
    "}\n\n"
    "Paper text:\n---\n{text}\n---\n"
)# ---------------------------------------------------------------------------
# API caller
# ---------------------------------------------------------------------------

_CLIENT = openai.OpenAI(api_key=_API_KEY, base_url=_API_BASE)

def call_openai_chat(system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = _CLIENT.chat.completions.create(
                model=_MODEL,
                temperature=_TEMPERATURE,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
            return _extract_json(content)
        except openai.RateLimitError:
            wait = min(2 ** attempt, 60)
            print(f"  Rate limited. Waiting {wait}s...")
            time.sleep(wait)
        except openai.InternalServerError:
            print(f"  Server error, attempt {attempt}/{_MAX_RETRIES}")
            time.sleep(2 * attempt)
        except openai.APITimeoutError:
            print(f"  Timeout, attempt {attempt}/{_MAX_RETRIES}")
            time.sleep(2 * attempt)
        except openai.APIConnectionError as e:
            print(f"  Connection error: {e}")
            time.sleep(2 * attempt)
        except openai.APIStatusError as e:
            print(f"  API error {e.status_code}: {str(e)[:200]}")
            return None
        except Exception as e:
            print(f"  Unexpected error: {e}")
            return None
    print("  All retries exhausted.")
    return None


def _extract_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        if m:
            text = m.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"  [WARN] Failed to parse JSON from: {text[:200]}...")
        return {"raw_response": text, "parse_error": True}
# ===========================================================================
# CORE ANALYSIS PIPELINE
# ===========================================================================

@dataclass
class AnalysisResult:
    filename: str
    has_photosensitizer: bool | None = None
    photosensitizers: list[dict] = field(default_factory=list)
    ros_types: list[dict] = field(default_factory=list)
    detection_methods: list[dict] = field(default_factory=list)
    chunk_results: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def analyze_paper(pdf_path: Path) -> AnalysisResult:
    """Two-pass analysis: (1) scan PS names, (2) PS-centric deep extraction."""
    result = AnalysisResult(filename=pdf_path.name)

    # Phase 1: Extract, clean, strip references
    print(f"\n[1/3] Extracting text from: {pdf_path.name}")
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        result.errors.append("Failed to extract any text from PDF")
        return result

    text = clean_text(raw_text)
    text = strip_references(text)
    print(f"  Body: ~{len(text):,} chars")

    # Phase 2: Two-pass LLM analysis
    print(f"\n[2/3] Pass 1 — Scanning photosensitizers...")
    pass1 = call_openai_chat(SYSTEM_PROMPT, PASS1_PROMPT.replace("{text}", text))
    if not pass1 or not pass1.get("has_photosensitizer"):
        if pass1:
            result.has_photosensitizer = False
        print("  No photosensitizer found.")
        return result

    result.has_photosensitizer = True
    ps_list = pass1.get("photosensitizers", [])
    if not ps_list:
        print("  PS list empty.")
        return result

    ps_names = [p.get("abbreviation") or p.get("name", "?") for p in ps_list]
    print(f"  Found {len(ps_names)} PS: {', '.join(ps_names)}")
    time.sleep(_DELAY_BETWEEN_CALLS)

    print(f"\n[3/3] Pass 2 — One-PS-at-a-time deep extraction...")
    pass2_results: list[dict[str, Any]] = []
    for idx, ps_item in enumerate(ps_list, start=1):
        target_ps = f'{ps_item.get("abbreviation") or ps_item.get("name", "")} ({ps_item.get("name", "")})'
        print(f"  [{idx}/{len(ps_list)}] Deep extraction for: {target_ps}")
        pass2 = call_openai_chat(
            SYSTEM_PROMPT,
            PASS2_PROMPT.replace("{target_ps}", target_ps).replace("{text}", text),
        )
        if not pass2:
            result.errors.append(f"Pass 2 failed for {target_ps}")
            continue

        pass2_results.append({"target_ps": target_ps, "pass2": pass2})
        ps = pass2.get("photosensitizer")
        if not ps:
            result.errors.append(f"Pass 2 returned no photosensitizer object for {target_ps}")
            continue

        result.photosensitizers.append(ps)
        ps_owner = ps.get("abbreviation") or ps.get("name") or target_ps

        ros_list = ps.get("ros_assessment", [])
        for ros in ros_list:
            ros["_ps_owner"] = ps_owner
            result.ros_types.append(ros)

        methods = ps.get("detection_methods", [])
        for m in methods:
            m["_ps_owner"] = ps_owner
            result.detection_methods.append(m)

        if idx < len(ps_list):
            time.sleep(_DELAY_BETWEEN_CALLS)

    result.chunk_results.append({"pass1": pass1, "pass2_by_photosensitizer": pass2_results})

    _cross_reference_dimensions(result)

    print(f"  PS count:   {len(result.photosensitizers)}")
    print(f"  ROS types:  {len(result.ros_types)}")
    print(f"  Methods:    {len(result.detection_methods)}")
    return result

def _cross_reference_dimensions(result: AnalysisResult) -> None:
    probe_to_ros: dict[str, list[str]] = {
        "dpbf":    ["singlet_oxygen"],
        "abda":    ["singlet_oxygen"],
        "sosg":    ["singlet_oxygen"],
        "dcfh-da": ["hydrogen_peroxide", "hydroxyl_radical", "superoxide"],
        "dhe":     ["superoxide"],
        "hpf":     ["hydroxyl_radical"],
        "apf":     ["hydroxyl_radical", "singlet_oxygen"],
        "temp":    ["singlet_oxygen"],
        "dmp":     ["superoxide", "hydroxyl_radical"],
        "mitosox": ["superoxide"],
    }

    detected_ros_from_probes: set[str] = set()
    for method in result.detection_methods:
        probe = method.get("probe_name", "").lower().strip()
        for known_probe, ros_list in probe_to_ros.items():
            if known_probe in probe:
                detected_ros_from_probes.update(ros_list)

    for ros_entry in result.ros_types:
        ros_name = ros_entry.get("ros_name", "").lower().strip()
        ros_entry["cross_validated"] = ros_name in detected_ros_from_probes

    reported_ros_names = {
        r.get("ros_name", "").lower().strip() for r in result.ros_types
    }
    missing_from_d2 = detected_ros_from_probes - reported_ros_names
    if missing_from_d2:
        for ros_name in missing_from_d2:
            result.ros_types.append({
                "ros_name": ros_name,
                "ros_label": ros_name,
                "is_generated": True, "status": "generated",
                "evidence": "Inferred from probe usage (cross-reference with D3)",
                "quantum_yield": None,
                "quantum_yield_standard": None,
                "cross_validated": "inferred_from_probes",
            })


# ===========================================================================
# OUTPUT
# ===========================================================================

def save_results(result: AnalysisResult) -> Path:
    out_dir = Path(_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.filename).stem
    out_path = out_dir / f"{stem}_analysis.json"
    data = {
        "filename": result.filename,
        "has_photosensitizer": result.has_photosensitizer,
        "photosensitizers": result.photosensitizers,
        "ros_types": result.ros_types,
        "detection_methods": result.detection_methods,
        "chunk_results": result.chunk_results,
        "errors": result.errors,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def generate_summary_table(all_results: list[AnalysisResult]) -> str:
    lines = [
        "# Photosensitizer & ROS Literature Analysis - Summary",
        "",
        "| Paper | PS | Type | ROS Generated | Detection Methods |",
        "|-------|----|------|---------------|-------------------|",
    ]
    for r in all_results:
        if not r.photosensitizers:
            has_ps = "No" if r.has_photosensitizer is False else "?"
            lines.append(f"| {r.filename[:40]} | {has_ps} | - | - | - |")
        else:
            first = True
            for ps in r.photosensitizers:
                ps_label = ps.get("abbreviation") or ps.get("name", "?")
                ps_type = ps.get("ps_classification", "?")
                ros_labels = ", ".join(
                    ros.get("ros_label", "?")
                    for ros in r.ros_types
                    if ros.get("_ps_owner") == ps_label
                ) or "-"
                methods = ", ".join(
                    m.get("probe_name", "?")
                    for m in r.detection_methods
                    if m.get("_ps_owner") == ps_label
                ) or "-"
                if first:
                    lines.append(
                        f"| {r.filename[:40]} | {ps_label[:20]} | {ps_type[:10]} | "
                        f"{ros_labels[:45]} | {methods[:55]} |"
                    )
                    first = False
                else:
                    lines.append(
                        f"| | {ps_label[:20]} | {ps_type[:10]} | "
                        f"{ros_labels[:45]} | {methods[:55]} |"
                    )
    return "\n".join(lines)# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    pdf_dir = Path(_PDF_DIR)
    if not pdf_dir.exists():
        print(f"[ERROR] PDF directory not found: {pdf_dir}")
        sys.exit(1)

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        print(f"[ERROR] No PDF files found in {pdf_dir}")
        sys.exit(1)

    print("=" * 70)
    print("  Photosensitizer & ROS Literature Analyzer")
    print("=" * 70)
    print(f"  Model:      {_MODEL}")
    print(f"  PDF dir:    {pdf_dir}")
    print(f"  PDFs found: {len(pdf_paths)}")
    for p in pdf_paths:
        print(f"    - {p.name}")
    print(f"  Output dir: {Path(_OUTPUT_DIR).resolve()}")
    print(f"  Two-pass mode (Pass 2 runs once per PS)")
    print("=" * 70)

    all_results: list[AnalysisResult] = []
    for i, pdf_path in enumerate(pdf_paths):
        print(f"\n{'='*70}")
        print(f"  Paper {i+1}/{len(pdf_paths)}: {pdf_path.name}")
        print(f"{'='*70}")
        try:
            result = analyze_paper(pdf_path)
            all_results.append(result)
            out_path = save_results(result)
            print(f"\n  Done. Results saved to: {out_path}")
        except Exception as e:
            print(f"\n  Failed to analyze {pdf_path.name}: {e}")
            all_results.append(AnalysisResult(
                filename=pdf_path.name, errors=[str(e)]
            ))

    summary = generate_summary_table(all_results)
    summary_path = Path(_OUTPUT_DIR) / "SUMMARY.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nSummary table saved to: {summary_path}")


if __name__ == "__main__":
    main()
