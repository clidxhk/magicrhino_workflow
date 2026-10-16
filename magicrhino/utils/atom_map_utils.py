"""Atom-level descriptor contribution mapping for RDKit-native and Mordred descriptors.

Uses a perturbation approach: for each heavy atom, replace it with H,
recompute the descriptor, measure the absolute change, and normalize the
per-atom changes into percentage shares of the selected descriptor mapping.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors

rdBase.DisableLog('rdApp.*')

# ---- Descriptor registries -----------------------------------------------

def get_rdkit_descriptor_names() -> set[str]:
    """Return the set of all RDKit-native descriptor names."""
    return {name for name, _ in Descriptors._descList}


_MORDRED_CALC = None
_MORDRED_NAMES = None
_MORDRED_NAME_MAP = None


def _ensure_mordred():
    """Lazily initialize the Mordred calculator and name mapping."""
    global _MORDRED_CALC, _MORDRED_NAMES, _MORDRED_NAME_MAP
    if _MORDRED_CALC is not None:
        return
    try:
        from mordred import Calculator, descriptors as mordred_descriptors
    except ImportError:
        _MORDRED_CALC = False
        return
    _MORDRED_CALC = Calculator(mordred_descriptors)
    _MORDRED_NAMES = {str(d) for d in _MORDRED_CALC.descriptors}
    _MORDRED_NAME_MAP = {}


def get_mordred_descriptor_names() -> set[str]:
    """Return the set of all Mordred descriptor names (or empty if not installed)."""
    _ensure_mordred()
    if _MORDRED_CALC is False:
        return set()
    return set(_MORDRED_NAMES)


def _resolve_mordred_name(csv_name: str) -> str | None:
    """Map a CSV column name to the corresponding Mordred descriptor name."""
    _ensure_mordred()
    if _MORDRED_CALC is False:
        return None
    if csv_name in _MORDRED_NAMES:
        return csv_name
    if csv_name in _MORDRED_NAME_MAP:
        return _MORDRED_NAME_MAP[csv_name]
    properties = ['are', 'pe', 'se', 'dv', 'c', 'd', 's', 'Z', 'm', 'v', 'p', 'i']
    for prop in sorted(properties, key=len, reverse=True):
        for prefix_len in range(len(csv_name) - 1, 0, -1):
            prefix = csv_name[:prefix_len]
            rest = csv_name[prefix_len:]
            if rest.startswith(prop):
                num_part = rest[len(prop):]
                if num_part and num_part.isdigit():
                    candidate = prefix + num_part + prop
                    if candidate in _MORDRED_NAMES:
                        _MORDRED_NAME_MAP[csv_name] = candidate
                        return candidate
    _MORDRED_NAME_MAP[csv_name] = None
    return None


def get_available_descriptor_names(csv_columns: list[str]) -> dict[str, str]:
    """Return a dict mapping CSV column name -> source label ('RDKit' or 'Mordred')."""
    rdkit_names = get_rdkit_descriptor_names()
    result = {}
    for col in csv_columns:
        if col in rdkit_names:
            result[col] = 'RDKit'
        elif _resolve_mordred_name(col) is not None:
            result[col] = 'Mordred'
    return result


def get_selected_features(artifact: dict) -> list[str]:
    """Extract feature names that survive the model pipeline's selection steps."""
    model = artifact['model']
    feature_names = np.asarray(artifact.get('feature_columns', []), dtype=object)
    if not hasattr(model, 'named_steps') or 'model' not in model.named_steps:
        return feature_names.tolist()
    names = feature_names.copy()
    for step_name in ('variance', 'select'):
        transformer = model.named_steps.get(step_name)
        if transformer is None:
            continue
        if hasattr(transformer, 'get_support'):
            names = names[transformer.get_support()]
    return names.tolist()


# ---- Descriptor evaluation -----------------------------------------------

def _eval_rdkit(mol, name):
    for n, func in Descriptors._descList:
        if n == name:
            return float(func(mol))
    raise ValueError(f"Unknown RDKit descriptor: {name}")


def _eval_mordred(mol, csv_name):
    _ensure_mordred()
    if _MORDRED_CALC is False:
        raise RuntimeError("Mordred is not installed")
    mordred_name = _resolve_mordred_name(csv_name)
    if mordred_name is None:
        raise ValueError(f"Unknown Mordred descriptor: {csv_name}")
    idx = list(str(d) for d in _MORDRED_CALC.descriptors).index(mordred_name)
    result = _MORDRED_CALC(mol)
    return float(result[idx])


def _eval_descriptor(mol, name, source):
    if source == 'RDKit':
        return _eval_rdkit(mol, name)
    return _eval_mordred(mol, name)


# ---- Perturbation core ----------------------------------------------------

def _perturb_mol(mol, atom_idx):
    """Return a copy of mol with atom_idx replaced by H (single bonds only)."""
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetIsAromatic(False)
    for b in rw.GetBonds():
        b.SetIsAromatic(False)

    atom_rw = rw.GetAtomWithIdx(atom_idx)
    atom_rw.SetAtomicNum(1)
    atom_rw.SetFormalCharge(0)
    atom_rw.SetNoImplicit(True)
    atom_rw.SetNumExplicitHs(0)

    neighbors = list(atom_rw.GetNeighbors())
    for nb in neighbors[1:]:
        rw.RemoveBond(atom_idx, nb.GetIdx())
    for nb in list(atom_rw.GetNeighbors()):
        bond = rw.GetBondBetweenAtoms(atom_idx, nb.GetIdx())
        if bond is not None and bond.GetBondTypeAsDouble() > 1.0:
            bond.SetBondType(Chem.BondType.SINGLE)

    mol_perturbed = rw.GetMol()
    Chem.SanitizeMol(mol_perturbed)
    return mol_perturbed


def _to_percentage_shares(values: list[float] | np.ndarray) -> list[float]:
    """Normalize non-negative contribution values to percentages summing to 100.

    This treats each atom's perturbation score as a share of the total
    atom-level perturbation signal for the selected descriptor. It avoids the
    misleading blow-up that happens when a descriptor value is close to zero.
    """
    arr = np.nan_to_num(
        np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 1e-12:
        return [0.0 for _ in arr]
    return (arr / total * 100.0).tolist()


def compute_atom_contributions(smiles: str, descriptor_name: str, source: str = 'RDKit'):
    """Compute per-atom percentage contributions to a descriptor via perturbation.

    The raw perturbation score is ``abs(original_descriptor - perturbed_descriptor)``.
    Scores are normalized by the sum of all atom perturbation scores, so the
    returned values are atom contribution shares that sum to ~100%.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    original = _eval_descriptor(mol, descriptor_name, source)
    raw_contributions: list[float] = []

    for i in range(mol.GetNumAtoms()):
        if mol.GetAtomWithIdx(i).GetAtomicNum() == 1:
            raw_contributions.append(0.0)
            continue
        try:
            mol_p = _perturb_mol(mol, i)
            perturbed = _eval_descriptor(mol_p, descriptor_name, source)
            diff = abs(float(original) - float(perturbed))
            if not np.isfinite(diff):
                diff = 0.0
            raw_contributions.append(diff)
        except Exception:
            raw_contributions.append(0.0)

    return _to_percentage_shares(raw_contributions), mol


# ---- Visualization --------------------------------------------------------

def draw_atom_heatmap_svg(mol, weights, size=(560, 520)):
    """Draw an editable SVG molecule with atoms colored by contribution percentages."""
    import re
    from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
    from rdkit.Chem.Draw import rdMolDraw2D

    weights_arr = np.nan_to_num(
        np.array(weights, dtype=float), nan=0.0, posinf=0.0, neginf=0.0
    )
    max_w = float(weights_arr.max()) if weights_arr.size else 0.0
    norm = Normalize(vmin=0, vmax=max(max_w, 1e-10))

    # Same red-blue colormap as the SHAP visualizations.
    _cdict = {
        "red": ((0.0, 0.117, 0.117), (1.0, 0.961, 0.961)),
        "green": ((0.0, 0.533, 0.533), (1.0, 0.153, 0.153)),
        "blue": ((0.0, 0.898, 0.898), (1.0, 0.341, 0.341)),
        "alpha": ((0.0, 1, 1), (0.5, 1, 1), (1.0, 1, 1)),
    }
    cmap = LinearSegmentedColormap("RedBlue", _cdict)

    highlight_atoms = list(range(len(weights_arr)))
    highlight_colors = {}
    highlight_radii = {}
    for i, w in enumerate(weights_arr):
        r, g, b, _ = cmap(norm(w))
        highlight_colors[i] = (r, g, b)
        highlight_radii[i] = 0.4

    width, height = size
    # Reserve a publication-style legend area with readable typography and
    # clear separation between title, ticks, and the color bar.
    legend_height = 124
    mol_height = max(260, height - legend_height)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, mol_height)
    opts = drawer.drawOptions()
    opts.atomHighlightsAreCircles = True
    opts.circleAtoms = True
    opts.fillHighlights = True
    opts.highlightRadius = 0.35
    opts.bondLineWidth = 1.5
    opts.clearBackground = False
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
    )
    drawer.FinishDrawing()
    mol_svg = drawer.GetDrawingText()

    # Strip the RDKit SVG wrapper and embed its vector content into our own
    # SVG, so the colorbar is included in the same editable SVG file.
    mol_body_match = re.search(r"<svg[^>]*>(.*)</svg>", mol_svg, flags=re.DOTALL | re.IGNORECASE)
    mol_body = mol_body_match.group(1) if mol_body_match else mol_svg
    mol_body = re.sub(r"<\?xml[^>]*\?>", "", mol_body).strip()

    bar_x = 72
    bar_y = mol_height + 48
    bar_w = width - 144
    bar_h = 18
    title_y = bar_y - 16
    tick_y1 = bar_y + bar_h
    tick_y2 = bar_y + bar_h + 7
    tick_label_y = tick_y2 + 20

    stops = []
    for pct in range(0, 101, 10):
        color = to_hex(cmap(pct / 100.0), keep_alpha=False)
        stops.append(f'<stop offset="{pct}%" stop-color="{color}"/>')
    tick_values = [0.0, max_w / 2.0, max_w]
    tick_positions = [bar_x, bar_x + bar_w / 2.0, bar_x + bar_w]
    ticks = []
    for value, x in zip(tick_values, tick_positions):
        ticks.append(
            f'<line x1="{x:.2f}" y1="{tick_y1:.2f}" x2="{x:.2f}" y2="{tick_y2:.2f}" '
            f'stroke="#0f172a" stroke-width="1"/>'
        )
        ticks.append(
            f'<text x="{x:.2f}" y="{tick_label_y:.2f}" text-anchor="middle" '
            f'font-family="Arial, Helvetica, sans-serif" font-size="15" font-weight="500" '
            f'fill="#111827">{value:.3g}%</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:rdkit="http://www.rdkit.org/xml" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="white"/>
  <g id="molecule">
    {mol_body}
  </g>
  <defs>
    <linearGradient id="atomContributionGradient" x1="0%" y1="0%" x2="100%" y2="0%">
      {''.join(stops)}
    </linearGradient>
  </defs>
  <g id="atom-contribution-colorbar">
    <text x="{width / 2:.2f}" y="{title_y:.2f}" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="700" fill="#111827">Atom contribution (%)</text>
    <rect x="{bar_x}" y="{bar_y}" width="{bar_w}" height="{bar_h}" fill="url(#atomContributionGradient)" stroke="#111827" stroke-width="1.1"/>
    {''.join(ticks)}
  </g>
</svg>"""
    return svg


def draw_atom_heatmap(mol, weights, size=(560, 520)):
    """Backward-compatible alias returning editable SVG text."""
    return draw_atom_heatmap_svg(mol, weights, size=size)

def contribution_table(mol, contributions):
    """Build a DataFrame of atom-level percentage contributions."""
    rows = []
    for i, atom in enumerate(mol.GetAtoms()):
        rows.append({
            "Atom": i,
            "Element": atom.GetSymbol(),
            "Contribution (%)": contributions[i],
        })
    return pd.DataFrame(rows)
