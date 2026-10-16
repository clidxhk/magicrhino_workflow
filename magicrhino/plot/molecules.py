"""Molecule drawing helpers using RDKit.

Return editable SVG strings so the UI layer can render and download them.
"""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


def _mol_to_svg(mol, size=(240, 240)):
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.bondLineWidth = 1.6
    opts.clearBackground = False
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def draw_molecule(df, index):
    """Return an SVG string for the molecule at ``index``."""
    molecule = df.loc[index, "Index"]
    mol = Chem.MolFromSmiles(molecule)
    if mol is None:
        return None
    return _mol_to_svg(mol)


def shap_draw_molecules(df, number):
    """Return an SVG string for the molecule at row ``number`` (indexed by SMILES)."""
    molecule = df.index[number]
    mol = Chem.MolFromSmiles(molecule)
    if mol is None:
        return None
    return _mol_to_svg(mol)
