"""Complete molecular similarity analysis for SMILES/type CSV input.

The script reads a CSV with `smiles` and `type` columns, computes Morgan
fingerprint similarities with RDKit, and writes diagnostic figures into an
output directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform


DEFAULT_CSV = Path(__file__).resolve().parent / "Molecular_Scaleframe.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figures"

CATEGORY_COLORS = {
    "AIE_1": "#b2182b",
    "AIE_2": "#d6604d",
    "AIE_3": "#f4a582",
    "BODIPY": "#92c5de",
    "Xanthene": "#4393c3",
    "Cyanines": "#2166ac",
    "Others": "#053061",
}

CATEGORY_SHAPES = {
    "AIE_1": "o",
    "AIE_2": "o",
    "AIE_3": "o",
    "BODIPY": ",",
    "Xanthene": "h",
    "Cyanines": "^",
    "Others": "D",
}

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster molecules by Morgan fingerprint similarity and save figures."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="CSV with smiles,type columns")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated figures are saved",
    )
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius")
    parser.add_argument("--nbits", type=int, default=2048, help="Morgan fingerprint bit length")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Tanimoto similarity threshold for network edges",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure resolution")
    parser.add_argument("--show", action="store_true", help="Show figures in addition to saving them")
    return parser.parse_args()


def validate_category(category: str, csv_path: Path, csv_line: int) -> None:
    if category not in CATEGORY_COLORS:
        raise ValueError(
            f"{csv_path}:{csv_line} has unknown type {category!r}. "
            f"Known types are {', '.join(sorted(CATEGORY_COLORS))}."
        )


def read_molecules_from_csv(csv_path: Path) -> dict:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = {"smiles", "type"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing_columns)}")

    molecules = []
    canonical_smiles = []
    categories = []

    for csv_line, (_, row) in enumerate(df.iterrows(), start=2):
        category = row.get("type")
        if pd.isna(category) or str(category).strip() == "":
            raise ValueError(f"{csv_path}:{csv_line} has an empty type")
        category = str(category).strip()
        validate_category(category, csv_path, csv_line)

        smiles = row.get("smiles")
        if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(f"{csv_path}:{csv_line} has an empty smiles field")
        smiles = smiles.strip()

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"{csv_path}:{csv_line} contains an unparseable SMILES: {smiles}")

        molecules.append(mol)
        canonical_smiles.append(Chem.MolToSmiles(mol))
        categories.append(category)

    category_counts = (
        pd.Series(Counter(categories))
        .reindex(CATEGORY_COLORS.keys())
        .fillna(0)
        .astype(int)
    )

    return {
        "molecules": molecules,
        "canonical_smiles": canonical_smiles,
        "categories": categories,
        "category_counts": category_counts,
    }


def compute_similarity_matrix(molecules: list, radius: int, nbits: int) -> np.ndarray:
    """Return a symmetric Morgan/Tanimoto similarity matrix with unit diagonal."""
    num_molecules = len(molecules)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fingerprints = [generator.GetFingerprint(mol) for mol in molecules]
    similarity_matrix = np.zeros((num_molecules, num_molecules))

    for i in range(num_molecules - 1):
        scores = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[i + 1:])
        similarity_matrix[i, i + 1:] = scores
        similarity_matrix[i + 1:, i] = scores

    np.fill_diagonal(similarity_matrix, 1.0)
    return similarity_matrix


def build_distance_linkage(similarity_matrix: np.ndarray, method: str = "average") -> np.ndarray:
    """Build hierarchical linkage from Tanimoto distance rather than raw similarity.

    Average linkage (UPGMA) is used because 1 - Tanimoto is a non-Euclidean
    distance and therefore incompatible with Ward linkage, which SciPy
    defines only for Euclidean distances.
    """
    distance_matrix = np.clip(1.0 - similarity_matrix, 0.0, 1.0)
    condensed_distances = squareform(distance_matrix, checks=False)
    return linkage(condensed_distances, method=method)


def finish_figure(fig, output_path: Path | None, dpi: int, show: bool) -> None:
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_category_counts(
    category_counts: pd.Series,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8 / 2.4, 8 / 2.4), dpi=dpi)
    counts = category_counts.reindex(CATEGORY_COLORS.keys()).fillna(0).astype(int)
    colors = [CATEGORY_COLORS[category] for category in counts.index]
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    finish_figure(fig, output_path, dpi, show)


def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10 / 2.54, 10 / 2.54), dpi=dpi)
    image = ax.imshow(
        similarity_matrix,
        cmap="RdBu_r",
        interpolation="nearest",
        vmax=0.55,
    )
    fig.colorbar(image, ax=ax)
    ax.set_xlabel("Molecule Index")
    ax.set_ylabel("Molecule Index")
    finish_figure(fig, output_path, dpi, show)


def plot_clustered_heatmap(
    similarity_matrix: np.ndarray,
    categories: list[str],
    category_counts: pd.Series,
    linkage_matrix: np.ndarray,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    category_colors = [CATEGORY_COLORS[category] for category in categories]
    legend_patches = [
        mpatches.Patch(color=CATEGORY_COLORS[category], label=category)
        for category, count in category_counts.items()
        if count > 0
    ]

    cluster_grid = sns.clustermap(
        similarity_matrix,
        row_linkage=linkage_matrix,
        col_linkage=linkage_matrix,
        cmap="RdBu_r",
        vmax=0.55,
        xticklabels=False,
        yticklabels=False,
        row_colors=category_colors,
        col_colors=category_colors,
        figsize=(10, 10),
        dendrogram_ratio=0.2,
        cbar_pos=(0.02, 0.80, 0.02, 0.18),
        cbar_kws={"label": "Tanimoto Similarity"},
    )
    # Legend sits in the empty top-left corner cell, right of the colorbar;
    # the widened dendrogram ratio keeps the column dendrogram clear of it.
    cluster_grid.figure.legend(
        handles=legend_patches,
        title="Category",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.105, 0.995),
        prop={"size": 11},
        title_fontsize=12,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_grid.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    else:
        plt.close(cluster_grid.fig)


def build_similarity_graph(similarity_matrix: np.ndarray, threshold: float) -> nx.Graph:
    num_molecules = similarity_matrix.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(num_molecules))
    rows, cols = np.triu_indices(num_molecules, k=1)
    edges = [
        (int(i), int(j))
        for i, j in zip(rows, cols)
        if similarity_matrix[int(i), int(j)] > threshold
    ]
    graph.add_edges_from(edges)
    return graph


def draw_category_nodes_with_shapes(
    ax,
    graph: nx.Graph,
    pos: dict,
    categories: list[str],
    node_size: int,
) -> None:
    for category in CATEGORY_COLORS:
        nodes = [i for i, item in enumerate(categories) if item == category]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_size=node_size,
            node_color=[CATEGORY_COLORS[category]] * len(nodes),
            node_shape=CATEGORY_SHAPES[category],
            ax=ax,
        )


def draw_curved_edges(ax, graph: nx.Graph, pos: dict, rad: float) -> None:
    for edge in graph.edges():
        start, end = pos[edge[0]], pos[edge[1]]
        patch = mpatches.ConnectionPatch(
            start,
            end,
            "data",
            "data",
            lw=0.3,
            color="black",
            connectionstyle=f"arc3, rad={rad}",
        )
        ax.add_patch(patch)


def add_network_legend(ax, category_counts: pd.Series) -> None:
    handles = [
        mpatches.Patch(color=CATEGORY_COLORS[category], label=category)
        for category, count in category_counts.items()
        if count > 0
    ]
    ax.legend(
        handles=handles,
        loc=(1.0, 0.7),
        title="Category",
        frameon=False,
        prop={"size": 12},
        title_fontsize=12,
    )


def plot_network(
    graph: nx.Graph,
    categories: list[str],
    category_counts: pd.Series,
    output_dir: Path,
    dpi: int,
    show: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10 / 2.54, 10 / 2.54), dpi=dpi)
    pos = nx.circular_layout(graph)
    draw_curved_edges(ax, graph, pos, rad=-0.3)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=100,
        node_color=[CATEGORY_COLORS[category] for category in categories],
        ax=ax,
    )
    add_network_legend(ax, category_counts)
    ax.axis("off")
    finish_figure(fig, output_dir / "network_circular.png", dpi, show)

    fig, ax = plt.subplots(figsize=(10 / 2.54, 10 / 2.54), dpi=dpi)
    pos = nx.spring_layout(graph, k=0.3, iterations=100, seed=42)
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color="black", width=0.3)
    draw_category_nodes_with_shapes(ax, graph, pos, categories, node_size=20)
    add_network_legend(ax, category_counts)
    ax.axis("off")
    finish_figure(fig, output_dir / "network_spring.png", dpi, show)


def main() -> None:
    args = parse_args()
    dataset = read_molecules_from_csv(args.csv)
    molecules = dataset["molecules"]
    categories = dataset["categories"]
    canonical_smiles = dataset["canonical_smiles"]
    category_counts = dataset["category_counts"]

    print(f"Loaded {len(molecules)} molecules from {args.csv}")
    duplicate_groups = sum(count - 1 for count in Counter(canonical_smiles).values() if count > 1)
    if duplicate_groups:
        print(f"Warning: {duplicate_groups} rows have a duplicate canonical SMILES")

    print("Category counts:")
    for category, count in category_counts.items():
        if count:
            print(f"  {category}: {count}")

    similarity_matrix = compute_similarity_matrix(molecules, radius=args.radius, nbits=args.nbits)
    upper = similarity_matrix[np.triu_indices(len(molecules), k=1)]
    print(
        "Similarity summary: "
        f"min={upper.min():.3f}, median={np.median(upper):.3f}, "
        f"mean={upper.mean():.3f}, max={upper.max():.3f}"
    )

    linkage_matrix = build_distance_linkage(similarity_matrix)
    graph = build_similarity_graph(similarity_matrix, threshold=args.threshold)
    print(
        f"Similarity graph: nodes={graph.number_of_nodes()}, "
        f"edges={graph.number_of_edges()}, components={nx.number_connected_components(graph)}"
    )

    plot_category_counts(category_counts, args.output_dir / "category_counts.png", args.dpi, args.show)
    plot_similarity_heatmap(similarity_matrix, args.output_dir / "similarity_heatmap.png", args.dpi, args.show)
    plot_clustered_heatmap(
        similarity_matrix,
        categories,
        category_counts,
        linkage_matrix,
        args.output_dir / "clustered_similarity.png",
        args.dpi,
        args.show,
    )
    plot_network(graph, categories, category_counts, args.output_dir, args.dpi, args.show)
    print(f"Figures saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
