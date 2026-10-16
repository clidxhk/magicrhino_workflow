"""Data utilities — pure logic, no UI dependencies.

Previously this module contained streamlit calls mixed into the business
logic (e.g. ``st.sidebar.button`` inside ``cluster``).  The UI bits have
been removed so the same functions can be reused from NiceGUI or any
other frontend.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import validation as sklearn_validation
import sklearn.utils as sklearn_utils

from utils.shap_utils import shap_cluster_color


def load_data(df: pd.DataFrame):
    """Split a labeled frame into ``X, y`` keyed by an ``Index`` column."""
    df = df.copy()
    if "Index" not in df.columns or "Y" not in df.columns:
        raise ValueError("The labeled dataset must contain Index and Y columns.")
    df.set_index("Index", inplace=True)
    return df.drop(columns=["Y"]), df[["Y"]]


def patch_sklearn_check_array_for_umap():
    """Allow older umap-learn releases to run with newer scikit-learn."""
    current = sklearn_validation.check_array
    if "force_all_finite" in inspect.signature(current).parameters:
        return
    if getattr(current, "_magicrhino_umap_compat", False):
        return

    def check_array_compat(*args, force_all_finite=None, ensure_all_finite=None, **kwargs):
        if ensure_all_finite is None:
            ensure_all_finite = force_all_finite
        if ensure_all_finite is None:
            return current(*args, **kwargs)
        return current(*args, ensure_all_finite=ensure_all_finite, **kwargs)

    check_array_compat._magicrhino_umap_compat = True
    sklearn_validation.check_array = check_array_compat
    sklearn_utils.check_array = check_array_compat


def _get_umap_reducer(n_components: int):
    patch_sklearn_check_array_for_umap()
    from umap import UMAP
    import umap.umap_ as umap_module

    umap_module.check_array = sklearn_validation.check_array
    return UMAP(n_components=n_components, random_state=42, n_jobs=1)


def reduce_features(option: str, n_components: int, perplexity: int | None = None):
    """Return a fitted dimensionality-reduction transformer."""
    n_components = int(n_components)
    if option == "PCA":
        return make_pipeline(
            StandardScaler(),
            PCA(n_components=n_components, random_state=42),
        )
    if option == "UMAP":
        return make_pipeline(
            StandardScaler(),
            _get_umap_reducer(n_components),
        )
    if option == "t-SNE":
        return make_pipeline(
            StandardScaler(),
            TSNE(
                n_components=n_components,
                perplexity=perplexity or 30,
                random_state=42,
                init="pca",
                learning_rate="auto",
            ),
        )
    raise ValueError(f"Unknown reduction option: {option}")


def _prepare_cluster_data(X, y):
    frame = pd.DataFrame(X).copy()
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(axis=1, how="all")
    if frame.empty:
        raise ValueError("No numeric features available for clustering.")

    if frame.isna().any().any():
        frame = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(frame),
            index=frame.index,
            columns=frame.columns,
        )

    target = np.asarray(y).reshape(-1)
    if len(frame) != len(target):
        raise ValueError("Feature and label row counts do not match.")
    if len(frame) < 3:
        raise ValueError("At least 3 samples are required for clustering.")
    return frame.to_numpy(dtype=float), target


def cluster(option: str, n_components: int, X, y, perplexity: int = 30):
    """Run PCA/UMAP/t-SNE + KMeans and return ``(fig, adjusted_rand)``."""
    X_clean, y = _prepare_cluster_data(X, y)
    max_components = min(X_clean.shape[0], X_clean.shape[1])
    if option == "t-SNE":
        max_components = min(max_components, 3)
    n_components = min(int(n_components), max_components)
    if n_components < 2:
        raise ValueError("Cluster visualization requires at least 2 numeric components.")
    if option == "t-SNE":
        perplexity = min(int(perplexity or 30), max(1, len(X_clean) - 1))
        if perplexity >= len(X_clean):
            perplexity = max(1, len(X_clean) - 1)
    reducer = reduce_features(option, n_components, perplexity)
    X_cluster = reducer.fit_transform(X_clean)
    if X_cluster.shape[1] < 2:
        raise ValueError("Cluster visualization requires at least 2 components.")
    X_cluster = np.asarray(X_cluster)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(X_cluster)
    y_kmeans = kmeans.predict(X_cluster)
    red_blue_solid = shap_cluster_color()

    axis_prefix = {
        "PCA": "Principal Component",
        "UMAP": "UMAP Component",
        "t-SNE": "t-SNE Component",
    }[option]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X_cluster[:, 0], X_cluster[:, 1], c=y_kmeans, cmap=red_blue_solid, label="Cluster")
    ax.scatter(X_cluster[:, 0], X_cluster[:, 1], c=np.ravel(y), cmap=red_blue_solid, label="True Label", marker="*")
    colors = ["#008BFB", "#FF004F"]
    for i in range(2):
        cluster_points = X_cluster[y_kmeans == i]
        center = kmeans.cluster_centers_[i]
        radius = cluster_points.std(axis=0)
        ellipse = patches.Ellipse(center, radius[0] * 4, radius[1] * 4, alpha=0.1, color=colors[i])
        ax.add_patch(ellipse)
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c="black", marker="x")
    ax.set_xlabel(f"{axis_prefix} 1")
    ax.set_ylabel(f"{axis_prefix} 2")
    ax.legend(loc="best")
    fig.tight_layout()

    score = adjusted_rand_score(np.ravel(y), y_kmeans)
    return fig, score
