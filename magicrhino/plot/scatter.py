"""Scatter plot helper. Returns a matplotlib Figure."""
from __future__ import annotations

import matplotlib.pyplot as plt


def plot_scatter(df, x, y):
    """Return a scatter Figure of ``x`` vs ``y`` from ``df``."""
    fig, ax = plt.subplots(figsize=(8, 6))
    if x == "Index":
        scatter = ax.scatter(range(len(df)), df[y], c=df[y], cmap="RdBu_r")
        ax.set_xlabel("Index")
    else:
        scatter = ax.scatter(df[x], df[y], c=df[y], cmap="RdBu_r")
        ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x if x != 'Index' else 'Index'} vs {y}")
    fig.colorbar(scatter, ax=ax)
    fig.tight_layout()
    return fig
