"""Correlation coefficient matrix plot.

Returns a matplotlib Figure instead of calling ``st.pyplot`` so the
caller (Streamlit or NiceGUI) can render it however it likes.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_matrix(df, method: str):
    """Return a Figure of the top-21 absolute-correlation heatmap."""
    df = df.set_index("Index", inplace=False)
    y_corr = df.corr(method=method).abs()["Y"].sort_values(ascending=False)[:21]
    sorted_index = sorted(y_corr.index, key=lambda x: df.columns.get_loc(x))
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        df[sorted_index].corr(),
        cmap="RdBu_r", linewidths=0.0, linecolor="white", ax=ax,
    )
    ax.set_title(f"Correlation matrix ({method})")
    fig.tight_layout()
    return fig
