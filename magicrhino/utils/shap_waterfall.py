"""SHAP waterfall variant with a pinned bottom "N other features" aggregate row.

Adapted from ``shap.plots._waterfall.waterfall`` (SHAP, MIT license) so a
user-selected subset of features is always shown individually while every
remaining feature is grouped into a single aggregate bar pinned at the
bottom of the plot, matching SHAP's native style for the aggregated row.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shap import Explanation
from shap.utils import format_value
from shap.plots._labels import labels
from shap.plots._style import get_style


def waterfall_selected_other(shap_values, selected_indices, show=True):
    """Waterfall plot showing ``selected_indices`` individually plus one bottom
    aggregate bar for all other features.

    Parameters
    ----------
    shap_values : Explanation
        A one-dimensional Explanation for a single prediction, containing the
        SHAP values of ALL features (not only the selected ones).

    selected_indices : list of int
        Feature column indices to display individually. They are sorted by
        SHAP magnitude, exactly like SHAP's native waterfall does.

    show : bool
        Whether ``plt.show()`` is called before returning.
    """
    style = get_style()
    # Turn off interactive plot
    if show is False:
        plt.ioff()

    # make sure the input is an Explanation object
    if not isinstance(shap_values, Explanation):
        emsg = "The waterfall plot requires an `Explanation` object as the `shap_values` argument."
        raise TypeError(emsg)

    # make sure we only have a single explanation to plot
    sv_shape = shap_values.shape
    if len(sv_shape) != 1:
        emsg = (
            "The waterfall plot can currently only plot a single explanation, but a "
            f"matrix of explanations (shape {sv_shape}) was passed!"
        )
        raise ValueError(emsg)

    base_values = float(shap_values.base_values)
    features = shap_values.display_data if shap_values.display_data is not None else shap_values.data
    feature_names = shap_values.feature_names
    lower_bounds = getattr(shap_values, "lower_bounds", None)
    upper_bounds = getattr(shap_values, "upper_bounds", None)
    values = np.asarray(shap_values.values, dtype=float)

    # unwrap pandas series
    if isinstance(features, pd.Series):
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values

    # fallback feature names
    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(len(values))])

    selected_indices = [int(i) for i in selected_indices]
    selected_set = set(selected_indices)
    num_individual = len(selected_indices)
    has_other = num_individual < len(values)

    # init variables we use for tracking the plot locations
    num_features = num_individual + 1 if has_other else num_individual
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    # selected features first (sorted by magnitude), everything else aggregated
    order = sorted(selected_indices, key=lambda idx: -abs(values[idx]))
    order = order + [i for i in range(len(values)) if i not in selected_set]
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for _ in range(num_features + 1)]

    # size the plot based on how many features we are plotting
    plt.gcf().set_size_inches(8, num_features * row_height + 1.5)

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot(
                [loc, loc],
                [rng[i] - 1 - 0.4, rng[i] + 0.4],
                color=style.vlines_color,
                linestyle="--",
                linewidth=0.5,
                zorder=-1,
            )
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            if np.issubdtype(type(features[order[i]]), np.number):
                yticklabels[rng[i]] = (
                    format_value(float(features[order[i]]), "%0.03f") + " = " + str(feature_names[order[i]])
                )
            else:
                yticklabels[rng[i]] = str(features[order[i]]) + " = " + str(feature_names[order[i]])

    # add the pinned bottom row representing the impact of all unselected features
    if has_other:
        yticklabels[0] = f"{len(values) - num_individual} other features"
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)

    points = (
        pos_lefts
        + list(np.array(pos_lefts) + np.array(pos_widths))
        + neg_lefts
        + list(np.array(neg_lefts) + np.array(neg_widths))
    )
    dataw = np.max(points) - np.min(points)

    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1 * dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(
        pos_inds,
        np.array(pos_widths) + label_padding + 0.02 * dataw,
        left=np.array(pos_lefts) - 0.01 * dataw,
        color=style.primary_color_positive,
        alpha=0,
    )
    label_padding = np.array([-0.1 * dataw if -w < 1 else 0 for w in neg_widths])
    plt.barh(
        neg_inds,
        np.array(neg_widths) + label_padding - 0.02 * dataw,
        left=np.array(neg_lefts) + 0.01 * dataw,
        color=style.primary_color_negative,
        alpha=0,
    )

    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen / width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()

    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(
            pos_lefts[i],
            pos_inds[i],
            dist - hl_scaled,
            0,
            head_length=min(dist, hl_scaled),
            color=style.primary_color_positive,
            width=bar_width,
            head_width=bar_width,
        )

        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i],
                pos_inds[i],
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=style.secondary_color_positive,
            )

        txt_obj = plt.text(
            pos_lefts[i] + 0.5 * dist,
            pos_inds[i],
            format_value(pos_widths[i], "%+0.02f"),
            horizontalalignment="center",
            verticalalignment="center",
            color=style.text_color,
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                pos_lefts[i] + (5 / 72) * bbox_to_xscale + dist,
                pos_inds[i],
                format_value(pos_widths[i], "%+0.02f"),
                horizontalalignment="left",
                verticalalignment="center",
                color=style.primary_color_positive,
                fontsize=12,
            )

    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]

        arrow_obj = plt.arrow(
            neg_lefts[i],
            neg_inds[i],
            -(-dist - hl_scaled),
            0,
            head_length=min(-dist, hl_scaled),
            color=style.primary_color_negative,
            width=bar_width,
            head_width=bar_width,
        )

        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i],
                neg_inds[i],
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=style.secondary_color_negative,
            )

        txt_obj = plt.text(
            neg_lefts[i] + 0.5 * dist,
            neg_inds[i],
            format_value(neg_widths[i], "%+0.02f"),
            horizontalalignment="center",
            verticalalignment="center",
            color=style.text_color,
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                neg_lefts[i] - (5 / 72) * bbox_to_xscale + dist,
                neg_inds[i],
                format_value(neg_widths[i], "%+0.02f"),
                horizontalalignment="right",
                verticalalignment="center",
                color=style.primary_color_negative,
                fontsize=12,
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ytick_pos = list(range(num_features)) + list(np.arange(num_features) + 1e-8)
    plt.yticks(ytick_pos, yticklabels[:-1] + [label.split("=")[-1] for label in yticklabels[:-1]], fontsize=13)

    # put horizontal lines for each feature row
    for i in range(num_features):
        plt.axhline(i, color=style.hlines_color, lw=0.5, dashes=(1, 5), zorder=-1)

    # mark the prior expected value and the model prediction
    plt.axvline(base_values, 0, 1 / num_features, color=style.vlines_color, linestyle="--", linewidth=0.5, zorder=-1)
    fx = base_values + values.sum()
    plt.axvline(fx, 0, 1, color=style.vlines_color, linestyle="--", linewidth=0.5, zorder=-1)

    # clean up the main axis
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("none")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    ax.tick_params(labelsize=13)

    # draw the E[f(X)] tick mark
    xmin, xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks(
        [base_values, base_values + min(1e-8, xmax * 1e-10)]
    )  # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    # However, for very small values, 1e-8 is disruptively large, so xmax * 1e-10 is used instead
    ax2.set_xticklabels(["\n$E[f(X)]$", "\n$ = " + format_value(base_values, "%0.03f") + "$"], fontsize=12, ha="left")
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # draw the f(x) tick mark
    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    ax3.set_xticks(
        [base_values + values.sum(), base_values + values.sum() + min(1e-8, xmax * 1e-10)]
    )  # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    # However, for very small values, 1e-8 is disruptively large, so xmax * 1e-10 is used instead
    ax3.set_xticklabels(["$f(x)$", "$ = " + format_value(fx, "%0.03f") + "$"], fontsize=12, ha="left")
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-10 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform() + matplotlib.transforms.ScaledTranslation(12 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_color(style.tick_labels_color)
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform() + matplotlib.transforms.ScaledTranslation(-20 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform()
        + matplotlib.transforms.ScaledTranslation(22 / 72.0, -1 / 72.0, fig.dpi_scale_trans)
    )

    tick_labels[1].set_color(style.tick_labels_color)

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color(style.tick_labels_color)

    if show:
        plt.show()
    else:
        return plt.gca()
