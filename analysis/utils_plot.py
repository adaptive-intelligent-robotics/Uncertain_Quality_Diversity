from itertools import cycle
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
from matplotlib.ticker import FixedLocator

MARKER_STYLES = [
    "o",  # circle
    "s",  # square
    "D",  # diamond
    "d",  # thin diamond
    "^",  # upward triangle
    "v",  # downward triangle
    "<",  # left triangle
    ">",  # right triangle
    "p",  # pentagon
    "*",  # star
    "h",  # hexagon 1
    "H",  # hexagon 2
    "8",  # octagon
    "X",  # filled X (added in newer versions of Matplotlib)
]


###################
# Utils functions #


def customize_axis(ax: Any) -> Any:
    """
    Customise axis for plots.
    """

    # Remove unused axis
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.tick_params(axis="y", length=0)

    # Offset the spines
    for spine in ax.spines.values():
        spine.set_position(("outward", 5))

    # Put the grid behind
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linestyle="--", linewidth=1.5)

    return ax


def intervalplot(
    y: str,
    data: pd.DataFrame,
    hue: str,
    hue_values: List,
    color_frame: pd.DataFrame,
    h: float,
    xlabel: str,
    ylabel: str,
    ax: Any,
) -> None:
    hue_values = np.flip(hue_values)
    for hue_idx, hue_value in enumerate(hue_values):

        # Get color
        hue_value_color = color_frame[color_frame["Label"] == hue_value][
            "Color"
        ].values[0]

        # Get values to plot
        values = np.expand_dims(data[data[hue] == hue_value][y].values, axis=1)
        values = values[values == values]

        # Safeguard in case empty
        if len(values) == 1 and values[0] == 0.0:
            ax.barh(
                y=hue_idx,
                width=0.0,
                height=h,
                left=0.0,
                color=hue_value_color,
                alpha=0.0,
                label=hue_value,
            )
            ax.vlines(
                x=0.0,
                ymin=hue_idx - (8 * h / 16),
                ymax=hue_idx + (7.98 * h / 16),
                label=hue_value,
                color="k",
                alpha=1.0,
                linewidth=2,
            )
            continue

        # Function for bootstrapped 95% confidence intervals
        def boot_percentile_ci(
            data: np.ndarray, confidence_level: float = 0.95
        ) -> Tuple[np.ndarray, np.darray]:
            res = scipy.stats.bootstrap(
                (data,),
                np.median,
                confidence_level=confidence_level,
                method="percentile",
                n_resamples=1000,
            )
            return res.confidence_interval.low, res.confidence_interval.high

        aggregate_values = scipy.stats.trim_mean(
            values.squeeze(), proportiontocut=0.25, axis=None
        )
        aggregate_values_cis = boot_percentile_ci(
            values.squeeze(),
        )

        # Plot interval estimates
        lower, upper = aggregate_values_cis
        ax.barh(
            y=hue_idx,
            width=upper - lower,
            height=h,
            left=lower,
            color=hue_value_color,
            alpha=0.8,
            label=hue_value,
        )

        # Plot point estimates
        ax.vlines(
            x=aggregate_values,
            ymin=hue_idx - (8 * h / 16),
            ymax=hue_idx + (7.98 * h / 16),
            label=hue_value,
            color="k",
            alpha=1.0,
            linewidth=2,
        )

    # Cosmetics
    ax.set_xlabel(None)
    ax.set_yticks(list(range(len(hue_values))))
    # ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.tick_params(axis="y", which="both", length=0.0)
    ax.tick_params(axis="x", which="both", length=6)
    if ylabel is not None and ylabel != "":
        ax.set_yticklabels(hue_values, weight="bold")
        ax.text(
            0.0,
            1.0,
            ylabel,
            transform=ax.transAxes,
            weight="bold",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
    else:
        ax.set_yticklabels([])

    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))


def intervalboxplot(
    y: str,
    data: pd.DataFrame,
    hue: str,
    hue_values: List,
    color_frame: pd.DataFrame,
    h: float,
    xlabel: str,
    ylabel: str,
    ax: Any,
) -> None:
    hue_values = np.flip(hue_values)
    for hue_idx, hue_value in enumerate(hue_values):

        # Get color
        hue_value_color = color_frame[color_frame["Label"] == hue_value][
            "Color"
        ].values[0]

        # Get values to plot
        values = np.expand_dims(data[data[hue] == hue_value][y].values, axis=1)
        values = values[values == values]

        # Safeguard in case empty
        if len(values) == 1 and values[0] == 0.0:
            continue

        ax.boxplot(
            values,
            vert=False,
            positions=[hue_idx],
            widths=h,
            patch_artist=True,
            boxprops=dict(facecolor=hue_value_color, color=hue_value_color, alpha=0.75),
            medianprops=dict(color="k", linewidth=2),
            whiskerprops=dict(linewidth=2),
            capprops=dict(linewidth=2),
            flierprops=dict(
                marker="o",
                color=hue_value_color,
                alpha=0.5,
                markersize=10,
                linestyle="none",
                linewidth=2,
            ),
        )

    # Cosmetics
    ax.set_xlabel(None)
    ax.set_yticks(list(range(len(hue_values))))
    # ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.tick_params(axis="y", which="both", length=0.0)
    ax.tick_params(axis="x", which="both", length=6)
    if ylabel is not None and ylabel != "":
        ax.set_yticklabels(hue_values, weight="bold")
        ax.text(
            0.0,
            1.0,
            ylabel,
            transform=ax.transAxes,
            weight="normal",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
    else:
        ax.set_yticklabels([])

    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))


def intervalboxplot_box(
    ax: Any,
    point_estimates: List,
    interval_estimates: List,
    datapoints: Dict,
    algorithms: List,
    colors: List,
    max_ticks: int,
    xlabel: str,
    ylabel: str,
    title: str,
    metric_title: str,
    plot_datapoints: bool = False,
) -> None:
    h = 0.6

    for alg_idx, algorithm in enumerate(algorithms):

        if algorithm not in datapoints.keys():
            continue

        # Plot interval estimates using boxplot
        algo_datapoints = datapoints[algorithm].squeeze()
        ax.boxplot(
            algo_datapoints,
            vert=False,
            positions=[alg_idx],
            widths=h,
            patch_artist=True,
            boxprops=dict(
                facecolor=colors[algorithm], color=colors[algorithm], alpha=0.75
            ),
            medianprops=dict(color="k", linewidth=2),
            whiskerprops=dict(linewidth=2),
            capprops=dict(linewidth=2),
            flierprops=dict(
                marker="o",
                color=colors[algorithm],
                alpha=0.5,
                markersize=10,
                linestyle="none",
                linewidth=2,
            ),
        )

        # Plot point estimates as vertical lines
        # ax.vlines(
        #     x=point_estimates[algorithm],
        #     ymin=alg_idx - h / 2,
        #     ymax=alg_idx + h / 2,
        #     color='k',
        #     alpha=0.5,
        #     linewidth=3,
        # )

        # Optionally, scatter plot datapoints
        if plot_datapoints:
            y = np.ones_like(algo_datapoints) * alg_idx
            ax.scatter(
                y=y,
                x=algo_datapoints,
                marker="o",
                color=colors[algorithm],
                s=200,
                alpha=0.25,
            )

    # Title
    ax.set_title(title, weight="bold")
    ax.set_xlabel(xlabel)

    # Set y-ticks and labels
    ax.set_yticks(list(range(len(algorithms))))
    if ylabel:
        ax.set_yticklabels(algorithms)
        ax.text(
            -0.2,
            1.0,
            metric_title,
            transform=ax.transAxes,
            fontsize=40,
            weight="bold",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
    else:
        ax.set_yticklabels([])

    # Set x-ticks and limits
    ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
    ax.tick_params(axis="y", which="both", length=0.0)
    ax.tick_params(axis="x", which="both", length=6)

    # Grid visual
    ax.grid(True, axis="x", alpha=0.25)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2)

    # Deal with ticks and the blank space at the origin
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    ax.yaxis.set_tick_params(pad=100)


#######################
# Main plot functions #


def plot(
    file_name: str,
    data_frame: pd.DataFrame,
    rows_columns: List,
    rows_columns_name: List,
    vlines: List,
    hlines: List,
    filllines: List,
    x_front: List,
    y_front: List,
    box_plot: bool,
    scatter_plot: bool,
    x: str,
    xlabel: str,
    major_locator: List,
    major_locator_name: List,
    size: float,
    sizes: List,
    hue: str,
    color_frame: pd.DataFrame,
    legend_columns: int,
    legend_bottom: float,
    markers: bool,
) -> None:
    """Main plot function."""

    # Extract lines and columns information
    nrows = len(rows_columns)
    ncols = len(rows_columns[0])
    figsize = (ncols * 7, nrows * 6)

    # Create figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)

    # Plot all subplots
    all_handles: List = []
    all_labels: List = []
    for nrow in range(nrows):

        # Set palette
        hue_values = data_frame[hue].drop_duplicates().values
        env_palette = color_frame[color_frame["Label"].isin(hue_values)]["Color"].values
        sns.set_palette(env_palette)

        # Plot each column
        for ncol in range(ncols):
            y = rows_columns[nrow][ncol]
            ylabel = rows_columns_name[nrow][ncol]

            # Get axis
            if nrows == 1 and ncols == 1:
                ax = axes
            elif nrows == 1:
                ax = axes[ncol]
            elif ncols == 1:
                ax = axes[nrow]
            else:
                ax = axes[nrow, ncol]

            # Add vline
            if vlines != []:
                for vline in vlines:
                    ax.axvline(vline[0], color=vline[1], dashes=vline[2])

            # Add hline
            if hlines != []:
                for hline in hlines:
                    ax.axhline(hline[0], color=hline[1], dashes=hline[2])

            # Add filllines
            if filllines != []:
                for fillline in filllines:
                    ax.axhspan(
                        fillline[0], fillline[1], color=fillline[2], alpha=fillline[3]
                    )

            # Add x_front y_front
            if x_front != [] and y_front != []:
                ax.plot(
                    x_front[ncol],
                    y_front[ncol],
                    color="b",
                    linestyle="--",
                    linewidth=2,
                    zorder=0,
                    marker="o",
                )

            # Plot
            if box_plot:
                sns.boxplot(
                    x=x,
                    y=y,
                    hue=hue,
                    data=data_frame,
                    width=1.0,
                    gap=0.3,
                    linewidth=2,
                    # whis=(25, 75),
                    ax=ax,
                )
                ax.set_xlabel(None, labelpad=7)
            elif scatter_plot:
                sns.scatterplot(
                    x=x,
                    y=y,
                    data=data_frame,
                    hue=hue,
                    size=size,
                    sizes=sizes,
                    edgecolor="black",
                    linewidth=2,
                    alpha=0.6,
                    ax=ax,
                )
                ax.set_xlabel(xlabel, labelpad=7)
            else:
                sns.lineplot(
                    x=x,
                    y=y,
                    data=data_frame,
                    hue=hue,
                    estimator=np.median,
                    errorbar=("pi", 50),
                    style=hue,
                    ax=ax,
                    markers=markers,
                    dashes=not (markers),
                )
                ax.set_xlabel(xlabel, labelpad=7)

            # Set labels
            ax.set_ylabel(ylabel, labelpad=7)
            customize_axis(ax)

            # Handle legends
            handles, labels = ax.get_legend_handles_labels()
            for i in range(len(labels)):
                if labels[i] not in all_labels:
                    all_handles.append(handles[i])
                    all_labels.append(labels[i])
            ax.legend_.remove()

            # Handle major locator
            if major_locator != [] and major_locator_name != []:
                if nrow == nrows - 1:
                    ax.xaxis.set_major_locator(FixedLocator(major_locator))
                    ax.xaxis.set_major_formatter(lambda pos, x: major_locator_name[x])

    # Add legend below graph
    plt.tight_layout()
    fig.subplots_adjust(bottom=legend_bottom)
    fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc="lower center",
        frameon=False,
        ncol=legend_columns,
    )

    # Save figure
    plt.savefig(file_name)
    plt.close()


def plot_rows_select(
    file_name: str,
    data_frame: pd.DataFrame,
    rows_select: List,
    rows_name: List,
    columns: List,
    columns_name: List,
    vlines: List,
    hlines: List,
    filllines: List,
    x_front: List,
    y_front: List,
    box_plot: bool,
    scatter_plot: bool,
    x: str,
    xlabel: str,
    major_locator: List,
    major_locator_name: List,
    size: float,
    sizes: List,
    hue: str,
    color_frame: pd.DataFrame,
    legend_columns: int,
    legend_bottom: float,
    markers: bool,
) -> None:
    """Main plot function."""

    # Extract lines and columns information
    nrows = len(rows_select)
    ncols = len(columns)
    figsize = (ncols * 8, nrows * 6)

    # Create figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)

    # Plot all subplots
    all_handles: List = []
    all_labels: List = []
    for nrow in range(nrows):

        # Get the data for this row
        row_data_frame = data_frame[
            data_frame[rows_select[nrow][0]] == rows_select[nrow][1]
        ]

        # Set palette
        hue_values = row_data_frame[hue].drop_duplicates().values
        env_palette = color_frame[color_frame["Label"].isin(hue_values)]["Color"].values
        sns.set_palette(env_palette)

        # Plot each column
        for ncol in range(ncols):
            y = columns[ncol]

            # Get axis
            if nrows == 1 and ncols == 1:
                ax = axes
            elif nrows == 1:
                ax = axes[ncol]
            elif ncols == 1:
                ax = axes[nrow]
            else:
                ax = axes[nrow, ncol]

            # Add vline
            if vlines != []:
                for vline in vlines:
                    ax.axvline(vline[0], color=vline[1], dashes=vline[2])

            # Add hline
            if hlines != []:
                for hline in hlines:
                    ax.axhline(hline[0], color=hline[1], dashes=hline[2])

            # Add filllines
            if filllines != []:
                for fillline in filllines:
                    ax.axhspan(
                        fillline[0], fillline[1], color=fillline[2], alpha=fillline[3]
                    )

            # Add x_front y_front
            if x_front != [] and y_front != []:
                ax.plot(
                    x_front[ncol],
                    y_front[ncol],
                    color="b",
                    linestyle="--",
                    linewidth=2,
                    zorder=0,
                    marker="o",
                )

            # Plot
            if box_plot:
                sns.boxplot(
                    x=x,
                    y=y,
                    hue=hue,
                    data=row_data_frame,
                    width=1.0,
                    gap=0.3,
                    linewidth=2,
                    # whis=(25, 75),
                    ax=ax,
                )
                ax.set_xlabel(None)
            elif scatter_plot:
                sns.scatterplot(
                    x=x,
                    y=y,
                    data=row_data_frame,
                    hue=hue,
                    size=size,
                    sizes=sizes,
                    edgecolor="black",
                    linewidth=2,
                    alpha=0.6,
                    ax=ax,
                )
                ax.set_xlabel(xlabel, labelpad=7)
            else:
                sns.lineplot(
                    x=x,
                    y=y,
                    data=row_data_frame,
                    hue=hue,
                    estimator=np.median,
                    errorbar=("pi", 50),
                    style=hue,
                    ax=ax,
                    markers=markers,
                    dashes=not (markers),
                )
                ax.set_xlabel(xlabel, labelpad=7)

            # Set labels and title
            if ncol == 0:
                ax.set_ylabel(rows_name[nrow], labelpad=7)
            else:
                ax.set_ylabel(None)
            if nrow == 0:
                ax.set_title(columns_name[ncol])
            customize_axis(ax)

            # Handle legends
            handles, labels = ax.get_legend_handles_labels()
            for i in range(len(labels)):
                if labels[i] not in all_labels:
                    all_handles.append(handles[i])
                    all_labels.append(labels[i])
            ax.legend_.remove()

            # Handle major locator
            if major_locator != [] and major_locator_name != []:
                if nrow == nrows - 1:
                    ax.xaxis.set_major_locator(FixedLocator(major_locator))
                    ax.xaxis.set_major_formatter(lambda pos, x: major_locator_name[x])

    # Add legend below graph
    plt.tight_layout()
    fig.subplots_adjust(bottom=legend_bottom)
    fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc="lower center",
        frameon=False,
        ncol=legend_columns,
    )

    # Save figure
    plt.savefig(file_name)
    plt.close()


def plot_columns_select(
    file_name: str,
    data_frame: pd.DataFrame,
    columns_select: List,
    columns_name: List,
    rows: List,
    rows_name: List,
    vlines: List,
    hlines: List,
    filllines: List,
    x_front: List,
    y_front: List,
    box_plot: bool,
    scatter_plot: bool,
    x: str,
    xlabel: str,
    major_locator: List,
    major_locator_name: List,
    size: float,
    sizes: List,
    hue: str,
    color_frame: pd.DataFrame,
    legend_columns: int,
    legend_bottom: float,
    markers: bool,
) -> None:
    """Main plot function."""

    # Extract lines and columns information
    nrows = len(rows)
    ncols = len(columns_select)
    figsize = (ncols * 8, nrows * 6)

    # Create figure
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey="row"
    )

    # Plot all subplots
    all_handles: List = []
    all_labels: List = []
    for ncol in range(ncols):

        # Get the data for this row
        column_data_frame = data_frame[
            data_frame[columns_select[ncol][0]] == columns_select[ncol][1]
        ]

        # Set palette
        hue_values = column_data_frame[hue].drop_duplicates().values
        env_palette = color_frame[color_frame["Label"].isin(hue_values)]["Color"].values
        sns.set_palette(env_palette)

        # Plot each column
        for nrow in range(nrows):
            y = rows[nrow]
            ylabel = rows_name[nrow]

            # Get axis
            if nrows == 1 and ncols == 1:
                ax = axes
            elif nrows == 1:
                ax = axes[ncol]
            elif ncols == 1:
                ax = axes[nrow]
            else:
                ax = axes[nrow, ncol]

            # Add vline
            if vlines != []:
                for vline in vlines:
                    ax.axvline(vline[0], color=vline[1], dashes=vline[2])

            # Add hline
            if hlines != []:
                for hline in hlines:
                    ax.axhline(hline[0], color=hline[1], dashes=hline[2])

            # Add filllines
            if filllines != []:
                for fillline in filllines:
                    ax.axhspan(
                        fillline[0], fillline[1], color=fillline[2], alpha=fillline[3]
                    )

            # Add x_front y_front
            if x_front != [] and y_front != []:
                ax.plot(
                    x_front[ncol],
                    y_front[ncol],
                    color="b",
                    linestyle="--",
                    linewidth=2,
                    zorder=0,
                    marker="o",
                )

            # Plot
            if box_plot:
                sns.boxplot(
                    x=x,
                    y=y,
                    hue=hue,
                    data=column_data_frame,
                    width=1.0,
                    gap=0.3,
                    linewidth=2,
                    # whis=(25, 75),
                    ax=ax,
                )
                ax.set_xlabel(None)
            elif scatter_plot:
                sns.scatterplot(
                    x=x,
                    y=y,
                    data=column_data_frame,
                    hue=hue,
                    size=size,
                    sizes=sizes,
                    edgecolor="black",
                    linewidth=2,
                    alpha=0.6,
                    ax=ax,
                )
                ax.set_xlabel(xlabel, labelpad=7)
            else:
                sns.lineplot(
                    x=x,
                    y=y,
                    data=column_data_frame,
                    hue=hue,
                    estimator=np.median,
                    errorbar=("pi", 50),
                    style=hue,
                    ax=ax,
                    markers=markers,
                    dashes=not (markers),
                )
                ax.set_xlabel(xlabel, labelpad=7)

            # Set labels and title
            if ncol == 0:
                ax.set_ylabel(ylabel, labelpad=7)
            else:
                ax.set_ylabel(None)
            if nrow == 0:
                ax.set_title(columns_name[ncol])
            customize_axis(ax)

            # Handle legends
            handles, labels = ax.get_legend_handles_labels()
            for i in range(len(labels)):
                if labels[i] not in all_labels:
                    all_handles.append(handles[i])
                    all_labels.append(labels[i])
            ax.legend_.remove()

            # Handle major locator
            if len(major_locator) > 0 and len(major_locator_name) == len(major_locator):
                if nrow == nrows - 1:
                    ax.xaxis.set_major_locator(FixedLocator(major_locator))
                    ax.xaxis.set_major_formatter(lambda pos, x: major_locator_name[x])

    # Add legend below graph
    plt.tight_layout()
    fig.subplots_adjust(bottom=legend_bottom)
    fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc="lower center",
        frameon=False,
        ncol=legend_columns,
    )

    # Save figure
    plt.savefig(file_name)
    plt.close()


def plot_rows_columns_select(
    file_name: str,
    data_frame: pd.DataFrame,
    rows_columns_metrics: List,
    rows_columns_metrics_name: List,
    rows_columns_select: List,
    rows_columns_title: List,
    vlines: List,
    hlines: List,
    filllines: List,
    x_front: List,
    y_front: List,
    box_plot: bool,
    scatter_plot: bool,
    interval_plot: bool,
    intervalbox_plot: bool,
    x: str,
    xlabel: str,
    major_locator: List,
    major_locator_name: List,
    size: float,
    sizes: List,
    hue: str,
    color_frame: pd.DataFrame,
    legend_columns: int,
    legend_bottom: float,
    markers: bool,
    remove_xaxis: bool = False,
    dashes: List = [],
) -> None:
    """Main plot function."""

    # Extract lines and columns information
    assert len(rows_columns_metrics) == len(
        rows_columns_metrics_name
    ), "!!!ERRROR!!! Inconsistant dims."
    assert len(rows_columns_metrics) == len(
        rows_columns_select
    ), "!!!ERRROR!!! Inconsistant dims."
    assert len(rows_columns_metrics) == len(
        rows_columns_title
    ), "!!!ERRROR!!! Inconsistant dims."
    assert len(rows_columns_metrics[0]) == len(
        rows_columns_metrics_name[0]
    ), "!!!ERRROR!!! Inconsistant dims."
    assert len(rows_columns_metrics[0]) == len(
        rows_columns_select[0]
    ), "!!!ERRROR!!! Inconsistant dims."
    assert len(rows_columns_metrics[0]) == len(
        rows_columns_title[0]
    ), "!!!ERRROR!!! Inconsistant dims."

    # Set palette and marker in common
    hue_values = data_frame[hue].drop_duplicates().values
    sub_color_frame = color_frame[color_frame["Label"].isin(hue_values)]
    env_palette = dict(zip(sub_color_frame["Label"], sub_color_frame["Color"]))
    if markers:
        markers = {
            hue_value: marker
            for hue_value, marker in zip(hue_values, cycle(MARKER_STYLES))
        }  # type: ignore
        dashes = False  # type: ignore
    else:
        if dashes == []:
            dashes = True  # type: ignore

    # Create figure
    nrows = len(rows_columns_metrics)
    ncols = len(rows_columns_metrics[0])
    if interval_plot or intervalboxplot:
        figsize = (ncols * 12, nrows * 0.7 * len(hue_values) + 3)
    else:
        figsize = (ncols * 8, nrows * 8)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    # Plot all subplots
    all_handles: List = []
    all_labels: List = []
    for nrow in range(nrows):

        for ncol in range(ncols):

            # Get the data for this row and column
            row_column_data_frame = data_frame[
                data_frame[rows_columns_select[nrow][ncol][0]]
                == rows_columns_select[nrow][ncol][1]
            ]

            # Get y and title
            y = rows_columns_metrics[nrow][ncol]
            ylabel = rows_columns_metrics_name[nrow][ncol]
            title = rows_columns_title[nrow][ncol]

            # Get axis
            if nrows == 1 and ncols == 1:
                ax = axes
            elif nrows == 1:
                ax = axes[ncol]
            elif ncols == 1:
                ax = axes[nrow]
            else:
                ax = axes[nrow, ncol]

            # Add x_front y_front
            if x_front != [] and y_front != []:
                ax.plot(
                    x_front[ncol],
                    y_front[ncol],
                    color="b",
                    linestyle="--",
                    linewidth=2,
                    zorder=0,
                    marker="o",
                )

            # Plot
            if box_plot:
                sns.boxplot(
                    x=x,
                    y=y,
                    hue=hue,
                    data=row_column_data_frame,
                    width=1.0,
                    gap=0.3,
                    linewidth=2,
                    # whis=(25, 75),
                    ax=ax,
                    palette=env_palette,
                )
                ax.set_xlabel(None)
                ax.set_ylabel(ylabel, labelpad=7)
                customize_axis(ax)
                ax.set_title(title)
            elif scatter_plot:
                sns.scatterplot(
                    x=x,
                    y=y,
                    data=row_column_data_frame,
                    hue=hue,
                    size=size,
                    sizes=sizes,
                    edgecolor="black",
                    linewidth=2,
                    alpha=0.6,
                    ax=ax,
                    palette=env_palette,
                )
                ax.set_xlabel(xlabel, labelpad=7)
                ax.set_ylabel(ylabel, labelpad=7)
                customize_axis(ax)
                ax.set_title(title)
            elif interval_plot:
                intervalplot(
                    y=y,
                    hue=hue,
                    hue_values=hue_values,
                    data=row_column_data_frame,
                    color_frame=color_frame,
                    h=0.6,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    ax=ax,
                )
                ax.set_title(title, weight="bold")
            elif intervalbox_plot:
                intervalboxplot(
                    y=y,
                    hue=hue,
                    hue_values=hue_values,
                    data=row_column_data_frame,
                    color_frame=color_frame,
                    h=0.6,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    ax=ax,
                )
                ax.set_title(title, weight="bold")
            else:
                sns.lineplot(
                    x=x,
                    y=y,
                    data=row_column_data_frame,
                    hue=hue,
                    estimator=np.median,
                    errorbar=("pi", 50),
                    style=hue,
                    ax=ax,
                    markers=markers,
                    dashes=dashes,
                    palette=env_palette,
                )
                ax.set_xlabel(xlabel, labelpad=7)
                ax.set_ylabel(ylabel, labelpad=7)
                customize_axis(ax)
                ax.set_title(title)

            # Set labels and title
            if remove_xaxis and not interval_plot and not intervalbox_plot:
                ax.get_xaxis().set_visible(False)
                # ax.spines["bottom"].set_visible(False)

            # Add vlines
            if vlines != []:
                if vlines[nrow][ncol] != []:
                    for vline in vlines[nrow][ncol]:
                        ax.axvline(vline[0], color=vline[1], dashes=vline[2])

            # Add hlines
            if hlines != []:
                if hlines[nrow][ncol] != []:
                    for hline in hlines[nrow][ncol]:
                        ax.axhline(hline[0], color=hline[1], dashes=hline[2])

            # Add filllines
            if filllines != []:
                if filllines[nrow][ncol] != []:
                    for fillline in filllines[nrow][ncol]:
                        if not interval_plot and not intervalbox_plot:
                            ax.axhspan(
                                fillline[0],
                                fillline[1],
                                color=fillline[2],
                                alpha=fillline[3],
                            )
                        else:
                            ax.axvspan(
                                fillline[0],
                                fillline[1],
                                color=fillline[2],
                                alpha=fillline[3],
                            )

            # Handle legends
            if not interval_plot and not intervalbox_plot:
                handles, labels = ax.get_legend_handles_labels()
                for i in range(len(labels)):
                    if labels[i] not in all_labels:
                        all_handles.append(handles[i])
                        all_labels.append(labels[i])
                ax.legend_.remove()

            # Handle major locator
            if major_locator != [] and major_locator_name != []:
                if nrow == nrows - 1:
                    ax.xaxis.set_major_locator(FixedLocator(major_locator))
                    ax.xaxis.set_major_formatter(lambda pos, x: major_locator_name[x])

            # Set y-axis in scientifix notation
            if not interval_plot and not intervalbox_plot:
                ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            else:
                if (
                    not "loss" in y
                    and not "samples" in y
                    and not "time" in y
                    and not "coverage" in y
                ):
                    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # Tight layout
    if not interval_plot and not intervalbox_plot:
        plt.tight_layout(h_pad=1.70)
    else:
        plt.tight_layout(w_pad=2.0)

    # Add legend below graph
    if not interval_plot and not intervalbox_plot:
        fig.subplots_adjust(bottom=legend_bottom)
        fig.legend(
            handles=all_handles,
            labels=all_labels,
            loc="lower center",
            frameon=False,
            ncol=legend_columns,
        )

    # Save figure
    plt.savefig(file_name)
    plt.close()
