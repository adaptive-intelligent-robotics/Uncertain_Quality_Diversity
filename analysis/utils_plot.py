from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator
from seaborn.relational import _LinePlotter

from analysis.utils_graphic import (
    adjust_box_widths,
    customize_axis,
    first_second_third_quartile,
)


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
    figsize = (ncols * 8, nrows * 6)

    # Create figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True)
    _LinePlotter.aggregate = first_second_third_quartile

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
                    ci=None,
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

    if box_plot:
        adjust_box_widths(fig, 0.6)

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
    _LinePlotter.aggregate = first_second_third_quartile

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
                    ci=None,
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

    if box_plot:
        adjust_box_widths(fig, 0.6)

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
    _LinePlotter.aggregate = first_second_third_quartile

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
                    ci=None,
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
            if major_locator != [] and major_locator_name != []:
                if nrow == nrows - 1:
                    ax.xaxis.set_major_locator(FixedLocator(major_locator))
                    ax.xaxis.set_major_formatter(lambda pos, x: major_locator_name[x])

    if box_plot:
        adjust_box_widths(fig, 0.6)

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
    nrows = len(rows_columns_metrics)
    ncols = len(rows_columns_metrics[0])
    figsize = (ncols * 8, nrows * 6)

    # Create figure
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    _LinePlotter.aggregate = first_second_third_quartile

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

            # Set palette
            hue_values = row_column_data_frame[hue].drop_duplicates().values
            env_palette = color_frame[color_frame["Label"].isin(hue_values)][
                "Color"
            ].values
            sns.set_palette(env_palette)

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
                    ax=ax,
                )
                ax.set_xlabel(None)
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
                )
                ax.set_xlabel(xlabel, labelpad=7)
            else:
                sns.lineplot(
                    x=x,
                    y=y,
                    data=row_column_data_frame,
                    hue=hue,
                    estimator=np.median,
                    ci=None,
                    style=hue,
                    ax=ax,
                    markers=markers,
                    dashes=not (markers),
                )
                ax.set_xlabel(xlabel, labelpad=7)

            # Set labels and title
            ax.set_ylabel(ylabel, labelpad=7)
            ax.set_title(title)
            customize_axis(ax)
            if remove_xaxis:
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
                        ax.axhspan(
                            fillline[0],
                            fillline[1],
                            color=fillline[2],
                            alpha=fillline[3],
                        )

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

            # Set y-axis in scientifix notation
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    if box_plot:
        adjust_box_widths(fig, 0.6)

    # Add legend below graph
    plt.tight_layout(h_pad=1.70)
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
