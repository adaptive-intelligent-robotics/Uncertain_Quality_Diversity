import os
import traceback
from collections import Counter
from itertools import cycle
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import trim_mean

from analysis.load_p_values import p_values as p_values_fn
from analysis.load_results import sort_data
from analysis.utils_plot import (
    MARKER_STYLES,
    customize_axis,
    plot_rows_columns_select,
    plot_rows_select,
)


def boxplot_line(algos: List, algo: float, dash: Any) -> List:
    single_line = [
        (algo / len(algos) - 0.5) * 1.0,  # position (multiply by 1.0, width of boxplot)
        "k",  # color
        dash,  # dash
    ]
    return single_line


def interval_line(algos: List, algo: float, dash: Any) -> List:
    single_line = [
        (-0.2 + len(algos) - algo) * (1.6 * 0.6),  # position (0.6 the h of lines)
        "k",  # color
        dash,  # dash
    ]
    return single_line


def plot_qdrl_families_convergence_split(
    plot_folder: str,
    all_convergence: pd.DataFrame,
    color_frame: pd.DataFrame,
    x_column: str,
    x_name: str,
    compare_size: str,
    compare_title: str,
    order: List,
    env_order: Dict,
    qdrl_families: Dict,
    legend_columns: int,
    legend_bottom: float,
    prefixe: str = "",
    prefixe_title: str = "",
    errors: bool = False,
) -> None:

    # First, print the convergence metric for each qdrl_families
    for family_key in qdrl_families.keys():

        # Extract data
        qdrl_family = qdrl_families[family_key]
        family_data = all_convergence[
            all_convergence["algo"].str.contains("|".join(qdrl_family))
        ].reset_index(drop=True)

        # Print the metrics per env
        for env in family_data["env"].drop_duplicates().values:

            # Extract and sort data
            env_convergence = family_data[family_data["env"] == env].reset_index(
                drop=True
            )
            env_convergence = sort_data(env_convergence, ["algo", "rep", "eval"], order)

            # Plot reeval across x_column with sizes as lines and metrics as columns
            size_values = env_convergence[compare_size].drop_duplicates().values
            size_values.sort()
            rows_select = [[compare_size, x] for x in size_values]
            rows_name = [f"{compare_title} {x}" for x in size_values]
            try:
                columns = [
                    f"{prefixe}reeval_qd_score",
                    f"{prefixe}reeval_coverage",
                    f"{prefixe}reeval_max_fitness",
                ]
                columns_name = [
                    f"{prefixe_title}Reeval QD Score",
                    f"{prefixe_title}Reeval Coverage in %",
                    f"{prefixe_title}Reeval Maximum fitness",
                ]
                plot_rows_select(
                    file_name=f"{plot_folder}/{env}-{family_key}-{prefixe}reeval_metrics-convergence.svg",
                    data_frame=env_convergence,
                    rows_select=rows_select,
                    rows_name=rows_name,
                    columns=columns,
                    columns_name=columns_name,
                    vlines=[],
                    hlines=[],
                    filllines=[],
                    x_front=[],
                    y_front=[],
                    box_plot=False,
                    scatter_plot=False,
                    x=x_column,
                    xlabel=x_name,
                    major_locator=[],
                    major_locator_name=[],
                    size=None,
                    sizes=[],
                    hue="algo",
                    color_frame=color_frame,
                    legend_columns=legend_columns,
                    legend_bottom=legend_bottom,
                    markers=True,
                )
            except Exception:
                print(
                    f"\n!!!WARNING!!! Cannot plot {prefixe}reeval_metrics-convergence for {family_key} and {env}."
                )
                if errors:
                    traceback.print_exc()


def plot_qdrl_families_convergence(
    plot_folder: str,
    all_convergence: pd.DataFrame,
    color_frame: pd.DataFrame,
    x_column: str,
    x_name: str,
    compare_size: str,
    compare_title: str,
    order: List,
    env_order: Dict,
    qdrl_families: Dict,
    legend_columns: int,
    legend_bottom: float,
    prefixe: str = "",
    prefixe_title: str = "",
    errors: bool = False,
) -> None:

    metrics = [
        f"{prefixe}reeval_qd_score",
        f"{prefixe}coverage",
        f"{prefixe}max_fitness",
    ]
    metrics_name = [
        f"{prefixe_title}Corrected QD-Score",
        f"{prefixe_title}Corrected Coverage",
        f"{prefixe_title}Corrected Max-Fitness",
    ]

    for idx in range(len(metrics)):

        try:
            file_name = f"{plot_folder}/{prefixe}reeval_metrics-all-convergence_{metrics[idx]}.svg"

            # Keep only the names that are in the list
            all_names_string = ""
            for family_key in qdrl_families.keys():
                all_names_string = "|".join(
                    [all_names_string, "|".join(qdrl_families[family_key])]
                )
            qdrl_frame = all_convergence[
                all_convergence["algo"].str.contains(all_names_string)
            ]
            qdrl_frame = sort_data(
                qdrl_frame,
                ["algo", "rep"],
                order,
            )

            # Extract the list of environments
            env_lists = qdrl_frame["env"].drop_duplicates().values

            # Create figure
            nrows = len(qdrl_families.keys())
            ncols = len(env_lists)
            figsize = (ncols * 6, nrows * 6)
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=figsize, sharey="col"
            )

            # Set palette and marker in common
            hue_values = qdrl_frame["algo"].drop_duplicates().values
            sub_color_frame = color_frame[color_frame["Label"].isin(hue_values)]
            env_palette = dict(zip(sub_color_frame["Label"], sub_color_frame["Color"]))
            markers = {
                hue_value: marker
                for hue_value, marker in zip(hue_values, cycle(MARKER_STYLES))
            }

            # Plot all subplots
            all_handles: List = []
            all_labels: List = []
            for nrow in range(nrows):

                for ncol in range(ncols):

                    # Get the data for this row and column
                    qdrl_family = qdrl_families[list(qdrl_families.keys())[nrow]]
                    env = env_lists[ncol]
                    row_column_qdrl_frame = qdrl_frame[
                        (qdrl_frame["algo"].str.contains("|".join(qdrl_family)))
                        & (qdrl_frame["env"] == env)
                    ]

                    # Get axis
                    if nrows == 1 and ncols == 1:
                        ax = axes
                    elif nrows == 1:
                        ax = axes[ncol]
                    elif ncols == 1:
                        ax = axes[nrow]
                    else:
                        ax = axes[nrow, ncol]

                    trimmed_mean_10 = lambda x: trim_mean(x, proportiontocut=0.1)
                    sns.lineplot(
                        x=x_column,
                        y=metrics[idx],
                        data=row_column_qdrl_frame,
                        hue="algo",
                        estimator=trimmed_mean_10,
                        errorbar=("pi", 95),
                        style="algo",
                        ax=ax,
                        markers=markers,
                        dashes=not (markers),
                        palette=env_palette,
                    )
                    ax.set_xlabel(x_name, labelpad=7)
                    if ncol == 0:
                        ax.set_ylabel(metrics_name[idx], labelpad=7)
                    else:
                        ax.set_ylabel(None)
                    customize_axis(ax)
                    if nrow == 0:
                        ax.set_title(env_order[env])

                    if nrow != nrows - 1:
                        ax.get_xaxis().set_visible(False)

                    # Handle legends
                    handles, labels = ax.get_legend_handles_labels()
                    for i in range(len(labels)):
                        if labels[i] not in all_labels:
                            all_handles.append(handles[i])
                            all_labels.append(labels[i])
                    ax.legend_.remove()

                    # Set y-axis in scientifix notation
                    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

            # Tight layout
            plt.tight_layout(h_pad=1.70)

            # Add legend below graph
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

        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot {prefixe}reeval_metrics-convergence..")
            if errors:
                traceback.print_exc()


def plot_qdrl_families_sample_complexity(
    plot_folder: str,
    all_convergence: pd.DataFrame,
    all_finals: pd.DataFrame,
    color_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    order: List,
    env_order: Dict,
    qdrl_families: Dict,
    legend_columns: int,
    legend_bottom: float,
    prefixe: str = "",
    prefixe_title: str = "",
    errors: bool = False,
    p_values: bool = False,
) -> None:
    print("\n\nsample_complexity")

    reach_families = [["PGA", "PE-PGA"], ["QDPG", "PE-QDPG"]]
    metrics = ["reeval_qd_score"]
    metrics_name = ["Corrected QD-Score"]

    for idx in range(len(metrics)):
        sample_complexity_frame = pd.DataFrame()

        for env in all_finals["env"].drop_duplicates().values:
            for family in reach_families:

                # Get target value as minimum from all seed of first approach
                family_finals = all_finals[
                    (all_finals["algo"] == family[0]) & (all_finals["env"] == env)
                ]
                target = family_finals[metrics[idx]].min()

                # Get max number of timestep for thi environment
                env_convergence = all_convergence[all_convergence["env"] == env]
                max_timesteps = env_convergence["timestep"].max()

                # Get time at which this value is reached by all algorithms in family
                for algo in family:
                    algo_convergence = env_convergence[
                        env_convergence["algo"] == algo
                    ].reset_index(drop=True)

                    for rep in algo_convergence["rep"].drop_duplicates().values:

                        rep_convergence = algo_convergence[
                            algo_convergence["rep"] == rep
                        ].reset_index(drop=True)

                        # Get infos
                        name = rep_convergence["name"][0]
                        algo = rep_convergence["algo"][0]
                        algo_batch = rep_convergence["algo_batch"][0]
                        env = rep_convergence["env"][0]
                        num_centroids = rep_convergence["num_centroids"][0]
                        num_reevals = rep_convergence["num_reevals"][0]
                        size = rep_convergence[compare_size][0]
                        batch_size = rep_convergence["batch_size"][0]
                        sampling_size = rep_convergence["sampling_size"][0]

                        # Compute metric
                        reached_line = rep_convergence[
                            rep_convergence[metrics[idx]] >= target
                        ]
                        if reached_line.empty:
                            print(f"Undefined for {algo} with target {target}.")
                            reached_timestep = 1
                        else:
                            reached_timestep = reached_line["timestep"].min()
                            reached_timestep = reached_timestep / max_timesteps

                        # Add to dataframe
                        frame: Dict[str, List[float]] = {}

                        frame["name"] = [name]
                        frame["algo"] = algo
                        frame["algo_batch"] = algo_batch
                        frame["num_centroids"] = num_centroids
                        frame["env"] = env
                        frame["num_reevals"] = num_reevals
                        frame[compare_size] = size
                        frame["batch_size"] = batch_size
                        frame["sampling_size"] = sampling_size
                        frame["rep"] = [float(rep)]

                        frame["sample_complexity"] = reached_timestep

                        # Concatenate all frames to existing ones
                        frame = pd.DataFrame.from_dict(frame)
                        sample_complexity_frame = pd.concat(
                            [sample_complexity_frame, frame], ignore_index=True
                        )

        # Sort datas
        sample_complexity_frame = sort_data(
            sample_complexity_frame, ["env", "algo", "rep"], order
        )

        all_envs = []
        for env_name in env_order:
            if env_name in sample_complexity_frame["env"].drop_duplicates().values:
                all_envs.append(env_name)
        for env_name in sample_complexity_frame["env"].drop_duplicates().values:
            if env_name not in all_envs:
                all_envs.append(env_name)

        # Build up the plot
        rows_columns_select = [
            [["env", env_name] for env_name in all_envs],
        ]
        rows_columns_title = [
            [env_order[env_name] for env_name in all_envs],
        ]
        rows_columns_metrics = [
            ["sample_complexity" for _ in all_envs],
        ]
        rows_columns_metrics_name = [
            ["Sampling Complexity"] + ["" for _ in range(len(all_envs) - 1)],
        ]

        try:
            plot_rows_columns_select(
                file_name=f"{plot_folder}/qdrl_families_sample_complexity.svg",
                data_frame=sample_complexity_frame,
                rows_columns_select=rows_columns_select,
                rows_columns_title=rows_columns_title,
                rows_columns_metrics=rows_columns_metrics,
                rows_columns_metrics_name=rows_columns_metrics_name,
                vlines=[],
                hlines=[],
                filllines=[],
                x_front=[],
                y_front=[],
                box_plot=False,
                scatter_plot=False,
                interval_plot=True,
                intervalbox_plot=False,
                x=metrics[idx],
                xlabel=metrics_name[idx],
                major_locator=[],
                major_locator_name=[],
                size=None,
                sizes=[],
                hue="algo",
                color_frame=color_frame,
                legend_columns=legend_columns,
                legend_bottom=legend_bottom,
                markers=True,
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot sample_complexity metrics.")
            if errors:
                traceback.print_exc()

    print("\n\n")

    # Second, compute p-value if asked
    if p_values:

        try:
            p_values_fn(
                plot_folder=plot_folder,
                compare_size=compare_size,
                stat=f"sample_complexity",
                dataframe=sample_complexity_frame,
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot p-values for sample_complexity.")
            if errors:
                traceback.print_exc()


def plot_qdrl_families_5pourcent(
    plot_folder: str,
    all_times: pd.DataFrame,
    env_order: Dict,
    qdrl_families: Dict,
    color_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    legend_columns: int,
    legend_bottom: float,
    p_values: bool,
    prefixe: str = "",
    prefixe_title: str = "",
    interval_plot: bool = True,
    errors: bool = False,
) -> None:

    # Extract algos that are in multiple categories as baselines
    name_counts = Counter([name for names in qdrl_families.values() for name in names])
    baselines_list = [name for name, count in name_counts.items() if count > 1]
    all_names_string = "|".join(baselines_list)
    other_lists = []
    for family_key in qdrl_families.keys():
        qdrl_family = [
            name for name in qdrl_families[family_key] if name not in baselines_list
        ]
        other_lists.append(qdrl_family)
        all_names_string = "|".join([all_names_string, "|".join(qdrl_family)])

    # Keep only the names that are in the list
    qdrl_frame = all_times[all_times["algo"].str.contains(all_names_string)]
    qdrl_frame = sort_data(
        qdrl_frame,
        ["algo", "rep"],
        baselines_list + [name for other_list in other_lists for name in other_list],
    )

    # Extract the list of environments
    env_lists = qdrl_frame["env"].drop_duplicates().values

    # Build the lines
    algos = qdrl_frame["algo"].drop_duplicates().values
    algo = 0
    for compare_algo in baselines_list:
        while algo < len(algos) and compare_algo in algos[algo]:
            algo += 1
    single_lines = [
        interval_line(algos, algo, "")
        if interval_plot
        else boxplot_line(algos, algo, "")
    ]
    for categories_list in other_lists:
        for compare_algo in categories_list:
            while algo < len(algos) and compare_algo in algos[algo]:
                algo += 1
        single_lines.append(
            interval_line(algos, algo, (5, 5))
            if interval_plot
            else boxplot_line(algos, algo, (5, 5))
        )

    lines = [
        [single_lines for env_name in env_lists],
    ]

    # Build up the plot
    rows_columns_select = [
        [["env", env_name] for env_name in env_lists],
    ]
    rows_columns_title = [
        [env_order[env_name] for env_name in env_lists],
    ]

    # First, plot QD-Score
    rows_columns_metrics = [
        [f"{prefixe}timestep" for env_name in env_lists],
    ]
    rows_columns_metrics_name = [
        [f"{prefixe_title}Timesteps"] + ["" for _ in range(len(env_lists) - 1)],
    ]
    sub_plot_qdrl_families(
        sub_file_name=f"{plot_folder}/{prefixe}qdrl_families_time_5pourcents",
        qdrl_frame=qdrl_frame,
        color_frame=color_frame,
        compare_size=compare_size,
        compare_title=compare_title,
        legend_columns=legend_columns,
        legend_bottom=legend_bottom,
        rows_columns_select=rows_columns_select,
        rows_columns_title=rows_columns_title,
        rows_columns_metrics=rows_columns_metrics,
        rows_columns_metrics_name=rows_columns_metrics_name,
        filllines=[],
        lines=lines,
        interval_plot=interval_plot,
        errors=errors,
    )

    # Second, compute p-value if asked
    if p_values:

        # Reeval QD-Score p-values
        p_values_frame = qdrl_frame[qdrl_frame["env"].isin(env_lists)]
        p_values_frame = sort_data(
            p_values_frame,
            ["algo", "rep"],
            baselines_list
            + [name for other_list in other_lists for name in other_list],
        )
        try:
            p_values_fn(
                plot_folder=plot_folder,
                compare_size=compare_size,
                stat=f"{prefixe}timestep",
                dataframe=p_values_frame,
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot p-values for {prefixe}time 5pourcents.")
            if errors:
                traceback.print_exc()


def plot_qdrl_families(
    plot_folder: str,
    all_finals: pd.DataFrame,
    env_order: Dict,
    qdrl_families: Dict,
    color_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    legend_columns: int,
    legend_bottom: float,
    p_values: bool,
    prefixe: str = "",
    prefixe_title: str = "",
    interval_plot: bool = True,
    errors: bool = False,
) -> None:

    # Extract algos that are in multiple categories as baselines
    name_counts = Counter([name for names in qdrl_families.values() for name in names])
    baselines_list = [name for name, count in name_counts.items() if count > 1]
    all_names_string = "|".join(baselines_list)
    other_lists = []
    for family_key in qdrl_families.keys():
        qdrl_family = [
            name for name in qdrl_families[family_key] if name not in baselines_list
        ]
        other_lists.append(qdrl_family)
        all_names_string = "|".join([all_names_string, "|".join(qdrl_family)])

    # Keep only the names that are in the list
    qdrl_frame = all_finals[all_finals["algo"].str.contains(all_names_string)]
    qdrl_frame = sort_data(
        qdrl_frame,
        ["algo", "rep"],
        baselines_list + [name for other_list in other_lists for name in other_list],
    )

    # Extract the list of environments
    env_lists = qdrl_frame["env"].drop_duplicates().values

    # Build the lines
    algos = qdrl_frame["algo"].drop_duplicates().values
    algo = 0
    for compare_algo in baselines_list:
        while algo < len(algos) and compare_algo in algos[algo]:
            algo += 1
    single_lines = [
        interval_line(algos, algo, "")
        if interval_plot
        else boxplot_line(algos, algo, "")
    ]
    for categories_list in other_lists:
        for compare_algo in categories_list:
            while algo < len(algos) and compare_algo in algos[algo]:
                algo += 1
        single_lines.append(
            interval_line(algos, algo, (5, 5))
            if interval_plot
            else boxplot_line(algos, algo, (5, 5))
        )

    lines = [
        [single_lines for env_name in env_lists],
    ]

    # Build up the plot
    rows_columns_select = [
        [["env", env_name] for env_name in env_lists],
    ]
    rows_columns_title = [
        [env_order[env_name] for env_name in env_lists],
    ]

    # First, plot QD-Score
    rows_columns_metrics = [
        [f"{prefixe}reeval_qd_score" for env_name in env_lists],
    ]
    rows_columns_metrics_name = [
        [f"{prefixe_title}Corrected QD-Score"]
        + ["" for _ in range(len(env_lists) - 1)],
    ]
    sub_plot_qdrl_families(
        sub_file_name=f"{plot_folder}/{prefixe}qdrl_families",
        qdrl_frame=qdrl_frame,
        color_frame=color_frame,
        compare_size=compare_size,
        compare_title=compare_title,
        legend_columns=legend_columns,
        legend_bottom=legend_bottom,
        rows_columns_select=rows_columns_select,
        rows_columns_title=rows_columns_title,
        rows_columns_metrics=rows_columns_metrics,
        rows_columns_metrics_name=rows_columns_metrics_name,
        filllines=[],
        lines=lines,
        interval_plot=interval_plot,
        errors=errors,
    )

    # Second, compute p-value if asked
    if p_values:

        # Reeval QD-Score p-values
        p_values_frame = qdrl_frame[qdrl_frame["env"].isin(env_lists)]
        p_values_frame = sort_data(
            p_values_frame,
            ["algo", "rep"],
            baselines_list
            + [name for other_list in other_lists for name in other_list],
        )
        try:
            p_values_fn(
                plot_folder=plot_folder,
                compare_size=compare_size,
                stat=f"{prefixe}reeval_qd_score",
                dataframe=p_values_frame,
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot p-values for {prefixe}reeval_qd_score.")
            if errors:
                traceback.print_exc()


def plot_qdrl_families_area(
    plot_folder: str,
    all_convergence: pd.DataFrame,
    env_order: Dict,
    qdrl_families: Dict,
    color_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    legend_columns: int,
    legend_bottom: float,
    p_values: bool,
    prefixe: str = "",
    prefixe_title: str = "",
    interval_plot: bool = True,
    errors: bool = False,
) -> None:

    # Extract algos that are in multiple categories as baselines
    name_counts = Counter([name for names in qdrl_families.values() for name in names])
    baselines_list = [name for name, count in name_counts.items() if count > 1]
    all_names_string = "|".join(baselines_list)
    other_lists = []
    for family_key in qdrl_families.keys():
        qdrl_family = [
            name for name in qdrl_families[family_key] if name not in baselines_list
        ]
        other_lists.append(qdrl_family)
        all_names_string = "|".join([all_names_string, "|".join(qdrl_family)])

    # Keep only the names that are in the list
    qdrl_frame = all_convergence[all_convergence["algo"].str.contains(all_names_string)]
    qdrl_frame = sort_data(
        qdrl_frame,
        ["algo", "rep", "eval"],
        baselines_list + [name for other_list in other_lists for name in other_list],
    )

    # Extract the list of environments
    env_lists = qdrl_frame["env"].drop_duplicates().values

    # Build the lines
    algos = qdrl_frame["algo"].drop_duplicates().values
    algo = 0
    for compare_algo in baselines_list:
        while algo < len(algos) and compare_algo in algos[algo]:
            algo += 1
    single_lines = [
        interval_line(algos, algo, "")
        if interval_plot
        else boxplot_line(algos, algo, "")
    ]
    for categories_list in other_lists:
        for compare_algo in categories_list:
            while algo < len(algos) and compare_algo in algos[algo]:
                algo += 1
        single_lines.append(
            interval_line(algos, algo, (5, 5))
            if interval_plot
            else boxplot_line(algos, algo, (5, 5))
        )

    lines = [
        [single_lines for env_name in env_lists],
    ]

    # Compute the area under the curve for each env and each algo
    final_qdrl_frame = pd.DataFrame()
    for env in qdrl_frame["env"].drop_duplicates().values:
        env_qdrl_frame = qdrl_frame[qdrl_frame["env"] == env]
        for algo in env_qdrl_frame["algo"].drop_duplicates().values:
            algo_qdrl_frame = env_qdrl_frame[env_qdrl_frame["algo"] == algo]
            for rep in algo_qdrl_frame["rep"].drop_duplicates().values:
                rep_qdrl_frame = algo_qdrl_frame[algo_qdrl_frame["rep"] == rep]
                area = pd.DataFrame.from_dict(
                    {
                        "env": [env],
                        "algo": [algo],
                        "rep": [rep],
                        "batch_size": [rep_qdrl_frame[f"batch_size"].values[0]],
                        "sampling_size": [rep_qdrl_frame[f"sampling_size"].values[0]],
                        "area": [
                            np.trapz(
                                rep_qdrl_frame[f"{prefixe}reeval_qd_score"].values,
                                x=rep_qdrl_frame[f"timestep"].values,
                            )
                        ],
                    }
                )
                final_qdrl_frame = pd.concat(
                    [final_qdrl_frame, area], ignore_index=True
                )

    # Build up the plot
    rows_columns_select = [
        [["env", env_name] for env_name in env_lists],
    ]
    rows_columns_title = [
        [env_order[env_name] for env_name in env_lists],
    ]

    # First, plot QD-Score area
    rows_columns_metrics = [
        [f"area" for env_name in env_lists],
    ]
    rows_columns_metrics_name = [
        [f"Area under Corrected QD-Score"] + ["" for _ in range(len(env_lists) - 1)],
    ]
    sub_plot_qdrl_families(
        sub_file_name=f"{plot_folder}/{prefixe}qdrl_families_area",
        qdrl_frame=final_qdrl_frame,
        color_frame=color_frame,
        compare_size=compare_size,
        compare_title=compare_title,
        legend_columns=legend_columns,
        legend_bottom=legend_bottom,
        rows_columns_select=rows_columns_select,
        rows_columns_title=rows_columns_title,
        rows_columns_metrics=rows_columns_metrics,
        rows_columns_metrics_name=rows_columns_metrics_name,
        filllines=[],
        lines=lines,
        interval_plot=interval_plot,
        errors=errors,
    )

    # Second, compute p-value if asked
    if p_values:

        # Reeval QD-Score p-values
        p_values_frame = final_qdrl_frame[final_qdrl_frame["env"].isin(env_lists)]
        p_values_frame = sort_data(
            p_values_frame,
            ["algo", "rep", "eval"],
            baselines_list
            + [name for other_list in other_lists for name in other_list],
        )
        try:
            p_values_fn(
                plot_folder=plot_folder,
                compare_size=compare_size,
                stat=f"area",
                dataframe=p_values_frame,
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot p-values for area.")
            if errors:
                traceback.print_exc()


def plot_qdrl_families_reprod(
    plot_folder: str,
    all_reprods: pd.DataFrame,
    env_order: Dict,
    qdrl_families: Dict,
    color_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    legend_columns: int,
    legend_bottom: float,
    p_values: bool,
    prefixe: str = "",
    prefixe_title: str = "",
    interval_plot: bool = True,
    errors: bool = False,
) -> None:

    # Extract algos that are in multiple categories as baselines
    name_counts = Counter([name for names in qdrl_families.values() for name in names])
    baselines_list = [name for name, count in name_counts.items() if count > 1]
    all_names_string = "|".join(baselines_list)
    other_lists = []
    for family_key in qdrl_families.keys():
        qdrl_family = [
            name for name in qdrl_families[family_key] if name not in baselines_list
        ]
        other_lists.append(qdrl_family)
        all_names_string = "|".join([all_names_string, "|".join(qdrl_family)])

    # Keep only the names that are in the list
    qdrl_frame = all_reprods[all_reprods["algo"].str.contains(all_names_string)]
    qdrl_frame = sort_data(
        qdrl_frame,
        ["algo", "rep"],
        baselines_list + [name for other_list in other_lists for name in other_list],
    )

    # Extract the list of environments
    env_lists = qdrl_frame["env"].drop_duplicates().values

    # Build the lines
    algos = qdrl_frame["algo"].drop_duplicates().values
    algo = 0
    for compare_algo in baselines_list:
        while algo < len(algos) and compare_algo in algos[algo]:
            algo += 1
    single_lines = [
        interval_line(algos, algo, "")
        if interval_plot
        else boxplot_line(algos, algo, "")
    ]
    for categories_list in other_lists:
        for compare_algo in categories_list:
            while algo < len(algos) and compare_algo in algos[algo]:
                algo += 1
        single_lines.append(
            interval_line(algos, algo, (5, 5))
            if interval_plot
            else boxplot_line(algos, algo, (5, 5))
        )

    lines = [
        [single_lines for env_name in env_lists],
    ]

    # Build up the plot
    rows_columns_select = [
        [["env", env_name] for env_name in env_lists],
    ]
    rows_columns_title = [
        [env_order[env_name] for env_name in env_lists],
    ]

    # First, plot Reproducibility-Score
    rows_columns_metrics = [
        [f"{prefixe}desc_reproducibilities_qd_score" for env_name in env_lists],
    ]
    rows_columns_metrics_name = [
        [f"{prefixe_title}Reproducibility-Score"]
        + ["" for _ in range(len(env_lists) - 1)],
    ]
    sub_plot_qdrl_families(
        sub_file_name=f"{plot_folder}/{prefixe}qdrl_families_reprod",
        qdrl_frame=qdrl_frame,
        color_frame=color_frame,
        compare_size=compare_size,
        compare_title=compare_title,
        legend_columns=legend_columns,
        legend_bottom=legend_bottom,
        rows_columns_select=rows_columns_select,
        rows_columns_title=rows_columns_title,
        rows_columns_metrics=rows_columns_metrics,
        rows_columns_metrics_name=rows_columns_metrics_name,
        filllines=[],
        lines=lines,
        interval_plot=interval_plot,
        errors=errors,
    )

    # Second, compute p-value if asked
    if p_values:

        # Reeval QD-Score p-values
        p_values_frame = qdrl_frame[qdrl_frame["env"].isin(env_lists)]
        p_values_frame = sort_data(
            p_values_frame,
            ["algo", "rep", "eval"],
            baselines_list
            + [name for other_list in other_lists for name in other_list],
        )
        try:
            p_values_fn(
                plot_folder=plot_folder,
                compare_size=compare_size,
                stat=f"{prefixe}desc_reproducibilities_qd_score",
                dataframe=p_values_frame,
            )
        except Exception:
            print(
                f"\n!!!WARNING!!! Cannot plot p-values for {prefixe}desc_reproducibilities_qd_score."
            )
            if errors:
                traceback.print_exc()


def sub_plot_qdrl_families(
    sub_file_name: str,
    qdrl_frame: pd.DataFrame,
    color_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    legend_columns: int,
    legend_bottom: float,
    rows_columns_select: List,
    rows_columns_title: List,
    rows_columns_metrics: List,
    rows_columns_metrics_name: List,
    filllines: List,
    lines: List,
    interval_plot: bool = True,
    errors: bool = False,
) -> None:

    # Plot metrics for all sizes
    try:
        plot_rows_columns_select(
            file_name=sub_file_name + f".svg",
            data_frame=qdrl_frame,
            rows_columns_select=rows_columns_select,
            rows_columns_title=rows_columns_title,
            rows_columns_metrics=rows_columns_metrics,
            rows_columns_metrics_name=rows_columns_metrics_name,
            vlines=[] if interval_plot else lines,
            hlines=lines if interval_plot else [],
            filllines=filllines,
            x_front=[],
            y_front=[],
            box_plot=not (interval_plot),
            scatter_plot=False,
            interval_plot=interval_plot,
            intervalbox_plot=False,
            x=compare_size,
            xlabel=compare_title,
            major_locator=[],
            major_locator_name=[],
            size=None,
            sizes=[],
            hue="algo",
            color_frame=color_frame,
            legend_columns=legend_columns,
            legend_bottom=legend_bottom,
            markers=True,
        )
    except Exception:
        print(f"\n!!!WARNING!!! Cannot plot qdrl_families plot in {sub_file_name}.svg.")
        if errors:
            traceback.print_exc()

    # Plot metrics for each size separatly
    try:
        if len(qdrl_frame[compare_size].drop_duplicates().values) > 1:
            size = 0
            for size in qdrl_frame[compare_size].drop_duplicates().values:
                size_qdrl_frame = qdrl_frame[qdrl_frame[compare_size] == size]
                plot_rows_columns_select(
                    file_name=sub_file_name + f"_{size}.svg",
                    data_frame=size_qdrl_frame,
                    rows_columns_select=rows_columns_select,
                    rows_columns_title=rows_columns_title,
                    rows_columns_metrics=rows_columns_metrics,
                    rows_columns_metrics_name=rows_columns_metrics_name,
                    vlines=[] if interval_plot else lines,
                    hlines=lines if interval_plot else [],
                    filllines=filllines,
                    x_front=[],
                    y_front=[],
                    box_plot=not (interval_plot),
                    scatter_plot=False,
                    interval_plot=False,
                    intervalbox_plot=interval_plot,
                    x=compare_size,
                    xlabel=None,
                    major_locator=[],
                    major_locator_name=[],
                    size=None,
                    sizes=[],
                    hue="algo",
                    color_frame=color_frame,
                    legend_columns=legend_columns,
                    legend_bottom=legend_bottom,
                    markers=True,
                    remove_xaxis=True,
                )
    except Exception:
        print(
            f"\n!!!WARNING!!! Cannot plot qdrl_families plot for size {size} in {sub_file_name}_{size}.svg."
        )
        if errors:
            traceback.print_exc()


def plot_qdrl_families_chaining(
    plot_folder: str,
    all_chaining_data: pd.DataFrame,
    env_order: Dict,
    qdrl_families: Dict,
    color_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    legend_columns: int,
    legend_bottom: float,
    p_values: bool,
    interval_plot: bool = True,
    errors: bool = False,
) -> None:

    print("\n\nChaining")

    # Extract algos that are in multiple categories as baselines
    name_counts = Counter([name for names in qdrl_families.values() for name in names])
    baselines_list = [name for name, count in name_counts.items() if count > 1]
    all_names_string = "|".join(baselines_list)
    other_lists = []
    for family_key in qdrl_families.keys():
        qdrl_family = [
            name for name in qdrl_families[family_key] if name not in baselines_list
        ]
        other_lists.append(qdrl_family)
        all_names_string = "|".join([all_names_string, "|".join(qdrl_family)])

    # Keep only the names that are in the list
    qdrl_frame = all_chaining_data[
        all_chaining_data["algo"].str.contains(all_names_string)
    ]
    qdrl_frame = sort_data(
        qdrl_frame,
        ["algo", "rep"],
        baselines_list + [name for other_list in other_lists for name in other_list],
    )

    # Extract the list of environments
    env_lists = qdrl_frame["env"].drop_duplicates().values

    # Build the lines
    algos = qdrl_frame["algo"].drop_duplicates().values
    algo = 0
    for compare_algo in baselines_list:
        while algo < len(algos) and compare_algo in algos[algo]:
            algo += 1
    single_lines = [
        interval_line(algos, algo, "")
        if interval_plot
        else boxplot_line(algos, algo, "")
    ]
    for categories_list in other_lists:
        for compare_algo in categories_list:
            while algo < len(algos) and compare_algo in algos[algo]:
                algo += 1
        single_lines.append(
            interval_line(algos, algo, (5, 5))
            if interval_plot
            else boxplot_line(algos, algo, (5, 5))
        )

    lines = [
        [single_lines for env_name in env_lists],
        # [single_lines for env_name in env_lists],
    ]

    # Build up the plot
    rows_columns_select = [
        [["env", env_name] for env_name in env_lists],
        # [["env", env_name] for env_name in env_lists],
    ]
    rows_columns_title = [
        [env_order[env_name] for env_name in env_lists],
        # [env_order[env_name] for env_name in env_lists],
    ]

    # First, plot Reproducibility-Score
    rows_columns_metrics = [
        ["qd_score" for env_name in env_lists],
        # [f"coverage" for env_name in env_lists],
    ]
    rows_columns_metrics_name = [
        ["Corrected-QD-Score"] + ["" for _ in range(len(env_lists) - 1)],
        # [f"Chained Coverage"]
        # + ["" for _ in range(len(env_lists) - 1)],
    ]
    sub_plot_qdrl_families(
        sub_file_name=f"{plot_folder}/qdrl_families_chaining",
        qdrl_frame=qdrl_frame,
        color_frame=color_frame,
        compare_size=compare_size,
        compare_title=compare_title,
        legend_columns=legend_columns,
        legend_bottom=legend_bottom,
        rows_columns_select=rows_columns_select,
        rows_columns_title=rows_columns_title,
        rows_columns_metrics=rows_columns_metrics,
        rows_columns_metrics_name=rows_columns_metrics_name,
        filllines=[],
        lines=lines,
        interval_plot=interval_plot,
        errors=errors,
    )

    # Second, compute p-value if asked
    if p_values:

        # Reeval QD-Score p-values
        try:
            p_values_fn(
                plot_folder=plot_folder,
                compare_size=compare_size,
                stat="qd_score",
                dataframe=qdrl_frame,
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot p-values for Chaining QD-Score.")
            if errors:
                traceback.print_exc()
    print("\n\n")


def plot_qdrl_families_partial_evaluation(
    plot_folder: str,
    config_frame: pd.DataFrame,
    x_column: str,
    x_name: str,
    compare_size: str,
    compare_title: str,
    order: List,
    env_order: Dict,
    color_frame: pd.DataFrame,
    legend_columns: int,
    legend_bottom: float,
    use_max_xaxis: bool,
    env_max_xaxis: Dict,
    p_values: bool,
    errors: bool = False,
) -> None:

    # Create the metrics dataframe
    finals_data = pd.DataFrame()
    rep = 0
    for line in range(config_frame.shape[0]):

        # Get the config for this line
        name = config_frame["name"][line]
        algo = config_frame["algo"][line]
        algo_batch = config_frame["algo_batch"][line]
        env = config_frame["env"][line]
        num_centroids = config_frame["num_centroids"][line]
        num_reevals = config_frame["num_reevals"][line]  # 0
        size = config_frame[compare_size][line]
        batch_size = config_frame["batch_size"][line]
        sampling_size = config_frame["sampling_size"][line]

        try:
            # Get the partial eval metrics file
            folder = config_frame["folder"][line]
            metrics_file = config_frame["metrics_file"][line]
            partial_eval_metrics_file = (
                "partial_eval_metrics_"
                + metrics_file[metrics_file.find("/metrics_") + 9 :]
            )
            partial_eval_metrics_file = os.path.join(folder, partial_eval_metrics_file)

            # Check if file exit
            if not os.path.isfile(partial_eval_metrics_file):
                continue

            # Read metrics
            data = pd.read_csv(partial_eval_metrics_file, index_col=False)

            # Read final metrics
            finals: Dict[str, List[float]] = {}
            for column in data.columns:
                # Add all final values
                finals[column] = data[column].mean()

            # Add config
            finals["name"] = [name]
            finals["algo"] = algo
            finals["algo_batch"] = algo_batch
            finals["num_centroids"] = num_centroids
            finals["env"] = env
            finals["num_reevals"] = num_reevals
            finals[compare_size] = size
            finals["batch_size"] = batch_size
            finals["sampling_size"] = sampling_size
            finals["rep"] = [float(rep)]

            # Concatenate all frames to existing ones
            finals = pd.DataFrame.from_dict(finals)
            finals_data = pd.concat([finals_data, finals], ignore_index=True)

            # Increment rep counter
            rep += 1

        except Exception:
            print("!!!WARNING!!! Cannot read", partial_eval_metrics_file, ".")
            if errors:
                traceback.print_exc()

            # try:

            #    # Reading value from parameters
            #    num_samples = extract_config_frame["num_samples"][line]

            #    # If the value is 1, just do not consider it
            #    if num_samples == 1:
            #        continue

            #    # Creating the frame
            #    finals: Dict[str, List[float]] = {}

            #    finals["average_length"] = episode_length
            #    finals["average_length_archive"] = episode_length
            #    finals["min_length_archive"] = episode_length
            #    finals["max_length_archive"] = episode_length
            #    finals["extract_average_length"] = episode_length
            #    finals["extract_max_length"] = episode_length
            #    finals["extract_min_length"] = episode_length
            #    finals["emit_average_length"] = episode_length
            #    finals["emit_max_length"] = episode_length
            #    finals["emit_min_length"] = episode_length

            #    finals["name"] = [name]
            #    finals["algo"] = algo
            #    finals["algo_batch"] = algo_batch
            #    finals["num_centroids"] = num_centroids
            #    finals["env"] = env
            #    finals["num_reevals"] = num_reevals
            #    finals[compare_size] = size
            #    finals["batch_size"] = batch_size
            #    finals["sampling_size"] = sampling_size
            #    finals["rep"] = [float(rep)]

            #    # Concatenate all frames to existing ones
            #    finals = pd.DataFrame.from_dict(finals)
            #    finals_data = pd.concat([finals_data, finals], ignore_index=True)

            #    # Increment rep counter
            #    rep += 1

            # except Exception:
            #    print(
            #        "!!!WARNING!!! Cannot use default sampling value from parameters either."
            #    )
            #    if errors:
            #        traceback.print_exc()

    if finals_data.empty:
        print("\n!!!WARNING!!! No Extract Archive Sampling to plot.")
        return

    # Sort datas
    finals_data = sort_data(finals_data, ["env", "algo", "rep"], order)

    all_envs = []
    for env_name in env_order:
        if env_name in finals_data["env"].drop_duplicates().values:
            all_envs.append(env_name)
    for env_name in finals_data["env"].drop_duplicates().values:
        if env_name not in all_envs:
            all_envs.append(env_name)

    # Build up the plot
    rows_columns_select = [
        [["env", env_name] for env_name in all_envs],
        # [["env", env_name] for env_name in all_envs],
    ]
    rows_columns_title = [
        [env_order[env_name] for env_name in all_envs],
        # ["" for env_name in all_envs],
    ]

    # First, plot top layer infos across time
    rows_columns_metrics = [
        ["average_length" for _ in all_envs],
        # ["extract_length" for _ in all_envs],
    ]
    rows_columns_metrics_name = [
        ["Average Env Steps"] + ["" for _ in range(len(all_envs) - 1)],
        # ["Average Env Steps"] + ["" for _ in range(len(all_envs) - 1)],
    ]

    vlines = [
        [[[1000, "k", (5, 5)]] for _ in range(len(all_envs))],
    ]

    # Second, plot top layer infos
    try:
        plot_rows_columns_select(
            file_name=f"{plot_folder}/qdrl_families_partial_evaluation.svg",
            data_frame=finals_data,
            rows_columns_select=rows_columns_select,
            rows_columns_title=rows_columns_title,
            rows_columns_metrics=rows_columns_metrics,
            rows_columns_metrics_name=rows_columns_metrics_name,
            vlines=vlines,
            hlines=[],
            filllines=[],
            x_front=[],
            y_front=[],
            box_plot=False,
            scatter_plot=False,
            interval_plot=True,
            intervalbox_plot=False,
            x=x_column,
            xlabel=x_name,
            major_locator=[],
            major_locator_name=[],
            size=None,
            sizes=[],
            hue="algo",
            color_frame=color_frame,
            legend_columns=legend_columns,
            legend_bottom=legend_bottom,
            markers=True,
        )
    except Exception:
        print(f"\n!!!WARNING!!! Cannot plot qdrl_families_partial_evaluation metrics.")
        if errors:
            traceback.print_exc()

    # Third, compute p-value if asked
    if p_values:

        try:
            p_values_fn(
                plot_folder=plot_folder,
                compare_size=compare_size,
                stat=f"average_length",
                dataframe=finals_data,
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot p-values.")
            if errors:
                traceback.print_exc()
