import os
import traceback
from copy import deepcopy
from random import randint
from typing import Any, Dict, List

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

from analysis.load_archives import get_folder_name
from analysis.load_p_values import p_values as p_values_fn
from analysis.load_results import sort_data
from analysis.utils_archive import plot_one_paper_archive
from analysis.utils_plot import plot_rows_columns_select

SIZE_ESTIMATION = 4096
ENV_ESTIMATION = [
    # "arm_gaussian_fit_nonoise",
    # "arm_gaussian_fit_fit0.1_desc0_params0",
    # "arm_gaussian_desc_fit0_desc0.05_params0",
    # "arm_fit0.01_desc0.01_params0.0",
    # "arm_gaussian_desc_fit0_desc0.01_params0",
    "hexapod_sin_omni_fit0.05_desc0.05_params0.0",
    "walker2d_uni",
    "ant_omni",
]
NAMES = {
    "arm_gaussian_fit_nonoise": "Arm No Noise",
    "arm_gaussian_fit_fit0.1_desc0_params0": "Arm Fitness Noise",
    "arm_gaussian_desc_fit0_desc0.05_params0": "Arm Desc Noise",
    "arm_gaussian_desc_fit0_desc0.01_params0": "Arm",
    "arm_fit0.01_desc0.01_params0.0": "Arm",
    "hexapod_sin_omni_fit0.05_desc0.05_params0.0": "Hexapod",
    "walker2d_uni": "Walker",
    "ant_omni": "Ant",
}

SIZE_QDRL = 128
ENV_QDRL = [
    "ant_uni",
    "walker2d_uni",
    "halfcheetah_uni",
]
NAMES_QDRL = {
    "ant_uni": "Ant",
    "walker2d_uni": "Walker",
    "halfcheetah_uni": "HalfCheetah",
}


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


def plot_extract(
    plot_folder: str,
    all_finals: pd.DataFrame,
    baselines_list: List,
    categories_list: List,
    baselines_as_list: List,
    categories_as_list: List,
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

    contains_all_envs = True
    for env_name in ENV_ESTIMATION:
        contains_all_envs = contains_all_envs and env_name in all_finals["env"].values
    contains_all_envs = (
        contains_all_envs and SIZE_ESTIMATION in all_finals[compare_size].values
    )

    if contains_all_envs:

        extract_frame = deepcopy(all_finals)

        # Keep only those environments
        extract_frame = extract_frame[extract_frame["env"].isin(ENV_ESTIMATION)]
        extract_frame = extract_frame[extract_frame[compare_size] == SIZE_ESTIMATION]
        extract_frame = sort_data(
            extract_frame,
            ["algo", "rep", "eval"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Build the lines
        # algos = extract_frame["algo"].drop_duplicates().values
        # algo = 0
        # for compare_algo in baselines_list:
        #    while algo < len(algos) and compare_algo in algos[algo]:
        #        algo += 1
        # single_lines = [
        #    interval_line(algos, algo, (5, 5))
        #    if interval_plot
        #    else boxplot_line(algos, algo, (5, 5))
        # ]
        # for compare_algo in categories_list:
        #    while algo < len(algos) and compare_algo in algos[algo]:
        #        algo += 1
        # single_lines.append(
        #    interval_line(algos, algo, "")
        #    if interval_plot
        #    else boxplot_line(algos, algo, "")
        # )
        # for compare_algo in baselines_as_list:
        #    while algo < len(algos) and compare_algo in algos[algo]:
        #        algo += 1
        # single_lines.append(
        #    interval_line(algos, algo, (5, 5))
        #    if interval_plot
        #    else boxplot_line(algos, algo, (5, 5))
        # )

        # lines = [
        #    [single_lines for env_name in ENV_ESTIMATION],
        #    [single_lines for env_name in ENV_ESTIMATION],
        # ]
        lines: List = []

        # Build up the plot
        rows_columns_select = [
            [["env", env_name] for env_name in ENV_ESTIMATION],
            [["env", env_name] for env_name in ENV_ESTIMATION],
        ]
        rows_columns_title = [
            [NAMES[env_name] for env_name in ENV_ESTIMATION],
            ["" for env_name in ENV_ESTIMATION],
        ]

        # First, plot QD-Score and Coverage
        rows_columns_metrics = [
            [f"{prefixe}reeval_qd_score" for env_name in ENV_ESTIMATION],
            [f"{prefixe}reeval_coverage" for env_name in ENV_ESTIMATION],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}Corrected-QD-Score"]
            + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
            [f"{prefixe_title}Corrected-Coverage"]
            + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
        ]
        sub_plot_extract(
            sub_file_name=f"{plot_folder}/{prefixe}extract",
            extract_frame=extract_frame,
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

        rows_columns_metrics = [
            [f"{prefixe}qd_score" for env_name in ENV_ESTIMATION],
            [f"{prefixe}coverage" for env_name in ENV_ESTIMATION],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}Illusory-QD-Score"]
            + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
            [f"{prefixe_title}Illusory-Coverage"]
            + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
        ]
        sub_plot_extract(
            sub_file_name=f"{plot_folder}/{prefixe}illusory_extract",
            extract_frame=extract_frame,
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

        # Second, plot QD-Score-Loss and Coverage-Loss
        rows_columns_metrics = [
            [f"loss_{prefixe}qd_score" for env_name in ENV_ESTIMATION],
            [f"loss_{prefixe}coverage" for env_name in ENV_ESTIMATION],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}QD-Score-Loss (%)"]
            + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
            [f"{prefixe_title}Coverage-Loss (%)"]
            + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
        ]
        sub_plot_extract(
            sub_file_name=f"{plot_folder}/{prefixe}extract_loss",
            extract_frame=extract_frame,
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

        # Third, compute p-value if asked
        if p_values:

            # Reeval QD-Score p-values
            try:
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"{prefixe}reeval_qd_score",
                    dataframe=extract_frame,
                )
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"{prefixe}reeval_coverage",
                    dataframe=extract_frame,
                )
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"loss_{prefixe}qd_score",
                    dataframe=extract_frame,
                )
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"loss_{prefixe}coverage",
                    dataframe=extract_frame,
                )
            except Exception:
                print(f"\n!!!WARNING!!! Cannot plot p-values.")
                if errors:
                    traceback.print_exc()

    else:
        print("!!!WARNING!!! Missing tasks to plot Extract Estimation.")

    contains_all_envs = True
    for env_name in ENV_QDRL:
        contains_all_envs = contains_all_envs and env_name in all_finals["env"].values
    contains_all_envs = (
        contains_all_envs and SIZE_QDRL in all_finals[compare_size].values
    )

    if contains_all_envs:

        extract_frame = deepcopy(all_finals)

        # Keep only those environments
        extract_frame = extract_frame[extract_frame["env"].isin(ENV_QDRL)]
        extract_frame = extract_frame[extract_frame[compare_size] == SIZE_QDRL]
        extract_frame = sort_data(
            extract_frame,
            ["algo", "rep", "eval"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Build the lines
        # algos = extract_frame["algo"].drop_duplicates().values
        # algo = 0
        # for compare_algo in baselines_list:
        #    while algo < len(algos) and compare_algo in algos[algo]:
        #        algo += 1
        # single_lines = [
        #    interval_line(algos, algo, (5, 5))
        #    if interval_plot
        #    else boxplot_line(algos, algo, (5, 5))
        # ]
        # for compare_algo in categories_list:
        #    while algo < len(algos) and compare_algo in algos[algo]:
        #        algo += 1
        # single_lines.append(
        #    interval_line(algos, algo, "")
        #    if interval_plot
        #    else boxplot_line(algos, algo, "")
        # )
        # for compare_algo in baselines_as_list:
        #    while algo < len(algos) and compare_algo in algos[algo]:
        #        algo += 1
        # single_lines.append(
        #    interval_line(algos, algo, (5, 5))
        #    if interval_plot
        #    else boxplot_line(algos, algo, (5, 5))
        # )

        # lines = [
        #    [single_lines for env_name in ENV_QDRL],
        #    [single_lines for env_name in ENV_QDRL],
        # ]
        lines = []

        # Build up the plot
        rows_columns_select = [
            [["env", env_name] for env_name in ENV_QDRL],
            [["env", env_name] for env_name in ENV_QDRL],
        ]
        rows_columns_title = [
            [NAMES_QDRL[env_name] for env_name in ENV_QDRL],
            ["" for env_name in ENV_QDRL],
        ]

        # First, plot QD-Score and Coverage
        rows_columns_metrics = [
            [f"{prefixe}reeval_qd_score" for env_name in ENV_QDRL],
            [f"{prefixe}reeval_coverage" for env_name in ENV_QDRL],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}Corrected-QD-Score"]
            + ["" for _ in range(len(ENV_QDRL) - 1)],
            [f"{prefixe_title}Corrected-Coverage"]
            + ["" for _ in range(len(ENV_QDRL) - 1)],
        ]
        sub_plot_extract(
            sub_file_name=f"{plot_folder}/{prefixe}extract_qdrl",
            extract_frame=extract_frame,
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

        # Second, plot QD-Score-Loss and Coverage-Loss
        rows_columns_metrics = [
            [f"loss_{prefixe}qd_score" for env_name in ENV_QDRL],
            [f"loss_{prefixe}coverage" for env_name in ENV_QDRL],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}QD-Score-Loss (%)"]
            + ["" for _ in range(len(ENV_QDRL) - 1)],
            [f"{prefixe_title}Coverage-Loss (%)"]
            + ["" for _ in range(len(ENV_QDRL) - 1)],
        ]
        sub_plot_extract(
            sub_file_name=f"{plot_folder}/{prefixe}extract_qdrl_loss",
            extract_frame=extract_frame,
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

        # Third, compute p-value if asked
        if p_values:

            # Reeval QD-Score p-values
            try:
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"{prefixe}reeval_qd_score",
                    dataframe=extract_frame,
                )
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"{prefixe}reeval_coverage",
                    dataframe=extract_frame,
                )
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"loss_{prefixe}qd_score",
                    dataframe=extract_frame,
                )
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"loss_{prefixe}coverage",
                    dataframe=extract_frame,
                )
            except Exception:
                print(f"\n!!!WARNING!!! Cannot plot p-values.")
                if errors:
                    traceback.print_exc()

    else:
        print("!!!WARNING!!! Missing tasks to plot Extract QDRL.")


def plot_extract_time(
    plot_folder: str,
    all_times: pd.DataFrame,
    baselines_list: List,
    categories_list: List,
    baselines_as_list: List,
    categories_as_list: List,
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

    contains_all_envs = True
    for env_name in ENV_ESTIMATION:
        contains_all_envs = contains_all_envs and env_name in all_times["env"].values
    contains_all_envs = (
        contains_all_envs and SIZE_ESTIMATION in all_times[compare_size].values
    )

    if contains_all_envs:

        extract_frame = deepcopy(all_times)

        # Keep only those environments
        extract_frame = extract_frame[extract_frame["env"].isin(ENV_ESTIMATION)]
        extract_frame = extract_frame[extract_frame[compare_size] == SIZE_ESTIMATION]
        extract_frame = sort_data(
            extract_frame,
            ["algo", "rep", "eval"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Change time in minutes
        extract_frame[f"{prefixe}time"] = extract_frame[f"{prefixe}time"].div(60 * 60)

        # Build up the plot
        rows_columns_select = [
            [["env", env_name] for env_name in ENV_ESTIMATION],
        ]
        rows_columns_title = [
            [NAMES[env_name] for env_name in ENV_ESTIMATION],
        ]

        # First, plot Time
        rows_columns_metrics = [
            [f"{prefixe}time" for env_name in ENV_ESTIMATION],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}Run Time (hours)"]
            + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
        ]
        sub_plot_extract(
            sub_file_name=f"{plot_folder}/{prefixe}extract_time",
            extract_frame=extract_frame,
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
            lines=[],
            interval_plot=interval_plot,
            errors=errors,
        )
        rows_columns_metrics_name = [
            [f"{prefixe_title}Time to Convergence (hours)"]
            + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
        ]
        sub_plot_extract(
            sub_file_name=f"{plot_folder}/{prefixe}extract_time_convergence",
            extract_frame=extract_frame,
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
            lines=[],
            interval_plot=interval_plot,
            errors=errors,
        )

        # Second, plot Time removing Adapt-ME
        no_adaptme_extract_frame = extract_frame[extract_frame["algo"] != "Adapt-ME"]
        sub_plot_extract(
            sub_file_name=f"{plot_folder}/{prefixe}extract_time_noadaptme",
            extract_frame=no_adaptme_extract_frame,
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
            lines=[],
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
                    stat=f"{prefixe}time",
                    dataframe=extract_frame,
                )
            except Exception:
                print(f"\n!!!WARNING!!! Cannot plot p-values.")
                if errors:
                    traceback.print_exc()

    else:
        print("!!!WARNING!!! Missing tasks to plot Extract Estimation.")

    contains_all_envs = True
    for env_name in ENV_QDRL:
        contains_all_envs = contains_all_envs and env_name in all_times["env"].values
    contains_all_envs = (
        contains_all_envs and SIZE_QDRL in all_times[compare_size].values
    )

    if contains_all_envs:

        extract_frame = deepcopy(all_times)

        # Keep only those environments
        extract_frame = extract_frame[extract_frame["env"].isin(ENV_QDRL)]
        extract_frame = extract_frame[extract_frame[compare_size] == SIZE_QDRL]
        extract_frame = sort_data(
            extract_frame,
            ["algo", "rep", "eval"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Change time in minutes
        extract_frame[f"{prefixe}time"] = extract_frame[f"{prefixe}time"].div(60 * 60)

        # Build up the plot
        rows_columns_select = [
            [["env", env_name] for env_name in ENV_ESTIMATION],
        ]
        rows_columns_title = [
            [NAMES[env_name] for env_name in ENV_ESTIMATION],
        ]

        # First, plot Time
        rows_columns_metrics = [
            [f"{prefixe}time" for env_name in ENV_ESTIMATION],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}Run Time (hours)"]
            + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
        ]
        sub_plot_extract(
            sub_file_name=f"{plot_folder}/{prefixe}extract_qdrl_time",
            extract_frame=extract_frame,
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
            lines=[],
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
                    stat=f"{prefixe}time",
                    dataframe=extract_frame,
                )
            except Exception:
                print(f"\n!!!WARNING!!! Cannot plot p-values.")
                if errors:
                    traceback.print_exc()

    else:
        print("!!!WARNING!!! Missing tasks to plot Extract QDRL.")


def sub_plot_extract(
    sub_file_name: str,
    extract_frame: pd.DataFrame,
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
            data_frame=extract_frame,
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
        print(f"\n!!!WARNING!!! Cannot plot extract plot.")
        if errors:
            traceback.print_exc()


def plot_extract_convergence(
    plot_folder: str,
    all_convergence: pd.DataFrame,
    baselines_list: List,
    categories_list: List,
    baselines_as_list: List,
    categories_as_list: List,
    x_column: str,
    x_name: str,
    color_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    legend_columns: int,
    legend_bottom: float,
    prefixe: str = "",
    prefixe_title: str = "",
    errors: bool = False,
) -> None:

    contains_all_envs = True
    for env_name in ENV_ESTIMATION:
        contains_all_envs = (
            contains_all_envs and env_name in all_convergence["env"].values
        )
    contains_all_envs = (
        contains_all_envs and SIZE_ESTIMATION in all_convergence[compare_size].values
    )

    if contains_all_envs:

        extract_frame = deepcopy(all_convergence)

        # Keep only those environments
        extract_frame = extract_frame[extract_frame["env"].isin(ENV_ESTIMATION)]
        extract_frame = extract_frame[extract_frame[compare_size] == SIZE_ESTIMATION]
        extract_frame = sort_data(
            extract_frame,
            ["algo", "rep", "eval"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Build up the plot
        rows_columns_select = [
            [["env", env_name] for env_name in ENV_ESTIMATION],
            [["env", env_name] for env_name in ENV_ESTIMATION],
        ]
        rows_columns_title = [
            [NAMES[env_name] for env_name in ENV_ESTIMATION],
            ["" for env_name in ENV_ESTIMATION],
        ]
        rows_columns_metrics = [
            [f"{prefixe}reeval_qd_score" for env_name in ENV_ESTIMATION],
            [f"{prefixe}reeval_coverage" for env_name in ENV_ESTIMATION],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}Corrected QD-Score"]
            + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
            [f"{prefixe_title}Corrected Coverage (%)"]
            + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
        ]

        # Plot
        try:
            plot_rows_columns_select(
                file_name=f"{plot_folder}/{prefixe}extract_convergence.svg",
                data_frame=extract_frame,
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
                interval_plot=False,
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
            print(f"\n!!!WARNING!!! Cannot plot extract convergence plots.")
            if errors:
                traceback.print_exc()

        # Do a second plot with both Illusory and Corrected on the same graph
        both_extract_frame = pd.concat(
            [
                extract_frame[
                    [
                        "algo",
                        f"{prefixe}qd_score",
                        f"{prefixe}coverage",
                        "env",
                        x_column,
                    ]
                ].rename(
                    columns={
                        f"{prefixe}qd_score": f"{prefixe}reeval_qd_score",
                        f"{prefixe}coverage": f"{prefixe}reeval_coverage",
                    }
                ),
                extract_frame[
                    [
                        "algo",
                        f"{prefixe}reeval_qd_score",
                        f"{prefixe}reeval_coverage",
                        "env",
                        x_column,
                    ]
                ].rename(
                    columns={
                        f"{prefixe}reeval_qd_score": f"{prefixe}reeval_qd_score",
                        f"{prefixe}reeval_coverage": f"{prefixe}reeval_coverage",
                    }
                ),
            ],
            ignore_index=True,
        )

        both_extract_frame.loc[: len(extract_frame) - 1, "algo"] += " - Illusory"
        both_extract_frame.loc[len(extract_frame) :, "algo"] += " - Corrected"

        # Create a matching color frame with the same color for both
        new_color_frame = []
        dashes = {}
        for _, row in color_frame.iterrows():
            label = row["Label"]
            color = row["Color"]
            new_color_frame.append({"Label": f"{label} - Illusory", "Color": color})
            new_color_frame.append({"Label": f"{label} - Corrected", "Color": color})
            dashes[f"{label} - Illusory"] = (2, 2)
            dashes[f"{label} - Corrected"] = ()
        new_color_frame = pd.DataFrame(new_color_frame)

        # Plot
        try:
            plot_rows_columns_select(
                file_name=f"{plot_folder}/{prefixe}extract_both_convergence.svg",
                data_frame=both_extract_frame,
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
                interval_plot=False,
                intervalbox_plot=False,
                x=x_column,
                xlabel=x_name,
                major_locator=[],
                major_locator_name=[],
                size=None,
                sizes=[],
                hue="algo",
                color_frame=new_color_frame,
                legend_columns=legend_columns,
                legend_bottom=legend_bottom,
                markers=False,
                dashes=dashes,
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot extract both convergence plots.")
            if errors:
                traceback.print_exc()
    else:
        print("!!!WARNING!!! Missing tasks to plot Extract.")

    contains_all_envs = True
    for env_name in ENV_QDRL:
        contains_all_envs = (
            contains_all_envs and env_name in all_convergence["env"].values
        )
    contains_all_envs = (
        contains_all_envs and SIZE_QDRL in all_convergence[compare_size].values
    )

    if contains_all_envs:

        extract_frame = deepcopy(all_convergence)

        # Keep only those environments
        extract_frame = extract_frame[extract_frame["env"].isin(ENV_QDRL)]
        extract_frame = extract_frame[extract_frame[compare_size] == SIZE_QDRL]
        extract_frame = sort_data(
            extract_frame,
            ["algo", "rep", "eval"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Build up the plot
        rows_columns_select = [
            [["env", env_name] for env_name in ENV_QDRL],
            [["env", env_name] for env_name in ENV_QDRL],
        ]
        rows_columns_title = [
            [NAMES_QDRL[env_name] for env_name in ENV_QDRL],
            ["" for env_name in ENV_QDRL],
        ]
        rows_columns_metrics = [
            [f"{prefixe}reeval_qd_score" for env_name in ENV_QDRL],
            [f"{prefixe}reeval_coverage" for env_name in ENV_QDRL],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}Corrected QD-Score"]
            + ["" for _ in range(len(ENV_QDRL) - 1)],
            [f"{prefixe_title}Corrected Coverage (%)"]
            + ["" for _ in range(len(ENV_QDRL) - 1)],
        ]

        # Plot
        try:
            plot_rows_columns_select(
                file_name=f"{plot_folder}/{prefixe}extract_convergence_qdrl.svg",
                data_frame=extract_frame,
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
                interval_plot=False,
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
            print(f"\n!!!WARNING!!! Cannot plot extract convergence plots.")
            if errors:
                traceback.print_exc()
    else:
        print("!!!WARNING!!! Missing tasks to plot Extract QDRL.")


def plot_extract_archives(
    plot_folder: str,
    single_compare_size: int,
    config_frame: pd.DataFrame,
    min_max_frame: pd.DataFrame,
    baselines_list: List,
    categories_list: List,
    baselines_as_list: List,
    categories_as_list: List,
    compare_size: str,
    compare_title: str,
    prefixe: str = "",
    prefixe_title: str = "",
    errors: bool = False,
) -> None:

    contains_all_envs = True
    for env_name in ENV_ESTIMATION:
        contains_all_envs = contains_all_envs and env_name in config_frame["env"].values

    if contains_all_envs:

        # Keep only those environments
        extract_config_frame = config_frame[config_frame["env"].isin(ENV_ESTIMATION)]
        extract_config_frame = sort_data(
            extract_config_frame,
            ["algo"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Reevaluated repertoire
        try:
            print(f"    Plotting Reeval Archive for {single_compare_size}.")
            sub_plot_archives(
                file_name=f"{plot_folder}/extract_{prefixe}reeval_archive_{single_compare_size}.png",
                single_compare_size=single_compare_size,
                config_frame=extract_config_frame,
                min_max_frame=min_max_frame,
                env_order=ENV_ESTIMATION,
                compare_size=compare_size,
                compare_title=compare_title,
                prefixe=prefixe,
                prefixe_title=prefixe_title,
                errors=errors,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
            if errors:
                traceback.print_exc()

        # Repertoire
        try:
            print(f"    Plotting Archive for {single_compare_size}.")
            sub_plot_archives(
                file_name=f"{plot_folder}/extract_{prefixe}archive_{single_compare_size}.png",
                single_compare_size=single_compare_size,
                config_frame=extract_config_frame,
                min_max_frame=min_max_frame,
                env_order=ENV_ESTIMATION,
                compare_size=compare_size,
                compare_title=compare_title,
                prefixe=prefixe,
                prefixe_title=prefixe_title,
                errors=errors,
                metric_name="",
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
            if errors:
                traceback.print_exc()

    else:
        print("!!!WARNING!!! Missing tasks to plot Extract archives.")

    contains_all_envs = True
    for env_name in ENV_QDRL:
        contains_all_envs = contains_all_envs and env_name in config_frame["env"].values

    if contains_all_envs:

        # Keep only those environments
        extract_config_frame = config_frame[config_frame["env"].isin(ENV_QDRL)]
        extract_config_frame = sort_data(
            extract_config_frame,
            ["algo"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Reevaluated repertoire
        try:
            print(f"    Plotting Reeval Archive for {single_compare_size}.")
            sub_plot_archives(
                file_name=f"{plot_folder}/extract_qdrl_{prefixe}reeval_archive_{single_compare_size}.png",
                single_compare_size=single_compare_size,
                config_frame=extract_config_frame,
                min_max_frame=min_max_frame,
                env_order=ENV_QDRL,
                compare_size=compare_size,
                compare_title=compare_title,
                prefixe=prefixe,
                prefixe_title=prefixe_title,
                errors=errors,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
            if errors:
                traceback.print_exc()

    else:
        print("!!!WARNING!!! Missing tasks to plot Extract QDRL archives.")


def sub_plot_archives(
    file_name: str,
    single_compare_size: int,
    config_frame: pd.DataFrame,
    min_max_frame: pd.DataFrame,
    env_order: List,
    compare_size: str,
    compare_title: str,
    prefixe: str = "",
    prefixe_title: str = "",
    errors: bool = False,
    metric_name="reeval_",
) -> None:

    # Get the algos
    algos = config_frame["algo"].drop_duplicates().values

    # Create the figure
    ncols = len(algos)
    nrows = len(env_order)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 8, nrows * 8))

    # For each environment
    for nrow in range(nrows):

        env = env_order[nrow]
        env_config_frame = config_frame[
            (config_frame["env"] == env)
            & (config_frame[compare_size] == single_compare_size)
        ].reset_index(drop=True)

        if env_config_frame.shape[0] == 0:
            print(
                f"\n!!!WARNING!!! No archives to plots for {env} and sampling-size {single_compare_size}.\n"
            )
            continue

        # Get all the corresponding min and max
        min_reeval_fitness = None
        max_reeval_fitness = None
        if min_max_frame is not None:
            env_min_max_frame = min_max_frame[min_max_frame["env"] == env]
            if not env_min_max_frame.empty:
                min_reeval_fitness = float(
                    env_min_max_frame[f"min_{metric_name}fitness"].values[0]
                )
                max_reeval_fitness = float(
                    env_min_max_frame[f"max_{metric_name}fitness"].values[0]
                )
        if "_" in env_config_frame["min_bd"][0]:
            min_bd_list = str(env_config_frame["min_bd"][0]).split("_")
        else:
            min_bd_list = str(env_config_frame["min_bd"][0])[1:-1].split(" ")
        if "" in min_bd_list:
            min_bd_list.remove("")
        if "_" in env_config_frame["max_bd"][0]:
            max_bd_list = str(env_config_frame["max_bd"][0]).split("_")
        else:
            max_bd_list = str(env_config_frame["max_bd"][0])[1:-1].split(" ")
        if "" in max_bd_list:
            max_bd_list.remove("")
        min_bd = [float(bd) for bd in min_bd_list]
        max_bd = [float(bd) for bd in max_bd_list]

        # For each algo for this env and size
        for ncol in range(ncols):

            algo = algos[ncol]
            algo_config_frame = env_config_frame[
                env_config_frame["algo"] == algo
            ].reset_index(drop=True)

            # Get the correpsonding axis
            if nrows == 1 and ncols == 1:
                ax = axes
            elif nrows == 1:
                ax = axes[ncol]
            elif ncols == 1:
                ax = axes[nrow]
            else:
                ax = axes[nrow, ncol]

            # Add colorbar to last row
            colorbar = False
            if ncol == ncols - 1:
                colorbar = True

            # Taking one randomly
            try:
                index = randint(0, max(1, algo_config_frame.shape[0] - 1))
                repertoire_folder = get_folder_name(
                    algo_config_frame, f"{prefixe}{metric_name}repertoire_folder", index
                )
                vmin = min_reeval_fitness  # type: ignore
                vmax = max_reeval_fitness  # type: ignore
                fitnesses = jnp.load(os.path.join(repertoire_folder, "fitnesses.npy"))
                descriptors = jnp.load(
                    os.path.join(repertoire_folder, "descriptors.npy")
                )
                centroids = jnp.load(os.path.join(repertoire_folder, "centroids.npy"))

                plot_one_paper_archive(
                    centroids=centroids,
                    descriptors=descriptors,
                    fitnesses=fitnesses,
                    ax=ax,
                    minval=min_bd,
                    maxval=max_bd,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar=colorbar,
                    colorbar_below=False,
                )

            except Exception:
                ax.tick_params(
                    left=False,
                    right=False,
                    labelleft=False,
                    labelbottom=False,
                    bottom=False,
                )
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)

                print(f"\n!!!WARNING!!! Cannot plot for {env} and {algo}.")
                if errors:
                    traceback.print_exc()

            # Add env name in first line
            if ncol == 0:
                title = env
                if env in NAMES.keys():
                    title = NAMES[env]
                elif env in NAMES_QDRL.keys():
                    title = NAMES_QDRL[env]
                ax.set_ylabel(title, fontsize=48)

            # Add algo in first column
            if nrow == 0:
                ax.set_title(algo, fontsize=48)

    # Finish figure
    plt.tight_layout(h_pad=1.70)
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()


def plot_extract_archive_sampling(
    plot_folder: str,
    config_frame: pd.DataFrame,
    x_column: str,
    x_name: str,
    compare_size: str,
    compare_title: str,
    order: List,
    color_frame: pd.DataFrame,
    legend_columns: int,
    legend_bottom: float,
    use_max_xaxis: bool,
    env_max_xaxis: Dict,
    p_values: bool,
    errors: bool = False,
) -> None:

    contains_all_envs = True
    for env_name in ENV_ESTIMATION:
        contains_all_envs = contains_all_envs and env_name in config_frame["env"].values
    contains_all_envs = (
        contains_all_envs and SIZE_ESTIMATION in config_frame[compare_size].values
    )

    if not contains_all_envs:
        print("!!!WARNING!!! Missing tasks to plot Extract Archive Sampling.")
        return

    extract_config_frame = deepcopy(config_frame)
    extract_config_frame = extract_config_frame[
        extract_config_frame["env"].isin(ENV_ESTIMATION)
    ]
    extract_config_frame = extract_config_frame[
        extract_config_frame[compare_size] == SIZE_ESTIMATION
    ]
    extract_config_frame = extract_config_frame.reset_index()

    if use_max_xaxis:
        print("\n\n!!!WARNING!!! Using max xaxis!")

    # Create the metrics dataframe
    all_data = pd.DataFrame()
    finals_data = pd.DataFrame()
    rep = 0
    for line in range(extract_config_frame.shape[0]):

        # Get the config for this line
        name = extract_config_frame["name"][line]
        algo = extract_config_frame["algo"][line]
        algo_batch = extract_config_frame["algo_batch"][line]
        env = extract_config_frame["env"][line]
        num_centroids = extract_config_frame["num_centroids"][line]
        num_reevals = extract_config_frame["num_reevals"][line]  # 0
        size = extract_config_frame[compare_size][line]
        batch_size = extract_config_frame["batch_size"][line]
        sampling_size = extract_config_frame["sampling_size"][line]

        try:
            # Get the partial eval metrics file
            folder = extract_config_frame["folder"][line]
            metrics_file = extract_config_frame["metrics_file"][line]
            archive_sampling_metrics_file = (
                "archive_sampling_metrics_"
                + metrics_file[metrics_file.find("/metrics_") + 9 :]
            )
            archive_sampling_metrics_file = os.path.join(
                folder, archive_sampling_metrics_file
            )

            # Read metrics
            data = pd.read_csv(archive_sampling_metrics_file, index_col=False)

            # Check if using maximum x-axis
            if use_max_xaxis:
                if env in env_max_xaxis.keys():
                    # Get from input dict
                    max_x = env_max_xaxis[env]
                    # print(f"\nFor {env}, from input dict: {max_x}.")
                else:
                    max_x = data["epoch"].max()
                    # print(f"\nNo max x axis in input dict for {env}, using all values.")

                data = data[data["epoch"] <= max_x]

            # Read final metrics
            max_eval = max(data["eval"])
            sub_data = data[data["eval"] == max_eval]
            finals: Dict[str, List[float]] = {}
            for column in sub_data.columns:
                # Add all final values
                finals[column] = sub_data[column].values[0]

            # Add config
            data["name"] = name
            data["algo"] = algo
            data["algo_batch"] = algo_batch
            data["num_centroids"] = num_centroids
            data["env"] = env
            data["num_reevals"] = num_reevals
            data[compare_size] = size
            data["batch_size"] = batch_size
            data["sampling_size"] = sampling_size
            data["rep"] = rep

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

            # Add timestep to dataframe
            if "timestep" not in data.columns:
                print("\n!!!WARNING!!! Timesteps not in metrics, infering it.")
                data["timestep"] = data["epoch"] * batch_size
                if "PartialEval" in name:
                    fixed_length_idx = name.find("-length")
                    fixed_length_end_idx = name.find("-smpl")
                    fixed_length = int(
                        name[fixed_length_idx + 7 : fixed_length_end_idx]
                    )
                    print(
                        f"Building timestep for PartialEval with fixed length {fixed_length}."
                    )
                    data["timestep"] = data["timestep"] * fixed_length
                else:
                    data["timestep"] = (
                        data["timestep"] * extract_config_frame["episode_length"][line]
                    )

            # Concatenate all frames to existing ones
            finals = pd.DataFrame.from_dict(finals)
            all_data = pd.concat([all_data, data], ignore_index=True)
            finals_data = pd.concat([finals_data, finals], ignore_index=True)

            # Increment rep counter
            rep += 1

        except Exception:

            try:

                # Reading value from parameters
                num_samples = extract_config_frame["num_samples"][line]

                # If the value is 1, just do not consider it
                if num_samples == 1:
                    continue

                # Creating the frame
                finals: Dict[str, List[float]] = {}
                data: Dict[str, List[float]] = {}

                data["top_average_samples"] = num_samples
                data["top_max_samples"] = num_samples
                data["overall_average_samples"] = num_samples
                data["overall_max_samples"] = num_samples

                finals["top_average_samples"] = num_samples
                finals["top_max_samples"] = num_samples
                finals["overall_average_samples"] = num_samples
                finals["overall_max_samples"] = num_samples

                data["name"] = [name]
                data["algo"] = algo
                data["algo_batch"] = algo_batch
                data["num_centroids"] = num_centroids
                data["env"] = env
                data["num_reevals"] = num_reevals
                data[compare_size] = size
                data["batch_size"] = batch_size
                data["sampling_size"] = sampling_size
                data["rep"] = [float(rep)]

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
                data = pd.DataFrame.from_dict(data)
                finals = pd.DataFrame.from_dict(finals)
                all_data = pd.concat([all_data, data], ignore_index=True)
                finals_data = pd.concat([finals_data, finals], ignore_index=True)

                # Increment rep counter
                rep += 1

            except Exception:
                print(
                    "!!!WARNING!!! Cannot use default sampling value from parameters either."
                )
                if errors:
                    traceback.print_exc()

    if all_data.empty:
        print("\n!!!WARNING!!! No Extract Archive Sampling to plot.")
        return

    # Sort datas
    all_data = sort_data(all_data, ["env", "algo", "rep", "eval"], order)
    finals_data = sort_data(finals_data, ["env", "algo", "rep"], order)

    # Build up the plot
    rows_columns_select = [
        [["env", env_name] for env_name in ENV_ESTIMATION],
        # [["env", env_name] for env_name in ENV_ESTIMATION],
    ]
    rows_columns_title = [
        [NAMES[env_name] for env_name in ENV_ESTIMATION],
        # ["" for env_name in ENV_ESTIMATION],
    ]

    # First, plot top layer infos across time
    rows_columns_metrics = [
        ["top_average_samples" for _ in ENV_ESTIMATION],
        # ["top_max_samples" for _ in ENV_ESTIMATION],
    ]
    rows_columns_metrics_name = [
        ["Average Samples"] + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
        # ["Max Samples"] + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
    ]
    try:
        plot_rows_columns_select(
            file_name=f"{plot_folder}/extract_archive_sampling_top_convergence.svg",
            data_frame=all_data,
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
            interval_plot=False,
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
        print(
            f"\n!!!WARNING!!! Cannot plot extract top archive_sampling convergence metrics."
        )
        if errors:
            traceback.print_exc()

    # Second, plot top layer infos
    try:
        plot_rows_columns_select(
            file_name=f"{plot_folder}/extract_archive_sampling_top.svg",
            data_frame=finals_data,
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
            interval_plot=False,
            intervalbox_plot=True,
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
        print(f"\n!!!WARNING!!! Cannot plot extract top archive_sampling metrics.")
        if errors:
            traceback.print_exc()

    # Third, compute p-value if asked
    if p_values:

        try:
            p_values_fn(
                plot_folder=plot_folder,
                compare_size=compare_size,
                stat=f"top_average_samples",
                dataframe=finals_data,
            )
            # p_values_fn(
            #     plot_folder=plot_folder,
            #     compare_size=compare_size,
            #     stat=f"top_max_samples",
            #     dataframe=finals_data,
            # )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot p-values.")
            if errors:
                traceback.print_exc()

    # Fourth, plot overall infos across time
    rows_columns_metrics = [
        ["overall_average_samples" for _ in ENV_ESTIMATION],
        # ["overall_max_samples" for _ in ENV_ESTIMATION],
    ]
    rows_columns_metrics_name = [
        ["Average Samples"] + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
        # ["Max Samples"] + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
    ]
    try:
        plot_rows_columns_select(
            file_name=f"{plot_folder}/extract_archive_sampling_overall_convergence.svg",
            data_frame=all_data,
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
            interval_plot=False,
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
        print(f"\n!!!WARNING!!! Cannot plot extract overall archive_sampling metrics.")
        if errors:
            traceback.print_exc()

    # Fifth, plot overall infos
    try:
        plot_rows_columns_select(
            file_name=f"{plot_folder}/extract_archive_sampling_overall.svg",
            data_frame=finals_data,
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
            interval_plot=False,
            intervalbox_plot=True,
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
        print(f"\n!!!WARNING!!! Cannot plot extract overall archive_sampling metrics.")
        if errors:
            traceback.print_exc()

    # Sixth, compute p-value if asked
    if p_values:

        try:
            p_values_fn(
                plot_folder=plot_folder,
                compare_size=compare_size,
                stat=f"overall_average_samples",
                dataframe=finals_data,
            )
            # p_values_fn(
            #     plot_folder=plot_folder,
            #     compare_size=compare_size,
            #     stat=f"overall_max_samples",
            #     dataframe=finals_data,
            # )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot p-values.")
            if errors:
                traceback.print_exc()
