import traceback
from copy import deepcopy
from typing import Any, List

import pandas as pd

from analysis.load_p_values import p_values as p_values_fn
from analysis.load_results import sort_data
from analysis.utils_plot import plot_rows_columns_select

ENV_ESTIMATION = [
    "arm_gaussian_fit_fit0.1_desc0_params0",
    "arm_multi_modal_fit_fit0.01_desc0.01_params0",
    "arm_gaussian_desc_fit0_desc0.01_params0",
    "arm_gaussian_desc_fit0_desc0.1_params0",
    "arm_multi_modal_desc_fit0.01_desc0.01_params0",
    "hexapod_sin_omni_fit0.05_desc0.05_params0.0",
    "ant_omni",
    "walker2d_uni",
]
NAMES = {
    "arm_gaussian_fit_fit0.1_desc0_params0": "Arm Gaussian Fit",
    "arm_multi_modal_fit_fit0.01_desc0.01_params0": "Arm Multi-Modal Fit",
    "arm_gaussian_desc_fit0_desc0.01_params0": "Arm Small Gaussian Desc",
    "arm_gaussian_desc_fit0_desc0.1_params0": "Arm Big Gaussian Desc",
    "arm_multi_modal_desc_fit0.01_desc0.01_params0": "Arm Multi-Modal Desc",
    "hexapod_sin_omni_fit0.05_desc0.05_params0.0": "Hexapod",
    "ant_omni": "Ant",
    "walker2d_uni": "Walker",
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


def plot_estimation_problem(
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

    if contains_all_envs:

        estimation_frame = deepcopy(all_finals)

        # Keep only those environments
        estimation_frame = estimation_frame[
            estimation_frame["env"].isin(ENV_ESTIMATION)
        ]
        estimation_frame = sort_data(
            estimation_frame,
            ["algo", "rep", "eval"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Build the lines
        algos = estimation_frame["algo"].drop_duplicates().values
        algo = 0
        for compare_algo in baselines_list:
            while algo < len(algos) and compare_algo in algos[algo]:
                algo += 1
        single_lines = [
            interval_line(algos, algo, (5, 5))
            if interval_plot
            else boxplot_line(algos, algo, (5, 5))
        ]
        for compare_algo in categories_list:
            while algo < len(algos) and compare_algo in algos[algo]:
                algo += 1
        single_lines.append(
            interval_line(algos, algo, "")
            if interval_plot
            else boxplot_line(algos, algo, "")
        )
        for compare_algo in baselines_as_list:
            while algo < len(algos) and compare_algo in algos[algo]:
                algo += 1
        single_lines.append(
            interval_line(algos, algo, (5, 5))
            if interval_plot
            else boxplot_line(algos, algo, (5, 5))
        )

        lines = [
            [single_lines for env_name in ENV_ESTIMATION],
            [single_lines for env_name in ENV_ESTIMATION],
        ]

        # Build up the plot
        rows_columns_select = [
            [["env", env_name] for env_name in ENV_ESTIMATION],
            [["env", env_name] for env_name in ENV_ESTIMATION],
        ]
        rows_columns_title = [
            [NAMES[env_name] for env_name in ENV_ESTIMATION],
            ["" for env_name in ENV_ESTIMATION],
        ]

        # First, plot QD-Score and
        rows_columns_metrics = [
            [f"{prefixe}reeval_qd_score" for env_name in ENV_ESTIMATION],
            [f"loss_{prefixe}qd_score" for env_name in ENV_ESTIMATION],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}Corrected QD-Score"]
            + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
            [f"{prefixe_title}Loss QD-Score"]
            + ["" for _ in range(len(ENV_ESTIMATION) - 1)],
        ]
        sub_plot_estimation_problem(
            sub_file_name=f"{plot_folder}/{prefixe}estimation",
            estimation_frame=estimation_frame,
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
                    stat=f"{prefixe}reeval_qd_score",
                    dataframe=estimation_frame,
                )
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"loss_{prefixe}qd_score",
                    dataframe=estimation_frame,
                )
            except Exception:
                print(f"\n!!!WARNING!!! Cannot plot p-values.")
                if errors:
                    traceback.print_exc()

    else:
        print("!!!WARNING!!! Missing tasks to plot Estimation.")


def sub_plot_estimation_problem(
    sub_file_name: str,
    estimation_frame: pd.DataFrame,
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
            data_frame=estimation_frame,
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
        print(f"\n!!!WARNING!!! Cannot plot estimation plot.")
        if errors:
            traceback.print_exc()

    # Plot metrics for each size separatly
    try:
        size = 0
        for size in estimation_frame[compare_size].drop_duplicates().values:
            size_estimation_frame = estimation_frame[
                estimation_frame[compare_size] == size
            ]
            plot_rows_columns_select(
                file_name=sub_file_name + f"_{size}.svg",
                data_frame=size_estimation_frame,
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
        print(f"\n!!!WARNING!!! Cannot plot estimation plot for size {size}.")
        if errors:
            traceback.print_exc()
