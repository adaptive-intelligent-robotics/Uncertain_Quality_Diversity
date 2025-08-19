import traceback
from functools import partial
from typing import List

import numpy as np
import pandas as pd

from analysis.load_results import sort_data
from analysis.utils_plot import plot, plot_rows_select


def plot_convergence_metrics(
    plot_folder: str,
    all_convergence: pd.DataFrame,
    all_times: pd.DataFrame,
    color_frame: pd.DataFrame,
    x_column: str,
    x_name: str,
    compare_size: str,
    compare_title: str,
    order: List,
    legend_columns: int,
    legend_bottom: float,
    prefixe: str = "",
    prefixe_title: str = "",
    errors: bool = False,
    additional: bool = False,
) -> None:

    plot_rows_select_fn = partial(
        plot_rows_select,
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

    # Print the metrics per env
    for env in all_convergence["env"].drop_duplicates().values:

        # Extract and sort data
        env_convergence = all_convergence[all_convergence["env"] == env].reset_index(
            drop=True
        )
        env_convergence = sort_data(env_convergence, ["algo", "rep", "eval"], order)
        env_times = all_times[all_times["env"] == env].reset_index(drop=True)
        env_times = sort_data(env_times, ["algo", "rep", compare_size], order)

        # Create the vlines
        vlines: List = []
        try:
            if x_column in env_times.columns:
                for algo in env_convergence["algo"].drop_duplicates().values:
                    vlines.append(
                        [
                            np.median(
                                env_times[env_times["algo"] == algo][x_column]
                            ),  # position
                            color_frame[color_frame["Label"] == algo]["Color"].values[
                                0
                            ],  # color
                            (5, 5),  # dash
                        ]
                    )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot compute vlines for {env}.")
            if errors:
                traceback.print_exc()

        # First, plot reeval across x_column with sizes as lines and metrics as columns
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
            plot_rows_select_fn(
                file_name=f"{plot_folder}/{env}-{prefixe}reeval_metrics-convergence.svg",
                data_frame=env_convergence,
                rows_select=rows_select,
                rows_name=rows_name,
                columns=columns,
                columns_name=columns_name,
                vlines=vlines,
                hlines=[],
                filllines=[],
                x_front=[],
                y_front=[],
                box_plot=False,
                scatter_plot=False,
                x=x_column,
                xlabel=x_name,
            )
        except Exception:
            print(
                f"\n!!!WARNING!!! Cannot plot {prefixe}reeval_metrics-convergence for {env}."
            )
            if errors:
                traceback.print_exc()

        # Second, plot metrics across x_column with size as lines and metrics as columns
        try:
            columns = [
                f"{prefixe}qd_score",
                f"{prefixe}coverage",
                f"{prefixe}max_fitness",
            ]
            columns_name = [
                f"{prefixe_title}QD Score",
                f"{prefixe_title}Coverage in %",
                f"{prefixe_title}Maximum fitness",
            ]
            plot_rows_select_fn(
                file_name=f"{plot_folder}/{env}-{prefixe}metrics-convergence.svg",
                data_frame=env_convergence,
                rows_select=rows_select,
                rows_name=rows_name,
                columns=columns,
                columns_name=columns_name,
                vlines=vlines,
                hlines=[],
                filllines=[],
                x_front=[],
                y_front=[],
                box_plot=False,
                scatter_plot=False,
                x=x_column,
                xlabel=x_name,
            )
        except Exception:
            print(
                f"\n!!!WARNING!!! Cannot plot {prefixe}metrics-convergence for {env}."
            )
            if errors:
                traceback.print_exc()

        if additional:

            # Third, plot additional across x_column with sizes as lines and metrics as columns
            try:
                columns = [
                    f"{prefixe}additional_qd_score",
                    f"{prefixe}additional_max_fitness",
                    f"{prefixe}additional_min_fitness",
                ]
                columns_name = [
                    f"Sum of {prefixe_title}Additional",
                    f"Max {prefixe_title}Additional",
                    f"Min {prefixe_title}Additional",
                ]
                plot_rows_select_fn(
                    file_name=f"{plot_folder}/{env}-{prefixe}additional-convergence.svg",
                    data_frame=env_convergence,
                    rows_select=rows_select,
                    rows_name=rows_name,
                    columns=columns,
                    columns_name=columns_name,
                    vlines=vlines,
                    hlines=[],
                    filllines=[],
                    x_front=[],
                    y_front=[],
                    box_plot=False,
                    scatter_plot=False,
                    x=x_column,
                    xlabel=x_name,
                )
            except Exception:
                print(
                    f"\n!!!WARNING!!! Cannot plot {prefixe}additional-convergence for {env}."
                )
                if errors:
                    traceback.print_exc()

            # Fourth, plot reeval additional across x_column with sizes as lines and metrics as columns
            try:
                columns = [
                    f"{prefixe}reeval_additional_qd_score",
                    f"{prefixe}reeval_additional_max_fitness",
                    f"{prefixe}reeval_additional_min_fitness",
                ]
                columns_name = [
                    f"Sum of {prefixe_title}Reeval Additional",
                    f"Max {prefixe_title}Reeval Additional",
                    f"Min {prefixe_title}Reeval Additional",
                ]
                plot_rows_select_fn(
                    file_name=f"{plot_folder}/{env}-{prefixe}reeval_additional-convergence.svg",
                    data_frame=env_convergence,
                    rows_select=rows_select,
                    rows_name=rows_name,
                    columns=columns,
                    columns_name=columns_name,
                    vlines=vlines,
                    hlines=[],
                    filllines=[],
                    x_front=[],
                    y_front=[],
                    box_plot=False,
                    scatter_plot=False,
                    x=x_column,
                    xlabel=x_name,
                )
            except Exception:
                print(
                    f"\n!!!WARNING!!! Cannot plot {prefixe}reeval_additional-convergence for {env}."
                )
                if errors:
                    traceback.print_exc()


def plot_metrics(
    plot_folder: str,
    all_finals: pd.DataFrame,
    color_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    order: List,
    legend_columns: int,
    legend_bottom: float,
    prefixe: str = "",
    prefixe_title: str = "",
    errors: bool = False,
    additional: bool = False,
) -> None:

    plot_fn = partial(
        plot,
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

    # Print the metrics per env
    for env in all_finals["env"].drop_duplicates().values:

        # Extract and sort data
        env_finals = all_finals[all_finals["env"] == env].reset_index(drop=True)
        env_finals = sort_data(env_finals, ["algo", "rep", compare_size], order)

        # First, plot reeval across size with metrics as lines
        try:
            rows_columns = [
                [f"{prefixe}reeval_qd_score"],
                [f"{prefixe}reeval_coverage"],
                [f"{prefixe}reeval_max_fitness"],
                [f"{prefixe}fit_reeval_qd_score"],
                [f"{prefixe}desc_reeval_coverage"],
            ]
            rows_columns_name = [
                [f"{prefixe_title}Reeval QD-Score"],
                [f"{prefixe_title}Reeval Coverage (%)"],
                [f"{prefixe_title}Reeval Max-Fitness"],
                [f"{prefixe_title}Reeval Fitness-only QD-Score"],
                [f"{prefixe_title}Reeval Desc-only Coverage (%)"],
            ]
            plot_fn(
                file_name=f"{plot_folder}/{env}-{prefixe}reeval_metrics.svg",
                data_frame=env_finals,
                rows_columns=rows_columns,
                rows_columns_name=rows_columns_name,
                vlines=[],
                hlines=[],
                filllines=[],
                x_front=[],
                y_front=[],
                box_plot=True,
                scatter_plot=False,
                x=compare_size,
                xlabel=compare_title,
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot {prefixe}reeval_metrics for {env}.")
            if errors:
                traceback.print_exc()

        # Second, plot metrics across size with metrics as lines
        try:
            rows_columns = [
                [f"{prefixe}qd_score"],
                [f"{prefixe}coverage"],
                [f"{prefixe}max_fitness"],
            ]
            rows_columns_name = [
                [f"{prefixe_title}QD-Score"],
                [f"{prefixe_title}Coverage (%)"],
                [f"{prefixe_title}Max-Fitness"],
            ]
            plot_fn(
                file_name=f"{plot_folder}/{env}-{prefixe}metrics.svg",
                data_frame=env_finals,
                rows_columns=rows_columns,
                rows_columns_name=rows_columns_name,
                vlines=[],
                hlines=[],
                filllines=[],
                x_front=[],
                y_front=[],
                box_plot=True,
                scatter_plot=False,
                x=compare_size,
                xlabel=compare_title,
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot {prefixe}metrics for {env}.")
            if errors:
                traceback.print_exc()

        # Third, plot metrics across size with the metrics as line and columns
        # try:
        #    rows_columns = [
        #        [f"{prefixe}qd_score", f"{prefixe}coverage", f"{prefixe}max_fitness"],
        #        [
        #            f"{prefixe}reeval_qd_score",
        #            f"{prefixe}reeval_coverage",
        #            f"{prefixe}reeval_max_fitness",
        #        ],
        #        [
        #            f"{prefixe}fit_reeval_qd_score",
        #            f"{prefixe}fit_reeval_coverage",
        #            f"{prefixe}fit_reeval_max_fitness",
        #        ],
        #        [
        #            f"{prefixe}desc_reeval_qd_score",
        #            f"{prefixe}desc_reeval_coverage",
        #            f"{prefixe}desc_reeval_max_fitness",
        #        ],
        #    ]
        #    rows_columns_name = [
        #        [
        #            f"{prefixe_title}QD Score",
        #            f"{prefixe_title}Coverage in %",
        #            f"{prefixe_title}Maximum fitness",
        #        ],
        #        [
        #            f"{prefixe_title}Reeval QD Score",
        #            f"{prefixe_title}Reeval Coverage in %",
        #            f"{prefixe_title}Reeval Maximum fitness",
        #        ],
        #        [
        #            f"{prefixe_title}Fit-Reeval QD Score",
        #            f"{prefixe_title}Fit-Reeval Coverage in %",
        #            f"{prefixe_title}Fit-Reeval Maximum fitness",
        #        ],
        #        [
        #            f"{prefixe_title}Desc-Reeval QD Score",
        #            f"{prefixe_title}Desc-Reeval Coverage in %",
        #            f"{prefixe_title}Desc-Reeval Maximum fitness",
        #        ],
        #    ]
        #    plot_fn(
        #        file_name=f"{plot_folder}/{env}-{prefixe}metrics-size.svg",
        #        data_frame=env_finals,
        #        rows_columns=rows_columns,
        #        rows_columns_name=rows_columns_name,
        #        vlines=[],
        #        hlines=[],
        #        filllines=[],
        #        x_front=[],
        #        y_front=[],
        #        box_plot=False,
        #        scatter_plot=False,
        #        x=compare_size,
        #        xlabel=compare_title,
        #    )
        # except Exception:
        #    print(f"\n!!!WARNING!!! Cannot plot {prefixe}metrics-size for {env}.")
        #    if errors:
        #        traceback.print_exc()

        # Fourth, plot loss across size with the metrics as line
        try:
            rows_columns = [
                [f"loss_{prefixe}qd_score"],
                [f"loss_{prefixe}coverage"],
            ]
            rows_columns_name = [
                [f"{prefixe_title}QD-Score Loss (%)"],
                [f"{prefixe_title}Coverage Loss (%)"],
            ]
            plot_fn(
                file_name=f"{plot_folder}/{env}-{prefixe}loss-size.svg",
                data_frame=env_finals,
                rows_columns=rows_columns,
                rows_columns_name=rows_columns_name,
                vlines=[],
                hlines=[],
                filllines=[],
                x_front=[],
                y_front=[],
                box_plot=True,
                scatter_plot=False,
                x=compare_size,
                xlabel=compare_title,
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot {prefixe}loss-size for {env}.")
            if errors:
                traceback.print_exc()

        if additional:

            # Fifth, plot additional across size with the metrics as line
            try:
                rows_columns = [
                    [f"{prefixe}additional_qd_score"],
                    [f"{prefixe}additional_average"],
                    [f"{prefixe}additional_max_fitness"],
                    [f"{prefixe}additional_min_fitness"],
                ]
                rows_columns_name = [
                    [f"{prefixe_title}Sum of Additional"],
                    [f"{prefixe_title}Average Additional"],
                    [f"{prefixe_title}Max Additional"],
                    [f"{prefixe_title}Min Additional"],
                ]
                plot_fn(
                    file_name=f"{plot_folder}/{env}-{prefixe}additional.svg",
                    data_frame=env_finals,
                    rows_columns=rows_columns,
                    rows_columns_name=rows_columns_name,
                    vlines=[],
                    hlines=[],
                    filllines=[],
                    x_front=[],
                    y_front=[],
                    box_plot=True,
                    scatter_plot=False,
                    x=compare_size,
                    xlabel=compare_title,
                )
            except Exception:
                print(f"\n!!!WARNING!!! Cannot plot {prefixe}additional for {env}.")
                if errors:
                    traceback.print_exc()

            # Sixth, plot reeval additional across size with the metrics as line
            try:
                rows_columns = [
                    [f"{prefixe}reeval_additional_qd_score"],
                    [f"{prefixe}reeval_additional_average"],
                    [f"{prefixe}reeval_additional_max_fitness"],
                    [f"{prefixe}reeval_additional_min_fitness"],
                ]
                rows_columns_name = [
                    [f"{prefixe_title}Sum of Reeval Additional"],
                    [f"{prefixe_title}Average Reeval Additional"],
                    [f"{prefixe_title}Max Reeval Additional"],
                    [f"{prefixe_title}Min Reeval Additional"],
                ]
                plot_fn(
                    file_name=f"{plot_folder}/{env}-{prefixe}reeval_additional.svg",
                    data_frame=env_finals,
                    rows_columns=rows_columns,
                    rows_columns_name=rows_columns_name,
                    vlines=[],
                    hlines=[],
                    filllines=[],
                    x_front=[],
                    y_front=[],
                    box_plot=True,
                    scatter_plot=False,
                    x=compare_size,
                    xlabel=compare_title,
                )
            except Exception:
                print(
                    f"\n!!!WARNING!!! Cannot plot {prefixe}reeval_additional for {env}."
                )
                if errors:
                    traceback.print_exc()
