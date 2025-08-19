import traceback
from functools import partial
from typing import List

import numpy as np
import pandas as pd

from analysis.load_results import sort_data
from analysis.utils_plot import plot, plot_rows_select


def plot_convergence_usage(
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
    errors: bool = False,
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

        # Plot usage across x_column with sizes as lines and metrics as columns
        size_values = env_convergence[compare_size].drop_duplicates().values
        size_values.sort()
        rows_select = [[compare_size, x] for x in size_values]
        rows_name = [f"{compare_title} {x}" for x in size_values]
        try:
            columns = [
                f"usage",
                f"explore_usage",
                f"exploit_usage",
            ]
            columns_name = [
                f"Usage",
                f"Explore usage",
                f"Exploit usage",
            ]
            plot_rows_select_fn(
                file_name=f"{plot_folder}/{env}-usage-convergence.svg",
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
            print(f"\n!!!WARNING!!! Cannot plot usage-convergencee for {env}.")
            if errors:
                traceback.print_exc()


def plot_usage(
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

        # Plot usage across size with metrics as lines
        try:
            rows_columns = [
                [f"usage"],
                [f"explore_usage"],
                [f"exploit_usage"],
            ]
            rows_columns_name = [
                [f"Usage"],
                [f"Explore usage"],
                [f"Exploit usage"],
            ]
            plot_fn(
                file_name=f"{plot_folder}/{env}-usage.svg",
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
            print(f"\n!!!WARNING!!! Cannot plot usage for {env}.")
            if errors:
                traceback.print_exc()
