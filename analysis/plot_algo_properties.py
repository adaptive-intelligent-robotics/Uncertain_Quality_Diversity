import traceback
from functools import partial
from typing import List

import pandas as pd

from analysis.load_results import sort_data
from analysis.utils_plot import plot_rows_select


def plot_algo_properties(
    plot_folder: str,
    all_data: pd.DataFrame,
    all_times: pd.DataFrame,
    color_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    order: List,
    legend_columns: int,
    legend_bottom: float,
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
        vlines=[],
        hlines=[],
        filllines=[],
        x_front=[],
        y_front=[],
        box_plot=False,
        scatter_plot=False,
        x="epoch",
        xlabel="Generations",
    )

    # Print the metrics per env
    size_values = all_data[compare_size].drop_duplicates().values
    size_values.sort()
    rows_select = [[compare_size, x] for x in size_values]
    rows_name = [f"{compare_title} {x}" for x in size_values]
    for env in all_data["env"].drop_duplicates().values:

        # Extract and sort data
        env_data = all_data[all_data["env"] == env].reset_index(drop=True)
        env_data = sort_data(env_data, ["algo", "rep", "eval"], order)
        env_times = all_times[all_times["env"] == env].reset_index(drop=True)
        env_times = sort_data(env_times, ["algo", "rep", compare_size], order)

        # First, plot num samples across epoch with sizes as lines and metrics as columns
        try:
            columns = [
                "num_samples_per_indiv",
                "num_samples_per_iter",
            ]
            columns_name = [
                "Number of samples spent on each indiv",
                "Number of samples per iteration",
            ]
            plot_rows_select_fn(
                file_name=f"{plot_folder}/{env}-num_samples-convergence.svg",
                data_frame=env_data,
                rows_select=rows_select,
                rows_name=rows_name,
                columns=columns,
                columns_name=columns_name,
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot num_samples-convergence for {env}.")
            traceback.print_exc()

        # Second, plot deep target across epoch with sizes as lines and metrics as columns
        try:

            # Only plot Deep-Target algos
            sub_data = env_data[env_data["algo"].str.contains("Deep-Target")]

            # Fix some values
            sub_data["average_target"] = (
                sub_data["target_qd_score"] / sub_data["num_centroids"]
            )
            sub_data["average_qd_score"] = (
                sub_data["qd_score"] / sub_data["num_centroids"]
            )
            sub_data["average_reeval_qd_score"] = (
                sub_data["reeval_qd_score"] / sub_data["num_centroids"]
            )

            # Plot
            columns = [
                "average_target",
                "average_qd_score",
                "average_reeval_qd_score",
                "target_max_fitness",
                "max_fitness",
                "reeval_max_fitness",
            ]
            columns_name = [
                "Target value",
                "Archive value",
                "Archive reeval value",
                "Max Target value",
                "Max archive value",
                "Max archive reeval value",
            ]
            plot_rows_select_fn(
                file_name=f"{plot_folder}/{env}-target-convergence.svg",
                data_frame=sub_data,
                rows_select=rows_select,
                rows_name=rows_name,
                columns=columns,
                columns_name=columns_name,
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot target-convergence for {env}.")
            traceback.print_exc()
