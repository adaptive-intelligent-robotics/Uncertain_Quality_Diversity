import traceback
from functools import partial
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis.utils_plot import plot_columns_select


def plot_summary(
    plot_folder: str,
    all_losses: pd.DataFrame,
    all_times: pd.DataFrame,
    all_var: pd.DataFrame,
    color_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    order: List,
    env_order: Dict,
    legend_columns: int,
    legend_bottom: float,
    prefixe: str = "",
    prefixe_title: str = "",
) -> None:

    plot_columns_select_fn = partial(
        plot_columns_select,
        size=None,
        sizes=[],
        hue="algo",
        color_frame=color_frame,
        legend_columns=legend_columns,
        legend_bottom=legend_bottom,
        markers=True,
    )

    # Concatenate the three dataframe
    all_data = pd.concat([all_times, all_losses, all_var], ignore_index=True)

    # Use env_name as column
    env_names = all_data["env"].drop_duplicates().values
    columns_select: List = []
    columns_name: List = []
    for env_name in env_order.keys():
        if env_name in env_names:
            columns_select.append(env_name)
            columns_name.append(env_order[env_name])
    for env_name in env_names:
        if env_name not in columns_select:
            columns_select.append(env_name)
            columns_name.append(env_name)
    columns_select = [["env", x] for x in columns_select]

    # Use log plot
    all_data["log_compare_size"] = np.log2(all_data[compare_size])

    # Change time in minutes
    all_data[f"{prefixe}time"] = all_data[f"{prefixe}time"].div(60)

    # Use compare_size as major locator
    compare_size_values = all_data[compare_size].drop_duplicates().values
    log_compare_size_values = all_data["log_compare_size"].drop_duplicates().values

    # Plot
    try:
        rows = [
            f"{prefixe}reeval_qd_score",
            f"{prefixe}reeval_max_fitness",
            f"loss_{prefixe}reeval_qd_score",
            f"{prefixe}avg_desc_var_qd_score",
            f"{prefixe}avg_fit_var_qd_score",
            f"{prefixe}time",
        ]
        rows_name = [
            f"{prefixe_title}Corrected QD-Score",
            f"{prefixe_title}Corrected Max-Fitness",
            f"{prefixe_title}Loss QD-Score",
            f"{prefixe_title}Descriptor-Variance",
            f"{prefixe_title}Fitness-Variance",
            f"{prefixe_title}Time to convergence(mins)",
        ]
        plot_columns_select_fn(
            file_name=f"{plot_folder}/all_env_{prefixe}metrics-size.svg",
            data_frame=all_data,
            columns_select=columns_select,
            columns_name=columns_name,
            rows=rows,
            rows_name=rows_name,
            vlines=[],
            hlines=[],
            filllines=[],
            x_front=[],
            y_front=[],
            box_plot=False,
            scatter_plot=False,
            x="log_compare_size",
            xlabel=compare_title,
            major_locator=log_compare_size_values,
            major_locator_name=compare_size_values,
        )
    except Exception:
        print(f"\n!!!WARNING!!! Cannot plot all_env_{prefixe}metrics-size.")
        traceback.print_exc()
