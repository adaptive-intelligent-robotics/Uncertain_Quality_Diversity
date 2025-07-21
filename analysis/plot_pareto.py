import traceback
from functools import partial
from typing import Dict, List

import numpy as np
import pandas as pd

from analysis.utils_plot import plot_columns_select


def plot_pareto(
    plot_folder: str,
    all_times: pd.DataFrame,
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
        hue="algo",
        color_frame=color_frame,
        legend_columns=legend_columns,
        legend_bottom=legend_bottom,
        markers=True,
    )

    # Create the pareto frame
    all_data = all_times[
        ["env", "algo", compare_size, f"{prefixe}time", f"{prefixe}reeval_qd_score"]
    ]
    all_data = all_data.groupby(
        by=["env", "algo", compare_size], as_index=False
    ).median()
    print(all_data)

    # Change time in minutes
    all_data[f"{prefixe}time"] = all_data[f"{prefixe}time"].div(60)

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

    # Create the pareto front per env
    x_front: List = []
    y_front: List = []
    for index_env in range(len(columns_select)):
        env_name = columns_select[index_env][1]
        env_data = all_data[all_data["env"] == env_name]

        # Find all points on pareto front
        front_bool = np.ones(env_data.shape[0])
        for indiv_1 in range(env_data.shape[0]):
            for indiv_2 in range(env_data.shape[0]):

                # If indiv_2 dominates indiv_1
                if (
                    env_data[f"{prefixe}reeval_qd_score"].values[indiv_1]
                    <= env_data[f"{prefixe}reeval_qd_score"].values[indiv_2]
                    and env_data[f"{prefixe}time"].values[indiv_1]
                    > env_data[f"{prefixe}time"].values[indiv_2]
                ) or (
                    env_data[f"{prefixe}reeval_qd_score"].values[indiv_1]
                    < env_data[f"{prefixe}reeval_qd_score"].values[indiv_2]
                    and env_data[f"{prefixe}time"].values[indiv_1]
                    >= env_data[f"{prefixe}time"].values[indiv_2]
                ):
                    front_bool[indiv_1] = 0
                    break

        # Store their x and y
        env_x_front: List = []
        env_y_front: List = []
        for indiv in range(env_data.shape[0]):
            if front_bool[indiv]:
                env_x_front.append(env_data[f"{prefixe}reeval_qd_score"].values[indiv])
                env_y_front.append(env_data[f"{prefixe}time"].values[indiv])
        env_x_front = np.array(env_x_front)
        env_y_front = np.array(env_y_front)
        indexes = np.argsort(x_front)
        env_x_front = env_x_front[indexes]
        env_y_front = env_y_front[indexes]

        # Add to common front
        x_front.append(env_x_front)
        y_front.append(env_y_front)

    # Create the point sizes
    def powspace(start: float, stop: float, power: int, num: int) -> np.array:
        start = np.power(start, 1 / float(power))
        stop = np.power(stop, 1 / float(power))
        return np.power(np.linspace(start, stop, num=num), power)

    sizes = all_data[compare_size].drop_duplicates().values
    sizes.sort()
    sizes = np.append(sizes, sizes[0] // 2)
    sizes.sort()
    dot_sizes = powspace(10, 2000, 4, len(sizes))
    scatter_sizes = dict(zip(sizes, dot_sizes))

    # Plot
    try:
        rows = [f"{prefixe}time"]
        rows_name = ["Time to convergence (mins)"]
        plot_columns_select_fn(
            file_name=f"{plot_folder}/all_env_{prefixe}pareto.svg",
            data_frame=all_data,
            columns_select=columns_select,
            columns_name=columns_name,
            rows=rows,
            rows_name=rows_name,
            vlines=[],
            hlines=[],
            filllines=[],
            x_front=x_front,
            y_front=y_front,
            box_plot=False,
            scatter_plot=True,
            x=f"{prefixe}reeval_qd_score",
            xlabel="Corrected QD-Score (%-max-value)",
            major_locator=[],
            major_locator_name=[],
            size=compare_size,
            sizes=scatter_sizes,
        )
    except Exception:
        print(f"\n!!!WARNING!!! Cannot plot all_env_{prefixe}pareto.")
        traceback.print_exc()
