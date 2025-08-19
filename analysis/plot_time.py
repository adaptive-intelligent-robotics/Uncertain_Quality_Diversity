import traceback
from functools import partial
from typing import List

import pandas as pd

from analysis.utils_plot import plot


def plot_time(
    plot_folder: str,
    all_times: pd.DataFrame,
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

    # Change time in minutes
    all_data = all_times[["env", "algo", compare_size, f"{prefixe}time"]]
    all_data[f"{prefixe}time"] = all_data[f"{prefixe}time"].div(60)

    # Print the metrics per env
    for env in all_data["env"].drop_duplicates().values:

        # Extract and sort data
        env_data = all_data[all_data["env"] == env].reset_index(drop=True)

        # Plot time across size with metrics as lines
        try:
            rows_columns = [
                [f"{prefixe}time"],
            ]
            rows_columns_name = [
                [f"Total time"],
            ]
            plot_fn(
                file_name=f"{plot_folder}/{env}-time.svg",
                data_frame=env_data,
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
