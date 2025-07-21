import traceback
from functools import partial
from typing import List

import pandas as pd

from analysis.load_results import sort_data
from analysis.utils_plot import plot


def plot_reproducibility_metrics(
    plot_folder: str,
    all_reproducibilities_data: pd.DataFrame,
    color_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    order: List,
    legend_columns: int,
    legend_bottom: float,
    prefixe: str = "",
    prefixe_title: str = "",
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
    for env in all_reproducibilities_data["env"].drop_duplicates().values:

        # Extract and sort data
        env_reproducibilities_data = all_reproducibilities_data[
            all_reproducibilities_data["env"] == env
        ].reset_index(drop=True)
        env_reproducibilities_data = sort_data(
            env_reproducibilities_data, ["algo", "rep", compare_size], order
        )

        # First, plot reproducibilities across size with metrics as lines
        try:
            rows_columns = [
                [f"{prefixe}fit_reproducibilities_qd_score"],
                [f"{prefixe}desc_reproducibilities_qd_score"],
            ]
            rows_columns_name = [
                [f"{prefixe_title}Fit-Reproducibility"],
                [f"{prefixe_title}Desc-Reproducibility"],
            ]
            plot_fn(
                file_name=f"{plot_folder}/{env}-{prefixe}reproducibilities.svg",
                data_frame=env_reproducibilities_data,
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
            print(f"\n!!!WARNING!!! Cannot plot {prefixe}reproducibilities for {env}.")
            traceback.print_exc()

        # Second, plot reeval reproducibilities across size with metrics as lines
        try:
            rows_columns = [
                [f"{prefixe}reeval_fit_reproducibilities_qd_score"],
                [f"{prefixe}reeval_desc_reproducibilities_qd_score"],
            ]
            rows_columns_name = [
                [f"{prefixe_title}Reeval Fit-Reproducibility"],
                [f"{prefixe_title}Reeval Desc-Reproducibility"],
            ]
            plot_fn(
                file_name=f"{plot_folder}/{env}-{prefixe}reeval_reproducibilities.svg",
                data_frame=env_reproducibilities_data,
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
                f"\n!!!WARNING!!! Cannot plot {prefixe}reeval_reproducibilities for {env}."
            )
            traceback.print_exc()
