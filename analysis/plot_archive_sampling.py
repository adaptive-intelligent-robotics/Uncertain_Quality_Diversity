import os
import traceback
from functools import partial
from typing import Dict, List

import pandas as pd

from analysis.load_results import sort_data
from analysis.utils_plot import plot_rows_columns_select


def plot_archive_sampling(
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
    errors: bool = False,
) -> None:

    plot_fn = partial(
        plot_rows_columns_select,
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

    # Create the metrics dataframe
    all_data = pd.DataFrame()

    # Function to add config info to a frame
    def add_config(
        frame: pd.DataFrame,
        name: List,
        algo: List,
        algo_batch: List,
        num_centroids: List,
        env: List,
        num_reevals: List,
        size: List,
        sampling_size: List,
        batch_size: List,
        rep: int,
    ) -> pd.DataFrame:
        frame["name"] = name
        frame["algo"] = algo
        frame["algo_batch"] = algo_batch
        frame["num_centroids"] = num_centroids
        frame["env"] = env
        frame["num_reevals"] = num_reevals
        frame[compare_size] = size
        frame["batch_size"] = batch_size
        frame["sampling_size"] = sampling_size
        frame["rep"] = rep
        return frame

    # Read all datas
    rep = 0
    for line in range(config_frame.shape[0]):

        try:
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

            # Get the partial eval metrics file
            folder = config_frame["folder"][line]
            metrics_file = config_frame["metrics_file"][line]
            archive_sampling_metrics_file = (
                "archive_sampling_metrics_"
                + metrics_file[metrics_file.find("/metrics_") + 9 :]
            )
            archive_sampling_metrics_file = os.path.join(
                folder, archive_sampling_metrics_file
            )

            # Check if file exit
            if not os.path.isfile(archive_sampling_metrics_file):
                # print(f"\n!!!WARNING!!! No partial eval for {algo} in {env}.")
                continue

            # Read metrics
            data = pd.read_csv(archive_sampling_metrics_file, index_col=False)
            data = add_config(
                frame=data,
                name=name,
                algo=algo,
                algo_batch=algo_batch,
                num_centroids=num_centroids,
                env=env,
                num_reevals=num_reevals,
                size=size,
                sampling_size=sampling_size,
                batch_size=batch_size,
                rep=rep,
            )

            # Check if using maximum x-axis
            if use_max_xaxis:
                if env in env_max_xaxis.keys():
                    # Get from input dict
                    max_x = env_max_xaxis[env]
                    print(f"\nFor {env}, from input dict: {max_x}.")
                else:
                    max_x = data["epoch"].max()
                    print(f"\nNo max x axis in input dict for {env}, using all values.")

                data = data[data["epoch"] <= max_x]

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
                        data["timestep"] * config_frame["episode_length"][line]
                    )

            # Concatenate all frames to existing ones
            all_data = pd.concat([all_data, data], ignore_index=True)

            # Increment rep counter
            rep += 1

        except Exception:
            print("\n!!!WARNING!!! Cannot read", archive_sampling_metrics_file, ".")
            if errors:
                traceback.print_exc()

    if all_data.empty:
        print("\n!!!WARNING!!! No Partial Evaluation to plot.")
        return

    # Sort datas
    all_data = sort_data(all_data, ["env", "algo", "rep", "eval"], order)

    # Get all sizes and environment values
    size_values = all_data[compare_size].drop_duplicates().values
    env_values = all_data["env"].drop_duplicates().values

    # Do a different plot per size
    for size in size_values:

        data = all_data[all_data[compare_size] == size]

        # Build up the plot
        sub_rows_columns_select = [["env", env] for env in env_values]
        sub_rows_columns_title = [env for env in env_values]

        # First, plot top layer infos
        rows_columns_select = [
            sub_rows_columns_select,
            sub_rows_columns_select,
            sub_rows_columns_select,
            sub_rows_columns_select,
            sub_rows_columns_select,
        ]
        rows_columns_title = [
            sub_rows_columns_title,
            sub_rows_columns_title,
            sub_rows_columns_title,
            sub_rows_columns_title,
            sub_rows_columns_title,
        ]
        rows_columns_metrics = [
            ["top_total_samples" for env in env_values],
            ["top_average_samples" for env in env_values],
            ["top_min_samples" for env in env_values],
            ["top_max_samples" for env in env_values],
            ["top_count_max" for env in env_values],
        ]
        rows_columns_metrics_name = [
            ["Top Layer - Total Samples"] + ["" for _ in range(len(env_values) - 1)],
            ["Top Layer - Average Samples"] + ["" for _ in range(len(env_values) - 1)],
            ["Top Layer - Min Samples"] + ["" for _ in range(len(env_values) - 1)],
            ["Top Layer - Max Samples"] + ["" for _ in range(len(env_values) - 1)],
            ["Top Layer - Count Max Samples"]
            + ["" for _ in range(len(env_values) - 1)],
        ]
        try:
            plot_fn(
                file_name=f"{plot_folder}/archive_sampling_top_{size}.svg",
                data_frame=data,
                rows_columns_select=rows_columns_select,
                rows_columns_title=rows_columns_title,
                rows_columns_metrics=rows_columns_metrics,
                rows_columns_metrics_name=rows_columns_metrics_name,
            )
        except Exception:
            print(
                f"\n!!!WARNING!!! Cannot plot top archive_sampling metrics for {size}."
            )
            if errors:
                traceback.print_exc()

        # Second, plot overall infos
        rows_columns_metrics = [
            ["overall_total_samples" for env in env_values],
            ["overall_average_samples" for env in env_values],
            ["overall_min_samples" for env in env_values],
            ["overall_max_samples" for env in env_values],
            ["overall_count_max" for env in env_values],
        ]
        rows_columns_metrics_name = [
            ["Overall Layer - Total Samples"]
            + ["" for _ in range(len(env_values) - 1)],
            ["Overall Layer - Average Samples"]
            + ["" for _ in range(len(env_values) - 1)],
            ["Overall Layer - Min Samples"] + ["" for _ in range(len(env_values) - 1)],
            ["Overall Layer - Max Samples"] + ["" for _ in range(len(env_values) - 1)],
            ["Overall Layer - Count Max Samples"]
            + ["" for _ in range(len(env_values) - 1)],
        ]
        try:
            plot_fn(
                file_name=f"{plot_folder}/archive_sampling_overall_{size}.svg",
                data_frame=data,
                rows_columns_select=rows_columns_select,
                rows_columns_title=rows_columns_title,
                rows_columns_metrics=rows_columns_metrics,
                rows_columns_metrics_name=rows_columns_metrics_name,
            )
        except Exception:
            print(
                f"\n!!!WARNING!!! Cannot plot overall archive_sampling metrics for {size}."
            )
            if errors:
                traceback.print_exc()
