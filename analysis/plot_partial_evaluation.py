import os
import traceback
from functools import partial
from typing import Dict, List

import pandas as pd

from analysis.load_results import sort_data
from analysis.utils_plot import plot_rows_columns_select


def plot_partial_evaluation(
    plot_folder: str,
    config_frame: pd.DataFrame,
    all_convergence: pd.DataFrame,
    x_column: str,
    x_name: str,
    compare_size: str,
    compare_title: str,
    order: List,
    env_order: Dict,
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

    # Read all datas
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
            episode_length = config_frame["episode_length"][line]
            rep = line

            try:
                # Get the partial eval metrics file
                folder = config_frame["folder"][line]
                metrics_file = config_frame["metrics_file"][line]
                partial_eval_metrics_file = (
                    "partial_eval_metrics_"
                    + metrics_file[metrics_file.find("/metrics_") + 9 :]
                )
                partial_eval_metrics_file = os.path.join(
                    folder, partial_eval_metrics_file
                )

                # Read metrics
                data = pd.read_csv(partial_eval_metrics_file, index_col=False)

            except Exception:
                print("\n!!!WARNING!!! Cannot read", partial_eval_metrics_file, ".")

                print("Using episode_length")
                sub_convergence = all_convergence[
                    (all_convergence["name"] == name)
                    & (all_convergence["env"] == env)
                    & (all_convergence["algo"] == algo)
                    & (all_convergence["num_centroids"] == num_centroids)
                    & (all_convergence["num_reevals"] == num_reevals)
                    & (all_convergence["batch_size"] == batch_size)
                ]

                data: Dict[str, List[float]] = {}
                data["epoch"] = sub_convergence["epoch"].drop_duplicates().values
                data["average_length"] = episode_length
                data["average_length_archive"] = episode_length
                data["min_length_archive"] = episode_length
                data["max_length_archive"] = episode_length
                data["extract_average_length"] = episode_length
                data["extract_max_length"] = episode_length
                data["extract_min_length"] = episode_length
                data["emit_average_length"] = episode_length
                data["emit_max_length"] = episode_length
                data["emit_min_length"] = episode_length

                data = pd.DataFrame.from_dict(data)

            # Complete data
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

        except Exception:
            print(
                "\n!!!WARNING!!! Cannot do anything for", partial_eval_metrics_file, "."
            )
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
        sub_rows_columns_title = [env_order[env] for env in env_values]

        # First, plot average over archive
        rows_columns_select = [
            sub_rows_columns_select,
            sub_rows_columns_select,
            sub_rows_columns_select,
        ]
        rows_columns_title = [
            sub_rows_columns_title,
            sub_rows_columns_title,
            sub_rows_columns_title,
        ]
        rows_columns_metrics = [
            ["average_length_archive" for env in env_values],
            ["min_length_archive" for env in env_values],
            ["max_length_archive" for env in env_values],
        ]
        rows_columns_metrics_name = [
            ["Archive - Average Length"] + ["" for _ in range(len(env_values) - 1)],
            ["Archive - Min Length"] + ["" for _ in range(len(env_values) - 1)],
            ["Archive - Max Length"] + ["" for _ in range(len(env_values) - 1)],
        ]
        try:
            plot_fn(
                file_name=f"{plot_folder}/partial_eval_archive_{size}.svg",
                data_frame=data,
                rows_columns_select=rows_columns_select,
                rows_columns_title=rows_columns_title,
                rows_columns_metrics=rows_columns_metrics,
                rows_columns_metrics_name=rows_columns_metrics_name,
            )
        except Exception:
            print(
                f"\n!!!WARNING!!! Cannot plot archive partial_evaluation metrics for {size}."
            )
            if errors:
                traceback.print_exc()

        # Second, plot average length and termination across batch
        rows_columns_select = [
            sub_rows_columns_select,
        ]
        rows_columns_title = [
            sub_rows_columns_title,
        ]
        rows_columns_metrics = [
            ["average_length" for env in env_values],
        ]
        rows_columns_metrics_name = [
            ["Batch - Average Length"] + ["" for _ in range(len(env_values) - 1)],
        ]
        try:
            plot_fn(
                file_name=f"{plot_folder}/partial_eval_average_batch_{size}.svg",
                data_frame=data,
                rows_columns_select=rows_columns_select,
                rows_columns_title=rows_columns_title,
                rows_columns_metrics=rows_columns_metrics,
                rows_columns_metrics_name=rows_columns_metrics_name,
            )
        except Exception:
            print(
                f"\n!!!WARNING!!! Cannot plot average batch partial_evaluation metrics for {size}."
            )
            if errors:
                traceback.print_exc()

        # Third, plot length and termination of extract and emit
        rows_columns_select = [
            sub_rows_columns_select,
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
            sub_rows_columns_title,
        ]
        rows_columns_metrics = [
            ["extract_average_length" for env in env_values],
            ["extract_max_length" for env in env_values],
            ["extract_min_length" for env in env_values],
            ["emit_average_length" for env in env_values],
            ["emit_max_length" for env in env_values],
            ["emit_min_length" for env in env_values],
        ]
        rows_columns_metrics_name = [
            ["Extract - Average Length"] + ["" for _ in range(len(env_values) - 1)],
            ["Extract - Max Length"] + ["" for _ in range(len(env_values) - 1)],
            ["Extract - Min Length"] + ["" for _ in range(len(env_values) - 1)],
            ["Emit - Average Length"] + ["" for _ in range(len(env_values) - 1)],
            ["Emit - Max Length"] + ["" for _ in range(len(env_values) - 1)],
            ["Emit - Min Length"] + ["" for _ in range(len(env_values) - 1)],
        ]
        try:
            plot_fn(
                file_name=f"{plot_folder}/partial_eval_extract_emit_{size}.svg",
                data_frame=data,
                rows_columns_select=rows_columns_select,
                rows_columns_title=rows_columns_title,
                rows_columns_metrics=rows_columns_metrics,
                rows_columns_metrics_name=rows_columns_metrics_name,
            )
        except Exception:
            print(
                f"\n!!!WARNING!!! Cannot plot extract-emit partial_evaluation metrics for {size}."
            )
            if errors:
                traceback.print_exc()
