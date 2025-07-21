import os
import traceback
from copy import deepcopy
from typing import Dict, List

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

from analysis.load_archives import get_folder_name
from analysis.load_p_values import p_values
from analysis.load_results import sort_data
from analysis.utils_archive import plot_one_paper_archive
from analysis.utils_plot import plot_rows_columns_select


def plot_benchmark(
    plot_folder: str,
    all_times: pd.DataFrame,
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
) -> None:

    # First, print for the UQD Reproducibility benchmark tasks
    if (
        "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0.0"
        in all_times["env"].values
        and "arm_gaussian_desc_fitprop_variance_nonoise" in all_times["env"].values
    ):
        # Keep only those environments
        benchmark_frame = deepcopy(all_times)
        benchmark_frame = benchmark_frame[
            benchmark_frame["env"].isin(
                [
                    "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0.0",
                    "arm_gaussian_desc_fitprop_variance_nonoise",
                ]
            )
        ]
        benchmark_frame = sort_data(
            benchmark_frame,
            ["algo", "rep", "eval"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Build up the plot
        rows_columns_select = [
            [
                ["env", "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0.0"],
                ["env", "arm_gaussian_desc_fitprop_variance_nonoise"],
            ],
            # [
            #    ["env", "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0.0"],
            #    ["env", "arm_gaussian_desc_fitprop_variance_nonoise"],
            # ],
        ]
        rows_columns_title = [
            ["Arm Bi-Variance", "Arm Fit-Prop-Variance"],
            # ["", ""],
        ]
        filllines = []

        # Build the vlines
        algos = benchmark_frame["algo"].drop_duplicates().values
        algo = 0
        for compare_algo in baselines_list:
            while compare_algo in algos[algo]:
                algo += 1
        single_vlines = [
            [
                (algo / len(algos) - 0.5)
                * 1.0,  # position (multiply by 1.0, width of boxplot)
                "k",  # color
                (5, 5),  # dash
            ]
        ]
        for compare_algo in categories_list:
            while compare_algo in algos[algo]:
                algo += 1
        single_vlines.append(
            [
                (algo / len(algos) - 0.5)
                * 1.0,  # position (multiply by 1.0, width of boxplot)
                "k",  # color
                "",  # dash
            ]
        )
        for compare_algo in baselines_as_list:
            while compare_algo in algos[algo]:
                algo += 1
        single_vlines.append(
            [
                (algo / len(algos) - 0.5)
                * 1.0,  # position (multiply by 1.0, width of boxplot)
                "k",  # color
                (5, 5),  # dash
            ]
        )

        vlines = [
            [single_vlines, single_vlines, single_vlines, single_vlines],
            [single_vlines, single_vlines, single_vlines, single_vlines],
        ]

        # First, plot raw additional metrics
        rows_columns_metrics = [
            [
                "complement_additional_average",
                "complement_additional_average",
            ],
            # [
            #    "reeval_coverage",
            #    "reeval_coverage",
            # ],
        ]
        rows_columns_metrics_name = [
            [
                "Average reproducibility",
                "",
            ],
            # [
            #    "Corrected Coverage (%)",
            #    "",
            # ],
        ]
        sub_plot_benchmark(
            sub_file_name=f"{plot_folder}/benchmark_reproducibility",
            benchmark_frame=benchmark_frame,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            legend_columns=legend_columns,
            legend_bottom=legend_bottom,
            rows_columns_select=rows_columns_select,
            rows_columns_title=rows_columns_title,
            rows_columns_metrics=rows_columns_metrics,
            rows_columns_metrics_name=rows_columns_metrics_name,
            filllines=filllines,
            vlines=vlines,
        )

        # Second, plot corrected additional metrics
        rows_columns_metrics = [
            [
                "complement_reeval_additional_average",
                "complement_reeval_additional_average",
            ],
            # [
            #    "reeval_coverage",
            #    "reeval_coverage",
            # ],
        ]
        rows_columns_metrics_name = [
            [
                "Average reproducibility",
                "",
            ],
            # [
            #    "Corrected Coverage (%)",
            #    "",
            # ],
        ]
        sub_plot_benchmark(
            sub_file_name=f"{plot_folder}/benchmark_reproducibility_reeval",
            benchmark_frame=benchmark_frame,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            legend_columns=legend_columns,
            legend_bottom=legend_bottom,
            rows_columns_select=rows_columns_select,
            rows_columns_title=rows_columns_title,
            rows_columns_metrics=rows_columns_metrics,
            rows_columns_metrics_name=rows_columns_metrics_name,
            filllines=filllines,
            vlines=vlines,
        )

        # Third, compute p-value if asked
        if p_values:

            all_p_values(
                plot_folder=plot_folder,
                compare_size=compare_size,
                stats=[
                    "complement_additional_average",
                    "complement_reeval_additional_average",
                    # "reeval_coverage",
                ],
                benchmark_frame=benchmark_frame,
            )
    else:
        print("!!!WARNING!!! Missing tasks to plot UQD Benchmark Reproducibility.")

    # Second, print for the UQD Trade-off benchmark tasks
    if (
        "direct_mapping_perfect_trade_off_0.02_nonoise" in all_times["env"].values
        and "direct_mapping_deceptive_0.1_nonoise" in all_times["env"].values
        and "direct_mapping_sharp_peak_bigger_0.2_nonoise" in all_times["env"].values
        and "direct_mapping_sharp_peak_smaller_0.02_nonoise" in all_times["env"].values
    ):

        # Keep only those environments
        benchmark_frame = deepcopy(all_times)
        benchmark_frame = benchmark_frame[
            benchmark_frame["env"].isin(
                [
                    "direct_mapping_perfect_trade_off_0.02_nonoise",
                    "direct_mapping_deceptive_0.1_nonoise",
                    "direct_mapping_sharp_peak_bigger_0.2_nonoise",
                    "direct_mapping_sharp_peak_smaller_0.02_nonoise",
                ]
            )
        ]
        benchmark_frame = sort_data(
            benchmark_frame,
            ["algo", "rep", "eval"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Build up the plot
        rows_columns_select = [
            [
                ["env", "direct_mapping_perfect_trade_off_0.02_nonoise"],
                ["env", "direct_mapping_deceptive_0.1_nonoise"],
                ["env", "direct_mapping_sharp_peak_bigger_0.2_nonoise"],
                ["env", "direct_mapping_sharp_peak_smaller_0.02_nonoise"],
            ],
            # [
            #    ["env", "direct_mapping_perfect_trade_off_0.02_nonoise"],
            #    ["env", "direct_mapping_deceptive_0.1_nonoise"],
            #    ["env", "direct_mapping_sharp_peak_bigger_0.2_nonoise"],
            #    ["env", "direct_mapping_sharp_peak_smaller_0.02_nonoise"],
            # ],
        ]
        rows_columns_title = [
            [
                "Linear trade-off",
                "Deceptive",
                "Avoidable Peak",
                "Unavoidable Peak",
            ],
            # ["", "", "", ""],
        ]

        # Create filllines for the desired subplots
        filllines = [
            [
                [[0.93, 0.97, "green", 0.3]],
                [[0.6, 0.7, "red", 0.3]],
                [[0.9, 1.0, "red", 0.3]],
                [[0.9, 1.0, "green", 0.3]],
            ],
            # [[], [], [], []],
        ]

        # Build the vlines
        algos = benchmark_frame["algo"].drop_duplicates().values
        algo = 0
        for compare_algo in baselines_list:
            while compare_algo in algos[algo]:
                algo += 1
        single_vlines = [
            [
                (algo / len(algos) - 0.5)
                * 1.0,  # position (multiply by 1.0, width of boxplot)
                "k",  # color
                (5, 5),  # dash
            ]
        ]
        for compare_algo in categories_list:
            while compare_algo in algos[algo]:
                algo += 1
        single_vlines.append(
            [
                (algo / len(algos) - 0.5)
                * 1.0,  # position (multiply by 1.0, width of boxplot)
                "k",  # color
                "",  # dash
            ]
        )
        for compare_algo in baselines_as_list:
            while compare_algo in algos[algo]:
                algo += 1
        single_vlines.append(
            [
                (algo / len(algos) - 0.5)
                * 1.0,  # position (multiply by 1.0, width of boxplot)
                "k",  # color
                (5, 5),  # dash
            ]
        )

        vlines = [
            [single_vlines, single_vlines, single_vlines, single_vlines],
            [single_vlines, single_vlines, single_vlines, single_vlines],
        ]

        # First, plot raw additional metrics
        rows_columns_metrics = [
            [
                "additional_average",
                "additional_average",
                "additional_average",
                "additional_average",
            ],
            # [
            #    "reeval_qd_score",
            #    "reeval_qd_score",
            #    "reeval_qd_score",
            #    "reeval_qd_score",
            # ],
        ]
        rows_columns_metrics_name = [
            [
                "Average fitness",
                "",
                "",
                "",
            ],
            # [
            #    "Corrected QD-Score",
            #    "",
            #    "",
            #    "",
            # ],
        ]
        sub_plot_benchmark(
            sub_file_name=f"{plot_folder}/benchmark",
            benchmark_frame=benchmark_frame,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            legend_columns=legend_columns,
            legend_bottom=legend_bottom,
            rows_columns_select=rows_columns_select,
            rows_columns_title=rows_columns_title,
            rows_columns_metrics=rows_columns_metrics,
            rows_columns_metrics_name=rows_columns_metrics_name,
            filllines=filllines,
            vlines=vlines,
        )

        # Second, plot corrected additional metrics
        rows_columns_metrics = [
            [
                "reeval_additional_average",
                "reeval_additional_average",
                "reeval_additional_average",
                "reeval_additional_average",
            ],
            # [
            #    "reeval_qd_score",
            #    "reeval_qd_score",
            #    "reeval_qd_score",
            #    "reeval_qd_score",
            # ],
        ]
        rows_columns_metrics_name = [
            [
                "Average fitness",
                "",
                "",
                "",
            ],
            # [
            #    "Corrected QD-Score",
            #    "",
            #    "",
            #    "",
            # ],
        ]
        sub_plot_benchmark(
            sub_file_name=f"{plot_folder}/benchmark_reeval",
            benchmark_frame=benchmark_frame,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            legend_columns=legend_columns,
            legend_bottom=legend_bottom,
            rows_columns_select=rows_columns_select,
            rows_columns_title=rows_columns_title,
            rows_columns_metrics=rows_columns_metrics,
            rows_columns_metrics_name=rows_columns_metrics_name,
            filllines=filllines,
            vlines=vlines,
        )

        # Third, compute p-value if asked
        if p_values:

            all_p_values(
                plot_folder=plot_folder,
                compare_size=compare_size,
                stats=[
                    "additional_average",
                    "reeval_additional_average",
                    # "reeval_qd_score",
                ],
                benchmark_frame=benchmark_frame,
            )
    else:
        print("!!!WARNING!!! Missing tasks to plot UQD Benchmark Trade-Off.")


def sub_plot_benchmark(
    sub_file_name: str,
    benchmark_frame: pd.DataFrame,
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
    vlines: List,
) -> None:

    # Plot metrics for all sizes
    try:
        plot_rows_columns_select(
            file_name=sub_file_name + f".svg",
            data_frame=benchmark_frame,
            rows_columns_select=rows_columns_select,
            rows_columns_title=rows_columns_title,
            rows_columns_metrics=rows_columns_metrics,
            rows_columns_metrics_name=rows_columns_metrics_name,
            vlines=vlines,
            hlines=[],
            filllines=filllines,
            x_front=[],
            y_front=[],
            box_plot=True,
            scatter_plot=False,
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
        print(f"\n!!!WARNING!!! Cannot plot benchmark plot.")
        traceback.print_exc()

    # Plot metrics for each size separatly
    try:
        size = 0
        for size in benchmark_frame[compare_size].drop_duplicates().values:
            size_benchmark_frame = benchmark_frame[
                benchmark_frame[compare_size] == size
            ]
            plot_rows_columns_select(
                file_name=sub_file_name + f"_{size}.svg",
                data_frame=size_benchmark_frame,
                rows_columns_select=rows_columns_select,
                rows_columns_title=rows_columns_title,
                rows_columns_metrics=rows_columns_metrics,
                rows_columns_metrics_name=rows_columns_metrics_name,
                vlines=vlines,
                hlines=[],
                filllines=filllines,
                x_front=[],
                y_front=[],
                box_plot=True,
                scatter_plot=False,
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
        print(f"\n!!!WARNING!!! Cannot plot benchmark plot for size {size}.")
        traceback.print_exc()


def all_p_values(
    plot_folder: str,
    compare_size: str,
    stats: List,
    benchmark_frame: pd.DataFrame,
) -> None:

    for stat in stats:
        try:
            p_values(
                plot_folder=plot_folder,
                compare_size=compare_size,
                stat=stat,
                dataframe=benchmark_frame,
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot p-values for {stat}.")
            traceback.print_exc()


def plot_benchmark_archives(
    plot_folder: str,
    config_frame: pd.DataFrame,
    single_compare_size: int,
    env_order: Dict,
    baselines_list: List,
    categories_list: List,
    baselines_as_list: List,
    categories_as_list: List,
    compare_size: str,
    compare_title: str,
) -> None:

    if (
        "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0.0"
        in config_frame["env"].values
        and "arm_gaussian_desc_fitprop_variance_nonoise" in config_frame["env"].values
    ):
        # Keep only those environments
        benchmark_frame = deepcopy(config_frame)
        benchmark_frame = benchmark_frame[
            benchmark_frame["env"].isin(
                [
                    "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0.0",
                    "arm_gaussian_desc_fitprop_variance_nonoise",
                ]
            )
        ]
        benchmark_frame = sort_data(
            benchmark_frame,
            ["algo"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        try:
            print(f"    Plotting Reeval Archive for {single_compare_size}.")
            sub_plot_archives(
                file_name=f"{plot_folder}/benchmark_reproducibility_{single_compare_size}.png",
                single_compare_size=single_compare_size,
                env_order=env_order,
                config_frame=benchmark_frame,
                compare_size=compare_size,
                compare_title=compare_title,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
            traceback.print_exc()

    # Second, print for the UQD Trade-off benchmark tasks
    if (
        "direct_mapping_perfect_trade_off_0.02_nonoise" in config_frame["env"].values
        and "direct_mapping_deceptive_0.1_nonoise" in config_frame["env"].values
        and "direct_mapping_sharp_peak_bigger_0.2_nonoise" in config_frame["env"].values
        and "direct_mapping_sharp_peak_smaller_0.02_nonoise"
        in config_frame["env"].values
    ):

        # Keep only those environments
        benchmark_frame = deepcopy(config_frame)
        benchmark_frame = benchmark_frame[
            benchmark_frame["env"].isin(
                [
                    "direct_mapping_perfect_trade_off_0.02_nonoise",
                    "direct_mapping_deceptive_0.1_nonoise",
                    "direct_mapping_sharp_peak_bigger_0.2_nonoise",
                    "direct_mapping_sharp_peak_smaller_0.02_nonoise",
                ]
            )
        ]
        benchmark_frame = sort_data(
            benchmark_frame,
            ["algo"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        try:
            print(f"    Plotting Reeval Archive for {single_compare_size}.")
            sub_plot_archives(
                file_name=f"{plot_folder}/benchmark_tradeoff_{single_compare_size}.png",
                single_compare_size=single_compare_size,
                env_order=env_order,
                config_frame=benchmark_frame,
                compare_size=compare_size,
                compare_title=compare_title,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
            traceback.print_exc()


def sub_plot_archives(
    file_name: str,
    single_compare_size: int,
    env_order: Dict,
    config_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
) -> None:

    # Get the envs and algos
    envs = config_frame["env"].drop_duplicates().values
    algos = config_frame["algo"].drop_duplicates().values

    # Create the figure
    ncols = len(algos)
    nrows = len(envs)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 8, nrows * 8))

    # For each environment
    for nrow in range(nrows):

        env = envs[nrow]
        env_config_frame = config_frame[
            (config_frame["env"] == env)
            & (config_frame[compare_size] == single_compare_size)
        ].reset_index(drop=True)

        if env_config_frame.shape[0] == 0:
            print(
                f"\n!!!WARNING!!! No archives to plots for {env} and sampling-size {single_compare_size}.\n"
            )
            continue

        min_bd_list = str(env_config_frame["min_bd"][0])[1:-1].split(" ")
        if "" in min_bd_list:
            min_bd_list.remove("")
        max_bd_list = str(env_config_frame["max_bd"][0])[1:-1].split(" ")
        if "" in max_bd_list:
            max_bd_list.remove("")
        min_bd = [float(bd) for bd in min_bd_list]
        max_bd = [float(bd) for bd in max_bd_list]

        # For each algo for this env and size
        for ncol in range(ncols):

            algo = algos[ncol]
            algo_config_frame = env_config_frame[
                env_config_frame["algo"] == algo
            ].reset_index(drop=True)

            # Get the correpsonding axis
            if nrows == 1 and ncols == 1:
                ax = axes
            elif nrows == 1:
                ax = axes[ncol]
            elif ncols == 1:
                ax = axes[nrow]
            else:
                ax = axes[nrow, ncol]

            # Add colorbar in last column
            colorbar = False
            if ncol == ncols - 1:
                colorbar = True

            # Taking the first one randomly
            try:
                if (
                    env == "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0.0"
                    or env == "arm_gaussian_desc_fitprop_variance_nonoise"
                ):
                    repertoire_folder = get_folder_name(
                        algo_config_frame, f"reeval_repertoire_folder", 0
                    )
                    vmin = -0.1
                    vmax = 0
                else:
                    repertoire_folder = get_folder_name(
                        algo_config_frame, f"reeval_additional_folder", 0
                    )
                    vmin = 0
                    vmax = 1
                fitnesses = jnp.load(os.path.join(repertoire_folder, "fitnesses.npy"))
                descriptors = jnp.load(
                    os.path.join(repertoire_folder, "descriptors.npy")
                )
                centroids = jnp.load(os.path.join(repertoire_folder, "centroids.npy"))

                plot_one_paper_archive(
                    centroids=centroids,
                    descriptors=descriptors,
                    fitnesses=fitnesses,
                    ax=ax,
                    minval=min_bd,
                    maxval=max_bd,
                    vmin=vmin,
                    vmax=vmax,
                    colorbar=colorbar,
                )

            except Exception:
                print(f"\n!!!WARNING!!! Cannot plot for {env} and {algo}.")
                traceback.print_exc()

            # Add algo name in first line
            if nrow == 0:
                ax.set_title(algo, fontsize=32)

            # Add env name in first column
            if ncol == 0:
                title = env
                if env in env_order.keys():
                    title = env_order[env]
                ax.set_ylabel(title, fontsize=32)

    # Finish figure
    # fig.suptitle(title_name, fontsize=32)
    plt.tight_layout(h_pad=1.70)
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()
