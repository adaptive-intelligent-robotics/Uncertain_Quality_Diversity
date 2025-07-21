import os
import traceback
from typing import Dict, List

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

from analysis.load_archives import get_folder_name
from analysis.load_p_values import p_values as p_values_fn
from analysis.load_results import sort_data
from analysis.utils_archive import plot_one_paper_archive
from analysis.utils_plot import plot_rows_columns_select


def plot_robotics(
    plot_folder: str,
    all_times: pd.DataFrame,
    all_reprods: pd.DataFrame,
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
    prefixe: str = "",
    prefixe_title: str = "",
) -> None:

    if (
        "hexapod_sin_omni_fit0.05_desc0.05_params0.0" in all_times["env"].values
        and "hexapod_sin_omni_fit0.05_desc0.05_params0.0" in all_reprods["env"].values
        and "ant_omni" in all_times["env"].values
        and "ant_omni" in all_reprods["env"].values
        and "walker2d_uni" in all_times["env"].values
        and "walker2d_uni" in all_reprods["env"].values
    ):
        # Merge all_times and all_reprods
        robotics_frame = pd.concat([all_times, all_reprods], ignore_index=True)

        # Keep only those environments
        robotics_frame = robotics_frame[
            robotics_frame["env"].isin(
                [
                    "hexapod_sin_omni_fit0.05_desc0.05_params0.0",
                    "ant_omni",
                    "walker2d_uni",
                ]
            )
        ]
        robotics_frame = sort_data(
            robotics_frame,
            ["algo", "rep", "eval"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Build the vlines
        algos = robotics_frame["algo"].drop_duplicates().values
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

        # Build up the plot
        rows_columns_select = [
            [
                ["env", "hexapod_sin_omni_fit0.05_desc0.05_params0.0"],
                ["env", "walker2d_uni"],
                ["env", "ant_omni"],
            ],
            [
                ["env", "hexapod_sin_omni_fit0.05_desc0.05_params0.0"],
                ["env", "walker2d_uni"],
                ["env", "ant_omni"],
            ],
        ]
        rows_columns_title = [
            ["Hexapod", "Walker", "Ant"],
            ["", "", ""],
        ]

        # First, plot QD-Score and
        rows_columns_metrics = [
            [
                f"{prefixe}reeval_qd_score",
                f"{prefixe}reeval_qd_score",
                f"{prefixe}reeval_qd_score",
            ],
            [
                # f"{prefixe}reeval_avg_desc_reproducibilities_qd_score",
                # f"{prefixe}reeval_avg_desc_reproducibilities_qd_score",
                # f"{prefixe}reeval_avg_desc_reproducibilities_qd_score",
                f"{prefixe}reeval_desc_reproducibilities_qd_score",
                f"{prefixe}reeval_desc_reproducibilities_qd_score",
                f"{prefixe}reeval_desc_reproducibilities_qd_score",
            ],
        ]
        rows_columns_metrics_name = [
            [
                f"{prefixe_title}Corrected QD-Score",
                "",
                "",
            ],
            [
                # f"{prefixe_title}Average Reproducibility",
                f"{prefixe_title}Reproducibility-Score",
                "",
                "",
            ],
        ]
        sub_plot_robotics(
            sub_file_name=f"{plot_folder}/{prefixe}robotics",
            robotics_frame=robotics_frame,
            color_frame=color_frame,
            compare_size=compare_size,
            compare_title=compare_title,
            legend_columns=legend_columns,
            legend_bottom=legend_bottom,
            rows_columns_select=rows_columns_select,
            rows_columns_title=rows_columns_title,
            rows_columns_metrics=rows_columns_metrics,
            rows_columns_metrics_name=rows_columns_metrics_name,
            filllines=[],
            vlines=vlines,
        )

        # Second, compute p-value if asked
        if p_values:

            # Reeval QD-Score p-values
            p_values_frame = all_times[
                all_times["env"].isin(
                    [
                        "hexapod_sin_omni_fit0.05_desc0.05_params0.0",
                        "ant_omni",
                        "walker2d_uni",
                    ]
                )
            ]
            p_values_frame = sort_data(
                p_values_frame,
                ["algo", "rep", "eval"],
                baselines_list
                + categories_list
                + baselines_as_list
                + categories_as_list,
            )
            try:
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"{prefixe}reeval_qd_score",
                    dataframe=p_values_frame,
                )
            except Exception:
                print(
                    f"\n!!!WARNING!!! Cannot plot p-values for {prefixe}reeval_qd_score."
                )
                traceback.print_exc()

            # Reproducibility-Score p-values
            p_values_frame = all_reprods[
                all_reprods["env"].isin(
                    [
                        "hexapod_sin_omni_fit0.05_desc0.05_params0.0",
                        "ant_omni",
                        "walker2d_uni",
                    ]
                )
            ]
            p_values_frame = sort_data(
                p_values_frame,
                ["algo", "rep"],
                baselines_list
                + categories_list
                + baselines_as_list
                + categories_as_list,
            )
            try:
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"{prefixe}reeval_desc_reproducibilities_qd_score",
                    dataframe=p_values_frame,
                )
            except Exception:
                print(
                    f"\n!!!WARNING!!! Cannot plot p-values for {prefixe}reeval_desc_reproducibilities_qd_score."
                )
                traceback.print_exc()

    else:
        print("!!!WARNING!!! Missing tasks to plot Robotics.")


def sub_plot_robotics(
    sub_file_name: str,
    robotics_frame: pd.DataFrame,
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
            data_frame=robotics_frame,
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
        print(f"\n!!!WARNING!!! Cannot plot robotics plot.")
        traceback.print_exc()

    # Plot metrics for each size separatly
    try:
        size = 0
        for size in robotics_frame[compare_size].drop_duplicates().values:
            size_robotics_frame = robotics_frame[robotics_frame[compare_size] == size]
            plot_rows_columns_select(
                file_name=sub_file_name + f"_{size}.svg",
                data_frame=size_robotics_frame,
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
        print(f"\n!!!WARNING!!! Cannot plot robotics plot for size {size}.")
        traceback.print_exc()


def plot_robotics_archives(
    plot_folder: str,
    single_compare_size: int,
    env_order: Dict,
    config_frame: pd.DataFrame,
    min_max_frame: pd.DataFrame,
    baselines_list: List,
    categories_list: List,
    baselines_as_list: List,
    categories_as_list: List,
    compare_size: str,
    compare_title: str,
    prefixe: str = "",
    prefixe_title: str = "",
) -> None:

    main_folder_name = f"{plot_folder}_{prefixe}reproducibility_repertoires/"

    if (
        "hexapod_sin_omni_fit0.05_desc0.05_params0.0" in config_frame["env"].values
        and "ant_omni" in config_frame["env"].values
        and "walker2d_uni" in config_frame["env"].values
    ):
        # Keep only those environments
        robotics_config_frame = config_frame[
            config_frame["env"].isin(
                [
                    "hexapod_sin_omni_fit0.05_desc0.05_params0.0",
                    "ant_omni",
                    "walker2d_uni",
                ]
            )
        ]
        robotics_config_frame = sort_data(
            robotics_config_frame,
            ["algo"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Reevaluated repertoire
        try:
            print(f"    Plotting Reeval Archive for {single_compare_size}.")
            sub_plot_archives(
                file_name=f"{plot_folder}/robotics_{prefixe}reeval_archive_{single_compare_size}.png",
                desc_reproducibility_repertoire=False,
                reeval_desc_reproducibility_repertoire=False,
                single_compare_size=single_compare_size,
                env_order=env_order,
                config_frame=robotics_config_frame,
                min_max_frame=min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                main_folder_name=main_folder_name,
                prefixe=prefixe,
                prefixe_title=prefixe_title,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
            traceback.print_exc()

        # Reproducibility repertoire
        try:
            print(f"    Plotting Reproducibility Archive for {single_compare_size}.")
            sub_plot_archives(
                file_name=f"{plot_folder}/robotics_{prefixe}reproducibility_archive_{single_compare_size}.png",
                desc_reproducibility_repertoire=True,
                reeval_desc_reproducibility_repertoire=False,
                single_compare_size=single_compare_size,
                env_order=env_order,
                config_frame=robotics_config_frame,
                min_max_frame=min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                main_folder_name=main_folder_name,
                prefixe=prefixe,
                prefixe_title=prefixe_title,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
            traceback.print_exc()

        # Reeval Reproducibility repertoire
        try:
            print(
                f"    Plotting Reeval Reproducibility Archive for {single_compare_size}."
            )
            sub_plot_archives(
                file_name=f"{plot_folder}/robotics_{prefixe}reeval_reproducibility_archive_{single_compare_size}.png",
                desc_reproducibility_repertoire=False,
                reeval_desc_reproducibility_repertoire=True,
                single_compare_size=single_compare_size,
                env_order=env_order,
                config_frame=robotics_config_frame,
                min_max_frame=min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                main_folder_name=main_folder_name,
                prefixe=prefixe,
                prefixe_title=prefixe_title,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
            traceback.print_exc()
    else:
        print("!!!WARNING!!! Missing tasks to plot Robotics.")


def sub_plot_archives(
    file_name: str,
    single_compare_size: int,
    env_order: Dict,
    desc_reproducibility_repertoire: bool,
    reeval_desc_reproducibility_repertoire: bool,
    config_frame: pd.DataFrame,
    min_max_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    main_folder_name: str,
    prefixe: str = "",
    prefixe_title: str = "",
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

        # Get all the corresponding min and max
        min_fitness = None
        max_fitness = None
        min_fit_var = None
        max_fit_var = None
        min_desc_var = None
        max_desc_var = None
        if min_max_frame is not None:
            env_min_max_frame = min_max_frame[min_max_frame["env"] == env]
            if not env_min_max_frame.empty:
                min_fitness = env_min_max_frame["min_fitness"].values[0]
                max_fitness = env_min_max_frame["max_fitness"].values[0]
                min_fit_var = env_min_max_frame["min_fit_var"].values[0]
                max_fit_var = env_min_max_frame["max_fit_var"].values[0]
                min_desc_var = env_min_max_frame["min_desc_var"].values[0]
                max_desc_var = env_min_max_frame["max_desc_var"].values[0]
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
                if desc_reproducibility_repertoire:
                    algo_name = algo.replace(" ", "_")
                    repertoire_folder = f"{main_folder_name}/{env}_{algo_name}_{single_compare_size}_desc_reproducibilities_repertoire"
                    vmin = 0
                    vmax = 1
                elif reeval_desc_reproducibility_repertoire:
                    algo_name = algo.replace(" ", "_")
                    repertoire_folder = f"{main_folder_name}/{env}_{algo_name}_{single_compare_size}_reeval_desc_reproducibilities_repertoire"
                    vmin = 0
                    vmax = 1
                else:
                    repertoire_folder = get_folder_name(
                        algo_config_frame, f"{prefixe}reeval_repertoire_folder", 0
                    )
                    vmin = min_fitness
                    vmax = max_fitness
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
    plt.tight_layout(h_pad=1.70)
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()
