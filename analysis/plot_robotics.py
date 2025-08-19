import os
import traceback
from typing import Any, Dict, List

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

from analysis.load_archives import get_folder_name
from analysis.load_p_values import p_values as p_values_fn
from analysis.load_results import sort_data
from analysis.utils_archive import plot_one_paper_archive
from analysis.utils_plot import plot_rows_columns_select

ENV_ROBOTICS = [
    "hexapod_sin_omni_fit0.05_desc0.05_params0.0",
    "ant_omni",
    "walker2d_uni",
]
NAMES = {
    "hexapod_sin_omni_fit0.05_desc0.05_params0.0": "Hexapod",
    "ant_omni": "Ant",
    "walker2d_uni": "Walker",
}


def boxplot_line(algos: List, algo: float, dash: Any) -> List:
    single_line = [
        (algo / len(algos) - 0.5) * 1.0,  # position (multiply by 1.0, width of boxplot)
        "k",  # color
        dash,  # dash
    ]
    return single_line


def interval_line(algos: List, algo: float, dash: Any) -> List:
    single_line = [
        (-0.2 + len(algos) - algo) * (1.6 * 0.6),  # position (0.6 the h of lines)
        "k",  # color
        dash,  # dash
    ]
    return single_line


def plot_robotics(
    plot_folder: str,
    all_finals: pd.DataFrame,
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
    interval_plot: bool = True,
    errors: bool = False,
) -> None:

    contains_all_envs = True
    for env_name in ENV_ROBOTICS:
        contains_all_envs = contains_all_envs and env_name in all_finals["env"].values
        contains_all_envs = contains_all_envs and env_name in all_reprods["env"].values

    if contains_all_envs:

        # Merge all_finals and all_reprods
        robotics_frame = pd.concat([all_finals, all_reprods], ignore_index=True)

        # Keep only those environments
        robotics_frame = robotics_frame[robotics_frame["env"].isin(ENV_ROBOTICS)]
        robotics_frame = sort_data(
            robotics_frame,
            ["algo", "rep", "eval"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Build the lines
        algos = robotics_frame["algo"].drop_duplicates().values
        algo = 0
        for compare_algo in baselines_list:
            while algo < len(algos) and compare_algo in algos[algo]:
                algo += 1
        # single_lines = [
        #    interval_line(algos, algo, (5, 5))
        #    if interval_plot
        #    else boxplot_line(algos, algo, (5, 5))
        # ]
        for compare_algo in categories_list:
            while algo < len(algos) and compare_algo in algos[algo]:
                algo += 1
        # single_lines.append(
        #    interval_line(algos, algo, "")
        #    if interval_plot
        #    else boxplot_line(algos, algo, "")
        # )
        single_lines = [
            interval_line(algos, algo, (5, 5))
            if interval_plot
            else boxplot_line(algos, algo, (5, 5))
        ]
        for compare_algo in baselines_as_list:
            while algo < len(algos) and compare_algo in algos[algo]:
                algo += 1
        # single_lines.append(
        #    interval_line(algos, algo, (5, 5))
        #    if interval_plot
        #    else boxplot_line(algos, algo, (5, 5))
        # )

        lines = [
            [single_lines, single_lines, single_lines, single_lines],
            [single_lines, single_lines, single_lines, single_lines],
        ]

        # First, plot QD-Score and Reproducibility-Score
        rows_columns_select = [
            [["env", env_name] for env_name in ENV_ROBOTICS],
            [["env", env_name] for env_name in ENV_ROBOTICS],
        ]
        rows_columns_title = [
            [NAMES[env_name] for env_name in ENV_ROBOTICS],
            ["" for env_name in ENV_ROBOTICS],
        ]
        rows_columns_metrics = [
            [f"{prefixe}reeval_qd_score" for env_name in ENV_ROBOTICS],
            [
                f"{prefixe}reeval_avg_desc_reproducibilities_qd_score"
                for env_name in ENV_ROBOTICS
            ],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}Corrected QD-Score"]
            + ["" for _ in range(len(ENV_ROBOTICS) - 1)],
            [f"{prefixe_title}Average Reproducibility"]
            + ["" for _ in range(len(ENV_ROBOTICS) - 1)],
        ]
        sub_plot_robotics(
            sub_file_name=f"{plot_folder}/{prefixe}robotics_summary",
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
            lines=lines,
            interval_plot=interval_plot,
            errors=errors,
        )

        # Second, plot QD-Score and Coverage
        rows_columns_select = [
            [["env", env_name] for env_name in ENV_ROBOTICS],
            [["env", env_name] for env_name in ENV_ROBOTICS],
        ]
        rows_columns_title = [
            [NAMES[env_name] for env_name in ENV_ROBOTICS],
            ["" for env_name in ENV_ROBOTICS],
        ]
        rows_columns_metrics = [
            [f"{prefixe}reeval_qd_score" for env_name in ENV_ROBOTICS],
            [f"{prefixe}reeval_coverage" for env_name in ENV_ROBOTICS],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}Corrected QD-Score"]
            + ["" for _ in range(len(ENV_ROBOTICS) - 1)],
            [f"{prefixe_title}Corrected Coverage"]
            + ["" for _ in range(len(ENV_ROBOTICS) - 1)],
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
            lines=lines,
            interval_plot=interval_plot,
            errors=errors,
        )

        # Third, plot Losses
        rows_columns_select = [
            [["env", env_name] for env_name in ENV_ROBOTICS],
            [["env", env_name] for env_name in ENV_ROBOTICS],
        ]
        rows_columns_title = [
            [NAMES[env_name] for env_name in ENV_ROBOTICS],
            ["" for env_name in ENV_ROBOTICS],
        ]
        rows_columns_metrics = [
            [f"loss_{prefixe}qd_score" for env_name in ENV_ROBOTICS],
            [f"loss_{prefixe}coverage" for env_name in ENV_ROBOTICS],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}QD-Score-Loss (%)"]
            + ["" for _ in range(len(ENV_ROBOTICS) - 1)],
            [f"{prefixe_title}Coverage-Loss (%)"]
            + ["" for _ in range(len(ENV_ROBOTICS) - 1)],
        ]
        sub_plot_robotics(
            sub_file_name=f"{plot_folder}/{prefixe}robotics_loss",
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
            lines=lines,
            interval_plot=interval_plot,
            errors=errors,
        )

        # Fourth, plot Reproducibility-Score
        rows_columns_select = [
            [["env", env_name] for env_name in ENV_ROBOTICS],
        ]
        rows_columns_title = [
            [NAMES[env_name] for env_name in ENV_ROBOTICS],
        ]

        rows_columns_metrics = [
            [
                f"{prefixe}avg_desc_reproducibilities_qd_score"
                for env_name in ENV_ROBOTICS
            ],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}Average Reproducibility"]
            + ["" for _ in range(len(ENV_ROBOTICS) - 1)],
        ]
        sub_plot_robotics(
            sub_file_name=f"{plot_folder}/{prefixe}robotics_reproducibility",
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
            lines=lines,
            interval_plot=interval_plot,
            errors=errors,
        )

        # Fifth, plot Reeval-Reproducibility-Score
        rows_columns_select = [
            [["env", env_name] for env_name in ENV_ROBOTICS],
        ]
        rows_columns_title = [
            [NAMES[env_name] for env_name in ENV_ROBOTICS],
        ]

        rows_columns_metrics = [
            [
                f"{prefixe}reeval_avg_desc_reproducibilities_qd_score"
                for env_name in ENV_ROBOTICS
            ],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}Average Reproducibility"]
            + ["" for _ in range(len(ENV_ROBOTICS) - 1)],
        ]
        sub_plot_robotics(
            sub_file_name=f"{plot_folder}/{prefixe}robotics_reeval_reproducibility",
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
            lines=lines,
            interval_plot=interval_plot,
            errors=errors,
        )

        # Sixth, plot Reeval-Fitness-Reproducibility-Score
        rows_columns_metrics = [
            [
                f"{prefixe}reeval_avg_fit_reproducibilities_qd_score"
                for env_name in ENV_ROBOTICS
            ],
        ]
        rows_columns_metrics_name = [
            [f"{prefixe_title}Average Fitness-Reproducibility"]
            + ["" for _ in range(len(ENV_ROBOTICS) - 1)],
        ]
        sub_plot_robotics(
            sub_file_name=f"{plot_folder}/{prefixe}robotics_reeval_fitness_reproducibility",
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
            lines=lines,
            interval_plot=interval_plot,
            errors=errors,
        )

        # Seventh, compute p-value if asked
        if p_values:

            # Corrected and Loss p-values
            p_values_frame = all_finals[all_finals["env"].isin(ENV_ROBOTICS)]
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
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"{prefixe}reeval_coverage",
                    dataframe=p_values_frame,
                )
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"loss_{prefixe}qd_score",
                    dataframe=p_values_frame,
                )
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"loss_{prefixe}coverage",
                    dataframe=p_values_frame,
                )
            except Exception:
                print(
                    f"\n!!!WARNING!!! Cannot plot p-values for Corrected and Loss metrics."
                )
                if errors:
                    traceback.print_exc()

            # Reproducibility-Score p-values
            p_values_frame = all_reprods[all_reprods["env"].isin(ENV_ROBOTICS)]
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
                    stat=f"{prefixe}avg_desc_reproducibilities_qd_score",
                    dataframe=p_values_frame,
                )
                p_values_fn(
                    plot_folder=plot_folder,
                    compare_size=compare_size,
                    stat=f"{prefixe}reeval_avg_desc_reproducibilities_qd_score",
                    dataframe=p_values_frame,
                )
            except Exception:
                print(
                    f"\n!!!WARNING!!! Cannot plot p-values for {prefixe}reeval_avg_desc_reproducibilities_qd_score."
                )
                if errors:
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
    lines: List,
    interval_plot: bool = True,
    errors: bool = False,
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
            vlines=[] if interval_plot else lines,
            hlines=lines if interval_plot else [],
            filllines=filllines,
            x_front=[],
            y_front=[],
            box_plot=not (interval_plot),
            scatter_plot=False,
            interval_plot=False,
            intervalbox_plot=interval_plot,
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
        if errors:
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
                vlines=[] if interval_plot else lines,
                hlines=lines if interval_plot else [],
                filllines=filllines,
                x_front=[],
                y_front=[],
                box_plot=not (interval_plot),
                scatter_plot=False,
                interval_plot=False,
                intervalbox_plot=interval_plot,
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
        if errors:
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
    errors: bool = False,
) -> None:

    main_folder_name = f"{plot_folder}_{prefixe}reproducibility_repertoires/"

    contains_all_envs = True
    for env_name in ENV_ROBOTICS:
        contains_all_envs = contains_all_envs and env_name in config_frame["env"].values

    if contains_all_envs:

        # Keep only those environments
        robotics_config_frame = config_frame[config_frame["env"].isin(ENV_ROBOTICS)]
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
                errors=errors,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
            if errors:
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
                errors=errors,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
            if errors:
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
                errors=errors,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
            if errors:
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
    errors: bool = False,
) -> None:

    # Get the envs and algos
    envs = config_frame["env"].drop_duplicates().values
    algos = config_frame["algo"].drop_duplicates().values

    # Create the figure
    ncols = len(algos)
    nrows = len(envs)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 6))

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
        min_reeval_fitness = None
        max_reeval_fitness = None
        if min_max_frame is not None:
            env_min_max_frame = min_max_frame[min_max_frame["env"] == env]
            if not env_min_max_frame.empty:
                min_reeval_fitness = float(
                    env_min_max_frame["min_reeval_fitness"].values[0]
                )
                max_reeval_fitness = float(
                    env_min_max_frame["max_reeval_fitness"].values[0]
                )
        if "_" in env_config_frame["min_bd"][0]:
            min_bd_list = str(env_config_frame["min_bd"][0]).split("_")
        else:
            min_bd_list = str(env_config_frame["min_bd"][0])[1:-1].split(" ")
        if "" in min_bd_list:
            min_bd_list.remove("")
        if "_" in env_config_frame["max_bd"][0]:
            max_bd_list = str(env_config_frame["max_bd"][0]).split("_")
        else:
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

            # Get the corresponding axis
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
                    vmin = 0.0
                    vmax = 1.0
                elif reeval_desc_reproducibility_repertoire:
                    algo_name = algo.replace(" ", "_")
                    repertoire_folder = f"{main_folder_name}/{env}_{algo_name}_{single_compare_size}_reeval_desc_reproducibilities_repertoire"
                    vmin = 0.0
                    vmax = 1.0
                else:
                    repertoire_folder = get_folder_name(
                        algo_config_frame, f"{prefixe}reeval_repertoire_folder", 0
                    )
                    vmin = min_reeval_fitness  # type: ignore
                    vmax = max_reeval_fitness  # type: ignore
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
                if errors:
                    traceback.print_exc()

            # Add algo name in first line
            if nrow == 0:
                ax.set_title(algo, fontsize=42)

            # Add env name in first column
            if ncol == 0:
                title = env
                if env in env_order.keys():
                    title = env_order[env]
                ax.set_ylabel(title, fontsize=42)

    # Finish figure
    plt.tight_layout(h_pad=1.30)
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()


def plot_robotics_per_approach_archives(
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
    errors: bool = False,
) -> None:

    contains_all_envs = True
    for env_name in ENV_ROBOTICS:
        contains_all_envs = contains_all_envs and env_name in config_frame["env"].values

    if contains_all_envs:

        # Keep only those environments
        robotics_config_frame = config_frame[config_frame["env"].isin(ENV_ROBOTICS)]
        robotics_config_frame = sort_data(
            robotics_config_frame,
            ["algo"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Reevaluated repertoire
        try:
            print(f"    Plotting Reeval Archive for {single_compare_size}.")
            sub_plot_per_approach_archives(
                partial_file_name=f"{plot_folder}/robotics_{prefixe}reeval_archive_{single_compare_size}",
                single_compare_size=single_compare_size,
                env_order=env_order,
                config_frame=robotics_config_frame,
                min_max_frame=min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                prefixe=prefixe,
                prefixe_title=prefixe_title,
                errors=errors,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
            if errors:
                traceback.print_exc()

    else:
        print("!!!WARNING!!! Missing tasks to plot Robotics.")


def sub_plot_per_approach_archives(
    partial_file_name: str,
    single_compare_size: int,
    env_order: Dict,
    config_frame: pd.DataFrame,
    min_max_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    prefixe: str = "",
    prefixe_title: str = "",
    errors: bool = False,
) -> None:

    # Get the envs and algos
    envs = config_frame["env"].drop_duplicates().values
    algos = config_frame["algo"].drop_duplicates().values

    # For each environment
    for env_idx in range(len(envs)):

        env = envs[env_idx]
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
        min_reeval_fitness = None
        max_reeval_fitness = None
        if min_max_frame is not None:
            env_min_max_frame = min_max_frame[min_max_frame["env"] == env]
            if not env_min_max_frame.empty:
                min_reeval_fitness = float(
                    env_min_max_frame["min_reeval_fitness"].values[0]
                )
                max_reeval_fitness = float(
                    env_min_max_frame["max_reeval_fitness"].values[0]
                )
        if "_" in env_config_frame["min_bd"][0]:
            min_bd_list = str(env_config_frame["min_bd"][0]).split("_")
        else:
            min_bd_list = str(env_config_frame["min_bd"][0])[1:-1].split(" ")
        if "" in min_bd_list:
            min_bd_list.remove("")
        if "_" in env_config_frame["max_bd"][0]:
            max_bd_list = str(env_config_frame["max_bd"][0]).split("_")
        else:
            max_bd_list = str(env_config_frame["max_bd"][0])[1:-1].split(" ")
        if "" in max_bd_list:
            max_bd_list.remove("")
        min_bd = [float(bd) for bd in min_bd_list]
        max_bd = [float(bd) for bd in max_bd_list]

        # For each algo for this env and size
        for algo_idx in range(len(algos)):

            algo = algos[algo_idx]
            algo_config_frame = env_config_frame[
                env_config_frame["algo"] == algo
            ].reset_index(drop=True)

            # Create the figure
            ncols = algo_config_frame.shape[0]
            nrows = 1
            fig, axes = plt.subplots(
                nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 6)
            )

            # For each rep
            for ncol in range(ncols):

                # Get the corresponding axis
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

                # Taking the archive for this rep
                try:
                    repertoire_folder = get_folder_name(
                        algo_config_frame, f"{prefixe}reeval_repertoire_folder", ncol
                    )
                    vmin = min_reeval_fitness  # type: ignore
                    vmax = max_reeval_fitness  # type: ignore

                    fitnesses = jnp.load(
                        os.path.join(repertoire_folder, "fitnesses.npy")
                    )
                    descriptors = jnp.load(
                        os.path.join(repertoire_folder, "descriptors.npy")
                    )
                    centroids = jnp.load(
                        os.path.join(repertoire_folder, "centroids.npy")
                    )

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
                    if errors:
                        traceback.print_exc()

                # Add env and algo name in the first column
                if ncol == 0:
                    title = env
                    if env in env_order.keys():
                        title = env_order[env]
                    ax.set_ylabel(f"{title} - {algo}", fontsize=42)

            # Finish figure
            file_name = f"{partial_file_name}_{env}_{algo}.png"
            plt.tight_layout(h_pad=1.30)
            plt.savefig(file_name, bbox_inches="tight")
            plt.close()
