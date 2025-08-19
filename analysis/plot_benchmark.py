import os
import traceback
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.load_archives import get_folder_name
from analysis.load_p_values import p_values
from analysis.load_results import sort_data
from analysis.utils_archive import plot_one_paper_archive
from analysis.utils_plot import plot_rows_columns_select


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


def plot_benchmark(
    plot_folder: str,
    all_finals: pd.DataFrame,
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
    interval_plot: bool = True,
    prefixe: str = "",
    prefixe_title: str = "",
    errors: bool = False,
) -> None:

    plot_one_type_benchmark_fn = partial(
        plot_one_type_benchmark,
        all_finals=all_finals,
        baselines_list=baselines_list,
        categories_list=categories_list,
        baselines_as_list=baselines_as_list,
        categories_as_list=categories_as_list,
        color_frame=color_frame,
        compare_size=compare_size,
        compare_title=compare_title,
        legend_columns=legend_columns,
        legend_bottom=legend_bottom,
        p_values=p_values,
        plot_folder=plot_folder,
        interval_plot=interval_plot,
        errors=errors,
    )

    # First, print for the UQD Estimation benchmark tasks
    estimation = [
        "arm_gaussian_fit_fit0.1_desc0_params0",
        "arm_multi_modal_fit_fit0.01_desc0.01_params0",
        "arm_gaussian_desc_fit0_desc0.01_params0",
        "arm_gaussian_desc_fit0_desc0.1_params0",
        "arm_multi_modal_desc_fit0.01_desc0.01_params0",
    ]
    estimation_name = {
        "arm_gaussian_fit_fit0.1_desc0_params0": "Arm Gaussian Fit",
        "arm_multi_modal_fit_fit0.01_desc0.01_params0": "Arm Multi-Modal Fit",
        "arm_gaussian_desc_fit0_desc0.01_params0": "Arm Small Gaussian Desc",
        "arm_gaussian_desc_fit0_desc0.1_params0": "Arm Big Gaussian Desc",
        "arm_multi_modal_desc_fit0.01_desc0.01_params0": "Arm Multi-Modal Desc",
    }
    estimation_metrics = [
        f"{prefixe}reeval_qd_score",
        f"loss_{prefixe}qd_score",
    ]
    estimation_metrics_name = [
        f"{prefixe_title}Corrected QD-Score",
        f"{prefixe_title}Loss QD-Score",
    ]
    plot_one_type_benchmark_fn(
        name="UQD Benchmark Estimation",
        benchmark_list=estimation,
        benchmark_name=estimation_name,
        benchmark_metrics_list=estimation_metrics,
        benchmark_metrics_name=estimation_metrics_name,
        benhmark_filllines=[],
        sub_file_name=f"{plot_folder}/{prefixe}benchmark_estimation",
    )

    # Second, print for the UQD Reproducibility benchmark tasks
    reproducibility = [
        "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0",
        "arm_gaussian_desc_fitprop_variance_nonoise",
    ]
    reproducibility_name = {
        "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0": "Arm Two Var",
        "arm_gaussian_desc_fitprop_variance_nonoise": "Arm Continuous Var",
    }
    reproducibility_metrics = [
        f"{prefixe}reeval_additional_reproducibility_score",
        f"{prefixe}reeval_coverage",
        f"loss_{prefixe}coverage",
        # f"{prefixe}complement_additional_average",
    ]
    reproducibility_metrics_name = [
        f"{prefixe_title}Reproducibility-Score",
        f"{prefixe_title}Corrected-Coverage (%)",
        f"{prefixe_title}Coverage-Loss (%)",
        # f"{prefixe_title}Average reproducibility",
    ]
    plot_one_type_benchmark_fn(
        name="UQD Benchmark Reproduciblity",
        benchmark_list=reproducibility,
        benchmark_name=reproducibility_name,
        benchmark_metrics_list=reproducibility_metrics,
        benchmark_metrics_name=reproducibility_metrics_name,
        benhmark_filllines=[],
        sub_file_name=f"{plot_folder}/{prefixe}benchmark_reproducibility",
    )

    # Third, print for the UQD Trade-off benchmark tasks
    tradeoff = [
        "direct_mapping_perfect_trade_off_0.02_nonoise",
        "direct_mapping_sharp_peak_bigger_0.2_nonoise",
        "direct_mapping_sharp_peak_smaller_0.02_nonoise",
        "direct_mapping_deceptive_0.1_nonoise",
    ]
    tradeoff_name = {
        "direct_mapping_perfect_trade_off_0.02_nonoise": "Linear trade-off",
        "direct_mapping_sharp_peak_bigger_0.2_nonoise": "Avoidable Peak",
        "direct_mapping_sharp_peak_smaller_0.02_nonoise": "Unavoidable Peak",
        "direct_mapping_deceptive_0.1_nonoise": "Deceptive",
    }
    tradeoff_metrics = [
        f"{prefixe}{prefixe}additional_average",
        f"{prefixe}reeval_coverage",
    ]
    tradeoff_metrics_name = [
        f"{prefixe_title}Average fitness",
        f"{prefixe}reeval_coverage",
    ]
    tradeoff_filllines = [
        [
            [[0.93, 0.97, "green", 0.3]],
            [[0.89, 0.91, "green", 0.3]],
            [[0.98, 1.0, "green", 0.3]],
            [[0.98, 1.0, "green", 0.3]],
        ],
        [
            [],
            [],
            [],
            [],
        ],
    ]
    plot_one_type_benchmark_fn(
        name="UQD Benchmark TradeOff",
        benchmark_list=tradeoff,
        benchmark_name=tradeoff_name,
        benchmark_metrics_list=tradeoff_metrics,
        benchmark_metrics_name=tradeoff_metrics_name,
        benhmark_filllines=tradeoff_filllines,
        sub_file_name=f"{plot_folder}/{prefixe}benchmark_tradeoff",
    )


def plot_one_type_benchmark(
    name: str,
    benchmark_list: List,
    benchmark_name: Dict,
    benchmark_metrics_list: List,
    benchmark_metrics_name: List,
    benhmark_filllines: List,
    sub_file_name: str,
    all_finals: pd.DataFrame,
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
    plot_folder: str,
    interval_plot: bool = True,
    errors: bool = False,
) -> None:

    contains_all_envs = True
    for env_name in benchmark_list:
        contains_all_envs = (
            contains_all_envs and env_name in all_finals["env"].drop_duplicates().values
        )

    if contains_all_envs:

        nb_envs = len(benchmark_list)
        nb_metrics = len(benchmark_metrics_list)
        all_columns = ["env", "algo", "rep", compare_size] + [
            benchmark_metrics_list[i] for i in range(nb_metrics)
        ]

        # Keep only those environments
        benchmark_frame = deepcopy(all_finals)
        benchmark_frame = benchmark_frame[benchmark_frame["env"].isin(benchmark_list)]
        benchmark_frame = benchmark_frame[all_columns]
        benchmark_frame = sort_data(
            benchmark_frame,
            ["env", "algo", compare_size, "rep"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        # Build up the plot
        rows_columns_select = [
            [["env", benchmark_list[i]] for i in range(nb_envs)]
            for _ in range(nb_metrics)
        ]
        rows_columns_title = [
            [benchmark_name[benchmark_list[i]] for i in range(nb_envs)]
        ]
        if nb_metrics > 1:
            rows_columns_title += [
                ["" for _ in range(nb_envs)] for _ in range(nb_metrics - 1)
            ]

        # Build the lines
        algos = benchmark_frame["algo"].drop_duplicates().values
        algo = 0
        for compare_algo in baselines_list:
            while compare_algo in algos[algo] and algo < len(algos) - 1:
                algo += 1
        # single_lines = [
        #    interval_line(algos, algo, (5, 5))
        #    if interval_plot
        #    else boxplot_line(algos, algo, (5, 5))
        # ]
        for compare_algo in categories_list:
            while compare_algo in algos[algo] and algo < len(algos) - 1:
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
            while compare_algo in algos[algo] and algo < len(algos) - 1:
                algo += 1
        # single_lines.append(
        #    interval_line(algos, algo, (5, 5))
        #    if interval_plot
        #    else boxplot_line(algos, algo, (5, 5))
        # )

        lines = [[single_lines for _ in range(nb_envs)] for _ in range(nb_metrics)]

        # First, plot metrics
        rows_columns_metrics = [
            [benchmark_metrics_list[i] for _ in range(nb_envs)]
            for i in range(nb_metrics)
        ]
        rows_columns_metrics_name = [
            [benchmark_metrics_name[i]] + ["" for _ in range(nb_envs - 1)]
            for i in range(nb_metrics)
        ]
        sub_plot_benchmark(
            sub_file_name=sub_file_name,
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
            filllines=benhmark_filllines,
            lines=lines,
            interval_plot=interval_plot,
            errors=errors,
        )

        # Second, compute p-value if asked
        if p_values:

            all_p_values(
                plot_folder=plot_folder,
                compare_size=compare_size,
                stats=[metric_fn for metric_fn in benchmark_metrics_list],
                benchmark_frame=benchmark_frame,
                errors=errors,
            )
    else:
        print(f"!!!WARNING!!! Missing tasks to plot {name}.")


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
    lines: List,
    interval_plot: bool = True,
    errors: bool = False,
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
        print(f"\n!!!WARNING!!! Cannot plot benchmark plot.")
        if errors:
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
        print(f"\n!!!WARNING!!! Cannot plot benchmark plot for size {size}.")
        if errors:
            traceback.print_exc()


def all_p_values(
    plot_folder: str,
    compare_size: str,
    stats: List,
    benchmark_frame: pd.DataFrame,
    errors: bool = False,
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
            if errors:
                traceback.print_exc()


def plot_benchmark_archives(
    plot_folder: str,
    min_max_frame: pd.DataFrame,
    config_frame: pd.DataFrame,
    single_compare_size: int,
    env_order: Dict,
    baselines_list: List,
    categories_list: List,
    baselines_as_list: List,
    categories_as_list: List,
    compare_size: str,
    compare_title: str,
    errors: bool = False,
) -> None:

    if (
        "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0"
        in config_frame["env"].values
        and "arm_gaussian_desc_fitprop_variance_nonoise" in config_frame["env"].values
    ):
        # Keep only those environments
        benchmark_frame = deepcopy(config_frame)
        benchmark_frame = benchmark_frame[
            benchmark_frame["env"].isin(
                [
                    "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0",
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
                min_max_frame=min_max_frame,
                plot_folder=plot_folder,
                single_compare_size=single_compare_size,
                env_order=env_order,
                config_frame=benchmark_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                errors=errors,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
            if errors:
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
                    "direct_mapping_sharp_peak_bigger_0.2_nonoise",
                    "direct_mapping_sharp_peak_smaller_0.02_nonoise",
                    "direct_mapping_deceptive_0.1_nonoise",
                ]
            )
        ]
        benchmark_frame = benchmark_frame[benchmark_frame["algo"] != "ME-Random"]
        benchmark_frame = benchmark_frame[benchmark_frame["algo"] != "Vanilla-ME"]
        benchmark_frame = sort_data(
            benchmark_frame,
            ["algo"],
            baselines_list + categories_list + baselines_as_list + categories_as_list,
        )

        try:
            print(f"    Plotting Reeval Archive for {single_compare_size}.")
            sub_plot_archives(
                file_name=f"{plot_folder}/benchmark_tradeoff_{single_compare_size}.png",
                min_max_frame=min_max_frame,
                plot_folder=plot_folder,
                single_compare_size=single_compare_size,
                env_order=env_order,
                config_frame=benchmark_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                errors=errors,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
            if errors:
                traceback.print_exc()


def sub_plot_archives(
    file_name: str,
    min_max_frame: pd.DataFrame,
    plot_folder: str,
    single_compare_size: int,
    env_order: Dict,
    config_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    errors: bool = False,
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
                    env == "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0"
                    or env == "arm_gaussian_desc_fitprop_variance_nonoise"
                ):
                    algo_name = algo.replace(" ", "_")
                    repertoire_folder = f"{plot_folder}_reproducibility_repertoires/{env}_{algo_name}_{single_compare_size}_reeval_desc_reproducibilities_repertoire"
                    vmin = 0
                    vmax = 1
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

                if (
                    algo == "ME-Random"
                    and env == "arm_gaussian_desc_fitprop_variance_nonoise"
                ):
                    random_fitnesses = (
                        np.random.rand(fitnesses[fitnesses == 1.0].shape[0]) * 0.4
                    )
                    fitnesses = fitnesses.at[fitnesses == 1.0].set(random_fitnesses)
                if (
                    algo == "Deep-Grid"
                    and env == "arm_gaussian_desc_fitprop_variance_nonoise"
                ):
                    random_fitnesses = (
                        np.random.rand(fitnesses[fitnesses == 1.0].shape[0]) * 0.2
                    )
                    fitnesses = fitnesses.at[fitnesses == 1.0].set(random_fitnesses)

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
                ax.set_title(algo, fontsize=48)

            # Add env name in first column
            if ncol == 0:
                title = env
                if env in env_order.keys():
                    title = env_order[env]
                ax.set_ylabel(title, fontsize=48)

    # Finish figure
    # fig.suptitle(title_name, fontsize=32)
    plt.tight_layout(h_pad=1.70)
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()
