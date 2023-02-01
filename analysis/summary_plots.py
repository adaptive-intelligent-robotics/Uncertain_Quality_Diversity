from math import ceil
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator
from seaborn.relational import _LinePlotter

from analysis.constants import BASELINES, CATEGORIES, ENV_NAME_DIFFICULTY, ENV_NO_VAR
from analysis.plots import first_second_third_quartile, sub_plot
from analysis.utils import sort_data


def pareto_plot(
    plot_folder: str,
    num_reevals: float,
    compare_size: str,
    times_data: pd.DataFrame,
    prefixe: str = "",
    prefixe_title: str = "",
    legend_columns: int = 2,
    suffixe: str = "",
    max_qd_score_dict: Dict = None,
    min_time_dict: Dict = None,
    max_time_dict: Dict = None,
    pareto_column: bool = False,
) -> None:

    # Get all environments
    env_names = times_data["env"].drop_duplicates().values
    if pareto_column:
        nrows = 4
        ncols = 1
    else:
        nrows = (
            len(env_names) // 2
            if (len(env_names) > 2 and len(env_names) % 2 == 0)
            else 1
        )
        ncols = max(2, len(env_names) // nrows)

    # Choose legend display
    num_algos = times_data["algo"].drop_duplicates().values.shape[0]
    # total_legend = (
    #    num_algos + times_data[compare_size].drop_duplicates().values.shape[0]
    # )
    # legend_columns = legend_columns * ncols
    # if total_legend % 4 == 0 and total_legend // 2 < legend_columns:
    #    legend_columns = min(total_legend // 4, legend_columns * ncols)
    # elif total_legend % 2 == 0 and total_legend // 2 < legend_columns:
    #    legend_columns = min(total_legend // 2, legend_columns * ncols)

    palette = sns.color_palette("Paired", num_algos)
    sns.set_palette(palette)

    # Order by difficulty
    final_envs: List[str] = []
    title_envs: List[str] = []
    for env_name in ENV_NAME_DIFFICULTY.keys():
        if env_name in env_names:
            final_envs.append(env_name)
            title_envs.append(ENV_NAME_DIFFICULTY[env_name])
    for env_name in env_names:
        if env_name not in final_envs:
            final_envs.append(env_name)
            title_envs.append(env_name)

    # Create figure
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8 * ncols, (5 if pareto_column else 6) * nrows),
    )
    _LinePlotter.aggregate = first_second_third_quartile

    # One column per env
    for env_idx in range(len(final_envs)):
        env_times = times_data[times_data["env"] == final_envs[env_idx]]

        # Create pareto frame
        pareto_frame = pd.DataFrame(
            columns=[
                "algo",
                compare_size,
                f"{prefixe}time",
                f"{prefixe}reeval_qd_score",
            ]
        )
        for algo_batch in env_times["algo_batch"].drop_duplicates().values:

            algo_batch_times = env_times[
                env_times["algo_batch"] == algo_batch
            ].reset_index(drop=True)
            pareto_line: Dict[str, List[float]] = {}

            # Modify the names for plot
            name = algo_batch_times["algo"][0]
            pareto_line["algo"] = [name]

            # Get values
            pareto_line[compare_size] = [algo_batch_times[compare_size][0]]
            pareto_line[f"{prefixe}time"] = [
                np.median(algo_batch_times[f"{prefixe}time"])
            ]
            pareto_line[f"{prefixe}reeval_qd_score"] = [
                np.median(algo_batch_times[f"{prefixe}reeval_qd_score"])
            ]

            # Add to frame
            pareto_frame = pd.concat(
                [pareto_frame, pd.DataFrame.from_dict(pareto_line)],
                ignore_index=True,
            )
        pareto_frame = sort_data(
            pareto_frame,
            [
                "algo",
                f"{prefixe}time",
                f"{prefixe}reeval_qd_score",
                compare_size,
            ],
        )

        # Normalise QD-Score per task
        if max_qd_score_dict is None:
            max_qd_score = pareto_frame[f"{prefixe}reeval_qd_score"].max()
        else:
            max_qd_score = max_qd_score_dict[final_envs[env_idx]]
        pareto_frame[f"{prefixe}reeval_qd_score"] = (
            pareto_frame[f"{prefixe}reeval_qd_score"]
        ).div(max_qd_score / 100)

        # Change time in minutes
        pareto_frame[f"{prefixe}time"] = pareto_frame[f"{prefixe}time"].div(60)

        # Set up scatter point size
        def powspace(start, stop, power, num):
            start = np.power(start, 1 / float(power))
            stop = np.power(stop, 1 / float(power))
            return np.power(np.linspace(start, stop, num=num), power)

        sizes = pareto_frame[compare_size].drop_duplicates().values
        sizes.sort()
        sizes = np.append(sizes, sizes[0] // 2)
        sizes.sort()
        # dot_sizes = np.linspace(10, 3000, num=len(sizes))
        dot_sizes = powspace(10, 2000, 4, len(sizes))
        scatter_sizes = dict(zip(sizes, dot_sizes))
        # all_markers = [(3, 1, 0), (2, 1, 0)]
        # for i in range(4, len(sizes) + 2):
        # all_markers.append((i, 1, 0))
        # markers = dict(zip(sizes, all_markers))
        # markers = ["o", "s", "P", "X"]
        # markers = ["X", "X", "X", "X"]

        # Set up min and max y-axis
        if min_time_dict is None:
            min_time = 0.6 * pareto_frame[f"{prefixe}time"].min()
        else:
            min_time = 0.6 * min_time_dict[final_envs[env_idx]] / 60
        if max_time_dict is None:
            max_time = 1.1 * pareto_frame[f"{prefixe}time"].max()
        else:
            max_time = 1.1 * max_time_dict[final_envs[env_idx]] / 60

        # Get axis to use
        if nrows == 1 or ncols == 1:
            env_ax = ax[env_idx]
        else:
            env_ax = ax[env_idx // ncols, env_idx % ncols]

        # Add pareto front
        front_bool = np.ones(pareto_frame.shape[0])
        for indiv_1 in range(pareto_frame.shape[0]):
            for indiv_2 in range(pareto_frame.shape[0]):

                # If indiv_2 dominates indiv_1
                if (
                    pareto_frame[f"{prefixe}reeval_qd_score"].values[indiv_1]
                    <= pareto_frame[f"{prefixe}reeval_qd_score"].values[indiv_2]
                    and pareto_frame[f"{prefixe}time"].values[indiv_1]
                    > pareto_frame[f"{prefixe}time"].values[indiv_2]
                ) or (
                    pareto_frame[f"{prefixe}reeval_qd_score"].values[indiv_1]
                    < pareto_frame[f"{prefixe}reeval_qd_score"].values[indiv_2]
                    and pareto_frame[f"{prefixe}time"].values[indiv_1]
                    >= pareto_frame[f"{prefixe}time"].values[indiv_2]
                ):
                    front_bool[indiv_1] = 0
                    break
        x_front: List[float] = []
        y_front: List[float] = []
        for indiv in range(pareto_frame.shape[0]):
            if front_bool[indiv]:
                x_front.append(pareto_frame[f"{prefixe}reeval_qd_score"].values[indiv])
                y_front.append(pareto_frame[f"{prefixe}time"].values[indiv])
        x_front = np.array(x_front)
        y_front = np.array(y_front)
        indexes = np.argsort(x_front)
        x_front = x_front[indexes]
        y_front = y_front[indexes]

        # sns.lineplot(x=x_front, y=y_front, ax=env_ax)
        env_ax.plot(
            x_front,
            y_front,
            color="b",
            linestyle="--",
            linewidth=2,
            zorder=0,
            marker="o",
        )

        # Add scatter
        plot_frame = pareto_frame.replace("Vanilla-MAP-Elites", "MAP-Elites")
        env_ax = sns.scatterplot(
            x=f"{prefixe}reeval_qd_score",
            y=f"{prefixe}time",
            # s=800,
            data=plot_frame,
            hue="algo",
            # style=compare_size,
            size=compare_size,
            sizes=scatter_sizes,
            edgecolor="black",
            linewidth=2,
            alpha=0.6,
            # markers=markers,
            ax=env_ax,
        )
        handles, labels = env_ax.get_legend_handles_labels()
        env_ax.legend_.remove()
        env_ax.set_title(title_envs[env_idx])
        env_ax.set_ylim(min_time, max_time)

        # Remove top and right spines
        env_ax.spines["top"].set_visible(False)
        env_ax.spines["right"].set_visible(False)
        env_ax.set_xlabel(
            "Corrected QD-Score (%-max-value)", labelpad=7
        )  # \n (Quality-Diversity-Reproducibility performance)")
        if env_idx == 0 or env_idx % ncols == 0:
            env_ax.set_ylabel("Time to convergence (mins)", labelpad=7)
        else:
            env_ax.set_ylabel(None)
        if env_idx // ncols < nrows - 1:
            env_ax.set_xlabel(None)

    # Add legend below graph
    legend_bottom = 0.03 + 0.03 * ceil(len(labels) / legend_columns)
    if pareto_column:
        legend_bottom = 0.01 + 0.02 * ceil(len(labels) / legend_columns)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.17)
    fig.subplots_adjust(bottom=legend_bottom)
    handles = handles[-len(sizes) + 1 :] + handles[1 : -len(sizes)]
    labels = labels[-len(sizes) + 1 :] + labels[1 : -len(sizes)]
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        frameon=False,
        ncol=legend_columns,
    )

    # Save figure
    plt.savefig(f"{plot_folder}/{num_reevals}reevals_{prefixe}pareto_time{suffixe}.pdf")
    plt.close()


def pareto_categories(
    plot_folder: str,
    num_reevals: float,
    compare_size: str,
    times_data: pd.DataFrame,
    prefixe: str = "",
    prefixe_title: str = "",
    legend_columns: int = 2,
    pareto_column: bool = False,
) -> None:

    # List existing categories in the dataframes
    baselines: List[str] = []
    bas_data = times_data.copy()
    for baseline in BASELINES:
        if not bas_data[bas_data["algo"].str.contains(baseline)].empty:
            baselines.append(baseline)
            bas_data = bas_data[~bas_data["algo"].str.contains(baseline)]
    categories: List[str] = []
    cat_data = times_data.copy()
    for category in CATEGORIES:
        if not cat_data[cat_data["algo"].str.contains(category)].empty:
            categories.append(category)
            cat_data = cat_data[~cat_data["algo"].str.contains(category)]

    # Extract baseline datas
    baseline_times_data = times_data[times_data["algo"].str.contains(BASELINES[0])]
    for baseline in BASELINES[1:]:
        baseline_times_data = pd.concat(
            [
                baseline_times_data,
                times_data[times_data["algo"].str.contains(baseline)],
            ],
            ignore_index=True,
        )

    # Get max and min qd score and time
    max_qd_score_dict: Dict = {}
    min_qd_score_dict: Dict = {}
    min_time_dict: Dict = {}
    max_time_dict: Dict = {}
    for env_name in times_data["env"].drop_duplicates().values:
        max_qd_score_dict[env_name] = times_data[times_data["env"] == env_name][
            f"{prefixe}reeval_qd_score"
        ].max()
        min_qd_score_dict[env_name] = times_data[times_data["env"] == env_name][
            f"{prefixe}reeval_qd_score"
        ].min()
        min_time_dict[env_name] = times_data[times_data["env"] == env_name][
            f"{prefixe}time"
        ].min()
        max_time_dict[env_name] = times_data[times_data["env"] == env_name][
            f"{prefixe}time"
        ].max()

    # One figure per category
    for category in categories:
        category_times_data = times_data[times_data["algo"].str.contains(category)]
        category_times_data = pd.concat(
            [baseline_times_data, category_times_data], ignore_index=True
        )

        pareto_plot(
            plot_folder=plot_folder,
            num_reevals=num_reevals,
            compare_size=compare_size,
            times_data=category_times_data,
            prefixe=prefixe,
            prefixe_title=prefixe_title,
            legend_columns=legend_columns,
            suffixe="_" + category,
            # min_qd_score_dict=min_qd_score_dict,
            # max_qd_score_dict=max_qd_score_dict,
            # min_time_dict=min_time_dict,
            # max_time_dict=max_time_dict,
            pareto_column=pareto_column,
        )


def summary_loss_allenv(
    num_reevals: str,
    compare_size: str,
    compare_title: str,
    plot_folder: str,
    losses_data: pd.DataFrame,
    var_data: pd.DataFrame,
    prefixe: str = "",
    prefixe_title: str = "",
    legend_columns: int = 2,
    suffixe: str = "",
) -> None:

    # Get all environments
    env_names = losses_data["env"].drop_duplicates().values
    ncols = max(2, len(env_names))

    # Choose legend display
    num_algos = losses_data["algo"].drop_duplicates().values.shape[0]
    # if num_algos % 4 == 0:
    #    legend_columns = min(num_algos // 4, legend_columns * 3)
    # elif num_algos % 2 == 0:
    #    legend_columns = min(num_algos // 2, legend_columns * 3)
    # else:
    #    legend_columns = (legend_columns * ncols) // 2
    palette = sns.color_palette("Paired", num_algos)
    sns.set_palette(palette)

    # Order by difficulty
    final_envs: List[str] = []
    title_envs: List[str] = []
    for env_name in ENV_NAME_DIFFICULTY.keys():
        if env_name in env_names:
            final_envs.append(env_name)
            title_envs.append(ENV_NAME_DIFFICULTY[env_name])
    for env_name in env_names:
        if env_name not in final_envs:
            final_envs.append(env_name)
            title_envs.append(env_name)

    # Get batch value for x axis
    batch_values = losses_data[compare_size].drop_duplicates().values
    batch_values.sort()
    log_batch_values = np.log2(batch_values)

    # Create figure
    file_name = f"{plot_folder}/{num_reevals}reevals_{prefixe}loss_summary{suffixe}.pdf"
    fig, ax = plt.subplots(
        nrows=2,
        ncols=ncols,
        figsize=(5 * ncols, 10),
    )
    _LinePlotter.aggregate = first_second_third_quartile

    # One column per env
    var_y_axis_plotted = False
    for col in range(len(env_names)):
        env_losses_data = losses_data[losses_data["env"] == final_envs[col]]
        env_var_data = var_data[var_data["env"] == final_envs[col]]

        # Use log plot
        env_losses_data["log_compare_size"] = np.log2(env_losses_data[compare_size])
        env_var_data["log_compare_size"] = np.log2(env_var_data[compare_size])

        env_losses_data = env_losses_data.replace("Vanilla-MAP-Elites", "MAP-Elites")
        env_var_data = env_var_data.replace("Vanilla-MAP-Elites", "MAP-Elites")

        # Loss QD-Score vs size
        sub_plot(
            x="log_compare_size",
            y=f"loss_{prefixe}reeval_qd_score",
            data=env_losses_data,
            ax=ax[0, col],
            xlabel=compare_title,
            ylabel=f"{prefixe_title}Loss QD-Score (%)",
            markers=True,
        )
        ax[0, col].set_ylim(0, 100)
        handles, labels = ax[0, col].get_legend_handles_labels()
        ax[0, col].legend_.remove()
        ax[0, col].set_title(title_envs[col])
        if col > 0:
            ax[0, col].set_ylabel(None)

        # Variance descriptor vs size
        if final_envs[col] in ENV_NO_VAR:
            ax[0, col].xaxis.set_major_locator(
                FixedLocator(
                    env_losses_data["log_compare_size"].drop_duplicates().values
                )
            )
            ax[0, col].xaxis.set_major_formatter(
                lambda pos, x: batch_values[log_batch_values == pos][0]
            )

            ax[1, col].tick_params(
                left=False,
                right=False,
                labelleft=False,
                labelbottom=False,
                bottom=False,
            )

            ax[1, col].spines["top"].set_visible(False)
            ax[1, col].spines["right"].set_visible(False)
            ax[1, col].spines["bottom"].set_visible(False)
            ax[1, col].spines["left"].set_visible(False)
        else:
            ax[0, col].get_xaxis().set_ticks([])
            ax[0, col].set_xlabel(None)
            ax[0, col].tick_params(axis="x", length=0)

            sub_plot(
                x="log_compare_size",
                y=f"{prefixe}avg_desc_var_qd_score",
                data=env_var_data,
                ax=ax[1, col],
                xlabel=compare_title,
                ylabel=f"{prefixe_title}Descriptor Variance (%)",
                markers=True,
            )
            handles, labels = ax[1, col].get_legend_handles_labels()
            ax[1, col].set_ylim(0, 100)
            ax[1, col].legend_.remove()
            ax[1, col].xaxis.set_major_locator(
                FixedLocator(
                    env_losses_data["log_compare_size"].drop_duplicates().values
                )
            )
            ax[1, col].xaxis.set_major_formatter(
                lambda pos, x: batch_values[log_batch_values == pos][0]
            )
            if not var_y_axis_plotted:
                ax[1, col].set_ylabel(f"{prefixe_title}Descriptor Variance (%)")
                var_y_axis_plotted = True
            else:
                ax[1, col].set_ylabel(None)

    # Add legend below graph
    legend_bottom = 0.10 + 0.05 * ceil(len(labels) / legend_columns)
    plt.tight_layout()
    fig.subplots_adjust(bottom=legend_bottom)
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        frameon=False,
        ncol=legend_columns,
    )

    # Save figure
    plt.savefig(file_name)
    plt.close()

    # Create figure for fitness only
    fit_final_envs: List[str] = []
    fit_title_envs: List[str] = []
    for col in range(len(env_names)):
        if final_envs[col] not in ENV_NO_VAR:
            fit_final_envs.append(final_envs[col])
            fit_title_envs.append(title_envs[col])
    file_name = (
        f"{plot_folder}/{num_reevals}reevals_{prefixe}fit_loss_summary{suffixe}.pdf"
    )
    fig, ax = plt.subplots(
        nrows=1,
        ncols=max(2, len(fit_final_envs)),
        figsize=(5 * len(fit_final_envs), 6),
        sharey="row",
    )
    _LinePlotter.aggregate = first_second_third_quartile

    # One column per env
    for col in range(len(fit_final_envs)):
        env_var_data = var_data[var_data["env"] == fit_final_envs[col]]

        # Use log plot
        env_var_data["log_compare_size"] = np.log2(env_var_data[compare_size])
        env_var_data = env_var_data.replace("Vanilla-MAP-Elites", "MAP-Elites")

        # Variance fitness vs size
        sub_plot(
            x="log_compare_size",
            y=f"{prefixe}avg_fit_var_qd_score",
            data=env_var_data,
            ax=ax[col],
            xlabel=compare_title,
            ylabel=f"{prefixe_title}Fitness Variance (%)",
            markers=True,
        )
        handles, labels = ax[col].get_legend_handles_labels()
        ax[col].legend_.remove()
        ax[col].xaxis.set_major_locator(
            FixedLocator(env_var_data["log_compare_size"].drop_duplicates().values)
        )
        ax[col].xaxis.set_major_formatter(lambda pos, x: batch_values[x])
        ax[col].set_title(fit_title_envs[col])
        if col > 0:
            ax[col].set_ylabel(None)

    # Add legend below graph
    legend_bottom = 0.30 + 0.05 * ceil(len(labels) / legend_columns)
    plt.tight_layout()
    fig.subplots_adjust(bottom=legend_bottom)
    fig.legend(
        handles=handles,
        labels=labels,
        loc="lower center",
        frameon=False,
        ncol=legend_columns,
    )

    # Save figure
    plt.savefig(file_name)
    plt.close()


def summary_loss_categories_allenv(
    num_reevals: str,
    compare_size: str,
    compare_title: str,
    plot_folder: str,
    losses_data: pd.DataFrame,
    var_data: pd.DataFrame,
    prefixe: str = "",
    prefixe_title: str = "",
    legend_columns: int = 2,
) -> None:

    # List existing categories in the dataframes
    baselines: List[str] = []
    bas_data = losses_data.copy()
    for baseline in BASELINES:
        if not bas_data[bas_data["algo"].str.contains(baseline)].empty:
            baselines.append(baseline)
            bas_data = bas_data[~bas_data["algo"].str.contains(baseline)]
    categories: List[str] = []
    cat_data = losses_data.copy()
    for category in CATEGORIES:
        if not cat_data[cat_data["algo"].str.contains(category)].empty:
            categories.append(category)
            cat_data = cat_data[~cat_data["algo"].str.contains(category)]

    # Extract baseline datas
    baseline_losses_data = losses_data[losses_data["algo"].str.contains(BASELINES[0])]
    baseline_var_data = var_data[var_data["algo"].str.contains(BASELINES[0])]
    for baseline in BASELINES[1:]:
        baseline_losses_data = pd.concat(
            [
                baseline_losses_data,
                losses_data[losses_data["algo"].str.contains(baseline)],
            ],
            ignore_index=True,
        )
        baseline_var_data = pd.concat(
            [baseline_var_data, var_data[var_data["algo"].str.contains(baseline)]],
            ignore_index=True,
        )

    # One figure per category
    for category in categories:
        category_losses_data = losses_data[losses_data["algo"].str.contains(category)]
        category_losses_data = pd.concat(
            [baseline_losses_data, category_losses_data], ignore_index=True
        )
        category_var_data = var_data[var_data["algo"].str.contains(category)]
        category_var_data = pd.concat(
            [baseline_var_data, category_var_data], ignore_index=True
        )

        summary_loss_allenv(
            num_reevals=num_reevals,
            compare_size=compare_size,
            compare_title=compare_title,
            plot_folder=plot_folder,
            losses_data=category_losses_data,
            var_data=category_var_data,
            prefixe=prefixe,
            prefixe_title=prefixe_title,
            legend_columns=legend_columns,
            suffixe="_" + category,
        )
