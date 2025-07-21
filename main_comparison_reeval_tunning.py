import argparse
import os
import traceback
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator
from seaborn.relational import _LinePlotter

from analysis.plots import (
    adjust_box_widths,
    first_second_third_quartile,
    sub_box,
    sub_plot,
)
from analysis.summary_plots import ENV_NAME_DIFFICULTY


def replications_box_plot(
    file_name: str,
    dataframe: pd.DataFrame,
    max_replications_value: int,
) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 20), sharex=True)
    suffixe = f"average distance to {max_replications_value} replications"
    sub_box(
        x="replications",
        y="descriptor_dist",
        hue="label",
        data=dataframe,
        ax=axes[0],
        ylabel="Descriptor " + suffixe,
    )
    sub_box(
        x="replications",
        y="fitness_dist",
        hue="label",
        data=dataframe,
        ax=axes[1],
        ylabel="Fitness " + suffixe,
    )
    adjust_box_widths(fig, 0.8)
    plt.title(f"Average distance to {max_replications_value} replications")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def replications_line_plot(
    file_name: str, dataframe: pd.DataFrame, max_replications_value: int
) -> None:
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 20), sharex=True)
    _LinePlotter.aggregate = first_second_third_quartile
    suffixe = f"distance to {max_replications_value} stat"
    replications_values = [
        int(x) for x in dataframe["replications"].drop_duplicates().values
    ]
    sub_plot(
        x="replications",
        y="descriptor_dist",
        hue="label",
        data=dataframe,
        ax=axes[0],
        xlabel="Replications",
        ylabel="Descriptor " + suffixe,
    )
    axes[0].xaxis.set_major_locator(FixedLocator(replications_values))
    axes[0].tick_params(axis="x", length=0)
    sub_plot(
        x="replications",
        y="fitness_dist",
        hue="label",
        data=dataframe,
        ax=axes[1],
        xlabel="Replications",
        ylabel="Fitness " + suffixe,
    )
    axes[1].xaxis.set_major_locator(FixedLocator(replications_values))
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def replications_summary_plot(
    file_name: str,
    dataframe: pd.DataFrame,
    max_replications_value: int,
    legend_columns: int = 2,
    log: bool = False,
    all_cells: bool = True,
) -> None:

    # Get all environments
    env_names = dataframe["env"].drop_duplicates().values
    nrows = len(env_names)
    ncols = 2

    # Choose legend display
    num_stats = dataframe["label"].drop_duplicates().values.shape[0]
    if num_stats % 2 == 0 and num_stats > 2:
        legend_columns = min(num_stats // 2, legend_columns * 3)
        palette = sns.color_palette("Paired", num_stats)
    else:
        legend_columns = legend_columns * ncols
        palette = sns.color_palette("tab10", num_stats)
    sns.set_palette(palette)

    # Order by difficulty
    final_envs = []
    title_envs = []
    for env_name in ENV_NAME_DIFFICULTY.keys():
        if env_name in env_names:
            final_envs.append(env_name)
            title_envs.append(ENV_NAME_DIFFICULTY[env_name])
    for env_name in env_names:
        if env_name not in final_envs:
            final_envs.append(env_name)
            title_envs.append(env_name)

    # Get batch value for x axis
    replications_values = [
        int(x) for x in dataframe["replications"].drop_duplicates().values
    ]
    replications_values = [16, 1024, 2048, 4096, 8192, 16384]

    # Create figure
    if all_cells:
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(8 * ncols, 6 * nrows)
        )
    else:
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(8 * ncols, 6 * nrows), sharey=True
        )
    _LinePlotter.aggregate = first_second_third_quartile

    # One line per env
    for line in range(len(env_names)):
        env_dataframe = dataframe[dataframe["env"] == final_envs[line]]

        if all_cells:
            plot_dataframe = env_dataframe
            unit_label = ""
        else:
            # Normalise datas
            plot_dataframe = env_dataframe
            plot_dataframe = pd.DataFrame()
            for label in env_dataframe["label"].drop_duplicates().values:
                sub_env_dataframe = env_dataframe[
                    env_dataframe["label"] == label
                ].reset_index()
                for column in sub_env_dataframe.columns:
                    if sub_env_dataframe.dtypes[column] == "float64":
                        # print("Normlising for", final_envs[col], "and", label)
                        # print("Max:", sub_env_dataframe[column].max())
                        # print("Min:", sub_env_dataframe[column].min())
                        sub_env_dataframe[column] = (
                            (
                                sub_env_dataframe[column]
                                - sub_env_dataframe[column].min()
                            )
                            / (
                                sub_env_dataframe[column].max()
                                - sub_env_dataframe[column].min()
                            )
                            * 100
                        )
                plot_dataframe = pd.concat(
                    [plot_dataframe, sub_env_dataframe], ignore_index=True
                )
            unit_label = " (%)"

        sub_plot(
            x="replications",
            y="descriptor_dist",
            hue="label",
            data=plot_dataframe,
            ax=axes[line, 0],
            xlabel="Number of Reevaluations",
            ylabel=title_envs[line],
            markers=True,
        )
        if line == 0:
            axes[line, 0].set_title(f"Descriptor distance to ground truth{unit_label}")
        if log:
            axes[line, 0].set_xscale("log")
        elif line == nrows - 1:
            axes[line, 0].xaxis.set_major_locator(FixedLocator(replications_values))
            axes[line, 0].xaxis.set_tick_params(rotation=90)
        else:
            axes[line, 0].set_xlabel(None)
            axes[line, 0].xaxis.set_major_locator(FixedLocator(replications_values))
            axes[line, 0].xaxis.set_tick_params(rotation=90)
        #    axes[line, 0].tick_params(axis="x", length=0)
        #    axes[line, 0].get_xaxis().set_ticks([])
        axes[line, 0].legend_.remove()
        if not all_cells:
            axes[line, 0].axhline(15, c="r", linestyle="--", linewidth=3)
        # else:
        # axes[line, 0].axhline(5, c="r", linestyle="--", linewidth=3)

        sub_plot(
            x="replications",
            y="fitness_dist",
            hue="label",
            data=plot_dataframe,
            ax=axes[line, 1],
            xlabel="Number of Reevaluations",
            ylabel="",
            markers=True,
        )
        axes[line, 1].set_ylabel(None)
        if line == 0:
            axes[line, 1].set_title(f"Fitness distance to ground truth{unit_label}")
        if log:
            axes[line, 1].set_xscale("log")
        elif line == nrows - 1:
            axes[line, 1].xaxis.set_major_locator(FixedLocator(replications_values))
            axes[line, 1].xaxis.set_tick_params(rotation=90)
        else:
            axes[line, 1].set_xlabel(None)
            axes[line, 1].xaxis.set_major_locator(FixedLocator(replications_values))
            axes[line, 1].xaxis.set_tick_params(rotation=90)
        #    axes[line, 1].tick_params(axis="x", length=0)
        #    axes[line, 1].get_xaxis().set_ticks([])
        if not all_cells:
            axes[line, 1].axhline(15, c="r", linestyle="--", linewidth=3)
        # else:
        # axes[line, 1].axhline(5, c="r", linestyle="--", linewidth=3)
        # axes[line, 1].set_ylim(0, 25)
        handles, labels = axes[line, 1].get_legend_handles_labels()
        axes[line, 1].legend_.remove()

    # Add legend below graph
    legend_bottom = 0.10 + 0.01 * (len(labels) // legend_columns)
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


#########
# Input #

parser = argparse.ArgumentParser()

# Folder
parser.add_argument("--results", default="results", type=str)
parser.add_argument("--plots", default="plots", type=str)

# Analysis configuration
parser.add_argument("--paper-plot", action="store_true")
parser.add_argument("--algos", default="", type=str)
parser.add_argument("--excludes", default="", type=str)
parser.add_argument("--legend-columns", default=2, type=int)

# Process inputs
args = parser.parse_args()
save_folder = args.results
plot_folder = args.plots
plot_algos = args.algos.rstrip().split("|")
exclude_algos = args.excludes.rstrip().split("|")
assert os.path.exists(save_folder), "\n!!!ERROR!!! Empty result folder.\n"


################
# Find results #

# Opening all config files in the folder
print("\n\nOpening config files")
folders = [
    root
    for root, dirs, files in os.walk(save_folder)
    for name in files
    if "config.csv" in name
]
assert len(folders) > 0, "\n!!!ERROR!!! No config files in result folder.\n"
config_frame = pd.DataFrame()
for folder in folders:
    config_file = os.path.join(folder, "config.csv")
    sub_config_frame = pd.read_csv(config_file, index_col=False)
    sub_config_frame["folder"] = folder
    config_frame = pd.concat([config_frame, sub_config_frame], ignore_index=True)
assert config_frame.shape[0] != 0, "\n!!!ERROR!!! No runs refered in config files.\n"
print("    Found", config_frame.shape[0], "runs:")
print(config_frame["run"].drop_duplicates().reset_index(drop=True))

# Create results folder if needed
if not os.path.exists(plot_folder):
    os.mkdir(plot_folder)

# Customize matplotlib params
font_size = 28
params = {
    "axes.labelsize": font_size,
    "axes.titlesize": font_size,
    "legend.fontsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "text.usetex": False,
    "axes.titlepad": 10,
    "lines.linewidth": 3,
    "lines.markersize": 16,
}
mpl.rcParams.update(params)


##################
# Filter results #

# Filter algos that does not need to be ploted
if plot_algos != [""]:
    config_frame = config_frame[config_frame["run"].str.contains("|".join(plot_algos))]
if exclude_algos != [""]:
    config_frame = config_frame[
        ~config_frame["run"].str.contains("|".join(exclude_algos))
    ]
config_frame = config_frame.reset_index(drop=True)
print("\n    After filtering, left with", config_frame.shape[0], "runs:")
print(config_frame["run"].drop_duplicates().reset_index(drop=True))
assert config_frame.shape[0] != 0, "\n!!!ERROR!!! No algos left to plot.\n"


################
# Name results #

print("\nSetting up algorithms names")
use_in_name = []
not_name = [
    "folder",
    "run",
    "seed",
    "env",
    "num_iterations",
    "batch_size",
    "sampling_size",
    "sampling_use",
    "num_reevals",
    "init_time",
    "run_time",
    "metrics_file",
    "in_cell_metrics_file",
    "repertoire_folder",
    "reeval_repertoire_folder",
    "average_repertoire_folder",
    "fit_reeval_repertoire_folder",
    "fit_average_repertoire_folder",
    "desc_reeval_repertoire_folder",
    "desc_average_repertoire_folder",
    "fit_var_repertoire_folder",
    "desc_var_repertoire_folder",
    "min_bd",
    "max_bd",
    "depth",
    "num_samples",
    "episode_length",
    "params_variance",
    "desc_variance",
    "fit_variance",
]
for column in config_frame.columns:
    if column not in not_name:
        if (config_frame[column] != config_frame[column][0]).any():
            use_in_name.append(column)
print("\n    Differences between runs:", use_in_name)

# Add algo name to each line
algos = []
for line in range(config_frame.shape[0]):
    algo = config_frame["run"][line]
    algo = algo.replace("-depth-", " ")
    # algo = algo.replace("-sampling-", " smpl ")
    algo = algo.replace("-NonCircular", "")  # for old results
    algo = algo.replace(
        "Extended-Adaptive-Sampling-median", "Parallel-Adaptive-Sampling"
    )
    # algo = algo.replace("MAP-Sampling", "Archive-Sampling")
    for name in use_in_name:
        algo += " " + name + ":" + str(config_frame[name][line])
    algos.append(algo)
config_frame["algo"] = algos
print("\n    Get final names for graphs:")
print(config_frame["algo"].drop_duplicates())
config_frame = config_frame.reset_index(drop=True)


###########
# Loading #

print("\nEntering loading")

all_dataframe = pd.DataFrame()
all_cells_dataframe = pd.DataFrame()

# For each line
for line in range(config_frame.shape[0]):

    # Get the data path
    folder = config_frame["folder"][line]
    metrics_file = config_frame["metrics_file"][line]
    metrics_file = metrics_file[metrics_file.rfind("/") + 1 :]
    metrics_file = os.path.join(folder, metrics_file)

    try:
        # Open the data path
        print("  Opening datas in", metrics_file)
        raw_dataframe = pd.read_csv(metrics_file, index_col=False)

        cells_dataframe = raw_dataframe

        # Within each dataframe take the median of each label for each replication value
        dataframe = pd.DataFrame()
        for replication in raw_dataframe["replications"].drop_duplicates().values:
            for label in raw_dataframe["label"].drop_duplicates().values:
                sub_raw_dataframe = raw_dataframe[
                    (raw_dataframe["replications"] == replication)
                    & (raw_dataframe["label"] == label)
                ]
                descriptor_dist = sub_raw_dataframe["descriptor_dist"].median()
                fitness_dist = sub_raw_dataframe["fitness_dist"].median()

                dataframe_line: Dict[str, List[float]] = {}
                dataframe_line["env"] = [sub_raw_dataframe["env"].values[0]]
                dataframe_line["algo"] = [sub_raw_dataframe["algo"].values[0]]
                dataframe_line["seed"] = [sub_raw_dataframe["seed"].values[0]]
                dataframe_line["replications"] = [replication]
                dataframe_line["label"] = [label]
                dataframe_line["descriptor_dist"] = [descriptor_dist]
                dataframe_line["fitness_dist"] = [fitness_dist]

                dataframe = pd.concat(
                    [dataframe, pd.DataFrame.from_dict(dataframe_line)],
                    ignore_index=True,
                )

        all_cells_dataframe = pd.concat(
            [all_cells_dataframe, cells_dataframe], ignore_index=True
        )
        all_dataframe = pd.concat([all_dataframe, dataframe], ignore_index=True)
    except Exception:
        print("\n!!!WARNING!!! Cannot read", metrics_file, ".")
        print(traceback.format_exc(-1))


############
# Plotting #

print("Finished loading \n\nEntering ploting")

all_cells_dataframe["label"] = all_cells_dataframe["label"].str.capitalize()
all_dataframe["label"] = all_dataframe["label"].str.capitalize()

for algo in all_dataframe["algo"].drop_duplicates().values:
    algo_cells_dataframe = all_cells_dataframe[all_cells_dataframe["algo"] == algo]
    algo_dataframe = all_dataframe[all_dataframe["algo"] == algo]

    # Per env plots
    for env in algo_dataframe["env"].drop_duplicates().values:
        env_dataframe = algo_dataframe[algo_dataframe["env"] == env]
        max_replications_value = max(
            env_dataframe["replications"].drop_duplicates().values
        )
        file_name = f"{plot_folder}/{env}_{algo}_summary.pdf"
        replications_line_plot(
            file_name=file_name,
            dataframe=env_dataframe,
            max_replications_value=max_replications_value,
        )

    # Remove closer from dataframe to simplify
    clearer_cells_dataframe = algo_cells_dataframe[
        ~algo_cells_dataframe["label"].str.contains("Closer")
    ]
    clearer_dataframe = algo_dataframe[~algo_dataframe["label"].str.contains("Closer")]

    # Summary plots
    file_name = f"{plot_folder}/{algo}_summary.pdf"
    replications_summary_plot(
        file_name=file_name,
        dataframe=clearer_dataframe,
        max_replications_value=max_replications_value,
        legend_columns=args.legend_columns,
        log=False,
        all_cells=False,
    )
    file_name = f"{plot_folder}/{algo}_summary_allcells.pdf"
    replications_summary_plot(
        file_name=file_name,
        dataframe=clearer_cells_dataframe,
        max_replications_value=max_replications_value,
        legend_columns=args.legend_columns,
        log=False,
        all_cells=True,
    )

    # Log summary plot
    file_name = f"{plot_folder}/{algo}_log_summary.pdf"
    replications_summary_plot(
        file_name=file_name,
        dataframe=clearer_dataframe,
        max_replications_value=max_replications_value,
        legend_columns=args.legend_columns,
        log=True,
        all_cells=False,
    )
    file_name = f"{plot_folder}/{algo}_log_summary_allcells.pdf"
    replications_summary_plot(
        file_name=file_name,
        dataframe=clearer_cells_dataframe,
        max_replications_value=max_replications_value,
        legend_columns=args.legend_columns,
        log=True,
        all_cells=True,
    )

print("Finished ploting")
