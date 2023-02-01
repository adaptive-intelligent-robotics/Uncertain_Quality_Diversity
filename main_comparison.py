import argparse
import os
import traceback
from functools import partial
from typing import Dict, List

import jax
import jax.numpy as jnp
import matplotlib as mpl
import pandas as pd
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire

from analysis.archives import plot_env_paper_archives
from analysis.p_value import p_values
from analysis.summary_plots import (
    pareto_categories,
    pareto_plot,
    summary_loss_allenv,
    summary_loss_categories_allenv,
)
from analysis.utils import get_folder_name, sort_data, uniformise_xaxis
from analysis.visualisation import plot_visualisation_archive, save_html
from set_up_environment import ENV_CONTROL, ENV_OPTIMISATION, set_up_environment


def compute_loss(
    name: str,
    data: pd.DataFrame,
    losses: Dict[str, List[float]],
    prefixe: str = "",
) -> Dict[str, List[float]]:
    max_eval = max(data["eval"])
    original = data[data["eval"] == max_eval][prefixe + name].values[0]
    losses[prefixe + name] = [original]
    average = data[data["eval"] == max_eval][prefixe + "reeval_" + name].values[0]
    fit_average = data[data["eval"] == max_eval][prefixe + "fit_reeval_" + name].values[
        0
    ]
    desc_average = data[data["eval"] == max_eval][
        prefixe + "desc_reeval_" + name
    ].values[0]
    losses[prefixe + "reeval_" + name] = [average]
    losses[prefixe + "fit_reeval_" + name] = [fit_average]
    losses[prefixe + "desc_reeval_" + name] = [desc_average]
    if original == 0:
        if average == 0:
            losses["loss_" + prefixe + "reeval_" + name] = [0]
        else:
            losses["loss_" + prefixe + "reeval_" + name] = [100]
        if fit_average == 0:
            losses["loss_" + prefixe + "fit_reeval_" + name] = [0]
        else:
            losses["loss_" + prefixe + "fit_reeval_" + name] = [100]
        if desc_average == 0:
            losses["loss_" + prefixe + "desc_reeval_" + name] = [0]
        else:
            losses["loss_" + prefixe + "desc_reeval_" + name] = [100]
    else:
        losses["loss_" + prefixe + "reeval_" + name] = [
            (original - average) / original * 100
        ]
        losses["loss_" + prefixe + "fit_reeval_" + name] = [
            (original - fit_average) / original * 100
        ]
        losses["loss_" + prefixe + "desc_reeval_" + name] = [
            (original - desc_average) / original * 100
        ]
    return losses


def compute_time(
    data: pd.DataFrame,
    env_name: str,
    algo: str,
    size: int,
    times: Dict[str, List[float]],
    pourcent: float = 0.95,
    prefixe: str = "",
    pourcent_value: bool = False,
) -> Dict[str, List[float]]:
    """
    Return time to first stricly reach pourcent % of final QD-Score value for one
    replication of one given algorithm on one given task.

    Args:
        data: dataframe for one replication of one algo to extract value from.
        times: time dictionary to complement for this replication
        poucent: pourcent use
        prefixe: if qd_score column name requires a prefixe

    Returns:
        new complemented time dictionary
    """

    # Get final value of QD-Score
    max_eval = max(data["eval"])
    final_line = data[data["eval"] == max_eval]
    final_value = final_line[prefixe + "qd_score"].values[0]

    # Finding pourcent % value
    pourcent_value = (
        (pourcent * final_value)
        if (final_value > 0)
        else (2.0 - pourcent * final_value)
    )
    if pourcent == 1:
        pourcent_epoch = max(data["epoch"].values)
    else:
        min_epoch = data["epoch"].drop_duplicates().nsmallest(2).iloc[-1]
        pourcent_epoch = max(
            min_epoch,
            min(data[data[prefixe + "qd_score"] > pourcent_value]["epoch"].values),
        )
    pourcent_line = data[data["epoch"] == pourcent_epoch]

    # Finding corresponding eval, gen, time and values
    times[prefixe + "epoch"] = pourcent_epoch
    times[prefixe + "eval"] = pourcent_line["eval"].values[0]
    times[prefixe + "time"] = pourcent_line["time"].values[0]
    if pourcent_value:
        times[prefixe + "qd_score"] = pourcent_line[prefixe + "qd_score"].values[0]
        times[prefixe + "coverage"] = pourcent_line[prefixe + "coverage"].values[0]
        times[prefixe + "max_fitness"] = pourcent_line[prefixe + "max_fitness"].values[
            0
        ]
        times[prefixe + "reeval_qd_score"] = pourcent_line[
            prefixe + "reeval_qd_score"
        ].values[0]
        times[prefixe + "reeval_coverage"] = pourcent_line[
            prefixe + "reeval_coverage"
        ].values[0]
        times[prefixe + "reeval_max_fitness"] = pourcent_line[
            prefixe + "reeval_max_fitness"
        ].values[0]
    else:
        times[prefixe + "qd_score"] = final_line[prefixe + "qd_score"].values[0]
        times[prefixe + "coverage"] = final_line[prefixe + "coverage"].values[0]
        times[prefixe + "max_fitness"] = final_line[prefixe + "max_fitness"].values[0]
        times[prefixe + "reeval_qd_score"] = final_line[
            prefixe + "reeval_qd_score"
        ].values[0]
        times[prefixe + "reeval_coverage"] = final_line[
            prefixe + "reeval_coverage"
        ].values[0]
        times[prefixe + "reeval_max_fitness"] = final_line[
            prefixe + "reeval_max_fitness"
        ].values[0]

    return times


def compute_var(
    data: pd.DataFrame,
    var: Dict[str, List[float]],
    num_centroids: int,
    prefixe: str = "",
) -> Dict[str, List[float]]:
    """
    Return final variance value for one replication of one given algorithm on
    one given task.

    Args:
        data: dataframe for one replication of one algo to extract value from.
        var: var dictionary to complement for this replication
        num_centroids: used to compute average variance
        prefixe: if qd_score columns name requires a prefixe

    Returns:
        new complemented var dictionary
    """

    # Get final value of Variances
    max_eval = max(data["eval"])
    final_fit_var_value = data[data["eval"] == max_eval][
        prefixe + "fit_var_qd_score"
    ].values[0]
    final_desc_var_value = data[data["eval"] == max_eval][
        prefixe + "desc_var_qd_score"
    ].values[0]
    coverage = data[data["eval"] == max_eval][prefixe + "coverage"].values[0]
    average_fit_var_value = final_fit_var_value / (coverage / 100 * num_centroids)
    average_desc_var_value = final_desc_var_value / (coverage / 100 * num_centroids)

    # Fill in Dict
    var[prefixe + "fit_var_qd_score"] = final_fit_var_value
    var[prefixe + "desc_var_qd_score"] = final_desc_var_value
    var[prefixe + "avg_fit_var_qd_score"] = average_fit_var_value
    var[prefixe + "avg_desc_var_qd_score"] = average_desc_var_value
    return var


#############################################################

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
parser.add_argument("--no-traceback", action="store_true")

# Metrics
parser.add_argument("--plot-paper-archives", action="store_true")
parser.add_argument("--plot-summary", action="store_true")
parser.add_argument("--plot-pareto", action="store_true")
parser.add_argument("--plot-p-values", action="store_true")

# Metrics parameters
parser.add_argument("--compare-batch-size", action="store_true")
parser.add_argument("--time-pourcent", default=0.95, type=float)
parser.add_argument("--pourcent-value", action="store_true")
parser.add_argument("--generations", action="store_true")
parser.add_argument("--time", action="store_true")
parser.add_argument("--metrics-legend-columns", default=2, type=int)
parser.add_argument("--summary-legend-columns", default=4, type=int)
parser.add_argument("--pareto-legend-columns", default=3, type=int)
parser.add_argument("--category-plots", action="store_true")
parser.add_argument("--paper-archives-prefixe", default="", type=str)

# Visualisation
parser.add_argument("--visualisation", action="store_true")
parser.add_argument("--save-html", action="store_true")
parser.add_argument("--deterministic", action="store_true")
parser.add_argument("--best-indiv", action="store_true")
parser.add_argument("--indiv", default=0, type=int)
parser.add_argument("--replications", default=256, type=int)

# Process inputs
args = parser.parse_args()
save_folder = args.results
plot_folder = args.plots
plot_algos = args.algos.rstrip().split("|")
exclude_algos = args.excludes.rstrip().split("|")
compare_size = "batch_size" if args.compare_batch_size else "sampling_size"
compare_title = "Batch-size" if args.compare_batch_size else "Sampling-size"
assert os.path.exists(save_folder), "\n!!!ERROR!!! Empty result folder.\n"

# Create results folder if needed
try:
    if not os.path.exists(plot_folder):
        os.mkdir(plot_folder)
except Exception:
    if not args.no_traceback:
        print("\n!!!WARNING!!! Cannot create folders for plots.")
        traceback.print_exc()


################
# Find results #

# If not, open all config files in the folder
else:
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
    assert (
        config_frame.shape[0] != 0
    ), "\n!!!ERROR!!! No runs refered in config files.\n"

print("    Found", config_frame.shape[0], "runs:")
print(config_frame["run"].drop_duplicates().reset_index(drop=True))


##################
# Filter results #

# Filter algos that does not need to be ploted
if plot_algos != [""]:
    config_frame = config_frame[config_frame["run"].isin(plot_algos)]
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
    "metrics_file",
    "in_cell_metrics_file",
    "repertoire_folder",
    "reeval_repertoire_folder",
    "fit_reeval_repertoire_folder",
    "desc_reeval_repertoire_folder",
    "fit_var_repertoire_folder",
    "desc_var_repertoire_folder",
    "in_cell_reeval_repertoire_folder",
    "in_cell_fit_reeval_repertoire_folder",
    "in_cell_desc_reeval_repertoire_folder",
    "in_cell_fit_var_repertoire_folder",
    "in_cell_desc_var_repertoire_folder",
    "min_bd",
    "max_bd",
    "depth",
    "num_samples",
    "episode_length",
]
for column in config_frame.columns:
    if column not in not_name:
        if (config_frame[column] != config_frame[column][0]).any():
            use_in_name.append(column)
print("\n    Differences between runs:", use_in_name)

# Add algo name to each line
algos = []
algos_batch = []
for line in range(config_frame.shape[0]):
    algo = config_frame["run"][line]
    if "Deep-Grid" in algo and "sampling" in algo:
        algo = algo.replace("-sampling-", " smpl ")
        algo = algo.replace("Deep-Grid", "Deep-Grid-sampling")
    algo = algo.replace("-depth-", " ")
    algo = algo.replace("-archive-out-sampling-", "-out-smpl")
    for name in use_in_name:
        algo += " " + name + ":" + str(config_frame[name][line])
    algo_batch = algo + " - " + str(config_frame[compare_size][line])
    algos.append(algo)
    algos_batch.append(algo_batch)
config_frame["algo"] = algos
config_frame["algo_batch"] = algos_batch
print("\n    Get final names for graphs:")
print(config_frame["algo"].drop_duplicates())
config_frame = config_frame.reset_index(drop=True)


#################################
# Read progress and loss graphs #

try:
    if not args.plot_summary and not args.plot_pareto and not args.plot_p_values:
        error = False
        assert 0
    error = True

    print("\nReading metrics data")

    # Create the dataframe with maximum number of gens
    max_gen_frame = pd.DataFrame(columns=["env", "epoch"])

    # Create the replication dataframe
    replications_frame = pd.DataFrame(
        columns=["env", "num_reevals", "algo", compare_size, "num_rep"]
    )
    for env in config_frame["env"].drop_duplicates().values:
        for num_reevals in config_frame["num_reevals"].drop_duplicates().values:
            for size in config_frame[compare_size].drop_duplicates().values:
                for algo in config_frame["algo"].drop_duplicates().values:
                    replications_frame = pd.concat(
                        [
                            replications_frame,
                            pd.DataFrame.from_dict(
                                {
                                    "env": [env],
                                    "num_reevals": [num_reevals],
                                    "algo": [algo],
                                    compare_size: [size],
                                    "num_rep": [0],
                                }
                            ),
                        ],
                        ignore_index=True,
                    )

    # Go through all metrics files to get the max number of epoch and replications first
    for line in range(config_frame.shape[0]):
        folder = config_frame["folder"][line]
        metrics_file = config_frame["metrics_file"][line]
        in_cell_metrics_file = config_frame["in_cell_metrics_file"][line]
        metrics_file = metrics_file[metrics_file.rfind("/") + 1 :]
        in_cell_metrics_file = in_cell_metrics_file[
            in_cell_metrics_file.rfind("/") + 1 :
        ]
        metrics_file = os.path.join(folder, metrics_file)
        in_cell_metrics_file = os.path.join(folder, in_cell_metrics_file)
        try:
            data = pd.read_csv(metrics_file, index_col=False)
            env = config_frame["env"][line]
            num_reevals = config_frame["num_reevals"][line]  # 0
            size = config_frame[compare_size][line]
            algo = config_frame["algo"][line]

            # Get the maximum number of generations for this line
            if env in max_gen_frame["env"].values:
                max_gen = min(
                    data["epoch"].max(),
                    max_gen_frame[max_gen_frame["env"] == env]["epoch"].values[0],
                )
                max_gen_frame.loc[max_gen_frame["env"] == env, "epoch"] = max_gen
            else:
                max_gen = data["epoch"].max()
                max_gen_frame = pd.concat(
                    [
                        max_gen_frame,
                        pd.DataFrame.from_dict({"env": [env], "epoch": [max_gen]}),
                    ],
                    ignore_index=True,
                )

            # Add replication to frame
            replications_frame.loc[
                (replications_frame["env"] == env)
                & (replications_frame["num_reevals"] == num_reevals)
                & (replications_frame["algo"] == algo)
                & (replications_frame[compare_size] == size),
                "num_rep",
            ] += 1

        except Exception:
            if not args.no_traceback:
                print("\n!!!WARNING!!! Cannot read", metrics_file, ".")
                traceback.print_exc()

    print("\nMax epoch for each environment:")
    print(max_gen_frame)
    print("\n")

    # Remove empty replications from replications_frame
    replications_frame = replications_frame[replications_frame["num_rep"] != 0]

    # Save replications frame as csv
    replications_frame = replications_frame.sort_values(
        ["env", "num_reevals"], ignore_index=True
    )
    replications_frame = sort_data(replications_frame, ["algo", compare_size])
    print("\nReplications:")
    print(replications_frame)
    print("\n")
    replications_frame.to_csv(
        f"{plot_folder}/replications_frame.csv",
        index=None,
        sep=",",
    )

    # Create the metrics dataframe
    all_data = pd.DataFrame()
    all_losses = pd.DataFrame()
    all_times = pd.DataFrame()
    all_var = pd.DataFrame()

    # Go through all metrics files
    rep = 0
    print(config_frame)
    for line in range(config_frame.shape[0]):
        folder = config_frame["folder"][line]
        metrics_file = config_frame["metrics_file"][line]
        in_cell_metrics_file = config_frame["in_cell_metrics_file"][line]
        metrics_file = metrics_file[metrics_file.rfind("/") + 1 :]
        in_cell_metrics_file = in_cell_metrics_file[
            in_cell_metrics_file.rfind("/") + 1 :
        ]
        metrics_file = os.path.join(folder, metrics_file)
        in_cell_metrics_file = os.path.join(folder, in_cell_metrics_file)
        try:
            data = pd.read_csv(metrics_file, index_col=False)

            # Add corresponding in_cell metrics
            in_cell_data = pd.read_csv(in_cell_metrics_file, index_col=False)
            data["in_cell_qd_score"] = in_cell_data["in_cell_qd_score"]
            data["in_cell_coverage"] = in_cell_data["in_cell_coverage"]
            data["in_cell_max_fitness"] = in_cell_data["in_cell_max_fitness"]
            data["in_cell_reeval_qd_score"] = in_cell_data["in_cell_reeval_qd_score"]
            data["in_cell_reeval_coverage"] = in_cell_data["in_cell_reeval_coverage"]
            data["in_cell_reeval_max_fitness"] = in_cell_data[
                "in_cell_reeval_max_fitness"
            ]
            data["in_cell_fit_reeval_qd_score"] = in_cell_data[
                "in_cell_fit_reeval_qd_score"
            ]
            data["in_cell_fit_reeval_coverage"] = in_cell_data[
                "in_cell_fit_reeval_coverage"
            ]
            data["in_cell_fit_reeval_max_fitness"] = in_cell_data[
                "in_cell_fit_reeval_max_fitness"
            ]
            data["in_cell_desc_reeval_qd_score"] = in_cell_data[
                "in_cell_desc_reeval_qd_score"
            ]
            data["in_cell_desc_reeval_coverage"] = in_cell_data[
                "in_cell_desc_reeval_coverage"
            ]
            data["in_cell_desc_reeval_max_fitness"] = in_cell_data[
                "in_cell_desc_reeval_max_fitness"
            ]
            data["in_cell_fit_var_qd_score"] = in_cell_data["in_cell_fit_var_qd_score"]
            data["in_cell_fit_var_coverage"] = in_cell_data["in_cell_fit_var_coverage"]
            data["in_cell_fit_var_max_fitness"] = in_cell_data[
                "in_cell_fit_var_max_fitness"
            ]
            data["in_cell_desc_var_qd_score"] = in_cell_data[
                "in_cell_desc_var_qd_score"
            ]
            data["in_cell_desc_var_coverage"] = in_cell_data[
                "in_cell_desc_var_coverage"
            ]
            data["in_cell_desc_var_max_fitness"] = in_cell_data[
                "in_cell_desc_var_max_fitness"
            ]

            # Filter datas after max_gen
            env = config_frame["env"][line]
            num_reevals = config_frame["num_reevals"][line]  # 0
            size = config_frame[compare_size][line]
            algo = config_frame["algo"][line]
            algo_batch = config_frame["algo_batch"][line]
            data = data[
                data["epoch"]
                <= max_gen_frame[max_gen_frame["env"] == env]["epoch"].values[0]
            ]

            # Add losses to frames
            losses: Dict[str, List[float]] = {}
            losses = compute_loss("qd_score", data, losses)
            losses = compute_loss("coverage", data, losses)
            losses = compute_loss("max_fitness", data, losses)
            losses = compute_loss("qd_score", data, losses, prefixe="in_cell_")
            losses = compute_loss("coverage", data, losses, prefixe="in_cell_")
            losses = compute_loss("max_fitness", data, losses, prefixe="in_cell_")

            # Add time to pourcent % convergence to frames
            times: Dict[str, List[float]] = {}
            times = compute_time(
                data,
                env,
                algo,
                size,
                times,
                pourcent=args.time_pourcent,
                pourcent_value=args.pourcent_value,
            )
            times = compute_time(
                data,
                env,
                algo,
                size,
                times,
                pourcent=args.time_pourcent,
                prefixe="in_cell_",
                pourcent_value=args.pourcent_value,
            )

            # Add final variances to frames
            num_centroids = config_frame["num_centroids"][line]
            var: Dict[str, List[float]] = {}
            var = compute_var(data, var, num_centroids)
            var = compute_var(data, var, num_centroids, prefixe="in_cell_")

            # Add run info to frames
            data["algo"] = algo
            data["algo_batch"] = algo_batch
            losses["algo"] = [algo]
            losses["algo_batch"] = [algo_batch]
            times["algo"] = [algo]
            times["algo_batch"] = [algo_batch]
            var["algo"] = [algo]
            var["algo_batch"] = [algo_batch]
            data["env"] = env
            losses["env"] = [env]
            times["env"] = [env]
            var["env"] = [env]
            data["num_reevals"] = num_reevals
            losses["num_reevals"] = [num_reevals]
            times["num_reevals"] = [num_reevals]
            var["num_reevals"] = [num_reevals]
            data[compare_size] = size
            losses[compare_size] = [size]
            times[compare_size] = [size]
            var[compare_size] = [size]
            data["rep"] = rep
            losses["rep"] = [rep]
            times["rep"] = [rep]
            var["rep"] = [rep]

            # Concatenate all frames to existing ones
            all_data = pd.concat([all_data, data], ignore_index=True)
            all_losses = pd.concat(
                [all_losses, pd.DataFrame.from_dict(losses)], ignore_index=True
            )
            all_times = pd.concat(
                [all_times, pd.DataFrame.from_dict(times)], ignore_index=True
            )
            all_var = pd.concat(
                [all_var, pd.DataFrame.from_dict(var)], ignore_index=True
            )

            rep += 1

        except Exception:
            if not args.no_traceback:
                print("\n!!!WARNING!!! Cannot read", metrics_file, ".")
                traceback.print_exc()

    # Normalise all_var per env
    final_all_var = pd.DataFrame()
    for env_name in all_var["env"].drop_duplicates().values:
        env_all_var = all_var[all_var["env"] == env_name].reset_index(drop=True)
        for column in env_all_var.columns:
            if env_all_var.dtypes[column] == "float64":
                env_all_var[column] = (
                    (env_all_var[column] - env_all_var[column].min())
                    / (env_all_var[column].max() - env_all_var[column].min())
                    * 100
                )
        final_all_var = pd.concat([final_all_var, env_all_var], ignore_index=True)
    all_var = final_all_var

    try:
        # Uniformise time values across replications
        all_data = uniformise_xaxis(all_data, "time")
    except Exception:
        if not args.no_traceback:
            print("\n!!!WARNING!!! Could not uniformise the time values across reps.")
            traceback.print_exc()

except Exception:
    if not args.no_traceback and error:
        print("\n!!!WARNING!!! Cannot read progress graphs.")
        traceback.print_exc()

#################
# Plot p-values #


try:
    if not args.plot_p_values:
        error = False
        assert 0
    error = True

    # Calling function to plot all p-values
    print("\nPlotting p-values")
    p_values(
        plot_folder=plot_folder,
        compare_size=compare_size,
        times_data=all_times,
        losses_data=all_losses,
        var_data=all_var,
    )
    p_values(
        plot_folder=plot_folder,
        compare_size=compare_size,
        times_data=all_times,
        losses_data=all_losses,
        var_data=all_var,
        prefixe="in_cell_",
        prefixe_title="In-Cell ",
    )


except Exception:
    if not args.no_traceback and error:
        print("\n!!!WARNING!!! Cannot plot p-values.")
        traceback.print_exc()

#######################
# Plot summary graphs #

try:
    if not args.plot_summary:
        error = False
        assert 0
    error = True

    # Customize matplotlib params
    print("\nPlotting summary graphs")
    font_size = 22
    params = {
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "axes.titlepad": 10,
        "lines.linewidth": 2,
        # "lines.markeredgecolor": "black",
        # "lines.markeredgewidth": 2,
        "lines.markersize": 8,
        # "xtick.major.pad": 20,
        # "ytick.major.pad": 20,
    }
    mpl.rcParams.update(params)

    # Print the metrics per num_reevals
    for num_reevals in all_data["num_reevals"].drop_duplicates().values:

        # Extract and sort data
        num_reeval_losses = all_losses[all_losses["num_reevals"] == num_reevals]
        num_reeval_losses = sort_data(num_reeval_losses, ["algo", "rep", compare_size])
        num_reeval_times = all_times[all_times["num_reevals"] == num_reevals]
        num_reeval_times = sort_data(num_reeval_times, ["algo", "rep", compare_size])
        num_reeval_var = all_var[all_var["num_reevals"] == num_reevals]
        num_reeval_var = sort_data(num_reeval_var, ["algo", "rep", compare_size])

        # Summary graph loss only (for paper)
        print("   Loss summary", compare_title, "graphs", num_reevals)
        if not args.category_plots:
            summary_loss_allenv(
                num_reevals=num_reevals,
                compare_size=compare_size,
                compare_title=compare_title,
                plot_folder=plot_folder,
                losses_data=num_reeval_losses,
                var_data=num_reeval_var,
                legend_columns=args.summary_legend_columns,
            )
            summary_loss_allenv(
                num_reevals=num_reevals,
                compare_size=compare_size,
                compare_title=compare_title,
                plot_folder=plot_folder,
                losses_data=num_reeval_losses,
                var_data=num_reeval_var,
                prefixe="in_cell_",
                legend_columns=args.summary_legend_columns,
            )
        else:
            summary_loss_categories_allenv(
                num_reevals=num_reevals,
                compare_size=compare_size,
                compare_title=compare_title,
                plot_folder=plot_folder,
                losses_data=num_reeval_losses,
                var_data=num_reeval_var,
                legend_columns=args.summary_legend_columns,
            )
            summary_loss_categories_allenv(
                num_reevals=num_reevals,
                compare_size=compare_size,
                compare_title=compare_title,
                plot_folder=plot_folder,
                losses_data=num_reeval_losses,
                var_data=num_reeval_var,
                prefixe="in_cell_",
                legend_columns=args.summary_legend_columns,
            )


except Exception:
    if not args.no_traceback and error:
        print("\n!!!WARNING!!! Cannot plot summary graphs.")
        traceback.print_exc()


#######################
# Plot pareto graphs #

try:
    if not args.plot_pareto:
        error = False
        assert 0
    error = True

    # Customize matplotlib params
    print("\nPlotting pareto graphs")
    font_size = 15
    params = {
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "axes.titlepad": 10,
        "lines.linewidth": 2,
        # "lines.markeredgecolor": "black",
        # "lines.markeredgewidth": 2,
        "lines.markersize": 8,
        # "xtick.major.pad": 20,
        # "ytick.major.pad": 20,
    }
    mpl.rcParams.update(params)

    # Print the metrics per num_reevals
    for num_reevals in all_data["num_reevals"].drop_duplicates().values:

        # Extract and sort data
        num_reeval_losses = all_losses[all_losses["num_reevals"] == num_reevals]
        num_reeval_losses = sort_data(num_reeval_losses, ["algo", "rep", compare_size])
        num_reeval_times = all_times[all_times["num_reevals"] == num_reevals]
        num_reeval_times = sort_data(num_reeval_times, ["algo", "rep", compare_size])
        num_reeval_var = all_var[all_var["num_reevals"] == num_reevals]
        num_reeval_var = sort_data(num_reeval_var, ["algo", "rep", compare_size])

        # Pareto graph
        print("   Pareto graphs", num_reevals)
        if not args.category_plots:
            pareto_plot(
                plot_folder=plot_folder,
                num_reevals=num_reevals,
                compare_size=compare_size,
                times_data=num_reeval_times,
                legend_columns=args.pareto_legend_columns,
                pareto_column=True,
            )
            pareto_plot(
                plot_folder=plot_folder,
                num_reevals=num_reevals,
                compare_size=compare_size,
                times_data=num_reeval_times,
                prefixe="in_cell_",
                legend_columns=args.pareto_legend_columns,
                pareto_column=True,
            )
        else:
            pareto_categories(
                plot_folder=plot_folder,
                num_reevals=num_reevals,
                compare_size=compare_size,
                times_data=num_reeval_times,
                legend_columns=args.pareto_legend_columns,
                pareto_column=args.category_plots,
            )
            pareto_categories(
                plot_folder=plot_folder,
                num_reevals=num_reevals,
                compare_size=compare_size,
                times_data=num_reeval_times,
                prefixe="in_cell_",
                legend_columns=args.pareto_legend_columns,
                pareto_column=args.category_plots,
            )

except Exception:
    if not args.no_traceback and error:
        print("\n!!!WARNING!!! Cannot plot pareto graphs.")
        traceback.print_exc()

#######################
# Plot paper archives #

try:
    if not args.plot_paper_archives:
        error = False
        assert 0
    error = True

    print("\nPlotting paper archive")

    # Customize matplotlib params
    font_size = 25
    params = {
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "axes.titlepad": 10,
        "lines.linewidth": 2,
        # "lines.markeredgecolor": "black",
        # "lines.markeredgewidth": 2,
        "lines.markersize": 8,
        # "xtick.major.pad": 20,
        # "ytick.major.pad": 20,
    }
    mpl.rcParams.update(params)

    # Create dataframe
    repertoire_folders = pd.DataFrame()
    env_min_max = pd.DataFrame()
    prefixe = args.paper_archives_prefixe

    # Separate files for each env
    for env in config_frame["env"].drop_duplicates().values:
        for num_reevals in config_frame["num_reevals"].drop_duplicates().values:
            try:
                print("\n    Reading for", env)

                env_config_frame = config_frame[
                    (config_frame["env"] == env)
                    & (config_frame["num_reevals"] == num_reevals)
                ].reset_index(drop=True)

                # Finding min and max for all plots
                min_fitness = jnp.inf
                max_fitness = -jnp.inf
                min_fit_var = jnp.inf
                max_fit_var = -jnp.inf
                min_desc_var = jnp.inf
                max_desc_var = -jnp.inf
                min_bd = [
                    float(bd) for bd in env_config_frame["min_bd"][0][1:-1].split(" ")
                ]
                max_bd = [
                    float(bd) for bd in env_config_frame["max_bd"][0][1:-1].split(" ")
                ]

                # One archive for each algorithm
                for algo in env_config_frame["algo"].drop_duplicates().values:
                    algo_config_frame = env_config_frame[
                        env_config_frame["algo"] == algo
                    ].reset_index(drop=True)
                    for size in (
                        algo_config_frame[compare_size].drop_duplicates().values
                    ):
                        if algo_config_frame[
                            algo_config_frame[compare_size] == size
                        ].empty:
                            print("    Size", size, "does not exist for", algo)
                            continue

                        size_config_frame = algo_config_frame[
                            algo_config_frame[compare_size] == size
                        ].reset_index(drop=True)
                        try:
                            reeval_repertoire_folder = get_folder_name(
                                size_config_frame,
                                f"{prefixe}reeval_repertoire_folder",
                                0,
                            )
                            fit_var_repertoire_folder = get_folder_name(
                                size_config_frame,
                                f"{prefixe}fit_var_repertoire_folder",
                                0,
                            )
                            desc_var_repertoire_folder = get_folder_name(
                                size_config_frame,
                                f"{prefixe}desc_var_repertoire_folder",
                                0,
                            )

                            # Open reeval repertoire to find min and max fitness
                            reeval_fitnesses = jnp.load(
                                os.path.join(reeval_repertoire_folder, "fitnesses.npy")
                            )
                            reeval_fitnesses_inf = jnp.where(
                                reeval_fitnesses == -jnp.inf, jnp.inf, reeval_fitnesses
                            )
                            min_fitness = min(
                                min_fitness, float(min(reeval_fitnesses_inf))
                            )
                            max_fitness = max(max_fitness, float(max(reeval_fitnesses)))

                            # Open fit_var repertoiret to find min and max fit_var
                            variances = jnp.load(
                                os.path.join(fit_var_repertoire_folder, "fitnesses.npy")
                            )
                            variances_inf = jnp.where(
                                variances == -jnp.inf, jnp.inf, variances
                            )
                            min_fit_var = min(min_fit_var, float(min(variances_inf)))
                            max_fit_var = max(max_fit_var, float(max(variances)))

                            # Open desc_var repertoiret to find min and max desc_var
                            variances = jnp.load(
                                os.path.join(
                                    desc_var_repertoire_folder, "fitnesses.npy"
                                )
                            )
                            variances_inf = jnp.where(
                                variances == -jnp.inf, jnp.inf, variances
                            )
                            min_desc_var = min(min_desc_var, float(min(variances_inf)))
                            max_desc_var = max(max_desc_var, float(max(variances)))

                            # Update frame
                            repertoire_folders = pd.concat(
                                [
                                    repertoire_folders,
                                    pd.DataFrame.from_dict(
                                        {
                                            "env": [env],
                                            "num_reevals": [num_reevals],
                                            "algo": [algo],
                                            compare_size: [size],
                                            "reeval_repertoire_folder": [
                                                reeval_repertoire_folder
                                            ],
                                            "fit_var_repertoire_folder": [
                                                fit_var_repertoire_folder
                                            ],
                                            "desc_var_repertoire_folder": [
                                                desc_var_repertoire_folder
                                            ],
                                        }
                                    ),
                                ],
                                ignore_index=True,
                            )
                            repertoire_folders = sort_data(
                                repertoire_folders,
                                ["env", "num_reevals", "algo", compare_size],
                            )
                        except Exception:
                            if not args.no_traceback:
                                print(
                                    "\n!!!WARNING!!! Cannot find folder name for",
                                    algo,
                                    "and",
                                    size,
                                )
                                traceback.print_exc()

                # Min-max frame for archives
                env_min_max = pd.concat(
                    [
                        env_min_max,
                        pd.DataFrame.from_dict(
                            {
                                "env": [env],
                                "min_bd": [min_bd],
                                "max_bd": [max_bd],
                                "min_fitness": min_fitness,
                                "max_fitness": max_fitness,
                                "min_fit_var": min_fit_var,
                                "max_fit_var": max_fit_var,
                                "min_desc_var": min_desc_var,
                                "max_desc_var": max_desc_var,
                            }
                        ),
                    ],
                    ignore_index=True,
                )

            except Exception:
                if not args.no_traceback and error:
                    print(
                        "\n!!!WARNING!!! Cannot plot paper archives for",
                        env,
                        "and",
                        num_reevals,
                    )
                    traceback.print_exc()

    # One final common file for all envs
    for num_reevals in config_frame["num_reevals"].drop_duplicates().values:
        size = 16384

        try:
            env_repertoire_folders = repertoire_folders[
                (repertoire_folders[compare_size] == size)
                & (repertoire_folders["num_reevals"] == num_reevals)
            ]
            plot_env_paper_archives_fn = partial(
                plot_env_paper_archives,
                repertoire_folders=env_repertoire_folders,
                env_min_max=env_min_max,
                no_traceback=args.no_traceback,
            )

            # First print all reevaluated archives in a common file
            file_name = f"{plot_folder}/{num_reevals}reevals_paper_archives.png"
            plot_env_paper_archives_fn("reeval_repertoire_folder", file_name, "fitness")

            # Second print all fit-var archives in a common file
            file_name = f"{plot_folder}/{num_reevals}reevals_fit_var_paper_archives.png"
            plot_env_paper_archives_fn(
                "fit_var_repertoire_folder", file_name, "fit_var"
            )

            # Third print all desc-var archives in a common file
            file_name = (
                f"{plot_folder}/{num_reevals}reevals_desc_var_paper_archives.png"
            )
            plot_env_paper_archives_fn(
                "desc_var_repertoire_folder", file_name, "desc_var"
            )

        except Exception:
            if not args.no_traceback and error:
                print("\n!!!WARNING!!! Cannot plot env paper archives for", num_reevals)
                traceback.print_exc()


except Exception:
    if not args.no_traceback and error:
        print("\n!!!WARNING!!! Cannot plot paper archives.")
        traceback.print_exc()


#######################
# Save visualisations #

try:
    if not args.visualisation:
        error = False
        assert 0
    error = True

    print("\nVisualising")

    # Customize matplotlib params
    print("\nPlotting metrics graphs")
    font_size = 16
    params = {
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
        "axes.titlepad": 10,
        "lines.linewidth": 3,
    }
    mpl.rcParams.update(params)

    # For each line in config file
    for line in range(config_frame.shape[0]):
        print("\n    Visualising results in line", line)

        try:
            # Init a random key
            seed = config_frame["seed"][line]
            env_name = config_frame["env"][line]
            size = config_frame[compare_size][line]
            algo = config_frame["algo"][line].replace(" ", "_")
            random_key = jax.random.PRNGKey(seed)
            policy_hidden_layer_sizes = config_frame["policy_hidden_layer_sizes"][line]
            policy_hidden_layer_sizes = (
                tuple([int(x) for x in policy_hidden_layer_sizes.split("_")])
                if type(policy_hidden_layer_sizes) == str
                else tuple([policy_hidden_layer_sizes])
            )

            # Init environment
            (
                env,
                scoring_fn,
                policy_structure,
                init_policies,
                _,
                _,
                min_bd,
                max_bd,
                qd_offset,
                num_descriptors,
                random_key,
            ) = set_up_environment(
                deterministic=args.deterministic,
                env_name=env_name,
                episode_length=config_frame["episode_length"][line],
                batch_size=args.replications,
                policy_hidden_layer_sizes=policy_hidden_layer_sizes,
                random_key=random_key,
            )

            # Reconstruction function
            print("      Loading repertoire")
            init_policy = jax.tree_map(lambda x: x[0], init_policies)
            _, reconstruction_fn = jax.flatten_util.ravel_pytree(init_policy)

            # Load repertoire
            repertoire_folder = get_folder_name(config_frame, "repertoire_folder", line)
            repertoire_folder += "/"
            repertoire = MapElitesRepertoire.load(
                reconstruction_fn=reconstruction_fn,
                path=repertoire_folder,
            )

            if args.indiv > 0:
                # Take given indiv
                print("      Visualising given individual")
                indiv_idx = args.indiv
                assert (
                    indiv_idx < repertoire.fitnesses.shape[0]
                    and repertoire.fitnesses[indiv_idx] > -jnp.inf
                ), "!!!ERROR!!! No indiv here."
            elif args.best_indiv:
                # Take best indiv
                print("      Visualising best individual")
                indiv_idx = jnp.argmax(repertoire.fitnesses)
            else:
                # Take random indiv
                print("      Visualising random individual")
                indices = jnp.arange(0, repertoire.fitnesses.shape[0], 1)
                repertoire_empty = repertoire.fitnesses == -jnp.inf
                p = (1.0 - repertoire_empty) / jnp.sum(1.0 - repertoire_empty)
                random_key, subkey = jax.random.split(random_key)
                indiv_idx = jax.random.choice(subkey, indices, shape=(1,), p=p)[0]

            print("      Visualising individual with indexe:", indiv_idx)
            before_fitness = repertoire.fitnesses[indiv_idx]
            min_fitness = min(repertoire.fitnesses[repertoire.fitnesses > -jnp.inf])
            max_fitness = max(repertoire.fitnesses)
            before_descriptor = repertoire.descriptors[indiv_idx]

            #######################
            # Saving as HTML file #

            if args.save_html and env_name in ENV_OPTIMISATION:
                print("!!!WARNING!!! No html save for optimisation tasks.")
            elif args.save_html:
                my_params = jax.tree_util.tree_map(
                    lambda x: x[indiv_idx].squeeze(), repertoire.genotypes
                )
                file_name = (
                    f"{plot_folder}/{env_name}_visualisation_"
                    + f"{algo}_{size}_{seed}_{indiv_idx}.html"
                )
                print("      Saving visualisation in html file:", file_name)
                save_html(
                    file_name=file_name,
                    env=env,
                    my_params=my_params,
                    policy_structure=policy_structure,
                    random_key=random_key,
                    is_env_control=env_name in ENV_CONTROL,
                )

            ############################################
            # Perform replications for archive display #

            print("      Starting replications")
            random_key, subkey = jax.random.split(random_key)
            my_params = jax.tree_util.tree_map(
                lambda x: jnp.repeat(
                    jnp.expand_dims(x[indiv_idx], axis=0), args.replications, axis=0
                ),
                repertoire.genotypes,
            )
            fitnesses, descriptors, _, _ = scoring_fn(my_params, subkey)

            # Display replications bd results in a grid
            file_name = (
                f"{plot_folder}/{env_name}_{args.replications}rep"
                + f"_{algo}_{size}_{seed}_{indiv_idx}.png"
            )
            print("         Saving replication results in:", file_name)
            plot_visualisation_archive(
                file_name=file_name,
                replications=args.replications,
                repertoire=repertoire,
                fitnesses=fitnesses,
                descriptors=descriptors,
                before_descriptor=before_descriptor,
                before_fitness=before_fitness,
                min_bd=min_bd,
                max_bd=max_bd,
                vmin=min_fitness,
                vmax=max_fitness,
                paper_plot=args.paper_plot,
                average=not args.paper_plot,
                median=not args.paper_plot,
            )

            # If paper, display also the original grid
            if args.paper_plot:
                file_name = f"{plot_folder}/{env_name}_{algo}_{size}_{seed}.png"
                print("         Saving replication results in:", file_name)
                plot_visualisation_archive(
                    file_name=file_name,
                    replications=0,
                    repertoire=repertoire,
                    fitnesses=jnp.delete(repertoire.fitnesses, indiv_idx, axis=0),
                    descriptors=jnp.delete(repertoire.descriptors, indiv_idx, axis=0),
                    before_descriptor=before_descriptor,
                    before_fitness=before_fitness,
                    min_bd=min_bd,
                    max_bd=max_bd,
                    vmin=min_fitness,
                    vmax=max_fitness,
                    paper_plot=True,
                )

            print("\n    Finished with line", line)

        except Exception:
            if not args.no_traceback:
                print(
                    "\n!!!WARNING!!! Cannot process with visualisation for line:", line
                )
                traceback.print_exc()


except Exception:
    if not args.no_traceback and error:
        print("\n!!!WARNING!!! Cannot process with visualisation.")
        traceback.print_exc()

print("Finished plotting")
