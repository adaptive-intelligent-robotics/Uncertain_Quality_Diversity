import os
import time
import traceback
from typing import Dict, List, Tuple

import jax.numpy as jnp
import pandas as pd
from natsort import natsort_keygen


def load_results(
    save_folder: str,
    plot_folder: str,
    plot_algos: List,
    exclude_algos: List,
    exclude_sizes: List,
    not_name: List,
    replace_name: Dict,
    time_pourcent: float,
    pourcent_value: bool,
    compare_size: str,
    order: List,
) -> Tuple:
    """Main function to aload all results."""

    ############################
    # 1. Find all config files #

    # If already an all_config file in the folder, use it
    file_name = f"{plot_folder}_csv/all_configs.csv"
    if os.path.exists(file_name):
        print("Loading existing all_configs in", file_name)
        config_frame = pd.read_csv(file_name, header=0, index_col=False)

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
            config_frame = pd.concat(
                [config_frame, sub_config_frame], ignore_index=True
            )
        assert (
            config_frame.shape[0] != 0
        ), "\n!!!ERROR!!! No runs refered in config files.\n"

        # Name algorithms
        print("\nSetting up algorithms names")
        use_in_name = []
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
                algo = algo.replace("Deep-Grid", "Deep-Grid-sampling")

            for name in replace_name.keys():
                algo = algo.replace(name, replace_name[name])

            for name in use_in_name:
                algo += " " + name + ":" + str(config_frame[name][line])
            algo_batch = algo + " - " + str(config_frame[compare_size][line])
            algos.append(algo)
            algos_batch.append(algo_batch)

        config_frame["algo"] = algos
        config_frame["algo_batch"] = algos_batch
        config_frame = config_frame.reset_index(drop=True)

        # Write in file
        config_frame.to_csv(file_name, index=None)

    print("    Found", config_frame.shape[0], "runs, with algo names:")
    print(config_frame["algo"].drop_duplicates().reset_index(drop=True))

    ########################
    # 2. Filter algorithms #

    # Filter algos that does not need to be ploted
    if plot_algos != [""]:
        config_frame = config_frame[config_frame["run"].isin(plot_algos)]
    if exclude_algos != [""]:
        config_frame = config_frame[
            ~config_frame["run"].str.contains("|".join(exclude_algos))
        ]
    if exclude_sizes != []:
        config_frame = config_frame[~(config_frame[compare_size].isin(exclude_sizes))]
    config_frame = config_frame.reset_index(drop=True)
    print(
        "\n    After filtering, left with",
        config_frame.shape[0],
        "runs, with algo names:",
    )
    print(config_frame["algo"].drop_duplicates().reset_index(drop=True))
    print(
        "and with compare sizes:",
        config_frame[compare_size].drop_duplicates().reset_index(drop=True),
    )
    assert config_frame.shape[0] != 0, "\n!!!ERROR!!! No algos left to plot.\n"

    #####################################
    # 3. Try to load already saved data #

    file_name_data = f"{plot_folder}_csv/all_data.csv"
    file_name_losses = f"{plot_folder}_csv/all_losses.csv"
    file_name_times = f"{plot_folder}_csv/all_times.csv"
    file_name_var = f"{plot_folder}_csv/all_var.csv"
    if (
        os.path.exists(file_name_data)
        and os.path.exists(file_name_losses)
        and os.path.exists(file_name_times)
        and os.path.exists(file_name_var)
    ):
        print("\nLoading existing progress and loss datas in:")
        print(file_name_data)
        print(file_name_losses)
        print(file_name_times)
        print(file_name_var)
        all_data = pd.read_csv(file_name_data, header=0, index_col=False)
        all_losses = pd.read_csv(file_name_losses, header=0, index_col=False)
        all_times = pd.read_csv(file_name_times, header=0, index_col=False)
        all_var = pd.read_csv(file_name_var, header=0, index_col=False)

        # Re-order
        all_data = sort_data(all_data, ["algo", "rep", "eval"], order)
        all_losses = sort_data(all_losses, ["algo", "rep", compare_size], order)
        all_times = sort_data(all_times, ["algo", "rep", compare_size], order)
        all_var = sort_data(all_var, ["algo", "rep", compare_size], order)

        return config_frame, all_data, all_losses, all_times, all_var

    #######################################
    # 4. Fill in max_gen and replications #

    print("\nReading max gen and replications")
    start_t = time.time()

    max_gen_frame = pd.DataFrame(columns=["env", "epoch"])
    replications_frame = pd.DataFrame(
        columns=["env", "num_reevals", "algo", compare_size, "num_rep"]
    )

    # Initialise replications_frame
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

    # Go through all metrics files to fill max_gen_frame and replications_frame
    seeds = []
    for line in range(config_frame.shape[0]):
        try:
            # Get the config for this line
            env = config_frame["env"][line]
            num_reevals = config_frame["num_reevals"][line]  # 0
            size = config_frame[compare_size][line]
            algo = config_frame["algo"][line]

            if config_frame["seed"][line] in seeds:
                print("twice seed:", config_frame["seed"][line])
            seeds.append(config_frame["seed"][line])

            # Open data to compute maximum number of generations
            folder = config_frame["folder"][line]
            metrics_file = config_frame["metrics_file"][line]
            in_cell_metrics_file = config_frame["in_cell_metrics_file"][line]
            metrics_file = metrics_file[metrics_file.rfind("/") + 1 :]
            in_cell_metrics_file = in_cell_metrics_file[
                in_cell_metrics_file.rfind("/") + 1 :
            ]
            metrics_file = os.path.join(folder, metrics_file)
            in_cell_metrics_file = os.path.join(folder, in_cell_metrics_file)
            data = pd.read_csv(metrics_file, index_col=False)

            # Add maximum number of generations to frame
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
            print("\n!!!WARNING!!! Cannot read line", line, ":")
            print(config_frame.loc[line])
            traceback.print_exc()

    print("\nMax epoch for each environment:")
    print(max_gen_frame)
    print("\n")

    # Remove empty replications from replications_frame
    replications_frame = replications_frame[replications_frame["num_rep"] != 0]
    replications_frame = replications_frame.sort_values(
        ["env", "num_reevals"], ignore_index=True
    )
    replications_frame = sort_data(replications_frame, ["algo", compare_size], order)

    # Save replications frame as csv
    print("\nReplications:")
    print(replications_frame)
    print("\n")
    replications_frame.to_csv(
        f"{plot_folder}/replications_frame.csv",
        index=None,
        sep=",",
    )

    print("Time to read replications and max_gen_frame:", time.time() - start_t)

    ################
    # 5. Load data #

    print("\nReading data")

    # Create the metrics dataframe
    all_data = pd.DataFrame()
    all_losses = pd.DataFrame()
    all_times = pd.DataFrame()
    all_var = pd.DataFrame()

    # Function to add config info to a frame
    def add_config(
        frame: pd.DataFrame,
        run: List,
        algo: List,
        algo_batch: List,
        num_centroids: List,
        env: List,
        num_reevals: List,
        size: List,
        sampling_size: List,
        batch_size: List,
        rep: List,
    ) -> pd.DataFrame:
        frame["run"] = run
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

    # Go through all metrics files
    start_t = time.time()
    rep = 0
    for line in range(config_frame.shape[0]):

        try:
            # Get the config for this line
            run = config_frame["run"][line]
            algo = config_frame["algo"][line]
            algo_batch = config_frame["algo_batch"][line]
            env = config_frame["env"][line]
            num_centroids = config_frame["num_centroids"][line]
            num_reevals = config_frame["num_reevals"][line]  # 0
            size = config_frame[compare_size][line]
            batch_size = config_frame["batch_size"][line]
            sampling_size = config_frame["sampling_size"][line]

            folder = config_frame["folder"][line]
            metrics_file = config_frame["metrics_file"][line]
            in_cell_metrics_file = config_frame["in_cell_metrics_file"][line]
            metrics_file = metrics_file[metrics_file.rfind("/") + 1 :]
            in_cell_metrics_file = in_cell_metrics_file[
                in_cell_metrics_file.rfind("/") + 1 :
            ]
            metrics_file = os.path.join(folder, metrics_file)
            in_cell_metrics_file = os.path.join(folder, in_cell_metrics_file)

            # Read metrics
            data = pd.read_csv(metrics_file, index_col=False)
            if data.empty:
                print(f"!!!WARNING!!! {metrics_file} is empty.")
                continue

            # Read in-cell metrics
            in_cell_data = pd.read_csv(in_cell_metrics_file, index_col=False)

            # Merge metrics and in-cell metrics in the same frame
            to_merge = [  # [Name in data, Name in in_cell_data]
                ["in_cell_qd_score", "in_cell_qd_score"],
                ["in_cell_coverage", "in_cell_coverage"],
                ["in_cell_max_fitness", "in_cell_max_fitness"],
                ["in_cell_reeval_qd_score", "in_cell_reeval_qd_score"],
                ["in_cell_reeval_coverage", "in_cell_reeval_coverage"],
                ["in_cell_reeval_max_fitness", "in_cell_reeval_max_fitness"],
                ["in_cell_fit_reeval_qd_score", "in_cell_fit_reeval_qd_score"],
                ["in_cell_fit_reeval_coverage", "in_cell_fit_reeval_coverage"],
                ["in_cell_fit_reeval_max_fitness", "in_cell_fit_reeval_max_fitness"],
                ["in_cell_desc_reeval_qd_score", "in_cell_desc_reeval_qd_score"],
                ["in_cell_desc_reeval_coverage", "in_cell_desc_reeval_coverage"],
                ["in_cell_desc_reeval_max_fitness", "in_cell_desc_reeval_max_fitness"],
                ["in_cell_fit_var_qd_score", "in_cell_fit_var_qd_score"],
                ["in_cell_fit_var_coverage", "in_cell_fit_var_coverage"],
                ["in_cell_fit_var_max_fitness", "in_cell_fit_var_max_fitness"],
                ["in_cell_desc_var_qd_score", "in_cell_desc_var_qd_score"],
                ["in_cell_desc_var_coverage", "in_cell_desc_var_coverage"],
                ["in_cell_desc_var_max_fitness", "in_cell_desc_var_max_fitness"],
                [
                    "in_cell_reeval_additional_qd_score",
                    "in_cell_reeval_additional_qd_score",
                ],
                [
                    "in_cell_reeval_additional_max_fitness",
                    "in_cell_reeval_additional_max_fitness",
                ],
                [
                    "in_cell_reeval_additional_min_fitness",
                    "in_cell_reeval_additional_min_fitness",
                ],
            ]
            for column_to_merge in to_merge:
                if column_to_merge[0] in in_cell_data.columns:
                    data[column_to_merge[0]] = in_cell_data[column_to_merge[1]]

            # Filter datas after max_gen
            data = data[
                data["epoch"]
                <= max_gen_frame[max_gen_frame["env"] == env]["epoch"].values[0]
            ]

            # Add run info to frame
            data = add_config(
                frame=data,
                run=run,
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

            # Compute losses
            losses: Dict[str, List[float]] = {}
            losses = compute_loss("qd_score", data, losses)
            losses = compute_loss("coverage", data, losses)
            losses = compute_loss("max_fitness", data, losses)
            losses = compute_loss("qd_score", data, losses, prefixe="in_cell_")
            losses = compute_loss("coverage", data, losses, prefixe="in_cell_")
            losses = compute_loss("max_fitness", data, losses, prefixe="in_cell_")

            # Compute times to % convergence
            times: Dict[str, List[float]] = {}
            times = compute_time(
                data,
                env,
                algo,
                size,
                times,
                pourcent=time_pourcent,
                pourcent_value=pourcent_value,
            )
            times = compute_time(
                data,
                env,
                algo,
                size,
                times,
                pourcent=time_pourcent,
                prefixe="in_cell_",
                pourcent_value=pourcent_value,
            )

            # Compute final variance
            var: Dict[str, List[float]] = {}
            var = compute_var(data, var, num_centroids)
            var = compute_var(data, var, num_centroids, prefixe="in_cell_")

            # Add run info to frame
            losses = add_config(
                frame=losses,
                run=[run],
                algo=[algo],
                algo_batch=[algo_batch],
                num_centroids=[num_centroids],
                env=[env],
                num_reevals=[num_reevals],
                size=[size],
                sampling_size=[sampling_size],
                batch_size=[batch_size],
                rep=[rep],
            )
            times = add_config(
                frame=times,
                run=[run],
                algo=[algo],
                algo_batch=[algo_batch],
                num_centroids=[num_centroids],
                env=[env],
                num_reevals=[num_reevals],
                size=[size],
                sampling_size=[sampling_size],
                batch_size=[batch_size],
                rep=[rep],
            )
            var = add_config(
                frame=var,
                run=[run],
                algo=[algo],
                algo_batch=[algo_batch],
                num_centroids=[num_centroids],
                env=[env],
                num_reevals=[num_reevals],
                size=[size],
                sampling_size=[sampling_size],
                batch_size=[batch_size],
                rep=[rep],
            )
            losses = pd.DataFrame.from_dict(losses)
            times = pd.DataFrame.from_dict(times)
            var = pd.DataFrame.from_dict(var)

            # Concatenate all frames to existing ones
            all_data = pd.concat([all_data, data], ignore_index=True)
            all_losses = pd.concat([all_losses, losses], ignore_index=True)
            all_times = pd.concat([all_times, times], ignore_index=True)
            all_var = pd.concat([all_var, var], ignore_index=True)

            # Increment rep counter
            rep += 1

        except Exception:
            print("\n!!!WARNING!!! Cannot read", metrics_file, ".")
            traceback.print_exc()

    print("Time to read all metrics frames:", time.time() - start_t)
    start_t = time.time()

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
    print("Time to normalise all variances:", time.time() - start_t)
    start_t = time.time()

    # Compute complement to additional for all env
    if len(all_times["env"].drop_duplicates().values) > 1:
        all_times["normalised_additional_average"] = all_times.groupby(
            "env", group_keys=False
        ).apply(
            lambda g: (g["additional_average"] - g["additional_average"].min())
            / (g["additional_average"].max() - g["additional_average"].min())
        )
        all_times["normalised_reeval_additional_average"] = all_times.groupby(
            "env", group_keys=False
        ).apply(
            lambda g: (
                g["reeval_additional_average"] - g["reeval_additional_average"].min()
            )
            / (
                g["reeval_additional_average"].max()
                - g["reeval_additional_average"].min()
            )
        )
        all_times["complement_additional_average"] = all_times.groupby(
            "env", group_keys=False
        ).apply(lambda g: 1 - g["normalised_additional_average"])
        all_times["complement_reeval_additional_average"] = all_times.groupby(
            "env", group_keys=False
        ).apply(lambda g: 1 - g["normalised_reeval_additional_average"])
    else:
        all_times["normalised_additional_average"] = (
            all_times["additional_average"] - all_times["additional_average"].min()
        ) / (
            all_times["additional_average"].max()
            - all_times["additional_average"].min()
        )
        all_times["normalised_reeval_additional_average"] = (
            all_times["reeval_additional_average"]
            - all_times["reeval_additional_average"].min()
        ) / (
            all_times["reeval_additional_average"].max()
            - all_times["reeval_additional_average"].min()
        )
        all_times["complement_additional_average"] = (
            1 - all_times["normalised_additional_average"]
        )
        all_times["complement_reeval_additional_average"] = (
            1 - all_times["normalised_reeval_additional_average"]
        )

    print("Time to take complement of all additional:", time.time() - start_t)
    start_t = time.time()

    # try:
    #    # Uniformise time values across replications
    #    all_data = uniformise_xaxis(all_data, "time")
    # except Exception:
    #    print("\n!!!WARNING!!! Could not uniformise the time values across reps.")
    #    traceback.print_exc()
    # print("Time to uniformise all time:", time.time() - start_t)
    # start_t = time.time()
    print("\n!!!WARNING!!! Not doing the time uniformisation to save some time.")

    # Sort datas
    all_data = sort_data(all_data, ["algo", "rep", "eval"], order)
    all_losses = sort_data(all_losses, ["algo", "rep", compare_size], order)
    all_times = sort_data(all_times, ["algo", "rep", compare_size], order)
    all_var = sort_data(all_var, ["algo", "rep", compare_size], order)

    # Save datas as csv
    all_data.to_csv(file_name_data, index=None)
    all_losses.to_csv(file_name_losses, index=None)
    all_times.to_csv(file_name_times, index=None)
    all_var.to_csv(file_name_var, index=None)

    return config_frame, all_data, all_losses, all_times, all_var


#################################
# Metrics computation functions #


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
    if original == 0.0 or original == -jnp.inf:
        if average == 0.0:
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

    # Handle Arm as a specific case
    # The compilation is usually longer than the actual run
    # So time comparison is just going to be compilation difference
    # As Parallel-Adaptive-Sampling is the only non-jitted approach
    # It suffers from this difference.
    offset_time = 0
    if "arm" in env_name and "Parallel" in algo:
        if size == 256:
            offset_time = 80
        elif size == 1024:
            offset_time = 36
        elif size == 4096:
            offset_time = 14
        elif size == 16384:
            offset_time = 4
        """
        # Offset the time from the three first points
        min_eval = data["eval"].drop_duplicates().nsmallest(1).iloc[-1]
        offset_time = data[data["eval"] == min_eval]["time"].values[0]
        data = data[data["eval"] > min_eval]
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
        if data[data[prefixe + "qd_score"] > pourcent_value]["epoch"].empty:
            pourcent_epoch = min_epoch
        else:
            pourcent_epoch = max(
                min_epoch,
                min(data[data[prefixe + "qd_score"] > pourcent_value]["epoch"].values),
            )
    pourcent_line = data[data["epoch"] == pourcent_epoch]

    # Finding corresponding eval, gen, time
    times[prefixe + "epoch"] = pourcent_epoch
    times[prefixe + "eval"] = pourcent_line["eval"].values[0]
    times[prefixe + "time"] = pourcent_line["time"].values[0] - offset_time

    # Setting all values
    if pourcent_value:
        line = pourcent_line
    else:
        line = final_line

    values_to_get = [
        prefixe + "qd_score",
        prefixe + "coverage",
        prefixe + "max_fitness",
        prefixe + "reeval_qd_score",
        prefixe + "reeval_coverage",
        prefixe + "reeval_min_fitness",
        prefixe + "reeval_max_fitness",
        prefixe + "fit_reeval_qd_score",
        prefixe + "fit_reeval_coverage",
        prefixe + "fit_reeval_min_fitness",
        prefixe + "fit_reeval_max_fitness",
        prefixe + "desc_reeval_qd_score",
        prefixe + "desc_reeval_coverage",
        prefixe + "desc_reeval_min_fitness",
        prefixe + "desc_reeval_max_fitness",
        prefixe + "additional_qd_score",
        prefixe + "additional_coverage",
        prefixe + "additional_min_fitness",
        prefixe + "additional_max_fitness",
        prefixe + "reeval_additional_qd_score",
        prefixe + "reeval_additional_coverage",
        prefixe + "reeval_additional_min_fitness",
        prefixe + "reeval_additional_max_fitness",
    ]

    for value in values_to_get:
        if value in line.columns:
            times[value] = line[value].values[0]
        else:
            times[value] = 0.0

    # Add the average value as well
    times[prefixe + "additional_average"] = times[prefixe + "additional_qd_score"] / (
        line["num_centroids"].values[0] * line[prefixe + "coverage"].values[0] / 100
    )
    times[prefixe + "reeval_additional_average"] = times[
        prefixe + "reeval_additional_qd_score"
    ] / (
        line["num_centroids"].values[0]
        * line[prefixe + "reeval_coverage"].values[0]
        / 100
    )

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
    if coverage == 0:
        average_fit_var_value = 0.0
        average_desc_var_value = 0.0
    else:
        average_fit_var_value = final_fit_var_value / (coverage / 100 * num_centroids)
        average_desc_var_value = final_desc_var_value / (coverage / 100 * num_centroids)

    # Fill in Dict
    var[prefixe + "fit_var_qd_score"] = final_fit_var_value  # type: ignore
    var[prefixe + "desc_var_qd_score"] = final_desc_var_value  # type: ignore
    var[prefixe + "avg_fit_var_qd_score"] = average_fit_var_value  # type: ignore
    var[prefixe + "avg_desc_var_qd_score"] = average_desc_var_value  # type: ignore
    return var  # type: ignore


###########################
# Data handling functions #


def extract_algo(data: pd.DataFrame, algos: str, columns: List) -> pd.DataFrame:
    sub_data = data[data["algo"].str.contains(algos)].reset_index(drop=True)
    if sub_data.empty:
        return sub_data
    sub_data = sub_data.sort_values(columns, key=natsort_keygen(), ignore_index=True)
    return sub_data


def extract_nonalgo(data: pd.DataFrame, algos: str, columns: List) -> pd.DataFrame:
    sub_data = data[~data["algo"].str.contains(algos)].reset_index(drop=True)
    if sub_data.empty:
        return sub_data
    sub_data = sub_data.sort_values(columns, key=natsort_keygen(), ignore_index=True)
    return sub_data


def sort_data(
    data: pd.DataFrame,
    columns: List,
    order: List,
) -> pd.DataFrame:
    final_data = extract_algo(data, order[0], columns=columns)
    left_data = extract_nonalgo(data, order[0], columns=columns)
    added_names = order[0]
    for i in range(1, len(order)):
        final_data = pd.concat(
            [
                final_data,
                extract_algo(left_data, order[i], columns=columns),
            ],
            ignore_index=True,
        )
        added_names += "|" + order[i]
        left_data = extract_nonalgo(data, added_names, columns=columns)
    final_data = pd.concat([final_data, left_data], ignore_index=True)
    return final_data


def uniformise_xaxis(data: pd.DataFrame, xaxis: str) -> pd.DataFrame:
    for exp in data["env"].drop_duplicates().values:
        for variant in data["algo"].drop_duplicates().values:
            sub_data = data[(data["env"] == exp) & (data["algo"] == variant)]
            sub_data = sub_data.sort_values(["rep", xaxis], ignore_index=True)
            replications = sub_data["rep"].drop_duplicates().values
            replications_evals = []  # type: List
            evals = []  # type: List
            need_rewrite = False
            for i, repl in enumerate(replications):
                replications_evals.append(
                    sub_data[sub_data["rep"] == repl][xaxis].values
                )
                if len(replications_evals[i]) > len(evals):
                    evals = replications_evals[i]
                elif any(
                    [
                        replications_evals[i][j] != evals[j]
                        for j in range(len(replications_evals[i]))
                    ]
                ):
                    need_rewrite = True
            if not need_rewrite:
                continue
            try:
                assert len(evals) > 0
                for i, repl in enumerate(replications):
                    for j in range(len(replications_evals[i])):
                        data.loc[
                            (data["env"] == exp)
                            & (data["algo"] == variant)
                            & (data["rep"] == repl)
                            & (data[xaxis] == replications_evals[i][j]),
                            xaxis,
                        ] = evals[j]
            except Exception:
                print(
                    "\n!!!WARNING!!! Cannot uniformise",
                    xaxis,
                    "for",
                    variant,
                    "in",
                    exp,
                )
                print(traceback.format_exc(-1))
    return data
