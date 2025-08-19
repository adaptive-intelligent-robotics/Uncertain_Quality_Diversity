import os
import time
import traceback
from typing import Dict, List, Tuple

import jax.numpy as jnp
import pandas as pd
from natsort import natsort_keygen

LOSSES = [
    "qd_score",
    "coverage",
]

# TODO: used to unformise old and new datas
UNIFORMISE_FOLDER_NAMES = {
    "env": "env_name",
    "name": "run",
    "repertoire_folder": "results_repertoire",
    "projected_repertoire_folder": "results_projected_repertoire",
    "target_repertoire_folder": "results_target_repertoire",
    "reeval_repertoire_folder": "results_reeval_repertoire",
    "average_repertoire_folder": "results_average_repertoire",
    "fit_reeval_repertoire_folder": "results_fit_reeval_repertoire",
    "fit_average_repertoire_folder": "results_fit_average_repertoire",
    "desc_reeval_repertoire_folder": "results_desc_reeval_repertoire",
    "desc_average_repertoire_folder": "results_desc_average_repertoire",
    "fit_var_repertoire_folder": "results_fit_var_repertoire",
    "desc_var_repertoire_folder": "results_desc_var_repertoire",
    "additional_folder": "results_additional",
    "reeval_additional_folder": "results_reeval_additional",
    "in_cell_reeval_repertoire_folder": "results_in_cell_reeval_repertoire",
    "in_cell_fit_reeval_repertoire_folder": "results_in_cell_fit_reeval_repertoire",
    "in_cell_desc_reeval_repertoire_folder": "results_in_cell_desc_reeval_repertoire",
    "in_cell_fit_var_repertoire_folder": "results_in_cell_fit_var_repertoire",
    "in_cell_desc_var_repertoire_folder": "results_in_cell_desc_var_repertoire",
    "reeval_fit_var_repertoire_folder": "results_reeval_fit_var_repertoire",
    "reeval_desc_var_repertoire_folder": "results_reeval_desc_var_repertoire",
    "in_cell_reeval_fit_var_repertoire_folder": "results_in_cell_reeval_fit_var_repertoire",
    "in_cell_reeval_desc_var_repertoire_folder": "results_in_cell_reeval_desc_var_repertoire",
    "in_cell_reeval_additional_repertoire_folder": "results_in_cell_reeval_additional_repertoire",
    "in_cell_reeval_additional_repertoire": "results_in_cell_reeval_additional_repertoire",
}


def load_results(
    save_folder: str,
    plot_folder: str,
    plot_algos: List,
    exclude_algos: List,
    exclude_sizes: List,
    always_name: List,
    not_name: List,
    replace_name: Dict,
    time_pourcent: float,
    compare_size: str,
    order: List,
    use_max_xaxis: bool,
    env_max_xaxis: Dict,
    x_column: str,
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

            # TODO: uniformise old and new
            for key in UNIFORMISE_FOLDER_NAMES.keys():
                if UNIFORMISE_FOLDER_NAMES[key] in sub_config_frame.columns:
                    sub_config_frame[key] = sub_config_frame[
                        UNIFORMISE_FOLDER_NAMES[key]
                    ]

            # TODO: env naming uniformisation
            if any(
                sub_config_frame["env"] == "arm_gaussian_fit0.01_desc0.01_params0.0"
            ):
                sub_config_frame["env"] = "arm_fit0.01_desc0.01_params0"
            if any(
                sub_config_frame["env"]
                == "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0.0"
            ):
                sub_config_frame[
                    "env"
                ] = "arm_gaussian_desc_bi_variance_fit0.01_desc0.1_params0"
            if any(
                sub_config_frame["env"]
                == "direct_mapping_deceptive_0.1_fit0_desc0.1_params0"
            ):
                sub_config_frame["env"] = "direct_mapping_deceptive_0.1_nonoise"
            if any(
                sub_config_frame["env"]
                == "direct_mapping_perfect_trade_off_0.02_fit0_desc0.2_params0"
            ):
                sub_config_frame[
                    "env"
                ] = "direct_mapping_perfect_trade_off_0.02_nonoise"
            if any(
                sub_config_frame["env"]
                == "direct_mapping_sharp_peak_bigger_0.2_fit0_desc0.05_params0"
            ):
                sub_config_frame["env"] = "direct_mapping_sharp_peak_bigger_0.2_nonoise"
            if any(
                sub_config_frame["env"]
                == "direct_mapping_sharp_peak_smaller_0.02_fit0_desc0.05_params0"
            ):
                sub_config_frame[
                    "env"
                ] = "direct_mapping_sharp_peak_smaller_0.02_nonoise"

            config_frame = pd.concat(
                [config_frame, sub_config_frame], ignore_index=True
            )
        assert (
            config_frame.shape[0] != 0
        ), "\n!!!ERROR!!! No runs refered in config files.\n"

        # Name algorithms
        print("\nSetting up algorithms names")
        use_in_name: Dict = {}
        for name_algo in config_frame["name"].drop_duplicates().values:
            sub_config_frame = config_frame[config_frame["name"] == name_algo]
            use_in_name[name_algo] = []
            for column in sub_config_frame.columns:
                if column in always_name:
                    use_in_name[name_algo].append(column)
                elif column not in not_name:
                    all_values = sub_config_frame[column]
                    all_values = all_values[all_values == all_values]  # remove NaN
                    if not all_values.empty and all_values.nunique() > 1:
                        use_in_name[name_algo].append(column)
        print("\n    Differences between runs:", use_in_name)

        # Add algo name to each line
        no_replace_algos = []
        algos = []
        algos_batch = []
        for line in range(config_frame.shape[0]):
            algo = config_frame["name"][line]

            for name in use_in_name[config_frame["name"][line]]:
                if config_frame[name][line] == config_frame[name][line]:
                    algo += " " + name + ":" + str(config_frame[name][line])
            if algo not in no_replace_algos:
                no_replace_algos.append(algo)

            for name in replace_name.keys():
                algo = algo.replace(name, replace_name[name])

            algo_batch = algo + " - " + str(config_frame[compare_size][line])
            algos.append(algo)
            algos_batch.append(algo_batch)

        print("    Names before replace:")
        print(no_replace_algos)

        config_frame["algo"] = algos
        config_frame["algo_batch"] = algos_batch
        config_frame = config_frame.reset_index(drop=True)

        # Write in file
        config_frame.to_csv(file_name, index=None)

    print("    Found", config_frame.shape[0], "runs, with algo names:")
    print(config_frame["name"].drop_duplicates().reset_index(drop=True))
    print(config_frame["algo"].drop_duplicates().reset_index(drop=True))

    ########################
    # 2. Filter algorithms #

    # Filter algos that does not need to be ploted
    if plot_algos != [""]:
        config_frame = config_frame[config_frame["algo"].isin(plot_algos)]
    if exclude_algos != [""]:
        config_frame = config_frame[
            ~config_frame["algo"].str.contains("|".join(exclude_algos))
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

    file_name_convergence = f"{plot_folder}_csv/all_convergence.csv"
    file_name_finals = f"{plot_folder}_csv/all_finals.csv"
    file_name_times = f"{plot_folder}_csv/all_times.csv"
    file_name_var = f"{plot_folder}_csv/all_var.csv"
    if (
        os.path.exists(file_name_convergence)
        and os.path.exists(file_name_finals)
        and os.path.exists(file_name_times)
        and os.path.exists(file_name_var)
    ):
        print("\nLoading existing progress and loss datas in:")
        print(file_name_convergence)
        print(file_name_finals)
        print(file_name_times)
        print(file_name_var)
        all_convergence = pd.read_csv(file_name_convergence, header=0, index_col=False)
        all_finals = pd.read_csv(file_name_finals, header=0, index_col=False)
        all_times = pd.read_csv(file_name_times, header=0, index_col=False)
        all_var = pd.read_csv(file_name_var, header=0, index_col=False)

        # Re-order
        all_convergence = sort_data(all_convergence, ["algo", "rep", "eval"], order)
        all_finals = sort_data(all_finals, ["algo", "rep", compare_size], order)
        all_times = sort_data(all_times, ["algo", "rep", compare_size], order)
        all_var = sort_data(all_var, ["algo", "rep", compare_size], order)

        return config_frame, all_convergence, all_finals, all_times, all_var

    #######################################
    # 4. Fill in max_x and replications #

    print("\nReading max gen and replications")
    start_t = time.time()

    if use_max_xaxis:
        print("\n\n!!!WARNING!!! Using max xaxis!")
        max_x_frame = pd.DataFrame(columns=["env", x_column])
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

    # Go through all metrics files to fill max_x_frame and replications_frame
    seeds: Dict = {}
    for line in range(config_frame.shape[0]):
        try:
            # Get the config for this line
            env = config_frame["env"][line]
            num_reevals = config_frame["num_reevals"][line]
            size = config_frame[compare_size][line]
            name = config_frame["name"][line]
            algo = config_frame["algo"][line]
            seed = config_frame["seed"][line]

            if seed in seeds.keys():
                if seeds[seed][0] == env and seeds[seed][1] == name:
                    if seeds[seed][2] == size:
                        print(f"Twice seed for {env}, {name} and {size}.")
                    else:
                        print(f"Twice seed for {env} and {name}.")
                        print(f"Size are {seeds[seed][2]} and {size} respectively.")
            seeds[seed] = [env, name, size]

            if use_max_xaxis:

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

                if env in env_max_xaxis.keys():
                    # Get from input dict
                    max_epoch = env_max_xaxis[env]
                    sub_data = data[data["epoch"] <= max_epoch]
                    max_x = sub_data[x_column].max()
                    # print(f"    For {env}, from input dict: {max_x}.")
                else:
                    # Get max
                    max_x = data[x_column].max()
                    # print(f"    For {env}, from loading data: {max_x}.")

                # Add maximum number of generations to frame
                if env in max_x_frame["env"].values:
                    max_x = min(
                        max_x,
                        max_x_frame[max_x_frame["env"] == env][x_column].values[0],
                    )
                    max_x_frame.loc[max_x_frame["env"] == env, x_column] = max_x
                else:
                    max_x_frame = pd.concat(
                        [
                            max_x_frame,
                            pd.DataFrame.from_dict({"env": [env], x_column: [max_x]}),
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

    if use_max_xaxis:
        print(f"\nMax {x_column} for each environment:")
        print(max_x_frame)
        print("\n")

    # Remove empty replications from replications_frame
    replications_frame = replications_frame[replications_frame["num_rep"] != 0]
    replications_frame = sort_data(
        replications_frame, ["env", "num_reevals", "algo", compare_size], order
    )

    # Save replications frame as csv
    print("\nReplications:")
    print(replications_frame)
    print("\n")
    replications_frame.to_csv(
        f"{plot_folder}/replications_frame.csv",
        index=None,
        sep=",",
    )

    print("Time to read replications and max_x_frame:", time.time() - start_t)

    ################
    # 5. Load data #

    print("\nReading data")

    # Create the metrics dataframe
    all_convergence = pd.DataFrame()
    all_finals = pd.DataFrame()
    all_times = pd.DataFrame()
    all_var = pd.DataFrame()

    # Go through all metrics files
    start_t = time.time()
    rep = 0
    for line in range(config_frame.shape[0]):

        try:
            # First, get the config for this line
            name = config_frame["name"][line]
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

            # Second, read metrics
            data = pd.read_csv(metrics_file, index_col=False)
            if data.empty:
                print(f"!!!WARNING!!! {metrics_file} is empty.")
                continue

            # Third, read in-cell metrics
            in_cell_data = pd.read_csv(in_cell_metrics_file, index_col=False)

            # TODO: Old data
            if "in_cell_qd_score" in in_cell_data.columns:
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
                    [
                        "in_cell_fit_reeval_max_fitness",
                        "in_cell_fit_reeval_max_fitness",
                    ],
                    ["in_cell_desc_reeval_qd_score", "in_cell_desc_reeval_qd_score"],
                    ["in_cell_desc_reeval_coverage", "in_cell_desc_reeval_coverage"],
                    [
                        "in_cell_desc_reeval_max_fitness",
                        "in_cell_desc_reeval_max_fitness",
                    ],
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
                    if column_to_merge[1] in in_cell_data.columns:
                        data[column_to_merge[0]] = in_cell_data[column_to_merge[1]]

            # TODO: New data
            else:
                to_merge = [  # [Name in data, Name in in_cell_data]
                    ["in_cell_qd_score", "qd_score"],
                    ["in_cell_coverage", "coverage"],
                    ["in_cell_max_fitness", "max_fitness"],
                    ["in_cell_reeval_qd_score", "reeval_qd_score"],
                    ["in_cell_reeval_coverage", "reeval_coverage"],
                    ["in_cell_reeval_max_fitness", "reeval_max_fitness"],
                    ["in_cell_fit_reeval_qd_score", "fit_reeval_qd_score"],
                    ["in_cell_fit_reeval_coverage", "fit_reeval_coverage"],
                    ["in_cell_fit_reeval_max_fitness", "fit_reeval_max_fitness"],
                    ["in_cell_desc_reeval_qd_score", "desc_reeval_qd_score"],
                    ["in_cell_desc_reeval_coverage", "desc_reeval_coverage"],
                    ["in_cell_desc_reeval_max_fitness", "desc_reeval_max_fitness"],
                    ["in_cell_fit_var_qd_score", "fit_var_qd_score"],
                    ["in_cell_fit_var_coverage", "fit_var_coverage"],
                    ["in_cell_fit_var_max_fitness", "fit_var_max_fitness"],
                    ["in_cell_desc_var_qd_score", "desc_var_qd_score"],
                    ["in_cell_desc_var_coverage", "desc_var_coverage"],
                    ["in_cell_desc_var_max_fitness", "desc_var_max_fitness"],
                    [
                        "in_cell_reeval_additional_qd_score",
                        "reeval_additional_qd_score",
                    ],
                    [
                        "in_cell_reeval_additional_max_fitness",
                        "reeval_additional_max_fitness",
                    ],
                    [
                        "in_cell_reeval_additional_min_fitness",
                        "reeval_additional_min_fitness",
                    ],
                ]
                for column_to_merge in to_merge:
                    if column_to_merge[1] in in_cell_data.columns:
                        data[column_to_merge[0]] = in_cell_data[column_to_merge[1]]

            # TODO: Old data
            # Fourth, add timestep to dataframe
            if "timestep" not in data.columns:
                data["timestep"] = data["epoch"] * batch_size
                if "PartialEval" in name:
                    print("!!!WARNING!!! Timesteps not in metrics, infering it.")
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

            # Fifth, filter datas after max_x if required
            if use_max_xaxis:
                data = data[
                    data[x_column]
                    <= max_x_frame[max_x_frame["env"] == env][x_column].values[0]
                ]

            # Anyway print if shorter than should be
            if env in env_max_xaxis:
                max_x_column = max(data[x_column])
                if max_x_column < env_max_xaxis[env]:
                    print(
                        f"!!!TOCHECK!!! Incomplete run for {algo} and {env} in {folder}, got {max_x_column}."
                    )
            else:
                print(
                    f"!!!WARNING!!! Env {env} not in env_max_xaxis, cannot check if all runs finished."
                )

            # Sixth, compute frame with all final values and losses
            finals: Dict[str, List[float]] = {}
            max_eval = max(data["eval"])
            sub_data = data[data["eval"] == max_eval]

            for column in sub_data.columns:
                # Add all final values
                finals[column] = sub_data[column].values[0]

            # TODO: condition for old
            if f"additional_qd_score" in sub_data.columns:
                for corrected_name in ["", "reeval_"]:
                    # Add all average additional
                    coverage = sub_data[f"{corrected_name}coverage"].values[0]
                    additional = sub_data[
                        f"{corrected_name}additional_qd_score"
                    ].values[0]
                    if coverage != 0:
                        finals[f"{corrected_name}additional_average"] = additional / (
                            num_centroids * coverage / 100
                        )

            for prefixe in ["", "in_cell_"]:
                for column in LOSSES:
                    # Add all losses
                    illusory = sub_data[f"{prefixe}{column}"].values[0]
                    corrected = sub_data[f"{prefixe}reeval_{column}"].values[0]
                    if illusory == 0.0 or illusory == -jnp.inf:
                        finals[f"loss_{prefixe}{column}"] = (
                            [0.0] if corrected == 0.0 else [100.0]
                        )
                    else:
                        finals[f"loss_{prefixe}{column}"] = [
                            (illusory - corrected) / illusory * 100
                        ]

            # Seventh, compute frame with all vars
            var: Dict[str, List[float]] = {}
            for prefixe in ["", "in_cell_"]:
                for column in ["fit_var", "desc_var"]:
                    # Add all qd-score and average var
                    coverage = sub_data[f"{prefixe}coverage"].values[0]
                    var_value = sub_data[f"{prefixe}{column}_qd_score"].values[0]
                    if coverage == 0:
                        var[f"{prefixe}{column}_avg"] = [0.0]
                    else:
                        var[f"{prefixe}{column}_avg"] = var_value / (
                            coverage / 100 * num_centroids
                        )
                    var[f"{prefixe}{column}_qd_score"] = var_value

            # Eighth, compute frame with times to % convergence
            times: Dict[str, List[float]] = {}

            for prefixe in ["", "in_cell_"]:
                # Add times to reach time_pourcent % of final value
                if time_pourcent == 1:
                    pourcent_epoch = max(data["epoch"].values)
                else:
                    final_value = sub_data[f"{prefixe}qd_score"].values[0]
                    if final_value > 0:
                        pourcent_value = time_pourcent * final_value
                    else:
                        pourcent_value = 2.0 - time_pourcent * final_value
                    min_epoch = data["epoch"].drop_duplicates().nsmallest(2).iloc[-1]
                    pourcent_epoch = min_epoch
                    pourcent_epoch_candidates = data[
                        data[f"{prefixe}qd_score"] < pourcent_value
                    ]["epoch"]
                    if not (pourcent_epoch_candidates.empty):
                        pourcent_epoch = max(
                            min_epoch, max(pourcent_epoch_candidates.values)
                        )
                    else:
                        print(f"Taking min epoch for {name}.")

                pourcent_line = data[data["epoch"] == pourcent_epoch]
                times[f"{prefixe}epoch"] = pourcent_epoch
                times[f"{prefixe}eval"] = pourcent_line["eval"].values[0]
                times[f"{prefixe}time"] = pourcent_line["time"].values[0]
                times[f"{prefixe}timestep"] = pourcent_line["timestep"].values[0]
                times[f"{prefixe}reeval_qd_score"] = pourcent_line[
                    f"{prefixe}reeval_qd_score"
                ].values[0]

            # Ninth, merge everything in main dataframes
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

            finals["name"] = [name]
            finals["algo"] = [algo]
            finals["algo_batch"] = [algo_batch]
            finals["num_centroids"] = [num_centroids]
            finals["env"] = [env]
            finals["num_reevals"] = [num_reevals]
            finals[compare_size] = [size]
            finals["batch_size"] = [batch_size]
            finals["sampling_size"] = [sampling_size]
            finals["rep"] = [rep]

            times["name"] = [name]
            times["algo"] = [algo]
            times["algo_batch"] = [algo_batch]
            times["num_centroids"] = [num_centroids]
            times["env"] = [env]
            times["num_reevals"] = [num_reevals]
            times[compare_size] = [size]
            times["batch_size"] = [batch_size]
            times["sampling_size"] = [sampling_size]
            times["rep"] = [rep]

            var["name"] = [name]
            var["algo"] = [algo]
            var["algo_batch"] = [algo_batch]
            var["num_centroids"] = [num_centroids]
            var["env"] = [env]
            var["num_reevals"] = [num_reevals]
            var[compare_size] = [size]
            var["batch_size"] = [batch_size]
            var["sampling_size"] = [sampling_size]
            var["rep"] = [rep]

            finals = pd.DataFrame.from_dict(finals)
            times = pd.DataFrame.from_dict(times)
            var = pd.DataFrame.from_dict(var)

            all_convergence = pd.concat([all_convergence, data], ignore_index=True)
            all_finals = pd.concat([all_finals, finals], ignore_index=True)
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
    if "additional_average" in all_finals.columns:
        if len(all_finals["env"].drop_duplicates().values) > 1:

            # Average
            all_finals["normalised_additional_average"] = all_finals.groupby(
                "env", group_keys=False
            ).apply(
                lambda g: (g["additional_average"] - g["additional_average"].min())
                / (g["additional_average"].max() - g["additional_average"].min())
            )
            all_finals["complement_additional_average"] = all_finals.groupby(
                "env", group_keys=False
            ).apply(lambda g: 1 - g["normalised_additional_average"])
            all_finals["additional_reproducibility_score"] = all_finals.groupby(
                "env", group_keys=False
            ).apply(lambda g: g["coverage"] * g["complement_additional_average"])

            # Reeval Average
            all_finals["normalised_reeval_additional_average"] = all_finals.groupby(
                "env", group_keys=False
            ).apply(
                lambda g: (
                    g["reeval_additional_average"]
                    - g["reeval_additional_average"].min()
                )
                / (
                    g["reeval_additional_average"].max()
                    - g["reeval_additional_average"].min()
                )
            )
            all_finals["complement_reeval_additional_average"] = all_finals.groupby(
                "env", group_keys=False
            ).apply(lambda g: 1 - g["normalised_reeval_additional_average"])
            all_finals["reeval_additional_reproducibility_score"] = all_finals.groupby(
                "env", group_keys=False
            ).apply(
                lambda g: g["reeval_coverage"]
                * g["complement_reeval_additional_average"]
            )

        else:
            # Average
            all_finals["normalised_additional_average"] = (
                all_finals["additional_average"]
                - all_finals["additional_average"].min()
            ) / (
                all_finals["additional_average"].max()
                - all_finals["additional_average"].min()
            )
            all_finals["complement_additional_average"] = (
                1 - all_finals["normalised_additional_average"]
            )
            all_finals["additional_reproducibility_score"] = (
                all_finals["coverage"] * all_finals["complement_additional_average"]
            )

            # Reeval Average
            all_finals["normalised_reeval_additional_average"] = (
                all_finals["reeval_additional_average"]
                - all_finals["reeval_additional_average"].min()
            ) / (
                all_finals["reeval_additional_average"].max()
                - all_finals["reeval_additional_average"].min()
            )
            all_finals["complement_reeval_additional_average"] = (
                1 - all_finals["normalised_reeval_additional_average"]
            )
            all_finals["reeval_additional_reproducibility_score"] = (
                all_finals["reeval_coverage"]
                * all_finals["complement_reeval_additional_average"]
            )

    print("Time to take complement of all additional:", time.time() - start_t)
    start_t = time.time()

    # try:
    #    # Uniformise time values across replications
    #    all_convergence = uniformise_xaxis(all_convergence, "time")
    # except Exception:
    #    print("\n!!!WARNING!!! Could not uniformise the time values across reps.")
    #    traceback.print_exc()
    # print("Time to uniformise all time:", time.time() - start_t)
    # start_t = time.time()
    print("\n!!!WARNING!!! Not doing the time uniformisation to save some time.")

    # Sort datas
    all_convergence = sort_data(all_convergence, ["algo", "rep", "eval"], order)
    all_finals = sort_data(all_finals, ["algo", "rep", compare_size], order)
    all_times = sort_data(all_times, ["algo", "rep", compare_size], order)
    all_var = sort_data(all_var, ["algo", "rep", compare_size], order)

    # Save datas as csv
    all_convergence.to_csv(file_name_convergence, index=None)
    all_finals.to_csv(file_name_finals, index=None)
    all_times.to_csv(file_name_times, index=None)
    all_var.to_csv(file_name_var, index=None)

    return config_frame, all_convergence, all_finals, all_times, all_var


###########################
# Data handling functions #


def sort_data(data: pd.DataFrame, columns: list, order: list) -> pd.DataFrame:
    """
    Sort the Pandas DataFrame "data" that contains an "algo" columns, so
    that the "algo" column follows the inputed "order" and the rest of the
    columns are sorted following the order inputed in "columns".
    """

    # Initialize an empty list to store the sorted DataFrames for each algorithm
    sorted_data = []

    # Process each algorithm in the order list
    for algo in order:
        # Filter data for the current algorithm and sort it
        algo_data = data[data["algo"].str.contains(algo, regex=False)].sort_values(
            columns, key=natsort_keygen(), ignore_index=True
        )
        sorted_data.append(algo_data)

        # Remove rows that contain the current algorithm from the data for future iterations
        data = data[~data["algo"].str.contains(algo, regex=False)]

    # Finally, add the remaining rows (non-matching any algorithm in 'order')
    sorted_data.append(
        data.sort_values(columns, key=natsort_keygen(), ignore_index=True)
    )

    # Concatenate the sorted DataFrames
    return pd.concat(sorted_data, ignore_index=True)


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
