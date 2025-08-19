import os
import traceback
from typing import Dict, List, Tuple

import pandas as pd

# Paths from main_comparison_chaining.py, to improve
all_configs_path = "plots_chaining_all_csv/all_configs.csv"


def load_chaining(
    plot_folder: str,
    compare_size: str,
    always_name: List,
    not_name: List,
    replace_name: Dict,
    order: List,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    WARNING: this function assumes that the chaining data have
    already been generated using main_comparison_chaining.py.
    """

    # If already a frame in the folder, use it
    file_name_config_frame = f"{plot_folder}_csv/chaining_config_frame.csv"
    file_name_all_chaining_data = f"{plot_folder}_csv/all_chaining_data.csv"
    file_name_all_data = f"{plot_folder}_csv/all_data.csv"
    if (
        os.path.exists(file_name_config_frame)
        and os.path.exists(file_name_all_chaining_data)
        and os.path.exists(file_name_all_data)
    ):
        print("Loading existing chaining datas in:")
        print(file_name_config_frame)
        print(file_name_all_chaining_data)
        print(file_name_all_data)
        config_frame = pd.read_csv(file_name_config_frame, header=0, index_col=False)
        all_chaining_data = pd.read_csv(
            file_name_all_chaining_data, header=0, index_col=False
        )
        all_data = pd.read_csv(file_name_all_data, header=0, index_col=False)
        return config_frame, all_chaining_data, all_data

    # If not, create it and populate it
    error_msg = "Please generates the data using main_comparison_chaining.py or edit the default paths."

    # Load everything
    if not os.path.exists(all_configs_path):
        print(f"\n!!!ERROR!!! Cannot open {all_configs_path}.", error_msg)
        assert 0

    print("\n\nLoading all_configs in", all_configs_path)
    config_frame = pd.read_csv(all_configs_path, header=0, index_col=False)

    # Name algorithms
    not_name = not_name + [
        "env_config_folder",
        "chaining_file_name",
    ]
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
    config_frame["rep"] = [i for i in range(config_frame.shape[0])]

    print("    Found", config_frame.shape[0], "runs, with algo names:")
    print(config_frame["name"].drop_duplicates().reset_index(drop=True))
    print(config_frame["algo"].drop_duplicates().reset_index(drop=True))

    # Populate the dataframes
    all_chaining_data = pd.DataFrame()
    all_data = pd.DataFrame()

    # For each line in config
    for line in range(config_frame.shape[0]):

        # Get config
        env = config_frame["env"][line]
        episode_length = config_frame["episode_length"][line]
        policy_hidden_layer_sizes = config_frame["policy_hidden_layer_sizes"][line]
        name = config_frame["name"][line]
        algo = config_frame["algo"][line]
        rep = config_frame["rep"][line]
        algo_batch = config_frame["algo_batch"][line]
        batch_size = config_frame["batch_size"][line]
        sampling_size = config_frame["sampling_size"][line]
        chaining_file_name = config_frame["chaining_file_name"][line]

        print(f"    Loading {line} / {config_frame.shape[0]}: {algo} in {env}.")

        # Check that the files exist
        if not os.path.isfile(chaining_file_name):
            # print(f"\n!!!WARNING!!! No chaining data for {algo} in {env}.")
            continue

        # Open the chaining dataframe
        try:
            chaining_data = pd.read_csv(chaining_file_name, header=0, index_col=False)
            chaining_data["env"] = env
            chaining_data["episode_length"] = episode_length
            chaining_data["policy_hidden_layer_sizes"] = policy_hidden_layer_sizes
            chaining_data["name"] = name
            chaining_data["algo"] = algo
            chaining_data["algo_batch"] = algo_batch
            chaining_data["batch_size"] = batch_size
            chaining_data["sampling_size"] = sampling_size
            chaining_data["rep"] = rep
            all_chaining_data = pd.concat(
                [all_chaining_data, chaining_data], ignore_index=True
            )
        except Exception:
            print(
                f"\n!!!WARNING!!! Cannot open {algo} in {env} in {chaining_file_name}."
            )
            traceback.print_exc()

    config_frame.to_csv(file_name_config_frame, index=None)
    all_chaining_data.to_csv(file_name_all_chaining_data, index=None)
    all_data.to_csv(file_name_all_data, index=None)
    print("    Done Loading chaining")

    return config_frame, all_chaining_data, all_data
