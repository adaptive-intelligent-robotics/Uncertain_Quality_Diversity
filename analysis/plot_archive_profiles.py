import os
import traceback
from typing import List

import jax.numpy as jnp
import numpy as np
import pandas as pd

from analysis.load_archives import get_folder_name
from analysis.load_results import sort_data
from analysis.utils_plot import plot_rows_select


def plot_archive_profiles(
    plot_folder: str,
    config_frame: pd.DataFrame,
    min_max_frame: pd.DataFrame,
    color_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    order: List,
    legend_columns: int,
    legend_bottom: float,
    prefixe: str = "",
    prefixe_title: str = "",
    errors: bool = False,
) -> None:

    # Create the archive profile dataframe
    all_profiles = pd.DataFrame()

    rep = 0
    for env in config_frame["env"].drop_duplicates().values:
        sub_config_frame = config_frame[config_frame["env"] == env].reset_index(
            drop=True
        )
        min_fitness = min_max_frame[min_max_frame["env"] == env]["min_fitness"]
        max_fitness = min_max_frame[min_max_frame["env"] == env]["max_fitness"]
        fitnesses_thresholds = np.linspace(
            start=min_fitness, stop=max_fitness, num=50, endpoint=True
        )

        for line in range(sub_config_frame.shape[0]):
            try:
                repertoire_folder = get_folder_name(
                    sub_config_frame, "repertoire_folder", line
                )
                fitnesses = jnp.load(os.path.join(repertoire_folder, "fitnesses.npy"))

                # Compute profile
                profiles = []
                for fitness_threshold in fitnesses_thresholds:
                    profile = int(np.sum(fitnesses > fitness_threshold))
                    profiles.append(profile)

                # Store as dataframe
                data = pd.DataFrame()
                data["threshold"] = fitnesses_thresholds.ravel()
                data["profile"] = profiles

                # Do the same for reeval if defined
                if "{prefixe}reeval_repertoire_folder" in config_frame.columns:
                    try:
                        reeval_repertoire_folder = get_folder_name(
                            sub_config_frame, "{prefixe}reeval_repertoire_folder", line
                        )
                        reeval_fitnesses = jnp.load(
                            os.path.join(reeval_repertoire_folder, "fitnesses.npy")
                        )
                        reeval_profiles = []
                        for fitness_threshold in fitnesses_thresholds:
                            reeval_profile = int(
                                np.sum(reeval_fitnesses > fitness_threshold)
                            )
                            reeval_profiles.append(reeval_profile)
                        data["reeval_profile"] = reeval_profiles
                    except Exception:
                        print("\n!!!WARNING!!! Cannot open reeval repertoire.")
                        print(sub_config_frame)
                        if errors:
                            traceback.print_exc()
                        data["reeval_profile"] = np.zeros_like(profiles)
                else:
                    data["reeval_profile"] = np.zeros_like(profiles)

                # Do the same for fitness reeval if defined
                if "{prefixe}fit_reeval_repertoire_folder" in config_frame.columns:
                    try:
                        fit_reeval_repertoire_folder = get_folder_name(
                            sub_config_frame,
                            "{prefixe}fit_reeval_repertoire_folder",
                            line,
                        )
                        fit_reeval_fitnesses = jnp.load(
                            os.path.join(fit_reeval_repertoire_folder, "fitnesses.npy")
                        )
                        fit_reeval_profiles = []
                        for fitness_threshold in fitnesses_thresholds:
                            fit_reeval_profile = int(
                                np.sum(fit_reeval_fitnesses > fitness_threshold)
                            )
                            fit_reeval_profiles.append(fit_reeval_profile)
                        data["fit_reeval_profile"] = fit_reeval_profiles
                    except Exception:
                        print("\n!!!WARNING!!! Cannot open fit-reeval repertoire.")
                        print(sub_config_frame)
                        if errors:
                            traceback.print_exc()
                        data["fit_reeval_profile"] = np.zeros_like(profiles)
                else:
                    data["fit_reeval_profile"] = np.zeros_like(profiles)

                # Do the same for desc reeval if defined
                if "{prefixe}desc_reeval_repertoire_folder" in config_frame.columns:
                    try:
                        desc_reeval_repertoire_folder = get_folder_name(
                            sub_config_frame,
                            "{prefixe}desc_reeval_repertoire_folder",
                            line,
                        )
                        desc_reeval_fitnesses = jnp.load(
                            os.path.join(desc_reeval_repertoire_folder, "fitnesses.npy")
                        )
                        desc_reeval_profiles = []
                        for fitness_threshold in fitnesses_thresholds:
                            desc_reeval_profile = int(
                                np.sum(desc_reeval_fitnesses > fitness_threshold)
                            )
                            desc_reeval_profiles.append(desc_reeval_profile)
                        data["desc_reeval_profile"] = desc_reeval_profiles
                    except Exception:
                        print("\n!!!WARNING!!! Cannot open desc-reeval repertoire.")
                        print(sub_config_frame)
                        if errors:
                            traceback.print_exc()
                        data["desc_reeval_profile"] = np.zeros_like(profiles)
                else:
                    data["desc_reeval_profile"] = np.zeros_like(profiles)

                # Add run info to frame
                data["algo"] = sub_config_frame["algo_batch"][line]
                data[compare_size] = sub_config_frame[compare_size][line]
                data["env"] = sub_config_frame["env"][line]
                data["num_reevals"] = sub_config_frame["num_reevals"][line]
                data["rep"] = rep
                rep += 1

                all_profiles = pd.concat([all_profiles, data], ignore_index=True)
            except Exception:
                print("\n!!!WARNING!!! Cannot plot archive profile.")
                print(sub_config_frame)
                if errors:
                    traceback.print_exc()

    # Plot the archive profile
    size_values = all_profiles[compare_size].drop_duplicates().values
    size_values.sort()
    rows_select = [[compare_size, x] for x in size_values]
    rows_name = [f"{compare_title} {x}" for x in size_values]
    columns = [
        "profile",
        "reeval_profile",
        "fit_reeval_profile",
        "desc_reeval_profile",
    ]
    columns_name = [
        "Number of individuals",
        "Number of average-reeval individuals",
        "Number of fitness-only average-reeval individuals",
        "Number of descriptor-only average-reeval individuals",
    ]

    print("\nPlotting archive profile")
    for env in all_profiles["env"].drop_duplicates().values:

        # Extract and sort data
        part_profiles = all_profiles[all_profiles["env"] == env]
        part_profiles = sort_data(part_profiles, ["algo", "rep", "threshold"], order)

        # Plot all
        print("    Archive profile", env)

        try:
            plot_rows_select(
                file_name=f"{plot_folder}/{env}-{prefixe}archive_profiles.svg",
                data_frame=all_profiles,
                rows_select=rows_select,
                rows_name=rows_name,
                columns=columns,
                columns_name=columns_name,
                major_locator=[],
                major_locator_name=[],
                size=None,
                sizes=[],
                hue="algo",
                color_frame=color_frame,
                legend_columns=legend_columns,
                legend_bottom=legend_bottom,
                markers=True,
                vlines=[],
                hlines=[],
                filllines=[],
                x_front=[],
                y_front=[],
                box_plot=False,
                scatter_plot=False,
                x="threshold",
                xlabel="{prefixe_title}Fitness threshold",
            )
        except Exception:
            print(f"\n!!!WARNING!!! Cannot plot {prefixe}archive_profiles for {env}.")
            if errors:
                traceback.print_exc()
