import os
import traceback

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from qdax.utils.plotting import plot_2d_map_elites_repertoire

from analysis.load_archives import get_folder_name


def plot_all_archives(
    plot_folder: str,
    config_frame: pd.DataFrame,
    min_max_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    prefixe: str = "",
    prefixe_title: str = "",
    errors: bool = False,
) -> None:

    # For each environment
    for env in config_frame["env"].drop_duplicates().values:

        print(f"    Plotting for env {env}")
        env_config_frame = config_frame[(config_frame["env"] == env)].reset_index(
            drop=True
        )

        # Get all the corresponding min and max
        min_fitness = None
        max_fitness = None
        min_reeval_fitness = None
        max_reeval_fitness = None
        min_fit_var = None
        max_fit_var = None
        min_desc_var = None
        max_desc_var = None
        min_additional = None
        max_additional = None
        if min_max_frame is not None:
            env_min_max_frame = min_max_frame[min_max_frame["env"] == env]
            if not env_min_max_frame.empty:
                min_fitness = env_min_max_frame["min_fitness"][0]
                max_fitness = env_min_max_frame["max_fitness"][0]
                min_reeval_fitness = env_min_max_frame["min_reeval_fitness"][0]
                max_reeval_fitness = env_min_max_frame["max_reeval_fitness"][0]
                min_fit_var = env_min_max_frame["min_fit_var"][0]
                max_fit_var = env_min_max_frame["max_fit_var"][0]
                min_desc_var = env_min_max_frame["min_desc_var"][0]
                max_desc_var = env_min_max_frame["max_desc_var"][0]
                min_additional = env_min_max_frame["min_additional"][0]
                max_additional = env_min_max_frame["max_additional"][0]
        min_bd = [0, 0]
        if env_config_frame["min_bd"].values[0] != []:
            if "_" in env_config_frame["min_bd"].values[0]:
                min_bd_list = str(env_config_frame["min_bd"].values[0]).split("_")
            else:
                min_bd_list = str(env_config_frame["min_bd"].values[0])[1:-1].split(" ")
            if "" in min_bd_list:
                min_bd_list.remove("")
            min_bd = [float(bd) for bd in min_bd_list]
        max_bd = [1, 1]
        if env_config_frame["max_bd"].values[0] != []:
            if "_" in env_config_frame["max_bd"].values[0]:
                max_bd_list = str(env_config_frame["max_bd"].values[0]).split("_")
            else:
                max_bd_list = str(env_config_frame["max_bd"].values[0])[1:-1].split(" ")
            if "" in max_bd_list:
                max_bd_list.remove("")
            max_bd = [float(bd) for bd in max_bd_list]

        # For each run for this env
        for line in range(env_config_frame.shape[0]):

            # Create the figure
            algo = env_config_frame["algo"][line].replace(" ", "_")
            size = env_config_frame[compare_size][line]
            file_name = f"{plot_folder}/{env}_{algo}_{size}_{line}_{prefixe}archive.png"
            fig, ax = plt.subplots(nrows=1, ncols=8, figsize=(70, 8), sharey=True)

            # Non-reevaluated repertoire
            try:
                folder_name = (
                    "projected_repertoire_folder"
                    if "MOME" in algo
                    else "repertoire_folder"
                )
                repertoire_folder = get_folder_name(env_config_frame, folder_name, line)
                fitnesses = jnp.load(os.path.join(repertoire_folder, "fitnesses.npy"))
                descriptors = jnp.load(
                    os.path.join(repertoire_folder, "descriptors.npy")
                )
                centroids = jnp.load(os.path.join(repertoire_folder, "centroids.npy"))

                _, _ = plot_2d_map_elites_repertoire(
                    centroids=centroids,
                    repertoire_fitnesses=fitnesses,
                    minval=min_bd,
                    maxval=max_bd,
                    vmin=min_fitness,
                    vmax=max_fitness,
                    repertoire_descriptors=descriptors,
                    ax=ax.flat[0],
                )
                ax.flat[0].set_title(f"{env} - {algo} - Original archive")

            except Exception:
                print("\n!!!WARNING!!! Cannot open non-reevaluated repertoire for:")
                print(env_config_frame.loc[line])
                if errors:
                    traceback.print_exc()

            # Reevaluated repertoire
            try:
                if f"{prefixe}reeval_repertoire_folder" in config_frame.columns:
                    repertoire_folder = get_folder_name(
                        env_config_frame, f"{prefixe}reeval_repertoire_folder", line
                    )
                    fitnesses = jnp.load(
                        os.path.join(repertoire_folder, "fitnesses.npy")
                    )
                    descriptors = jnp.load(
                        os.path.join(repertoire_folder, "descriptors.npy")
                    )
                    centroids = jnp.load(
                        os.path.join(repertoire_folder, "centroids.npy")
                    )

                    _, _ = plot_2d_map_elites_repertoire(
                        centroids=centroids,
                        repertoire_fitnesses=fitnesses,
                        minval=min_bd,
                        maxval=max_bd,
                        vmin=min_reeval_fitness,
                        vmax=max_reeval_fitness,
                        repertoire_descriptors=descriptors,
                        ax=ax.flat[1],
                    )
                    ax.flat[1].set_title(
                        f"{env} - {algo} - {prefixe_title}Reevaluated archive"
                    )

            except Exception:
                print("\n!!!WARNING!!! Cannot open reeval repertoire for:")
                print(env_config_frame.loc[line])
                if errors:
                    traceback.print_exc()

            # Additional repertoire
            try:
                if (
                    "additional_folder" in config_frame.columns
                    and env_config_frame["additional_folder"][line]
                    == env_config_frame["additional_folder"][line]
                ):
                    repertoire_folder = get_folder_name(
                        env_config_frame, "additional_folder", line
                    )
                    fitnesses = jnp.load(
                        os.path.join(repertoire_folder, "fitnesses.npy")
                    )
                    descriptors = jnp.load(
                        os.path.join(repertoire_folder, "descriptors.npy")
                    )
                    centroids = jnp.load(
                        os.path.join(repertoire_folder, "centroids.npy")
                    )
                    _, _ = plot_2d_map_elites_repertoire(
                        centroids=centroids,
                        repertoire_fitnesses=fitnesses,
                        minval=min_bd,
                        maxval=max_bd,
                        vmin=min_additional,
                        vmax=max_additional,
                        repertoire_descriptors=descriptors,
                        ax=ax.flat[2],
                    )
                    ax.flat[2].set_title(f"{env} - {algo} - Additional archive")
            except Exception:
                print("\n!!!WARNING!!! Cannot open additional repertoire for:")
                print(env_config_frame.loc[line])
                if errors:
                    traceback.print_exc()

            # Reevaluated fitness repertoire
            try:
                if f"{prefixe}fit_reeval_repertoire_folder" in config_frame.columns:
                    repertoire_folder = get_folder_name(
                        env_config_frame, f"{prefixe}fit_reeval_repertoire_folder", line
                    )
                    fitnesses = jnp.load(
                        os.path.join(repertoire_folder, "fitnesses.npy")
                    )
                    descriptors = jnp.load(
                        os.path.join(repertoire_folder, "descriptors.npy")
                    )
                    centroids = jnp.load(
                        os.path.join(repertoire_folder, "centroids.npy")
                    )

                    _, _ = plot_2d_map_elites_repertoire(
                        centroids=centroids,
                        repertoire_fitnesses=fitnesses,
                        minval=min_bd,
                        maxval=max_bd,
                        vmin=min_fitness,
                        vmax=max_fitness,
                        repertoire_descriptors=descriptors,
                        ax=ax.flat[4],
                    )
                    ax.flat[4].set_title(
                        f"{env} - {algo} - {prefixe_title}Reevaluated Fitness-only archive"
                    )

            except Exception:
                print("\n!!!WARNING!!! Cannot open fit-reeval repertoire for:")
                print(env_config_frame.loc[line])
                if errors:
                    traceback.print_exc()

            # Reevaluated desc repertoire
            try:
                if f"{prefixe}desc_reeval_repertoire_folder" in config_frame.columns:
                    repertoire_folder = get_folder_name(
                        env_config_frame,
                        f"{prefixe}desc_reeval_repertoire_folder",
                        line,
                    )
                    fitnesses = jnp.load(
                        os.path.join(repertoire_folder, "fitnesses.npy")
                    )
                    descriptors = jnp.load(
                        os.path.join(repertoire_folder, "descriptors.npy")
                    )
                    centroids = jnp.load(
                        os.path.join(repertoire_folder, "centroids.npy")
                    )

                    _, _ = plot_2d_map_elites_repertoire(
                        centroids=centroids,
                        repertoire_fitnesses=fitnesses,
                        minval=min_bd,
                        maxval=max_bd,
                        vmin=min_fitness,
                        vmax=max_fitness,
                        repertoire_descriptors=descriptors,
                        ax=ax.flat[5],
                    )
                    ax.flat[5].set_title(
                        f"{env} - {algo} - {prefixe_title}Reevaluated Descriptors-only archive"
                    )
            except Exception:
                print("\n!!!WARNING!!! Cannot open desc-reeval repertoire for:")
                print(env_config_frame.loc[line])
                if errors:
                    traceback.print_exc()

            # Variance fitness repertoire
            try:
                if f"{prefixe}fit_var_repertoire_folder" in config_frame.columns:
                    repertoire_folder = get_folder_name(
                        env_config_frame, f"{prefixe}fit_var_repertoire_folder", line
                    )
                    fitnesses = jnp.load(
                        os.path.join(repertoire_folder, "fitnesses.npy")
                    )
                    descriptors = jnp.load(
                        os.path.join(repertoire_folder, "descriptors.npy")
                    )
                    centroids = jnp.load(
                        os.path.join(repertoire_folder, "centroids.npy")
                    )

                    _, _ = plot_2d_map_elites_repertoire(
                        centroids=centroids,
                        repertoire_fitnesses=fitnesses,
                        minval=min_bd,
                        maxval=max_bd,
                        vmin=min_fit_var,
                        vmax=max_fit_var,
                        repertoire_descriptors=descriptors,
                        ax=ax.flat[6],
                    )
                    ax.flat[6].set_title(
                        f"{env} - {algo} - {prefixe_title}Fitness-Variance archive"
                    )
            except Exception:
                print("\n!!!WARNING!!! Cannot open fit-var repertoire for:")
                print(env_config_frame.loc[line])
                if errors:
                    traceback.print_exc()

            # Variance desc repertoire
            try:
                if f"{prefixe}desc_var_repertoire_folder" in config_frame.columns:
                    repertoire_folder = get_folder_name(
                        env_config_frame, f"{prefixe}desc_var_repertoire_folder", line
                    )
                    fitnesses = jnp.load(
                        os.path.join(repertoire_folder, "fitnesses.npy")
                    )
                    descriptors = jnp.load(
                        os.path.join(repertoire_folder, "descriptors.npy")
                    )
                    centroids = jnp.load(
                        os.path.join(repertoire_folder, "centroids.npy")
                    )

                    _, _ = plot_2d_map_elites_repertoire(
                        centroids=centroids,
                        repertoire_fitnesses=fitnesses,
                        minval=min_bd,
                        maxval=max_bd,
                        vmin=min_desc_var,
                        vmax=max_desc_var,
                        repertoire_descriptors=descriptors,
                        ax=ax.flat[7],
                    )
                    ax.flat[7].set_title(
                        f"{env} - {algo} - {prefixe_title}Descriptors-Variance archive"
                    )
            except Exception:
                print("\n!!!WARNING!!! Cannot open desc-var repertoire for:")
                print(env_config_frame.loc[line])
                if errors:
                    traceback.print_exc()

            # Finish figure
            plt.tight_layout()
            plt.savefig(file_name, bbox_inches="tight")
            plt.close()
