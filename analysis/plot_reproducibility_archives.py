import os
import traceback
from typing import Dict

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

from analysis.utils_archive import get_folder_name, plot_one_paper_archive


def plot_reproducibility_archives(
    plot_folder: str,
    single_compare_size: int,
    env_order: Dict,
    config_frame: pd.DataFrame,
    reprod_min_max_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    prefixe: str = "",
    prefixe_title: str = "",
    errors: bool = False,
) -> None:

    # Fitness reproducibility repertoire
    try:
        if f"{prefixe}fit_reproducibilities_repertoire_folder" in config_frame.columns:
            print(f"    Plotting Reeval Archive for {single_compare_size}.")
            folder_name = f"{prefixe}fit_reproducibilities_repertoire_folder"
            sub_plot_reproducibility_archives(
                file_name=f"{plot_folder}/{prefixe}fit_reprod_archive_{single_compare_size}.png",
                title_name="Fitness-Reproducibility archive",
                single_compare_size=single_compare_size,
                env_order=env_order,
                single_folder_name=folder_name,
                config_frame=config_frame,
                reprod_min_max_frame=reprod_min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                errors=errors,
            )

    except Exception:
        print("\n!!!WARNING!!! Cannot fit_reprod repertoire.")
        if errors:
            traceback.print_exc()

    # Descriptor reproducibility repertoire
    try:
        if f"{prefixe}desc_reproducibilities_repertoire_folder" in config_frame.columns:
            print(f"    Plotting Reeval Archive for {single_compare_size}.")
            folder_name = f"{prefixe}desc_reproducibilities_repertoire_folder"
            sub_plot_reproducibility_archives(
                file_name=f"{plot_folder}/{prefixe}desc_reprod_archive_{single_compare_size}.png",
                title_name="Desc-Reproducibility archive",
                single_compare_size=single_compare_size,
                env_order=env_order,
                single_folder_name=folder_name,
                config_frame=config_frame,
                reprod_min_max_frame=reprod_min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                errors=errors,
            )

    except Exception:
        print("\n!!!WARNING!!! Cannot desc_reprod repertoire.")
        if errors:
            traceback.print_exc()

    # Reeval-Fitness reproducibility repertoire
    try:
        if (
            f"{prefixe}reprod_fit_reproducibilities_repertoire_folder"
            in config_frame.columns
        ):
            print(f"    Plotting Reeval Archive for {single_compare_size}.")
            folder_name = f"{prefixe}reprod_fit_reproducibilities_repertoire_folder"
            sub_plot_reproducibility_archives(
                file_name=f"{plot_folder}/{prefixe}reprod_fit_reprod_archive_{single_compare_size}.png",
                title_name="Reeval-Fitness-Reproducibility archive",
                single_compare_size=single_compare_size,
                env_order=env_order,
                single_folder_name=folder_name,
                config_frame=config_frame,
                reprod_min_max_frame=reprod_min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                errors=errors,
            )

    except Exception:
        print("\n!!!WARNING!!! Cannot reprod_fit_reprod repertoire.")
        if errors:
            traceback.print_exc()

    # Reeval-Descriptor reproducibility repertoire
    try:
        if (
            f"{prefixe}reeval_desc_reproducibilities_repertoire_folder"
            in config_frame.columns
        ):
            print(f"    Plotting Reeval Archive for {single_compare_size}.")
            folder_name = f"{prefixe}reeval_desc_reproducibilities_repertoire_folder"
            sub_plot_reproducibility_archives(
                file_name=f"{plot_folder}/{prefixe}reeval_desc_reprod_archive_{single_compare_size}.png",
                title_name="Reeval-Desc-Reproducibility archive",
                single_compare_size=single_compare_size,
                env_order=env_order,
                single_folder_name=folder_name,
                config_frame=config_frame,
                reprod_min_max_frame=reprod_min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
                errors=errors,
            )

    except Exception:
        print("\n!!!WARNING!!! Cannot reeval_desc_reprod repertoire.")
        if errors:
            traceback.print_exc()


def sub_plot_reproducibility_archives(
    file_name: str,
    title_name: str,
    single_compare_size: int,
    env_order: Dict,
    single_folder_name: str,
    config_frame: pd.DataFrame,
    reprod_min_max_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    min_max_fit_var: bool = False,
    min_max_desc_var: bool = False,
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

        if env_config_frame.empty:
            print(f"\n!!!WARNING!!! Size {single_compare_size} undefined for {env}.")
            continue

        # Get all the corresponding min and max
        min_fitness = None
        max_fitness = None
        if reprod_min_max_frame is not None:
            env_reprod_min_max_frame = reprod_min_max_frame[
                reprod_min_max_frame["env"] == env
            ]
            if not env_reprod_min_max_frame.empty:
                if min_max_fit_var:
                    min_fitness = env_reprod_min_max_frame["min_fit_var"].values[0]
                    max_fitness = env_reprod_min_max_frame["max_fit_var"].values[0]
                elif min_max_desc_var:
                    min_fitness = env_reprod_min_max_frame["min_desc_var"].values[0]
                    max_fitness = env_reprod_min_max_frame["max_desc_var"].values[0]
                else:
                    min_fitness = env_reprod_min_max_frame["min_fitness"].values[0]
                    max_fitness = env_reprod_min_max_frame["max_fitness"].values[0]
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

        # For each algo for this env and size
        for ncol in range(ncols):

            algo = algos[ncol]
            algo_config_frame = env_config_frame[
                env_config_frame["algo"] == algo
            ].reset_index(drop=True)
            if algo_config_frame.empty:
                continue

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
                repertoire_folder = get_folder_name(
                    algo_config_frame, single_folder_name, 0
                )
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
                    vmin=min_fitness,
                    vmax=max_fitness,
                    colorbar=colorbar,
                )

            except Exception:
                print(
                    f"\n!!!WARNING!!! Cannot plot {single_folder_name} for {env} and {algo}."
                )
                if errors:
                    traceback.print_exc()

            # Add algo name in first line
            if nrow == 0:
                ax.set_title(algo, fontsize=32)

            # Add env name in first column
            if ncol == 0:
                title = env
                if env in env_order.keys():
                    title = env_order[env]
                ax.set_ylabel(title, fontsize=32)

    # Finish figure
    # fig.suptitle(title_name, fontsize=32)
    plt.tight_layout(h_pad=1.70)
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()
