import os
import traceback
from typing import Dict

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd

from analysis.load_archives import get_folder_name
from analysis.utils_archive import plot_one_paper_archive


def plot_archives(
    plot_folder: str,
    single_compare_size: int,
    env_order: Dict,
    config_frame: pd.DataFrame,
    min_max_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
    prefixe: str = "",
    prefixe_title: str = "",
) -> None:

    # Non-reevaluated repertoire
    try:
        print(f"    Plotting Archive for {single_compare_size}.")
        folder_name = "repertoire_folder"
        sub_plot_archives(
            file_name=f"{plot_folder}/{prefixe}archive_{single_compare_size}.png",
            title_name="Original archive",
            single_compare_size=single_compare_size,
            env_order=env_order,
            single_folder_name=folder_name,
            config_frame=config_frame,
            min_max_frame=min_max_frame,
            compare_size=compare_size,
            compare_title=compare_title,
        )

    except Exception:
        print("\n!!!WARNING!!! Cannot plot non-reevaluated repertoire.")
        traceback.print_exc()

    # Reevaluated repertoire
    try:
        if f"{prefixe}reeval_repertoire_folder" in config_frame.columns:
            print(f"    Plotting Reeval Archive for {single_compare_size}.")
            folder_name = f"{prefixe}reeval_repertoire_folder"
            sub_plot_archives(
                file_name=f"{plot_folder}/{prefixe}reeval_archive_{single_compare_size}.png",
                title_name="Corrected archive",
                single_compare_size=single_compare_size,
                env_order=env_order,
                single_folder_name=folder_name,
                config_frame=config_frame,
                min_max_frame=min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
            )

    except Exception:
        print("\n!!!WARNING!!! Cannot plot reeval repertoire.")
        traceback.print_exc()

    # Additional repertoire
    try:
        if "additional_folder" in config_frame.columns:
            print(f"    Plotting Additional Archive for {single_compare_size}.")
            folder_name = f"additional_folder"
            sub_plot_archives(
                file_name=f"{plot_folder}/{prefixe}additional_archive_{single_compare_size}.png",
                title_name="Additional archive",
                single_compare_size=single_compare_size,
                env_order=env_order,
                single_folder_name=folder_name,
                config_frame=config_frame,
                min_max_frame=min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
            )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot additional repertoire.")
        traceback.print_exc()

    # Reevaluated fitness repertoire
    try:
        if f"{prefixe}fit_reeval_repertoire_folder" in config_frame.columns:
            print(f"    Plotting Fit-Reeval Archive for {single_compare_size}.")
            folder_name = f"{prefixe}fit_reeval_repertoire_folder"
            sub_plot_archives(
                file_name=f"{plot_folder}/{prefixe}fit_reeval_archive_{single_compare_size}.png",
                title_name="Fit-Reeval archive",
                single_compare_size=single_compare_size,
                env_order=env_order,
                single_folder_name=folder_name,
                config_frame=config_frame,
                min_max_frame=min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
            )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot fit-reeval repee")
        traceback.print_exc()

    # Reevaluated desc repertoire
    try:
        if f"{prefixe}desc_reeval_repertoire_folder" in config_frame.columns:
            print(f"    Plotting Desc-Reeval Archive for {single_compare_size}.")
            folder_name = f"{prefixe}desc_reeval_repertoire_folder"
            sub_plot_archives(
                file_name=f"{plot_folder}/{prefixe}desc_reeval_archive_{single_compare_size}.png",
                title_name="Desc-Reeval archive",
                single_compare_size=single_compare_size,
                env_order=env_order,
                single_folder_name=folder_name,
                config_frame=config_frame,
                min_max_frame=min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
            )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot desc-reeval repertoire.")
        traceback.print_exc()

    # Variance fitness repertoire
    try:
        if f"{prefixe}fit_var_repertoire_folder" in config_frame.columns:
            print(f"    Plotting Fit-Var Archive for {single_compare_size}.")
            folder_name = f"{prefixe}fit_var_repertoire_folder"
            sub_plot_archives(
                file_name=f"{plot_folder}/{prefixe}fit_var_archive_{single_compare_size}.png",
                title_name="Fit-Var archive",
                single_compare_size=single_compare_size,
                env_order=env_order,
                single_folder_name=folder_name,
                config_frame=config_frame,
                min_max_frame=min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
            )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot fit-var repertoire.")
        traceback.print_exc()

    # Variance desc repertoire
    try:
        if f"{prefixe}desc_var_repertoire_folder" in config_frame.columns:
            print(f"    Plotting Desc-Var Archive for {single_compare_size}.")
            folder_name = f"{prefixe}desc_var_repertoire_folder"
            sub_plot_archives(
                file_name=f"{plot_folder}/{prefixe}desc_var_archive_{single_compare_size}.png",
                title_name="Desc-Var archive",
                single_compare_size=single_compare_size,
                env_order=env_order,
                single_folder_name=folder_name,
                config_frame=config_frame,
                min_max_frame=min_max_frame,
                compare_size=compare_size,
                compare_title=compare_title,
            )
    except Exception:
        print("\n!!!WARNING!!! Cannot plot desc-var repertoire.")
        traceback.print_exc()


def sub_plot_archives(
    file_name: str,
    title_name: str,
    single_compare_size: int,
    env_order: Dict,
    single_folder_name: str,
    config_frame: pd.DataFrame,
    min_max_frame: pd.DataFrame,
    compare_size: str,
    compare_title: str,
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

        # Get all the corresponding min and max
        min_fitness = None
        max_fitness = None
        min_fit_var = None
        max_fit_var = None
        min_desc_var = None
        max_desc_var = None
        if min_max_frame is not None:
            env_min_max_frame = min_max_frame[min_max_frame["env"] == env]
            if not env_min_max_frame.empty:
                min_fitness = env_min_max_frame["min_fitness"].values[0]
                max_fitness = env_min_max_frame["max_fitness"].values[0]
                min_fit_var = env_min_max_frame["min_fit_var"].values[0]
                max_fit_var = env_min_max_frame["max_fit_var"].values[0]
                min_desc_var = env_min_max_frame["min_desc_var"].values[0]
                max_desc_var = env_min_max_frame["max_desc_var"].values[0]
        min_bd_list = str(env_config_frame["min_bd"][0])[1:-1].split(" ")
        if "" in min_bd_list:
            min_bd_list.remove("")
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
                if single_folder_name == "repertoire_folder" and "MOME" in algo:
                    repertoire_folder = get_folder_name(
                        algo_config_frame, "projected_repertoire_folder", 0
                    )
                else:
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
