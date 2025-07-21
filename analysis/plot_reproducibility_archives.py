import os
import traceback

import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from qdax.utils.plotting import plot_2d_map_elites_repertoire


def plot_reproducibility_archives(
    plot_folder: str,
    reprod_repertoire_folders: pd.DataFrame,
    reprod_min_max_frame: pd.DataFrame,
    compare_size: str,
    prefixe: str = "",
    prefixe_title: str = "",
) -> None:

    # For each environment
    for env in reprod_repertoire_folders["env"].drop_duplicates().values:

        print(f"    Plotting for env {env}")
        env_reprod_repertoire_folders = reprod_repertoire_folders[
            (reprod_repertoire_folders["env"] == env)
        ].reset_index(drop=True)

        # Get all the corresponding min and max
        env_min_max_frame = reprod_min_max_frame[
            reprod_min_max_frame["env"] == env
        ].reset_index(drop=True)
        min_fit_reprod = env_min_max_frame["min_fit_reprod"][0]
        max_fit_reprod = env_min_max_frame["max_fit_reprod"][0]
        min_desc_reprod = env_min_max_frame["min_desc_reprod"][0]
        max_desc_reprod = env_min_max_frame["max_desc_reprod"][0]
        min_bd = env_min_max_frame["min_bd"][0]
        max_bd = env_min_max_frame["max_bd"][0]

        # For each run for this env
        for line in range(env_reprod_repertoire_folders.shape[0]):

            # Create the figure
            algo = env_reprod_repertoire_folders["algo"][line].replace(" ", "_")
            size = env_reprod_repertoire_folders[compare_size][line]
            file_name = f"{plot_folder}/{env}_{algo}_{size}_{line}_{prefixe}archive.png"
            fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(35, 8), sharey=True)

            # First, fit_reproduciblities repertoire
            try:
                repertoire_folder = env_reprod_repertoire_folders[
                    "fit_reproducibilities_repertoire_folder"
                ][line]
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
                    vmin=min_fit_reprod,
                    vmax=max_fit_reprod,
                    repertoire_descriptors=descriptors,
                    ax=ax.flat[0],
                )
                ax.flat[0].set_title(f"{prefixe_title}Fitness Reproducibilities")

            except Exception:
                print(
                    "\n!!!WARNING!!! Cannot open fit_reproduciblities repertoire for:"
                )
                print(env_reprod_repertoire_folders.loc[line])
                traceback.print_exc()

            # Second, desc_reproducibilities repertoire
            try:
                repertoire_folder = env_reprod_repertoire_folders[
                    "desc_reproducibilities_repertoire_folder"
                ][line]
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
                    vmin=min_desc_reprod,
                    vmax=max_desc_reprod,
                    repertoire_descriptors=descriptors,
                    ax=ax.flat[1],
                )
                ax.flat[1].set_title(f"{prefixe_title}Descriptor Reproducibilities")

            except Exception:
                print(
                    "\n!!!WARNING!!! Cannot open desc_reproducibilities repertoire for:"
                )
                print(env_reprod_repertoire_folders.loc[line])
                traceback.print_exc()

            # Third, reeval_fit_reproducibilities repertoire
            try:
                repertoire_folder = env_reprod_repertoire_folders[
                    "reeval_fit_reproducibilities_repertoire_folder"
                ][line]
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
                    vmin=min_fit_reprod,
                    vmax=max_fit_reprod,
                    repertoire_descriptors=descriptors,
                    ax=ax.flat[2],
                )
                ax.flat[2].set_title(f"{prefixe_title}Reeval Fitness Reproducibilities")

            except Exception:
                print(
                    "\n!!!WARNING!!! Cannot open reeval_fit_reproducibilities repertoire for:"
                )
                print(env_reprod_repertoire_folders.loc[line])
                traceback.print_exc()

            # Fourth, reeval_desc_reproducibilities repertoire
            try:
                repertoire_folder = env_reprod_repertoire_folders[
                    "reeval_desc_reproducibilities_repertoire_folder"
                ][line]
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
                    vmin=min_desc_reprod,
                    vmax=max_desc_reprod,
                    repertoire_descriptors=descriptors,
                    ax=ax.flat[3],
                )
                ax.flat[3].set_title(
                    f"{prefixe_title}Reeval Descriptor Reproducibilities"
                )

            except Exception:
                print(
                    "\n!!!WARNING!!! Cannot open reeval_desc_reproducibilities repertoire for:"
                )
                print(env_reprod_repertoire_folders.loc[line])
                traceback.print_exc()

            # Finish figure
            plt.tight_layout()
            plt.savefig(file_name, bbox_inches="tight")
            plt.close()
