import os
import traceback
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
    get_cells_indices,
)

from analysis.load_archives import get_folder_name
from analysis.load_results import sort_data
from analysis.utils_archive import plot_one_paper_archive


def load_reproducibilities(
    plot_folder: str,
    config_frame: pd.DataFrame,
    compare_size: str,
    order: List,
    reproducibility_fix: bool,
    prefixe: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Note: during thesis writting, realised that the computation of reeval reproducibility
    was incorrect, to avoid breaking results from paper, added an option for the fix as a
    boolean, reproducibility_fix.
    """

    main_folder_name = f"{plot_folder}_{prefixe}reproducibility_repertoires/"

    # If already an min_max_frame file in the folder, use it
    file_name_all_reproducibilities_data = (
        f"{plot_folder}_csv/{prefixe}all_reproducibilities_data.csv"
    )
    file_name_reprod_min_max_frame = (
        f"{plot_folder}_csv/{prefixe}reprod_min_max_frame.csv"
    )
    if os.path.exists(file_name_all_reproducibilities_data) and os.path.exists(
        file_name_reprod_min_max_frame
    ):
        print("Loading existing plot archive reproducibility paper datas in:")
        print(file_name_all_reproducibilities_data)
        print(file_name_reprod_min_max_frame)
        all_reproducibilities_data = pd.read_csv(
            file_name_all_reproducibilities_data, header=0, index_col=False
        )
        reprod_min_max_frame = pd.read_csv(
            file_name_reprod_min_max_frame, header=0, index_col=False
        )
        return (
            all_reproducibilities_data,
            reprod_min_max_frame,
        )

    # If not, create it and populate it
    all_reproducibilities_data = pd.DataFrame()
    reprod_min_max_frame = pd.DataFrame()

    # Go through each env
    for env in config_frame["env"].drop_duplicates().values:

        # Get data for env
        print("\n    Reading for", env)
        env_config_frame = config_frame[config_frame["env"] == env].reset_index(
            drop=True
        )

        # Get parameters for env
        if "_" in env_config_frame["min_bd"][0]:
            min_bd_list = str(env_config_frame["min_bd"][0]).split("_")
        else:
            min_bd_list = str(env_config_frame["min_bd"][0])[1:-1].split(" ")
        if "" in min_bd_list:
            min_bd_list.remove("")
        if "_" in env_config_frame["max_bd"][0]:
            max_bd_list = str(env_config_frame["max_bd"][0]).split("_")
        else:
            max_bd_list = str(env_config_frame["max_bd"][0])[1:-1].split(" ")
        if "" in max_bd_list:
            max_bd_list.remove("")
        min_bd = [float(bd) for bd in min_bd_list]
        max_bd = [float(bd) for bd in max_bd_list]
        num_centroids = env_config_frame["num_centroids"][0]
        num_init_cvt_samples = env_config_frame["num_init_cvt_samples"][0]
        num_descriptors = len(min_bd)
        seed = env_config_frame["seed"][0]

        # Creating an archive to store max variance per cell
        random_key = jax.random.PRNGKey(seed)
        centroids, random_key = compute_cvt_centroids(
            num_descriptors=num_descriptors,
            num_init_cvt_samples=num_init_cvt_samples,
            num_centroids=num_centroids,
            minval=min_bd,
            maxval=max_bd,
            random_key=random_key,
        )

        # Fill in reprod_min_max_frame
        sub_reprod_min_max: Dict = {}
        sub_reprod_min_max["env"] = env
        sub_reprod_min_max["min_fit_reprod"] = 0
        sub_reprod_min_max["max_fit_reprod"] = 1
        sub_reprod_min_max["min_desc_reprod"] = 0
        sub_reprod_min_max["max_desc_reprod"] = 1
        sub_reprod_min_max["min_bd"] = min_bd
        sub_reprod_min_max["max_bd"] = max_bd
        reprod_min_max_frame = pd.concat(
            [reprod_min_max_frame, pd.DataFrame.from_dict(sub_reprod_min_max)],
            ignore_index=True,
        )

        # Opening all archives
        print("      First, opening all archives for normalisation.")
        initialised = False
        for line in range(env_config_frame.shape[0]):

            # Loading var repertoires
            try:
                fit_var_repertoire_folder = get_folder_name(
                    env_config_frame,
                    f"{prefixe}fit_var_repertoire_folder",
                    line,
                )
                desc_var_repertoire_folder = get_folder_name(
                    env_config_frame,
                    f"{prefixe}desc_var_repertoire_folder",
                    line,
                )

                # Open fit_var repertoire
                fit_var_fitnesses = jnp.load(
                    os.path.join(fit_var_repertoire_folder, "fitnesses.npy")
                )
                fit_var_descriptors = jnp.load(
                    os.path.join(fit_var_repertoire_folder, "descriptors.npy")
                )
                fit_var_genotypes = jnp.load(
                    os.path.join(fit_var_repertoire_folder, "genotypes.npy")
                )

                # Open desc_var repertoire
                desc_var_fitnesses = jnp.load(
                    os.path.join(desc_var_repertoire_folder, "fitnesses.npy")
                )
                desc_var_descriptors = jnp.load(
                    os.path.join(desc_var_repertoire_folder, "descriptors.npy")
                )
                desc_var_genotypes = jnp.load(
                    os.path.join(desc_var_repertoire_folder, "genotypes.npy")
                )
            except Exception:
                print("\n!!!WARNING!!! Cannot open repertoire of line", line)
                traceback.print_exc()
                continue

            # Loading reeval var repertoires
            reeval_require_compute = False
            try:
                reeval_fit_var_repertoire_folder = get_folder_name(
                    env_config_frame,
                    f"{prefixe}reeval_fit_var_repertoire_folder",
                    line,
                )
                reeval_desc_var_repertoire_folder = get_folder_name(
                    env_config_frame,
                    f"{prefixe}reeval_desc_var_repertoire_folder",
                    line,
                )

                # Open reeval_fit_var repertoire
                reeval_fit_var_centroids = jnp.load(
                    os.path.join(reeval_fit_var_repertoire_folder, "centroids.npy")
                )
                reeval_fit_var_genotypes = jnp.load(
                    os.path.join(reeval_fit_var_repertoire_folder, "genotypes.npy")
                )
                reeval_fit_var_fitnesses = jnp.load(
                    os.path.join(reeval_fit_var_repertoire_folder, "fitnesses.npy")
                )
                reeval_fit_var_descriptors = jnp.load(
                    os.path.join(reeval_fit_var_repertoire_folder, "descriptors.npy")
                )

                # Open reeval_desc_var repertoire
                reeval_desc_var_centroids = jnp.load(
                    os.path.join(reeval_desc_var_repertoire_folder, "centroids.npy")
                )
                reeval_desc_var_genotypes = jnp.load(
                    os.path.join(reeval_desc_var_repertoire_folder, "genotypes.npy")
                )
                reeval_desc_var_fitnesses = jnp.load(
                    os.path.join(reeval_desc_var_repertoire_folder, "fitnesses.npy")
                )
                reeval_desc_var_descriptors = jnp.load(
                    os.path.join(reeval_desc_var_repertoire_folder, "descriptors.npy")
                )

            except Exception:
                print(
                    "\n!!!WARNING!!! Cannot open reeval var repertoire of line",
                    line,
                )
                print(
                    "Attempting to re-create them, this process might lead to memory overflow."
                )
                reeval_require_compute = True
                traceback.print_exc()

            # Recomputing reeval var repertoire if required
            # This code only exist because of old data from the original UQD paper
            if reeval_require_compute:
                try:
                    reeval_repertoire_folder = get_folder_name(
                        env_config_frame,
                        f"{prefixe}reeval_repertoire_folder",
                        line,
                    )

                    # Open reeval_repertoire to build fit_var_reeval and desc_var_reeval
                    reeval_centroids = jnp.load(
                        os.path.join(reeval_repertoire_folder, "centroids.npy")
                    )
                    reeval_genotypes = jnp.load(
                        os.path.join(reeval_repertoire_folder, "genotypes.npy")
                    )
                    reeval_fitnesses = jnp.load(
                        os.path.join(reeval_repertoire_folder, "fitnesses.npy")
                    )
                    reeval_descriptors = jnp.load(
                        os.path.join(reeval_repertoire_folder, "descriptors.npy")
                    )

                    # Build reeval fit and desc var
                    num_centroids = reeval_genotypes.shape[0]
                    num_bds = reeval_genotypes.shape[1]
                    transformed_fit_var_genotypes = jnp.reshape(
                        jnp.repeat(fit_var_genotypes, num_centroids, axis=0),
                        (num_centroids, num_centroids, num_bds),
                    )
                    transformed_reeval_genotypes = jnp.reshape(
                        jnp.repeat(
                            jnp.expand_dims(reeval_genotypes, axis=0),
                            num_centroids,
                            axis=0,
                        ),
                        (num_centroids, num_centroids, num_bds),
                    )
                    tested = jnp.where(
                        jnp.transpose(
                            jnp.all(
                                jnp.isclose(
                                    transformed_fit_var_genotypes,
                                    transformed_reeval_genotypes,
                                    rtol=1e-02,
                                    atol=1e-02,
                                ),
                                axis=2,
                            )
                        ),
                        jnp.arange(1, num_centroids + 1, dtype=np.intc),
                        0,
                    )
                    fit_var_added = jnp.nanmax(tested, axis=1) - 1
                    fit_var_added = jnp.array(fit_var_added, dtype=np.intc)

                    reeval_fit_var_fitnesses = jnp.where(
                        reeval_fitnesses > -jnp.inf,
                        fit_var_fitnesses.at[fit_var_added].get(),
                        -jnp.inf,
                    )
                    reeval_desc_var_fitnesses = jnp.where(
                        reeval_fitnesses > -jnp.inf,
                        desc_var_fitnesses.at[fit_var_added].get(),
                        -jnp.inf,
                    )

                    reeval_fit_var_genotypes = reeval_genotypes
                    reeval_desc_var_genotypes = reeval_genotypes
                    reeval_fit_var_descriptors = reeval_descriptors
                    reeval_desc_var_descriptors = reeval_descriptors
                    reeval_fit_var_centroids = reeval_centroids
                    reeval_desc_var_centroids = reeval_centroids

                except Exception:
                    print(
                        "\n!!!WARNING!!! Cannot recompute reeval var repertoire, giving up for this line."
                    )
                    traceback.print_exc()
                    continue

            # Saving min and max
            try:
                # Create the archive
                if not initialised:
                    max_fit_var_repertoire = MapElitesRepertoire.init(
                        genotypes=fit_var_genotypes,
                        fitnesses=fit_var_fitnesses,
                        descriptors=fit_var_descriptors,
                        centroids=centroids,
                        extra_scores={},
                    )
                    max_desc_var_repertoire = MapElitesRepertoire.init(
                        genotypes=desc_var_genotypes,
                        fitnesses=desc_var_fitnesses,
                        descriptors=desc_var_descriptors,
                        centroids=centroids,
                        extra_scores={},
                    )
                    max_reeval_fit_var_repertoire = MapElitesRepertoire.init(
                        genotypes=reeval_fit_var_genotypes,
                        fitnesses=reeval_fit_var_fitnesses,
                        descriptors=reeval_fit_var_descriptors,
                        centroids=centroids,
                        extra_scores={},
                    )
                    max_reeval_desc_var_repertoire = MapElitesRepertoire.init(
                        genotypes=reeval_desc_var_genotypes,
                        fitnesses=reeval_desc_var_fitnesses,
                        descriptors=reeval_desc_var_descriptors,
                        centroids=centroids,
                        extra_scores={},
                    )
                    initialised = True

                # Add to existing archive
                else:
                    max_fit_var_repertoire = max_fit_var_repertoire.add(
                        fit_var_genotypes,
                        fit_var_descriptors,
                        fit_var_fitnesses,
                        {},
                    )
                    max_desc_var_repertoire = max_desc_var_repertoire.add(
                        desc_var_genotypes,
                        desc_var_descriptors,
                        desc_var_fitnesses,
                        {},
                    )
                    max_reeval_fit_var_repertoire = max_reeval_fit_var_repertoire.add(
                        reeval_fit_var_genotypes,
                        reeval_fit_var_descriptors,
                        reeval_fit_var_fitnesses,
                        {},
                    )
                    max_reeval_desc_var_repertoire = max_reeval_desc_var_repertoire.add(
                        reeval_desc_var_genotypes,
                        reeval_desc_var_descriptors,
                        reeval_desc_var_fitnesses,
                        {},
                    )
            except Exception:
                print("\n!!!WARNING!!! Cannot open repertoire of line", line)
                traceback.print_exc()
                continue

        # Write the maximum var archives
        save_single_archive(
            main_folder_name=main_folder_name,
            sub_folder_name=f"{env}_max_fit_var_repertoire",
            repertoire=max_fit_var_repertoire,
        )
        save_single_archive(
            main_folder_name=main_folder_name,
            sub_folder_name=f"{env}_max_desc_var_repertoire",
            repertoire=max_desc_var_repertoire,
        )
        save_single_archive(
            main_folder_name=main_folder_name,
            sub_folder_name=f"{env}_max_reeval_fit_var_repertoire",
            repertoire=max_reeval_fit_var_repertoire,
        )
        save_single_archive(
            main_folder_name=main_folder_name,
            sub_folder_name=f"{env}_max_reeval_desc_var_repertoire",
            repertoire=max_reeval_desc_var_repertoire,
        )

        # Plot the maximum var archives
        plot_single_archive(
            file_name=f"{plot_folder}/{env}_max_fit_var_repertoire.png",
            title=f"{env} - Maximum variances",
            centroids=centroids,
            fitnesses=max_fit_var_repertoire.fitnesses,
            descriptors=max_fit_var_repertoire.descriptors,
            min_bd=min_bd,
            max_bd=max_bd,
        )
        plot_single_archive(
            file_name=f"{plot_folder}/{env}_max_desc_var_repertoire.png",
            title=f"{env} - Maximum variances",
            centroids=centroids,
            fitnesses=max_desc_var_repertoire.fitnesses,
            descriptors=max_desc_var_repertoire.descriptors,
            min_bd=min_bd,
            max_bd=max_bd,
        )
        plot_single_archive(
            file_name=f"{plot_folder}/{env}_max_reeval_fit_var_repertoire.png",
            title=f"{env} - Maximum variances",
            centroids=centroids,
            fitnesses=max_reeval_fit_var_repertoire.fitnesses,
            descriptors=max_reeval_fit_var_repertoire.descriptors,
            min_bd=min_bd,
            max_bd=max_bd,
        )
        plot_single_archive(
            file_name=f"{plot_folder}/{env}_max_reeval_desc_var_repertoire.png",
            title=f"{env} - Maximum variances",
            centroids=centroids,
            fitnesses=max_reeval_desc_var_repertoire.fitnesses,
            descriptors=max_reeval_desc_var_repertoire.descriptors,
            min_bd=min_bd,
            max_bd=max_bd,
        )
        print(
            "      Done opening files for normalisation, second computing reproducibility."
        )

        # Go again through all files to compute modified archive and new metrics
        for algo in env_config_frame["algo"].drop_duplicates().values:
            algo_config_frame = env_config_frame[
                env_config_frame["algo"] == algo
            ].reset_index(drop=True)

            for size in algo_config_frame[compare_size].drop_duplicates().values:
                if algo_config_frame[algo_config_frame[compare_size] == size].empty:
                    print("    Size", size, "does not exist for", algo)
                    continue

                size_config_frame = algo_config_frame[
                    algo_config_frame[compare_size] == size
                ].reset_index(drop=True)

                # Only save one archive per env algo and size to be ploted
                written = False

                # For each rep
                for line in range(size_config_frame.shape[0]):

                    # Loading var repertoires
                    try:
                        fit_var_repertoire_folder = get_folder_name(
                            size_config_frame,
                            f"{prefixe}fit_var_repertoire_folder",
                            line,
                        )
                        desc_var_repertoire_folder = get_folder_name(
                            size_config_frame,
                            f"{prefixe}desc_var_repertoire_folder",
                            line,
                        )

                        # Open fit_var repertoire
                        fit_var_centroids = jnp.load(
                            os.path.join(fit_var_repertoire_folder, "centroids.npy")
                        )
                        fit_var_genotypes = jnp.load(
                            os.path.join(fit_var_repertoire_folder, "genotypes.npy")
                        )
                        fit_var_fitnesses = jnp.load(
                            os.path.join(fit_var_repertoire_folder, "fitnesses.npy")
                        )
                        fit_var_descriptors = jnp.load(
                            os.path.join(fit_var_repertoire_folder, "descriptors.npy")
                        )

                        # Open desc_var repertoire
                        desc_var_centroids = jnp.load(
                            os.path.join(desc_var_repertoire_folder, "centroids.npy")
                        )
                        desc_var_genotypes = jnp.load(
                            os.path.join(desc_var_repertoire_folder, "genotypes.npy")
                        )
                        desc_var_fitnesses = jnp.load(
                            os.path.join(desc_var_repertoire_folder, "fitnesses.npy")
                        )
                        desc_var_descriptors = jnp.load(
                            os.path.join(desc_var_repertoire_folder, "descriptors.npy")
                        )

                    except Exception:
                        print(
                            "\n!!!WARNING!!! Cannot open repertoire of line",
                            line,
                        )
                        traceback.print_exc()
                        continue

                    # Loading reeval var repertoires
                    reeval_require_compute = False
                    try:
                        reeval_fit_var_repertoire_folder = get_folder_name(
                            size_config_frame,
                            f"{prefixe}reeval_fit_var_repertoire_folder",
                            line,
                        )
                        reeval_desc_var_repertoire_folder = get_folder_name(
                            size_config_frame,
                            f"{prefixe}reeval_desc_var_repertoire_folder",
                            line,
                        )

                        # Open reeval_fit_var repertoire
                        reeval_fit_var_centroids = jnp.load(
                            os.path.join(
                                reeval_fit_var_repertoire_folder, "centroids.npy"
                            )
                        )
                        reeval_fit_var_genotypes = jnp.load(
                            os.path.join(
                                reeval_fit_var_repertoire_folder, "genotypes.npy"
                            )
                        )
                        reeval_fit_var_fitnesses = jnp.load(
                            os.path.join(
                                reeval_fit_var_repertoire_folder, "fitnesses.npy"
                            )
                        )
                        reeval_fit_var_descriptors = jnp.load(
                            os.path.join(
                                reeval_fit_var_repertoire_folder, "descriptors.npy"
                            )
                        )

                        # Open reeval_desc_var repertoire
                        reeval_desc_var_centroids = jnp.load(
                            os.path.join(
                                reeval_desc_var_repertoire_folder, "centroids.npy"
                            )
                        )
                        reeval_desc_var_genotypes = jnp.load(
                            os.path.join(
                                reeval_desc_var_repertoire_folder, "genotypes.npy"
                            )
                        )
                        reeval_desc_var_fitnesses = jnp.load(
                            os.path.join(
                                reeval_desc_var_repertoire_folder, "fitnesses.npy"
                            )
                        )
                        reeval_desc_var_descriptors = jnp.load(
                            os.path.join(
                                reeval_desc_var_repertoire_folder, "descriptors.npy"
                            )
                        )

                    except Exception:
                        print(
                            "\n!!!WARNING!!! Cannot open reeval var repertoire of line",
                            line,
                        )
                        print(
                            "Attempting to re-create them, this process might lead to memory overflow."
                        )
                        reeval_require_compute = True
                        traceback.print_exc()

                    # Normalise fit and desc var
                    fit_var_indices = get_cells_indices(fit_var_descriptors, centroids)
                    matching_max_fit_var_repertoire = max_fit_var_repertoire.fitnesses[
                        fit_var_indices
                    ]
                    fit_reproducibilities = jnp.where(
                        matching_max_fit_var_repertoire == 0.0,
                        0.0,
                        jnp.divide(
                            fit_var_fitnesses,
                            matching_max_fit_var_repertoire,
                        ),
                    )
                    fit_reproducibilities = jnp.where(
                        matching_max_fit_var_repertoire > -jnp.inf,
                        fit_reproducibilities,
                        -jnp.inf,
                    )

                    desc_var_indices = get_cells_indices(
                        desc_var_descriptors, centroids
                    )
                    matching_max_desc_var_repertoire = (
                        max_desc_var_repertoire.fitnesses[desc_var_indices]
                    )
                    desc_reproducibilities = jnp.where(
                        matching_max_desc_var_repertoire == 0.0,
                        0.0,
                        jnp.divide(
                            desc_var_fitnesses,
                            matching_max_desc_var_repertoire,
                        ),
                    )
                    desc_reproducibilities = jnp.where(
                        matching_max_desc_var_repertoire > -jnp.inf,
                        desc_reproducibilities,
                        -jnp.inf,
                    )

                    # Recomputing reeval var repertoire if required
                    # This code only exist because of old data from the original UQD paper
                    if reeval_require_compute:
                        try:
                            reeval_repertoire_folder = get_folder_name(
                                size_config_frame,
                                f"{prefixe}reeval_repertoire_folder",
                                line,
                            )

                            # Open reeval_repertoire to build fit_var_reeval and desc_var_reeval
                            reeval_centroids = jnp.load(
                                os.path.join(reeval_repertoire_folder, "centroids.npy")
                            )
                            reeval_genotypes = jnp.load(
                                os.path.join(reeval_repertoire_folder, "genotypes.npy")
                            )
                            reeval_fitnesses = jnp.load(
                                os.path.join(reeval_repertoire_folder, "fitnesses.npy")
                            )
                            reeval_descriptors = jnp.load(
                                os.path.join(
                                    reeval_repertoire_folder, "descriptors.npy"
                                )
                            )

                            # Build reeval fit and desc reproducibilities
                            num_centroids = reeval_genotypes.shape[0]
                            num_bds = reeval_genotypes.shape[1]
                            transformed_fit_var_genotypes = jnp.reshape(
                                jnp.repeat(fit_var_genotypes, num_centroids, axis=0),
                                (num_centroids, num_centroids, num_bds),
                            )
                            transformed_reeval_genotypes = jnp.reshape(
                                jnp.repeat(
                                    jnp.expand_dims(reeval_genotypes, axis=0),
                                    num_centroids,
                                    axis=0,
                                ),
                                (num_centroids, num_centroids, num_bds),
                            )
                            tested = jnp.where(
                                jnp.transpose(
                                    jnp.all(
                                        jnp.isclose(
                                            transformed_fit_var_genotypes,
                                            transformed_reeval_genotypes,
                                            rtol=1e-02,
                                            atol=1e-02,
                                        ),
                                        axis=2,
                                    )
                                ),
                                jnp.arange(1, num_centroids + 1, dtype=np.intc),
                                0,
                            )
                            fit_var_added = jnp.nanmax(tested, axis=1) - 1
                            fit_var_added = jnp.array(fit_var_added, dtype=np.intc)

                            reeval_fit_reproducibilities = jnp.where(
                                reeval_fitnesses > -jnp.inf,
                                fit_reproducibilities.at[fit_var_added].get(),
                                -jnp.inf,
                            )
                            reeval_desc_reproducibilities = jnp.where(
                                reeval_fitnesses > -jnp.inf,
                                desc_reproducibilities.at[fit_var_added].get(),
                                -jnp.inf,
                            )

                            reeval_fit_var_genotypes = reeval_genotypes
                            reeval_desc_var_genotypes = reeval_genotypes
                            reeval_fit_var_descriptors = reeval_descriptors
                            reeval_desc_var_descriptors = reeval_descriptors
                            reeval_fit_var_centroids = reeval_centroids
                            reeval_desc_var_centroids = reeval_centroids

                        except Exception:
                            print(
                                "\n!!!WARNING!!! Cannot recompute reeval var repertoire, giving up for this line."
                            )
                            traceback.print_exc()
                            continue
                    else:

                        # Normalise reeval fit and desc var
                        # (If recomputing already using normalised values)
                        reeval_fit_var_indices = get_cells_indices(
                            reeval_fit_var_descriptors, centroids
                        )
                        if reproducibility_fix:
                            matching_max_reeval_fit_var_repertoire = (
                                max_reeval_fit_var_repertoire.fitnesses[
                                    reeval_fit_var_indices
                                ]
                            )
                        else:
                            matching_max_reeval_fit_var_repertoire = (
                                max_fit_var_repertoire.fitnesses[reeval_fit_var_indices]
                            )
                        reeval_fit_reproducibilities = jnp.where(
                            matching_max_reeval_fit_var_repertoire == 0.0,
                            0.0,
                            jnp.divide(
                                reeval_fit_var_fitnesses,
                                matching_max_reeval_fit_var_repertoire,
                            ),
                        )
                        reeval_fit_reproducibilities = jnp.where(
                            matching_max_reeval_fit_var_repertoire > -jnp.inf,
                            reeval_fit_reproducibilities,
                            -jnp.inf,
                        )

                        reeval_desc_var_indices = get_cells_indices(
                            reeval_desc_var_descriptors, centroids
                        )
                        if reproducibility_fix:
                            matching_max_reeval_desc_var_repertoire = (
                                max_reeval_desc_var_repertoire.fitnesses[
                                    reeval_desc_var_indices
                                ]
                            )
                        else:
                            matching_max_reeval_desc_var_repertoire = (
                                max_desc_var_repertoire.fitnesses[
                                    reeval_desc_var_indices
                                ]
                            )
                        reeval_desc_reproducibilities = jnp.where(
                            matching_max_reeval_desc_var_repertoire == 0.0,
                            0.0,
                            jnp.divide(
                                reeval_desc_var_fitnesses,
                                matching_max_reeval_desc_var_repertoire,
                            ),
                        )
                        reeval_desc_reproducibilities = jnp.where(
                            matching_max_reeval_desc_var_repertoire > -jnp.inf,
                            reeval_desc_reproducibilities,
                            -jnp.inf,
                        )

                    # Inverse values to not penalise new cells
                    fit_reproducibilities = jnp.where(
                        fit_reproducibilities == -jnp.inf, 1.0, fit_reproducibilities
                    )
                    fit_reproducibilities = 1.0 - fit_reproducibilities
                    desc_reproducibilities = jnp.where(
                        desc_reproducibilities == -jnp.inf, 1.0, desc_reproducibilities
                    )
                    desc_reproducibilities = 1.0 - desc_reproducibilities
                    reeval_fit_reproducibilities = jnp.where(
                        reeval_fit_reproducibilities == -jnp.inf,
                        1.0,
                        reeval_fit_reproducibilities,
                    )
                    reeval_fit_reproducibilities = 1.0 - reeval_fit_reproducibilities
                    reeval_desc_reproducibilities = jnp.where(
                        reeval_desc_reproducibilities == -jnp.inf,
                        1.0,
                        reeval_desc_reproducibilities,
                    )
                    reeval_desc_reproducibilities = 1.0 - reeval_desc_reproducibilities

                    if any(fit_reproducibilities < 0):
                        print(
                            f"!!!WARNING!!! Got out of range fit_reproducibilities: smaller than 0 {algo} {env}"
                        )
                        print(fit_reproducibilities[fit_reproducibilities < 0])
                    if any(fit_reproducibilities > 1):
                        print(
                            f"!!!WARNING!!! Got out of range fit_reproducibilities: bigger than 1 {algo} {env}"
                        )
                        print(fit_reproducibilities[fit_reproducibilities > 1])
                    if any(reeval_fit_reproducibilities < 0):
                        print(
                            f"!!!WARNING!!! Got out of range reeval_fit_reproducibilities: smaller than 0 {algo} {env}"
                        )
                        print(
                            reeval_fit_reproducibilities[
                                reeval_fit_reproducibilities < 0
                            ]
                        )
                    if any(reeval_fit_reproducibilities > 1):
                        print(
                            f"!!!WARNING!!! Got out of range reeval_fit_reproducibilities: bigger than 1 {algo} {env}"
                        )
                        print(
                            reeval_fit_reproducibilities[
                                reeval_fit_reproducibilities > 1
                            ]
                        )

                    if any(desc_reproducibilities < 0):
                        print(
                            f"!!!WARNING!!! Got out of range desc_reproducibilities: smaller than 0 {algo} {env}"
                        )
                        print(desc_reproducibilities[desc_reproducibilities < 0])
                    if any(desc_reproducibilities > 1):
                        print(
                            f"!!!WARNING!!! Got out of range desc_reproducibilities: bigger than 1 {algo} {env}"
                        )
                        print(desc_reproducibilities[desc_reproducibilities > 1])
                    if any(reeval_desc_reproducibilities < 0):
                        print(
                            f"!!!WARNING!!! Got out of range reeval_desc_reproducibilities: smaller than 0 {algo} {env}"
                        )
                        print(
                            reeval_desc_reproducibilities[
                                reeval_desc_reproducibilities < 0
                            ]
                        )
                    if any(reeval_desc_reproducibilities > 1):
                        print(
                            f"!!!WARNING!!! Got out of range reeval_desc_reproducibilities: bigger than 1 {algo} {env}"
                        )
                        print(
                            reeval_desc_reproducibilities[
                                reeval_desc_reproducibilities > 1
                            ]
                        )

                    # Write one of the repertoire for later ploting
                    if not written:
                        # Create the repertoires
                        fit_reproducibilities_repertoire = MapElitesRepertoire.init(
                            genotypes=fit_var_genotypes,
                            fitnesses=fit_reproducibilities,
                            descriptors=fit_var_descriptors,
                            centroids=fit_var_centroids,
                            extra_scores={},
                        )
                        desc_reproducibilities_repertoire = MapElitesRepertoire.init(
                            genotypes=desc_var_genotypes,
                            fitnesses=desc_reproducibilities,
                            descriptors=desc_var_descriptors,
                            centroids=desc_var_centroids,
                            extra_scores={},
                        )
                        reeval_fit_reproducibilities_repertoire = (
                            MapElitesRepertoire.init(
                                genotypes=reeval_fit_var_genotypes,
                                fitnesses=reeval_fit_reproducibilities,
                                descriptors=reeval_fit_var_descriptors,
                                centroids=reeval_fit_var_centroids,
                                extra_scores={},
                            )
                        )
                        reeval_desc_reproducibilities_repertoire = (
                            MapElitesRepertoire.init(
                                genotypes=reeval_desc_var_genotypes,
                                fitnesses=reeval_desc_reproducibilities,
                                descriptors=reeval_desc_var_descriptors,
                                centroids=reeval_desc_var_centroids,
                                extra_scores={},
                            )
                        )
                        # plot_single_archive(
                        #    file_name=f"{plot_folder}/{env}_{algo}_test.png",
                        #    title=f"{env} - {algo}",
                        #    centroids=reeval_desc_reproducibilities_repertoire.centroids,
                        #    fitnesses=reeval_desc_reproducibilities_repertoire.fitnesses,
                        #    descriptors=reeval_desc_reproducibilities_repertoire.descriptors,
                        #    min_bd=min_bd,
                        #    max_bd=max_bd,
                        # )

                        algo_name = algo.replace(" ", "_")
                        fit_reproducibilities_folder_name = save_single_archive(
                            main_folder_name=main_folder_name,
                            sub_folder_name=f"{env}_{algo_name}_{size}_{prefixe}fit_reproducibilities_repertoire",
                            repertoire=fit_reproducibilities_repertoire,
                        )
                        desc_reproducibilities_folder_name = save_single_archive(
                            main_folder_name=main_folder_name,
                            sub_folder_name=f"{env}_{algo_name}_{size}_{prefixe}desc_reproducibilities_repertoire",
                            repertoire=desc_reproducibilities_repertoire,
                        )
                        reeval_fit_reproducibilities_folder_name = save_single_archive(
                            main_folder_name=main_folder_name,
                            sub_folder_name=f"{env}_{algo_name}_{size}_{prefixe}reeval_fit_reproducibilities_repertoire",
                            repertoire=reeval_fit_reproducibilities_repertoire,
                        )
                        reeval_desc_reproducibilities_folder_name = save_single_archive(
                            main_folder_name=main_folder_name,
                            sub_folder_name=f"{env}_{algo_name}_{size}_{prefixe}reeval_desc_reproducibilities_repertoire",
                            repertoire=reeval_desc_reproducibilities_repertoire,
                        )

                        written = True

                    # Compute new metrics
                    reprod_data: Dict[str, List[float]] = {}
                    reprod_data[f"{prefixe}fit_reproducibilities_qd_score"] = jnp.sum(
                        fit_reproducibilities
                    )
                    reprod_data[f"{prefixe}avg_fit_reproducibilities_qd_score"] = (
                        jnp.sum(fit_reproducibilities) / num_centroids
                    )
                    reprod_data[f"{prefixe}desc_reproducibilities_qd_score"] = jnp.sum(
                        desc_reproducibilities
                    )
                    reprod_data[f"{prefixe}avg_desc_reproducibilities_qd_score"] = (
                        jnp.sum(desc_reproducibilities) / num_centroids
                    )
                    reprod_data[
                        f"{prefixe}reeval_fit_reproducibilities_qd_score"
                    ] = jnp.sum(reeval_fit_reproducibilities)
                    reprod_data[
                        f"{prefixe}reeval_avg_fit_reproducibilities_qd_score"
                    ] = (jnp.sum(reeval_fit_reproducibilities) / num_centroids)
                    reprod_data[
                        f"{prefixe}reeval_desc_reproducibilities_qd_score"
                    ] = jnp.sum(reeval_desc_reproducibilities)
                    reprod_data[
                        f"{prefixe}reeval_avg_desc_reproducibilities_qd_score"
                    ] = (jnp.sum(reeval_desc_reproducibilities) / num_centroids)

                    reprod_data["algo"] = [algo]
                    reprod_data["env"] = [env]
                    reprod_data[compare_size] = [size]
                    reprod_data["rep"] = [line]

                    all_reproducibilities_data = pd.concat(
                        [
                            all_reproducibilities_data,
                            pd.DataFrame.from_dict(reprod_data),
                        ],
                        ignore_index=True,
                    )

        print("      Done computing reproducibility.")

    all_reproducibilities_data.to_csv(file_name_all_reproducibilities_data, index=None)
    reprod_min_max_frame.to_csv(file_name_reprod_min_max_frame, index=None)

    return all_reproducibilities_data, reprod_min_max_frame


def save_single_archive(
    main_folder_name: str,
    sub_folder_name: str,
    repertoire: MapElitesRepertoire,
) -> str:

    # Create main folder first
    if not os.path.exists(main_folder_name):
        os.mkdir(main_folder_name)

    # Create sub folder now
    folder_name = f"{main_folder_name}/{sub_folder_name}/"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    # Save repertoire
    repertoire.save(path=folder_name)

    return folder_name


def plot_single_archive(
    file_name: str,
    title: str,
    centroids: jnp.ndarray,
    fitnesses: jnp.ndarray,
    descriptors: jnp.ndarray,
    min_bd: jnp.ndarray,
    max_bd: jnp.ndarray,
    min_fitness: jnp.ndarray = None,
    max_fitness: jnp.ndarray = None,
) -> None:

    try:
        # Create a figure
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(70, 8), sharey=True)

        # Print
        plot_one_paper_archive(
            centroids=centroids,
            descriptors=descriptors,
            fitnesses=fitnesses,
            ax=ax,
            minval=min_bd,
            maxval=max_bd,
            vmin=min_fitness,
            vmax=max_fitness,
            colorbar=True,
        )

        # Add title
        ax.set_title(title)

        # Finish figure
        plt.tight_layout()
        plt.savefig(file_name, bbox_inches="tight")
        plt.close()

    except Exception:
        print(f"Failed to plot {title}.")
        traceback.print_exc()
