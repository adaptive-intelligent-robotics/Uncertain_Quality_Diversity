import os
import traceback
from typing import List, Optional, Tuple

import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qdax.utils.plotting import get_voronoi_finite_polygons_2d

from analysis.constants import BASELINES, CATEGORIES, ENV_NAME_DIFFICULTY, ENV_NO_VAR


def plot_env_paper_archives(
    name_column: str,
    file_name: str,
    min_max_column: str,
    repertoire_folders: pd.DataFrame,
    env_min_max: pd.DataFrame,
    no_traceback: int,
) -> None:

    # Order algos
    algos = repertoire_folders["algo"].drop_duplicates().values
    ranked_algos: List = []
    for baseline in BASELINES:
        for algo in algos:
            if baseline in algo:
                ranked_algos.append(algo)
    for category in CATEGORIES:
        for algo in algos:
            if category in algo:
                ranked_algos.append(algo)

    # Order envs
    env_names = repertoire_folders["env"].drop_duplicates().values
    final_envs: List = []
    title_envs: List = []
    for env_name in ENV_NAME_DIFFICULTY.keys():
        if "_var" in name_column:
            if env_name in ENV_NO_VAR:
                continue
        if env_name in env_names:
            final_envs.append(env_name)
            title_envs.append(ENV_NAME_DIFFICULTY[env_name])
    for env_name in env_names:
        if "_var" in name_column:
            if env_name in ENV_NO_VAR:
                continue
        if env_name not in final_envs:
            final_envs.append(env_name)
            title_envs.append(env_name)

    # Create figure
    num_algos = len(repertoire_folders["algo"].drop_duplicates().values)
    num_envs = max(2, len(final_envs))
    fig, ax = plt.subplots(
        nrows=num_envs, ncols=num_algos, figsize=(4 * num_algos, 4.1 * num_envs)
    )
    col = 0

    # One column per algo
    for algo in ranked_algos:
        line = 0

        # One line per env
        for env_idx in range(num_envs):
            try:
                if repertoire_folders[
                    (repertoire_folders["algo"] == algo)
                    & (repertoire_folders["env"] == final_envs[env_idx])
                ].empty:
                    print("    Env", title_envs[env_idx], "does not exist for", algo)

                    # Display empty graph
                    ax[line, col].tick_params(
                        left=False,
                        right=False,
                        labelleft=False,
                        labelbottom=False,
                        bottom=False,
                    )

                    ax[line, col].spines["top"].set_visible(False)
                    ax[line, col].spines["right"].set_visible(False)
                    ax[line, col].spines["bottom"].set_visible(False)
                    ax[line, col].spines["left"].set_visible(False)

                    ax[line, col].set_aspect("equal")
                    if line == 0:
                        title = algo.replace("Sampling", "Smpl")
                        title = algo.replace("Vanilla-MAP-Elites", "MAP-Elites")
                        title = algo.replace("sampling", "smpl")
                        ax[0, col].set_title(title)
                    line += 1
                    continue

                # Find min and max
                vmin = env_min_max[env_min_max["env"] == final_envs[env_idx]][
                    "min_" + min_max_column
                ].values[0]
                vmax = env_min_max[env_min_max["env"] == final_envs[env_idx]][
                    "max_" + min_max_column
                ].values[0]
                min_bd = env_min_max[env_min_max["env"] == final_envs[env_idx]][
                    "min_bd"
                ].values[0]
                max_bd = env_min_max[env_min_max["env"] == final_envs[env_idx]][
                    "max_bd"
                ].values[0]

                # Find file
                repertoire_folder = repertoire_folders[
                    (repertoire_folders["algo"] == algo)
                    & (repertoire_folders["env"] == final_envs[env_idx])
                ][name_column].values[0]

                # Open file
                fitnesses = jnp.load(os.path.join(repertoire_folder, "fitnesses.npy"))
                descriptors = jnp.load(
                    os.path.join(repertoire_folder, "descriptors.npy")
                )
                centroids = jnp.load(os.path.join(repertoire_folder, "centroids.npy"))

                # Plot
                _, _ = plot_one_paper_archive(
                    centroids=centroids,
                    repertoire_fitnesses=fitnesses,
                    minval=min_bd,
                    maxval=max_bd,
                    vmin=vmin,
                    vmax=vmax,
                    repertoire_descriptors=descriptors,
                    ax=ax[line, col],
                    colorbar=False,
                )
                if line == 0:
                    title = algo.replace("Sampling", "Smpl")
                    title = algo.replace("Vanilla-MAP-Elites", "MAP-Elites")
                    title = algo.replace("sampling", "smpl")
                    ax[0, col].set_title(title)
                if col == 0:
                    ax[line, 0].set_ylabel(title_envs[env_idx])
                else:
                    ax[line, col].set_ylabel(None)

            except Exception:
                if not no_traceback:
                    print(
                        "\n!!!WARNING!!! Cannot open file for",
                        algo,
                        "and",
                        title_envs[env_idx],
                    )
                    traceback.print_exc()

            line += 1
        col += 1

    # Save figure
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()


def plot_one_paper_archive(
    centroids: jnp.ndarray,
    repertoire_fitnesses: jnp.ndarray,
    minval: jnp.ndarray,
    maxval: jnp.ndarray,
    repertoire_descriptors: Optional[jnp.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    vmin: Optional[float] = None,
    colorbar: Optional[bool] = True,
    vmax: Optional[float] = None,
) -> Tuple[Optional[Figure], Axes]:
    """Plot a visual representation of a 2d map elites repertoire.

    TODO: Use repertoire as input directly. Because this
    function is very specific to repertoires.

    Args:
        centroids: the centroids of the repertoire
        repertoire_fitnesses: the fitness of the repertoire
        minval: minimum values for the descritors
        maxval: maximum values for the descriptors
        repertoire_descriptors: the descriptors. Defaults to None.
        ax: a matplotlib axe for the figure to plot. Defaults to None.
        vmin: minimum value for the fitness. Defaults to None. If not given,
            the value will be set to the minimum fitness in the repertoire.
        vmax: maximum value for the fitness. Defaults to None. If not given,
            the value will be set to the maximum fitness in the repertoire.

    Raises:
        NotImplementedError: does not work for descriptors dimension different
        from 2.

    Returns:
        A figure and axes object, corresponding to the visualisation of the
        repertoire.
    """

    # TODO: check it and fix it if needed
    grid_empty = repertoire_fitnesses == -jnp.inf
    num_descriptors = centroids.shape[1]
    if num_descriptors != 2:
        raise NotImplementedError("Grid plot supports 2 descriptors only for now.")

    my_cmap = cm.viridis

    fitnesses = repertoire_fitnesses
    if vmin is None:
        vmin = float(jnp.min(fitnesses[~grid_empty]))
    if vmax is None:
        vmax = float(jnp.max(fitnesses[~grid_empty]))

    # create the plot object
    fig = None
    if ax is None:
        fig, ax = plt.subplots(facecolor="white", edgecolor="white")

    assert (
        len(np.array(minval).shape) < 2
    ), f"minval : {minval} should be float or couple of floats"
    assert (
        len(np.array(maxval).shape) < 2
    ), f"maxval : {maxval} should be float or couple of floats"

    if len(np.array(minval).shape) == 0 and len(np.array(maxval).shape) == 0:
        ax.set_xlim(minval, maxval)
        ax.set_ylim(minval, maxval)
    else:
        ax.set_xlim(minval[0], maxval[0])
        ax.set_ylim(minval[1], maxval[1])

    ax.set(adjustable="box", aspect="equal")

    # create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(centroids)

    norm = Normalize(vmin=vmin, vmax=vmax)

    # fill the plot with contours
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)

    # fill the plot with the colors
    for idx, fitness in enumerate(fitnesses):
        if fitness > -jnp.inf:
            region = regions[idx]
            polygon = vertices[region]

            ax.fill(*zip(*polygon), alpha=0.8, color=my_cmap(norm(fitness)))

    # if descriptors are specified, add points location
    if repertoire_descriptors is not None:
        descriptors = repertoire_descriptors[~grid_empty]
        ax.scatter(
            descriptors[:, 0],
            descriptors[:, 1],
            c=fitnesses[~grid_empty],
            cmap=my_cmap,
            s=10,
            zorder=0,
        )

    # aesthetic
    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
        cbar.ax.tick_params(labelsize=20)

    ax.tick_params(
        left=False, right=False, labelleft=False, labelbottom=False, bottom=False
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_aspect("equal")

    return fig, ax
