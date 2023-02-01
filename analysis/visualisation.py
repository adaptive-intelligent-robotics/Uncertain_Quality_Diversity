from typing import Any

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from brax.io import html
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qdax.core.containers.repertoire import Repertoire
from qdax.types import Genotype, RNGKey
from qdax.utils.plotting import get_voronoi_finite_polygons_2d


def save_html(
    file_name: str,
    env: Any,
    my_params: Genotype,
    policy_structure: Any,
    random_key: RNGKey,
    is_env_control: bool = False,
) -> None:
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_structure.apply)
    rollout = []
    next_state = jit_env_reset(rng=random_key)
    timestep = 0
    while not next_state.done:
        state = next_state
        rollout.append(state)
        if is_env_control:
            action = jit_inference_fn(my_params, state.obs, timestep)
        else:
            action = jit_inference_fn(my_params, state.obs)
        next_state = jit_env_step(state, action)
        timestep += 1

    # Save html
    html_file = html.render(env.sys, [s.qp for s in rollout])
    f = open(file_name, "w")
    f.write(html_file)
    f.close()


def plot_visualisation_archive(
    file_name: str,
    replications: int,
    repertoire: Repertoire,
    fitnesses: jnp.ndarray,
    descriptors: jnp.ndarray,
    before_descriptor: jnp.ndarray,
    before_fitness: float,
    min_bd: jnp.ndarray,
    max_bd: jnp.ndarray,
    vmin: float,
    vmax: float,
    original: bool = True,
    average: bool = False,
    average_descriptor: jnp.ndarray = None,
    average_fitness: jnp.ndarray = None,
    average_both: jnp.ndarray = None,
    median: bool = False,
    median_descriptor: jnp.ndarray = None,
    median_fitness: jnp.ndarray = None,
    median_both: jnp.ndarray = None,
    paper_plot: bool = False,
) -> None:

    # Set the parameters
    my_cmap = cm.viridis
    font_size = 12
    params = {
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
    }
    mpl.rcParams.update(params)

    # Create the plot object
    fig, ax = plt.subplots(facecolor="white", edgecolor="white", figsize=(10, 10))
    if len(np.array(min_bd).shape) == 0 and len(np.array(max_bd).shape) == 0:
        ax.set_xlim(min_bd, max_bd)
        ax.set_ylim(min_bd, max_bd)
    else:
        ax.set_xlim(min_bd[0], max_bd[0])
        ax.set_ylim(min_bd[1], max_bd[1])
    ax.set(adjustable="box", aspect="equal")

    # Create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(repertoire.centroids)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Fill the plot with contours
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)

    # Fill the plot with the colors
    for idx, fitness in enumerate(repertoire.fitnesses):
        if fitness > -jnp.inf:
            region = regions[idx]
            polygon = vertices[region]

            ax.fill(*zip(*polygon), alpha=0.3, color=my_cmap(norm(fitness)))

    # Add replications as dots
    ax.scatter(
        descriptors[:, 0],
        descriptors[:, 1],
        c=fitnesses,
        norm=norm,
        cmap=my_cmap,
        s=40 if not paper_plot else 600,
        zorder=0,
    )
    if original:
        ax.scatter(
            before_descriptor[0],
            before_descriptor[1],
            c=before_fitness,
            norm=norm,
            marker="X",
            edgecolor="r",
            s=80 if not paper_plot else 550,
            zorder=0,
            label="Original",
        )

    # Add average descriptor
    if average and not (
        average_descriptor is not None
        and average_fitness is not None
        and average_both is not None
    ):
        both = jnp.concatenate(
            [descriptors, jnp.expand_dims(fitnesses, axis=1)], axis=1
        )
        average_descriptor = jnp.mean(descriptors, axis=0)
        average_fitness = jnp.mean(fitnesses, axis=0)
        average_both = jnp.mean(both, axis=0)
    if (
        average_descriptor is not None
        and average_fitness is not None
        and average_both is not None
    ):
        ax.scatter(
            average_descriptor[0],
            average_descriptor[1],
            c=average_fitness,
            norm=norm,
            marker="P",
            edgecolor="r",
            s=80 if not paper_plot else 550,
            zorder=0,
            label="Average",
        )
        ax.scatter(
            average_both[0],
            average_both[1],
            c=average_both[2],
            norm=norm,
            marker="p",
            edgecolor="r",
            s=80 if not paper_plot else 550,
            zorder=0,
            label="Average both",
        )

    # Add median descriptor
    if median and not (
        median_descriptor is not None
        and median_fitness is not None
        and median_both is not None
    ):
        both = jnp.concatenate(
            [descriptors, jnp.expand_dims(fitnesses, axis=1)], axis=1
        )
        median_descriptor = jnp.median(descriptors, axis=0)
        median_fitness = jnp.median(fitnesses, axis=0)
        median_both = jnp.median(both, axis=0)
    if (
        median_descriptor is not None
        and median_fitness is not None
        and median_both is not None
    ):
        ax.scatter(
            median_descriptor[0],
            median_descriptor[1],
            c=median_fitness,
            norm=norm,
            marker="o",
            edgecolor="r",
            s=80 if not paper_plot else 550,
            zorder=0,
            label="Median",
        )
        ax.scatter(
            median_both[0],
            median_both[1],
            c=median_both[2],
            norm=norm,
            marker="s",
            edgecolor="r",
            s=80 if not paper_plot else 550,
            zorder=0,
            label="Median both",
        )

    if not paper_plot:
        plt.legend()

    # Aesthetic
    if paper_plot:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    else:
        ax.set_xlabel("Behavior Dimension 1")
        ax.set_ylabel("Behavior Dimension 2")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
        cbar.ax.tick_params(labelsize=font_size)
        ax.set_title(str(replications) + " replications")
    ax.set_aspect("equal")

    # Save figure
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()


def plot_simple_visualisation_archive(
    file_name: str,
    replications: int,
    repertoire: Repertoire,
    fitnesses: jnp.ndarray,
    descriptors: jnp.ndarray,
    before_descriptor: jnp.ndarray,
    before_fitness: float,
    min_bd: jnp.ndarray,
    max_bd: jnp.ndarray,
    vmin: float = None,
    vmax: float = None,
) -> None:
    """
    Plot an archive with all replications as scatter plot.
    Args:
        file_name: where to save the image
        replications: number of replications performed (used for title only)
        repertoire: original repertoire (used to put it in background)
        fitnesses: replications fitnesses
        descriptors: replications descriptors
        before_descriptor: descriptor as orginally estimated by the algo
        before_fitness: fitness as orginally estimated by the algo
        min_bd: minimum bd for plot
        max_bd: maximum bd for plot
    """

    # Set the parameters
    my_cmap = cm.viridis
    font_size = 12
    params = {
        "axes.labelsize": font_size,
        "legend.fontsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "text.usetex": False,
    }
    mpl.rcParams.update(params)

    # Create the plot object
    fig, ax = plt.subplots(facecolor="white", edgecolor="white", figsize=(10, 10))
    if len(np.array(min_bd).shape) == 0 and len(np.array(max_bd).shape) == 0:
        ax.set_xlim(min_bd, max_bd)
        ax.set_ylim(min_bd, max_bd)
    else:
        ax.set_xlim(min_bd[0], max_bd[0])
        ax.set_ylim(min_bd[1], max_bd[1])
    ax.set(adjustable="box", aspect="equal")

    # Create the regions and vertices from centroids
    regions, vertices = get_voronoi_finite_polygons_2d(repertoire.centroids)
    if vmin is None:
        vmin = min(
            jnp.nanmin(jnp.where(fitnesses == -jnp.inf, jnp.inf, fitnesses)),
            jnp.nanmin(
                jnp.where(
                    repertoire.fitnesses == -jnp.inf, jnp.inf, repertoire.fitnesses
                )
            ),
        )
    elif vmax is None:
        vmax = max(jnp.nanmax(fitnesses), jnp.nanmax(repertoire.fitnesses))
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Fill the plot with contours
    for region in regions:
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor="black", facecolor="white", lw=1)

    # Fill the plot with the colors
    for idx, fitness in enumerate(repertoire.fitnesses):
        if fitness > -jnp.inf:
            region = regions[idx]
            polygon = vertices[region]

            ax.fill(*zip(*polygon), alpha=0.3, color=my_cmap(norm(fitness)))

    # Add replications as dots
    ax.scatter(
        descriptors[:, 0],
        descriptors[:, 1],
        c=fitnesses,
        norm=norm,
        cmap=my_cmap,
        s=40,
        zorder=0,
    )
    ax.scatter(
        before_descriptor[0],
        before_descriptor[1],
        c=before_fitness,
        norm=norm,
        marker="X",
        edgecolor="r",
        s=80,
        zorder=0,
        label="Original",
    )

    plt.legend()

    # Aesthetic
    ax.set_xlabel("Behavior Dimension 1")
    ax.set_ylabel("Behavior Dimension 2")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=my_cmap), cax=cax)
    cbar.ax.tick_params(labelsize=font_size)
    ax.set_title(str(replications) + " replications")
    ax.set_aspect("equal")

    # Save figure
    plt.tight_layout()
    plt.savefig(file_name, bbox_inches="tight")
    plt.close()
