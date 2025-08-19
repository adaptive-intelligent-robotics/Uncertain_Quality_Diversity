import traceback
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.core.containers.repertoire import Repertoire
from qdax.utils.plotting import get_voronoi_finite_polygons_2d
from scipy.spatial.distance import cdist, euclidean

from analysis.plot_archives import get_folder_name
from environments_manager.set_up_environment import set_up_environment


def plot_replay_archive(
    plot_folder: str,
    config_frame: pd.DataFrame,
    min_max_frame: pd.DataFrame,
    compare_size: str,
    deterministic: bool,
    replications: int,
    bd: List,
    paper_plot: bool,
    errors: bool = False,
) -> None:

    # For each line in config file
    for line in range(config_frame.shape[0]):
        print("\n    Visualising results in line", line)

        try:
            # Get necessary configs
            seed = config_frame["seed"][line]
            env_name = config_frame["env"][line]
            size = config_frame[compare_size][line]
            min_fitness = min_max_frame[min_max_frame["env"] == env_name]["min_fitness"]
            max_fitness = min_max_frame[min_max_frame["env"] == env_name]["max_fitness"]
            algo = config_frame["algo"][line].replace(" ", "_")
            policy_hidden_layer_sizes = config_frame["policy_hidden_layer_sizes"][line]
            policy_hidden_layer_sizes = (
                tuple([int(x) for x in policy_hidden_layer_sizes.split("_")])
                if type(policy_hidden_layer_sizes) == str
                else tuple([policy_hidden_layer_sizes])
            )

            # Init a random key
            random_key = jax.random.PRNGKey(seed)

            # Init environment
            if "_nonoise" in env_name:
                env_name = env_name[: env_name.find("_nonoise")]
                fit_std = 0.0
                desc_std = 0.0
                params_std = 0.0
                gaussian_pos = False
                gaussian_vel = False
                print("      Loading environment:", env_name, "without noise.")
            elif "_fit" in env_name and "_desc" in env_name and "_params" in env_name:
                fit_idx = env_name.find("_fit")
                desc_idx = env_name.find("_desc")
                params_idx = env_name.find("_params")
                fit_std = float(env_name[fit_idx + 4 : desc_idx])
                desc_std = float(env_name[desc_idx + 5 : params_idx])
                params_std = float(env_name[params_idx + 7 :])
                gaussian_pos = False
                gaussian_vel = False
                env_name = env_name[:fit_idx]
                print("      Loading environment:", env_name)
                print("        With fitness std:", fit_std)
                print("        With desc std:", desc_std)
                print("        With parameters std:", params_std)
            elif "_velnormal" in env_name or "_posnormal" in env_name:
                if "_posnormal" in env_name:
                    gaussian_pos = True
                    env_name = env_name[: -len("_posnormal")]
                if "_velnormal" in env_name:
                    gaussian_vel = True
                    env_name = env_name[: -len("_velnormal")]
            else:
                fit_std = 0
                desc_std = 0
                params_std = 0
                gaussian_pos = False
                gaussian_vel = False
            if params_std == 0 and "params_std" in config_frame.columns:
                if (
                    config_frame["params_std"].values[line]
                    == config_frame["params_std"].values[line]
                ):
                    params_std = config_frame["params_std"].values[line]
            (
                env_name,
                env,
                _,
                _,
                scoring_fn,
                _,
                _,
                _,
                _,
                policy_structure,
                _,
                init_policies_fn,
                _,
                _,
                _,
                min_bd,
                max_bd,
                qd_offset,
                num_descriptors,
                _,
                _,
                random_key,
            ) = set_up_environment(
                env_name=env_name,
                episode_length=config_frame["episode_length"][line],
                batch_size=replications,
                policy_hidden_layer_sizes=policy_hidden_layer_sizes,
                random_key=random_key,
                deterministic=deterministic,
                fit_std=fit_std,
                desc_std=desc_std,
                params_std=params_std,
                gaussian_pos=gaussian_pos,
                gaussian_vel=gaussian_vel,
            )

            # Reconstruction function
            print("      Loading repertoire")
            init_policies, random_key = init_policies_fn(1, random_key)
            init_policy = jax.tree_map(lambda x: x[0], init_policies)
            _, reconstruction_fn = jax.flatten_util.ravel_pytree(init_policy)

            # Load repertoire
            repertoire_folder = get_folder_name(config_frame, "repertoire_folder", line)
            repertoire_folder += "/"
            repertoire = MapElitesRepertoire.load(
                reconstruction_fn=reconstruction_fn,
                path=repertoire_folder,
            )

            # Take indiv corresponding to BD
            indices = get_cells_indices(jnp.asarray([bd]), repertoire.centroids)
            before_fitness = repertoire.fitnesses[indices].squeeze()
            if min_fitness.values.size == 0:
                min_fitness = min(repertoire.fitnesses[repertoire.fitnesses > -jnp.inf])
            if max_fitness.values.size == 0:
                max_fitness = max(repertoire.fitnesses)
            before_descriptor = repertoire.descriptors[indices].squeeze()

            ############################################
            # Perform replications for archive display #

            print("      Starting replications")
            random_key, subkey = jax.random.split(random_key)
            params = jax.tree_util.tree_map(
                lambda x: jnp.repeat(
                    jnp.expand_dims(x[indices].squeeze(), axis=0), replications, axis=0
                ),
                repertoire.genotypes,
            )
            fitnesses, descriptors, _, _ = scoring_fn(params, subkey)

            # Display replications bd results in a grid
            file_name = (
                f"{plot_folder}/{env_name}_{replications}rep"
                + f"_{algo}_{size}_{seed}_{indices[0]}.png"
            )
            print("         Saving replication results in:", file_name)
            sub_plot_replay_archive(
                file_name=file_name,
                replications=replications,
                repertoire=repertoire,
                fitnesses=fitnesses,
                descriptors=descriptors,
                before_descriptor=before_descriptor,
                before_fitness=before_fitness,
                min_bd=min_bd,
                max_bd=max_bd,
                vmin=min_fitness,
                vmax=max_fitness,
                paper_plot=paper_plot,
                average=not paper_plot,
                median=not paper_plot,
                closermedian=not paper_plot,
                geometricmedian=not paper_plot,
            )

            # If paper, display also the original grid
            if paper_plot:
                file_name = f"{plot_folder}/{env_name}_{algo}_{size}_{seed}.png"
                print("         Saving replication results in:", file_name)
                sub_plot_replay_archive(
                    file_name=file_name,
                    replications=0,
                    repertoire=repertoire,
                    fitnesses=jnp.delete(repertoire.fitnesses, indices, axis=0),
                    descriptors=jnp.delete(repertoire.descriptors, indices, axis=0),
                    before_descriptor=before_descriptor,
                    before_fitness=before_fitness,
                    min_bd=min_bd,
                    max_bd=max_bd,
                    vmin=min_fitness,
                    vmax=max_fitness,
                    paper_plot=True,
                )

            print("\n    Finished with line", line)

        except Exception:
            print("\n!!!WARNING!!! Cannot process with visualisation for:")
            print(config_frame.loc[line])
            if errors:
                traceback.print_exc()


def geometric_median(values: jnp.ndarray, axis: int, eps: float = 1e-5) -> jnp.ndarray:
    """Compute a genometric median."""

    def one_dim_geometric_median(values: jnp.ndarray) -> jnp.ndarray:
        y = np.mean(values, axis=0)

        while True:
            d = cdist(values, [y])
            nonzeros = (d != 0)[:, 0]

            dinv = 1 / d[nonzeros]
            dinvs = np.sum(dinv)
            w = dinv / dinvs
            t = np.sum(w * values[nonzeros], axis=0)

            num_zeros = len(values) - np.sum(nonzeros)
            if num_zeros == 0:
                y1 = t
            elif num_zeros == len(values):
                return y
            else:
                vec_r = (t - y) * dinvs
                r = np.linalg.norm(vec_r)
                rinv = 0 if r == 0 else num_zeros / r
                y1 = max(0, 1 - rinv) * t + min(1, rinv) * y

            if euclidean(y, y1) < eps:
                return y1

            y = y1
        return y

    if axis == 0:
        return one_dim_geometric_median(values)
    if axis > 1:
        print("!!!WARNING!!! Not sure about the code with axis > 1 at the moment.")
    num_dim = values.shape[0]
    median = []
    for dim in range(num_dim):
        median.append(one_dim_geometric_median(values[dim]))
    return jnp.array(median)


def closer_median(values: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Compute a closer median."""

    def one_dim_closer_median(values: jnp.ndarray) -> jnp.ndarray:
        def distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return jnp.sqrt(jnp.sum(jnp.square(x - y)))

        distances = jax.vmap(
            jax.vmap(partial(distance), in_axes=(None, 0)), in_axes=(0, None)
        )(values, values)
        distances = jnp.mean(distances, axis=0)
        min_index = jnp.argmin(distances)
        return values[min_index]

    if axis == 0:
        return one_dim_closer_median(values)
    if axis > 1:
        print("!!!WARNING!!! Not sure about the code with axis > 1 at the moment.")
    return jax.vmap(one_dim_closer_median)(values)


def sub_plot_replay_archive(
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
    closermedian: bool = False,
    closer_descriptor: jnp.ndarray = None,
    closer_fitness: jnp.ndarray = None,
    closer_both: jnp.ndarray = None,
    geometricmedian: bool = False,
    geometric_descriptor: jnp.ndarray = None,
    geometric_fitness: jnp.ndarray = None,
    geometric_both: jnp.ndarray = None,
    paper_plot: bool = False,
) -> None:
    """Main function to save a visualisation archive."""
    my_cmap = cm.viridis
    font_size = 12

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

    # Add closer to median
    if closermedian and not (
        closer_descriptor is not None
        and closer_fitness is not None
        and closer_both is not None
    ):
        both = jnp.concatenate(
            [descriptors, jnp.expand_dims(fitnesses, axis=1)], axis=1
        )
        closer_descriptor = closer_median(descriptors, axis=0)
        closer_fitness = closer_median(fitnesses, axis=0)
        closer_both = closer_median(both, axis=0)
    if (
        closer_descriptor is not None
        and closer_fitness is not None
        and closer_both is not None
    ):
        ax.scatter(
            closer_descriptor[0],
            closer_descriptor[1],
            c=closer_fitness,
            norm=norm,
            marker="D",
            edgecolor="r",
            s=80 if not paper_plot else 550,
            zorder=0,
            label="Closer",
        )
        ax.scatter(
            closer_both[0],
            closer_both[1],
            c=closer_both[2],
            norm=norm,
            marker="*",
            edgecolor="r",
            s=80 if not paper_plot else 550,
            zorder=0,
            label="Closer both",
        )

    # Add geometric median descriptor
    if geometricmedian and not (
        geometric_descriptor is not None
        and geometric_fitness is not None
        and geometric_both is not None
    ):
        both = jnp.concatenate(
            [descriptors, jnp.expand_dims(fitnesses, axis=1)], axis=1
        )
        geometric_descriptor = geometric_median(descriptors, axis=0)
        geometric_fitness = jnp.median(fitnesses, axis=0)  # Fitness is one-dimensional
        geometric_both = geometric_median(both, axis=0)
    if (
        geometric_descriptor is not None
        and geometric_fitness is not None
        and geometric_both is not None
    ):
        ax.scatter(
            geometric_descriptor[0],
            geometric_descriptor[1],
            c=geometric_fitness,
            norm=norm,
            marker="v",
            edgecolor="r",
            s=80 if not paper_plot else 550,
            zorder=0,
            label="Geometric",
        )
        ax.scatter(
            geometric_both[0],
            geometric_both[1],
            c=geometric_both[2],
            norm=norm,
            marker="^",
            edgecolor="r",
            s=80 if not paper_plot else 550,
            zorder=0,
            label="Geometric both",
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
