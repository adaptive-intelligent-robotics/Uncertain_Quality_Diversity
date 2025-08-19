import traceback
from typing import Any, List

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brax.v1.io import html
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.types import Genotype, RNGKey
from qdax.utils.plotting import get_voronoi_finite_polygons_2d

from analysis.plot_archives import get_folder_name
from environments_manager.set_up_environment import (
    ENV_CONTROL,
    ENV_TIMESTEP_LIST,
    set_up_environment,
)


def plot_replay_trajectory(
    plot_folder: str,
    config_frame: pd.DataFrame,
    min_max_frame: pd.DataFrame,
    compare_size: str,
    deterministic: bool,
    replications: int,
    bd: List,
    paper_plot: bool,
    save_html: bool,
    errors: bool = False,
) -> None:

    # For each line in config file
    for line in range(config_frame.shape[0]):
        print("\n    Visualising results in line", line)

        try:
            # Get necessary configs
            seed = config_frame["seed"][line]
            env_name = config_frame["env"][line]
            min_fitness = min_max_frame[min_max_frame["env"] == env_name]["min_fitness"]
            max_fitness = min_max_frame[min_max_frame["env"] == env_name]["max_fitness"]
            size = config_frame[compare_size][line]
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
            fit_std = 0.0
            desc_std = 0.0
            params_std = 0.0
            gaussian_pos = False
            gaussian_vel = False
            if "_params-var" in env_name:
                params_idx = env_name.find("_params-var")
                params_std = float(env_name[params_idx + 11 :])
            elif "params_std" in config_frame.columns:
                if (
                    config_frame["params_std"].values[line]
                    == config_frame["params_std"].values[line]
                ):
                    params_std = config_frame["params_std"].values[line]
            if "_fit" in env_name and "_desc" in env_name and "_params" in env_name:
                fit_idx = env_name.find("_fit")
                desc_idx = env_name.find("_desc")
                params_idx = env_name.find("_params")
                fit_std = float(env_name[fit_idx + 4 : desc_idx])
                desc_std = float(env_name[desc_idx + 5 : params_idx])
                params_std = float(env_name[params_idx + 7 :])
                env_name = env_name[:fit_idx]
            if "_posnormal" in env_name:
                gaussian_pos = True
                env_name = env_name[: -len("_posnormal")]
            if "_velnormal" in env_name:
                gaussian_vel = True
                env_name = env_name[: -len("_velnormal")]

            # Ensure env is valid
            if env_name not in ENV_TIMESTEP_LIST:
                print(
                    f"!!!WARNING!!! Invalid environment for trajectory distribution: {env_name}."
                )
                continue

            (
                _,
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
                batch_size=1,
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
            repertoire_folder = get_folder_name(
                config_frame, "reeval_repertoire_folder", line
            )
            repertoire_folder += "/"
            repertoire = MapElitesRepertoire.load(
                reconstruction_fn=reconstruction_fn,
                path=repertoire_folder,
            )

            # Take indiv corresponding to BD
            indices = get_cells_indices(jnp.asarray([bd]), repertoire.centroids)
            if repertoire.fitnesses[indices] == -jnp.inf:
                print("!!!WARNING!!! This cell is empty for this rep.")
                continue

            params = jax.tree_util.tree_map(
                lambda x: x[indices].squeeze(), repertoire.genotypes
            )
            before_descriptor = repertoire.descriptors[indices].squeeze()

            # Generate trajectory
            file_name = (
                f"{plot_folder}/{env_name}_trajectory_distribution_"
                + f"{algo}_{size}_{seed}_{indices[0]}"
            )
            print("      Saving trajectory in file:", file_name)
            roll_out(
                file_name=file_name,
                env=env,
                repertoire=repertoire,
                min_bd=min_bd,
                max_bd=max_bd,
                vmin=min_fitness,
                vmax=max_fitness,
                bd=before_descriptor,
                params=params,
                policy_structure=policy_structure,
                random_key=random_key,
                replications=replications,
                is_env_control=env_name in ENV_CONTROL,
                paper_plot=paper_plot,
                save_html=save_html,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot process with trajectory_distribution for:")
            print(config_frame.loc[line])
            if errors:
                traceback.print_exc()


def roll_out(
    file_name: str,
    env: Any,
    repertoire: MapElitesRepertoire,
    min_bd: jnp.ndarray,
    max_bd: jnp.ndarray,
    vmin: float,
    vmax: float,
    bd: List,
    params: Genotype,
    policy_structure: Any,
    random_key: RNGKey,
    replications: int,
    is_env_control: bool,
    paper_plot: bool,
    save_html: bool,
) -> None:
    """Main function to save visualiation as html."""
    my_cmap = cm.viridis
    font_size = 12

    # Do all the replications
    trajectories = []
    for rep in range(replications):

        random_key, subkey = jax.random.split(random_key)

        # Run
        jit_env_reset = jax.jit(env.reset)
        jit_env_step = jax.jit(env.step)
        jit_inference_fn = jax.jit(policy_structure.apply)
        rollout = []
        trajectory = []
        next_state = jit_env_reset(rng=subkey)
        timestep = 0
        while not next_state.done:
            state = next_state
            rollout.append(state)
            trajectory.append(state.info["state_descriptor"])
            if is_env_control:
                action = jit_inference_fn(params, state.obs, timestep)
            else:
                action = jit_inference_fn(params, state.obs)
            next_state = jit_env_step(state, action)
            timestep += 1
        trajectory.append(next_state.info["state_descriptor"])

        # Save html
        if save_html:
            html_file_name = file_name + f"_rep{rep}.html"
            html_file = html.render(env.sys, [s.qp for s in rollout])
            f = open(html_file_name, "w")
            f.write(html_file)
            f.close()

        # Store all trajectories
        trajectory = np.asarray(trajectory)
        trajectory = trajectory.transpose()
        trajectories.append(trajectory)

    # Save trajectory distribution
    trajectory_file_name = file_name + ".png"
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

    # Add the trajectories
    for rep in range(replications):
        plt.plot(trajectories[rep][0], trajectories[rep][1])

    # Add cross at initial BD
    ax.scatter(
        bd[0],
        bd[1],
        c="r",
        norm=norm,
        marker="X",
        edgecolor="r",
        s=500 if paper_plot else 100,
        zorder=0,
        label="Original",
    )

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
    plt.savefig(trajectory_file_name, bbox_inches="tight")
    plt.close()
