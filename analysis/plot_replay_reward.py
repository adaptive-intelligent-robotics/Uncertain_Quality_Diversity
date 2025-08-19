import traceback
from typing import Any, List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brax.io import html
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    get_cells_indices,
)
from qdax.types import Genotype, RNGKey

from analysis.plot_archives import get_folder_name
from environments_manager.set_up_environment import (
    ENV_NEUROEVOLUTION,
    ENV_NEUROEVOLUTION_GENERALISED,
    set_up_environment,
)


def plot_replay_reward(
    plot_folder: str,
    config_frame: pd.DataFrame,
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

            # Ensure env is valid
            if env_name not in ENV_NEUROEVOLUTION + ENV_NEUROEVOLUTION_GENERALISED:
                print(
                    f"!!!WARNING!!! Invalid environment for reward distribution: {env_name}."
                )
                continue

            # Init environment
            params_std = 0
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
            if "_posnormal" in env_name:
                gaussian_pos = True
                env_name = env_name[: -len("_posnormal")]
            if "_velnormal" in env_name:
                gaussian_vel = True
                env_name = env_name[: -len("_velnormal")]
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
                fit_std=0.0,
                desc_std=0.0,
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

            # Take best indiv
            best_indiv_idx = jnp.argmax(repertoire.fitnesses)
            best_params = jax.tree_util.tree_map(
                lambda x: x[best_indiv_idx].squeeze(), repertoire.genotypes
            )
            print("      Visualising best individual with indexe:", best_indiv_idx)

            file_name = (
                f"{plot_folder}/{env_name}_best_reward_distribution_"
                + f"{algo}_{size}_{seed}_{best_indiv_idx}"
            )
            print("      Saving best reward in file:", file_name)
            roll_out(
                file_name=file_name,
                env=env,
                params=best_params,
                policy_structure=policy_structure,
                random_key=random_key,
                is_env_control=False,
                save_html=save_html,
            )

            # Take indiv corresponding to BD
            indices = get_cells_indices(jnp.asarray([bd]), repertoire.centroids)
            params = jax.tree_util.tree_map(
                lambda x: x[indices].squeeze(), repertoire.genotypes
            )
            print("      Visualising individual with indexe:", indices)

            file_name = (
                f"{plot_folder}/{env_name}_reward_distribution_"
                + f"{algo}_{size}_{seed}_{indices[0]}"
            )
            print("      Saving best reward in file:", file_name)
            roll_out(
                file_name=file_name,
                env=env,
                params=params,
                policy_structure=policy_structure,
                random_key=random_key,
                is_env_control=False,
                save_html=save_html,
            )

        except Exception:
            print("\n!!!WARNING!!! Cannot process with reward_distribution for:")
            print(config_frame.loc[line])
            if errors:
                traceback.print_exc()


def roll_out(
    file_name: str,
    env: Any,
    params: Genotype,
    policy_structure: Any,
    random_key: RNGKey,
    is_env_control: bool,
    save_html: bool,
) -> None:

    # Run
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_structure.apply)
    rollout = []
    rewards = []
    next_state = jit_env_reset(rng=random_key)
    timestep = 0
    while not next_state.done:
        state = next_state
        rollout.append(state)
        rewards.append(state.reward)
        if is_env_control:
            action = jit_inference_fn(params, state.obs, timestep)
        else:
            action = jit_inference_fn(params, state.obs)
        next_state = jit_env_step(state, action)
        timestep += 1

    # Save html
    if save_html:
        html_file_name = file_name + ".html"
        html_file = html.render(env.sys, [s.pipeline_state for s in rollout])
        f = open(html_file_name, "w")
        f.write(html_file)
        f.close()

    # Save reward distribution
    figsize = (8, 6)
    reward_file_name = file_name + ".png"
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize, sharex=True)
    plt.plot(rewards, label="Rewards")
    plt.plot([r - 5 * r / 100 for r in rewards], label="Reward - 5 %")
    plt.plot([r + 5 * r / 100 for r in rewards], label="Reward + 5 %")
    axes.axhline(np.mean(rewards), label="Average", color="r")
    axes.axhline(np.mean(rewards[:50]), label="50 steps Average", color="g")
    axes.axhline(np.mean(rewards[:100]), label="100 steps Average", color="b")
    axes.axhline(np.mean(rewards[:150]), label="150 steps Average", color="y")
    plt.legend()
    plt.savefig(reward_file_name)
    plt.close()
