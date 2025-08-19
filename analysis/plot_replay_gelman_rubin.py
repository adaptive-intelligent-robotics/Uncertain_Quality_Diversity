import traceback
from functools import partial
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
from qdax.core.gelman_rubin import (
    compute_delta,
    compute_omnidirectional_revisited_gelman_rubin_convergence_estimators,
    compute_omnidirectional_revisited_gelman_rubin_estimators,
    compute_unidirectional_revisited_gelman_rubin_convergence_estimators,
    compute_unidirectional_revisited_gelman_rubin_estimators,
)
from qdax.types import Genotype, RNGKey

from analysis.plot_archives import get_folder_name
from environments_manager.set_up_environment import (
    ENV_TIMESTEP_LIST,
    set_up_environment,
)

gr_period = 50
gr_warm_start = 0
gr_epsilon = 0.10
gr_uni_convergence = compute_delta(3, epsilon=gr_epsilon)
gr_omni_convergence = compute_delta(3, epsilon=gr_epsilon)
gr_min_convergence = 0
first_batch_size = 20
second_batch_size = 8


def plot_replay_gelman_rubin(
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
                    f"!!!WARNING!!! Invalid environment for gelman rubin estimation: {env_name}."
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
            # Take best indiv
            best_indiv_idx = jnp.argmax(repertoire.fitnesses)
            best_params = jax.tree_util.tree_map(
                lambda x: x[best_indiv_idx].squeeze(), repertoire.genotypes
            )
            print("      Visualising best individual with indexe:", best_indiv_idx)

            file_name = (
                f"{plot_folder}/{env_name}_best_gelman_rubin_"
                + f"{algo}_{size}_{seed}_{best_indiv_idx}"
            )
            print("      Saving best gelman_rubin in file:", file_name)
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
                f"{plot_folder}/{env_name}_gelman_rubin_"
                + f"{algo}_{size}_{seed}_{indices[0]}"
            )
            print("      Saving gelman_rubin in file:", file_name)
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
            print("\n!!!WARNING!!! Cannot process with gelman_rubin for:")
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

    # Functions
    compute_unidirectional_revisited_gelman_rubin_estimators_fn = partial(
        compute_unidirectional_revisited_gelman_rubin_estimators,
        first_batch_size=first_batch_size,
        second_batch_size=second_batch_size,
    )
    compute_omnidirectional_revisited_gelman_rubin_estimators_fn = partial(
        compute_omnidirectional_revisited_gelman_rubin_estimators,
        first_batch_size=first_batch_size,
        second_batch_size=second_batch_size,
    )
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(policy_structure.apply)

    # Run
    rollout = []
    rewards = []
    state_descriptors = []
    uni_sigma_below = []
    uni_sigma_above = []
    uni_convergence_estimator = []
    omni_sigma_below = []
    omni_sigma_above = []
    omni_convergence_estimator = []
    omni_bd_sigma_below = []
    omni_bd_sigma_above = []
    omni_bd_convergence_estimator = []
    next_state = jit_env_reset(rng=random_key)
    timestep = 0
    while not next_state.done:

        # Get all info from state
        state = next_state
        rollout.append(state)
        rewards.append(state.reward)
        state_descriptors.append(state.info["state_descriptor"])

        # Get action
        if is_env_control:
            action = jit_inference_fn(params, state.obs, timestep)
        else:
            action = jit_inference_fn(params, state.obs)

        # Step environment
        next_state = jit_env_step(state, action)
        timestep += 1

        # Apply Gelman-Rubin estimator
        if timestep % gr_period == 0 and timestep > gr_warm_start:
            samples = jnp.asarray(rewards)[gr_warm_start:]
            (
                sigma_below,
                sigma_above,
            ) = compute_unidirectional_revisited_gelman_rubin_estimators_fn(
                samples=samples,
            )
            uni_sigma_below.append(sigma_below)
            uni_sigma_above.append(sigma_above)
            uni_convergence_estimator.append(
                compute_unidirectional_revisited_gelman_rubin_convergence_estimators(
                    samples=samples, sigma_below=sigma_below, sigma_above=sigma_above
                )
            )

            samples = jnp.concatenate(
                [
                    jnp.expand_dims(jnp.asarray(rewards), axis=1),
                    jnp.asarray(state_descriptors),
                ],
                axis=1,
            )[gr_warm_start:, :]
            (
                sigma_below,
                sigma_above,
            ) = compute_omnidirectional_revisited_gelman_rubin_estimators_fn(
                samples=samples,
            )
            omni_sigma_below.append(sigma_below)
            omni_sigma_above.append(sigma_above)
            omni_convergence_estimator.append(
                compute_omnidirectional_revisited_gelman_rubin_convergence_estimators(
                    samples=samples, sigma_below=sigma_below, sigma_above=sigma_above
                )
            )

            samples = jnp.asarray(state_descriptors)[gr_warm_start:, :]
            (
                sigma_below,
                sigma_above,
            ) = compute_omnidirectional_revisited_gelman_rubin_estimators_fn(
                samples=samples,
            )
            omni_bd_sigma_below.append(sigma_below)
            omni_bd_sigma_above.append(sigma_above)
            omni_bd_convergence_estimator.append(
                compute_omnidirectional_revisited_gelman_rubin_convergence_estimators(
                    samples=samples, sigma_below=sigma_below, sigma_above=sigma_above
                )
            )

    # Save html
    if save_html:
        html_file_name = file_name + ".html"
        html_file = html.render(env.sys, [s.pipeline_state for s in rollout])
        f = open(html_file_name, "w")
        f.write(html_file)
        f.close()

    # Save gelman rubin estimator
    state_descriptors = np.asarray(state_descriptors)
    uni_stop_condition = np.where(
        np.asarray(uni_convergence_estimator) < gr_uni_convergence
    )[0]
    if len(uni_stop_condition) > 0:
        if np.any(uni_stop_condition > (gr_min_convergence // gr_period)):
            uni_stop_condition = uni_stop_condition[
                uni_stop_condition > (gr_min_convergence // gr_period)
            ][0]
        else:
            uni_stop_condition = 0
    else:
        uni_stop_condition = 0
    omni_stop_condition = np.where(
        np.asarray(omni_convergence_estimator) < gr_omni_convergence
    )[0]
    if len(omni_stop_condition) > 0:
        if np.any(omni_stop_condition > (gr_min_convergence // gr_period)):
            omni_stop_condition = omni_stop_condition[
                omni_stop_condition > (gr_min_convergence // gr_period)
            ][0]
        else:
            omni_stop_condition = 0
    else:
        omni_stop_condition = 0
    omni_bd_stop_condition = np.where(
        np.asarray(omni_bd_convergence_estimator) < gr_omni_convergence
    )[0]
    if len(omni_bd_stop_condition) > 0:
        if np.any(omni_bd_stop_condition > (gr_min_convergence // gr_period)):
            omni_bd_stop_condition = omni_bd_stop_condition[
                omni_bd_stop_condition > (gr_min_convergence // gr_period)
            ][0]
        else:
            omni_bd_stop_condition = 0
    else:
        omni_bd_stop_condition = 0
    num_dim = state_descriptors.shape[1]
    print(
        f"Uni {uni_stop_condition * gr_period} {uni_convergence_estimator[uni_stop_condition]}"
    )
    print(
        f"Omni {omni_stop_condition * gr_period} {omni_convergence_estimator[omni_stop_condition]}"
    )
    print(
        f"OmniBD {omni_bd_stop_condition * gr_period} {omni_bd_convergence_estimator[omni_bd_stop_condition]}"
    )

    figsize = (30, 30)
    gr_file_name = file_name + ".png"
    fig, axes = plt.subplots(nrows=2, ncols=max(num_dim + 1, 4), figsize=figsize)

    axes[0, 0].plot(rewards, label="Rewards")
    axes[0, 0].axvline(x=(uni_stop_condition * gr_period), color="r", label="Uni stop")
    axes[0, 0].axvline(
        x=(omni_stop_condition * gr_period), color="b", label="Omni stop"
    )
    axes[0, 0].axvline(
        x=(omni_bd_stop_condition * gr_period), color="g", label="OmniBD stop"
    )
    axes[0, 0].legend()
    for i in range(num_dim):
        axes[0, i + 1].plot(state_descriptors[:, i], label=f"Desc {i}")
        axes[0, i + 1].axvline(
            x=uni_stop_condition * gr_period, color="r", label="Uni stop"
        )
        axes[0, i + 1].axvline(
            x=omni_stop_condition * gr_period, color="b", label="Omni stop"
        )
        axes[0, i + 1].axvline(
            x=omni_bd_stop_condition * gr_period, color="g", label="OmniBD stop"
        )
        axes[0, i + 1].legend()
    axes[1, 0].plot(uni_sigma_below, label="Uni GR sigma_below")
    axes[1, 0].plot(uni_sigma_above, label="Uni GR sigma_above")
    axes[1, 0].axvline(x=uni_stop_condition, color="r", label="Uni stop")
    axes[1, 0].axvline(x=omni_stop_condition, color="b", label="Omni stop")
    axes[1, 0].axvline(x=omni_bd_stop_condition, color="g", label="OmniBD stop")
    axes[1, 0].legend()
    axes[1, 1].plot(omni_sigma_below, label="Omni GR sigma_below")
    axes[1, 1].plot(omni_sigma_above, label="Omni GR sigma_above")
    axes[1, 1].axvline(x=uni_stop_condition, color="r", label="Uni stop")
    axes[1, 1].axvline(x=omni_stop_condition, color="b", label="Omni stop")
    axes[1, 1].axvline(x=omni_bd_stop_condition, color="g", label="OmniBD stop")
    axes[1, 1].legend()
    axes[1, 2].plot(omni_bd_sigma_below, label="OmniBD GR sigma_below")
    axes[1, 2].plot(omni_bd_sigma_above, label="OmniBD GR sigma_above")
    axes[1, 2].axvline(x=uni_stop_condition, color="r", label="Uni stop")
    axes[1, 2].axvline(x=omni_stop_condition, color="b", label="Omni stop")
    axes[1, 2].axvline(x=omni_bd_stop_condition, color="g", label="OmniBD stop")
    axes[1, 2].legend()
    axes[1, 3].plot(uni_convergence_estimator, label="Uni R")
    axes[1, 3].plot(omni_convergence_estimator, label="Omni R")
    axes[1, 3].plot(omni_bd_convergence_estimator, label="OmniBD R")
    axes[1, 3].axvline(x=uni_stop_condition, color="r", label="Uni stop")
    axes[1, 3].axvline(x=omni_stop_condition, color="b", label="Omni stop")
    axes[1, 3].axvline(x=omni_bd_stop_condition, color="g", label="OmniBD stop")
    axes[1, 3].legend()

    plt.savefig(gr_file_name)
    plt.close()
