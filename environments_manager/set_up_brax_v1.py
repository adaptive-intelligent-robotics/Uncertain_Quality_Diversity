from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from qdax.core.neuroevolution.buffers.buffer import DCRLTransition, Transition
from qdax.core.neuroevolution.networks.networks import MLP, MLPDC
from qdax.custom_types import EnvState, Genotype, RNGKey
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs

import tasks
from environments_manager.uncertainty_in_cell_metrics import (
    incell_reevaluation_function,
)
from environments_manager.uncertainty_metrics import reevaluation_function
from tasks.brax_control_envs import reset_based_scoring_function_time_brax_envs


def set_up_brax_v1(
    env_name: str,
    episode_length: int,
    batch_size: int,
    policy_hidden_layer_sizes: Tuple,
    random_key: RNGKey,
    deterministic: bool = False,
    params_std: float = 0.0,
    delta_fitness: float = 0.0,
    delta_reproducibility: float = 0.0,
) -> Tuple:

    # Init environment
    env = tasks.create(
        env_name,
        episode_length=episode_length,
        reset_noise_scale=params_std,
    )

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )
    policy_dc_network = MLPDC(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jax.nn.softmax,
    )

    # Init population of controllers
    def init_policies_fn(size: int, random_key: RNGKey) -> Tuple[jnp.ndarray, RNGKey]:
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, num=size)
        fake_batch = jnp.zeros(shape=(size, env.observation_size))
        init_policies = jax.vmap(policy_network.init)(keys, fake_batch)
        return init_policies, random_key

    # Define the fonction to play a step with the policy in the environment
    def brax_play_step_fn(
        env_state: EnvState,
        policy_params: Genotype,
        random_key: RNGKey,
        env: Any,
    ) -> Tuple[EnvState, Genotype, RNGKey, Transition]:

        """
        Play an environment step and return the updated state and the transition.
        """

        actions = policy_network.apply(policy_params, env_state.obs)
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = DCRLTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
            desc=jnp.zeros(
                env.behavior_descriptor_length,
            )
            * jnp.nan,
            desc_prime=jnp.zeros(
                env.behavior_descriptor_length,
            )
            * jnp.nan,
        )

        return next_state, policy_params, random_key, transition

    play_step_fn = partial(
        brax_play_step_fn,
        env=env,
    )

    # Prepare the scoring function
    bd_extraction_fn = tasks.behavior_descriptor_extractor[env_name]

    if deterministic:

        # Create the initial environment states
        random_key, subkey = jax.random.split(random_key)
        init_state = env.reset(subkey)

        # Define the function to deterministically reset the environment
        def deterministic_reset(key: RNGKey, init_state: EnvState) -> EnvState:
            return init_state

        play_reset_fn = partial(deterministic_reset, init_state=init_state)

    else:

        # Define the function to stochastically reset the environment
        play_reset_fn = partial(env.reset)

    # Use stochastic scoring function
    scoring_fn = partial(
        reset_based_scoring_function_brax_envs,
        episode_length=episode_length,
        play_reset_fn=play_reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = tasks.reward_offset[env_name]
    qd_offset = reward_offset * episode_length

    # Get number descriptor dimensions
    num_descriptors = env.behavior_descriptor_length

    # Get min and max bd
    min_bd, max_bd = env.behavior_descriptor_limits

    # Get min and max genotypes
    hard_limit_genotype = False
    min_genotype = -2
    max_genotype = 2

    # Add noise values to name
    new_env_name = env_name

    # Return of neuroevolution env
    return (
        new_env_name,
        env,
        play_reset_fn,
        brax_play_step_fn,
        scoring_fn,
        reevaluation_function,
        incell_reevaluation_function,
        policy_network,
        policy_dc_network,
        init_policies_fn,
        hard_limit_genotype,
        min_genotype,
        max_genotype,
        min_bd,
        max_bd,
        qd_offset,
        num_descriptors,
        delta_fitness,
        delta_reproducibility,
        random_key,
    )


def set_up_brax_v1_control(
    env_name: str,
    episode_length: int,
    batch_size: int,
    random_key: RNGKey,
    deterministic: bool = False,
    params_std: float = 0.0,
    fit_std: float = 0.0,
    desc_std: float = 0.0,
    delta_fitness: float = 0.0,
    delta_reproducibility: float = 0.0,
) -> Tuple:

    # Set up necessary depending on type of task
    if env_name == "hexapod_sin_omni" or env_name == "hexapod_control_sin_omni":
        dim_control = 24
        if env_name == "hexapod_sin_omni":
            env_name_brax = "hexapod_omni"
        if env_name == "hexapod_control_sin_omni":
            env_name_brax = "hexapod_control_omni"

        # Define the fonction to infer the next action
        def simple_sine_controller(
            amplitude: jnp.ndarray, phase: jnp.ndarray, t: int
        ) -> jnp.ndarray:
            return amplitude * jnp.sin(
                (2 * t * jnp.pi / 25) + phase * jnp.pi
            )  # in degrees for brax

        def inference(params: Genotype, state: EnvState, timestep: int) -> jnp.ndarray:
            amplitudes_top = params.at[jnp.asarray([0, 1, 2, 3, 4, 5])].get()
            phases_top = params.at[jnp.asarray([6, 7, 8, 9, 10, 11])].get()
            amplitudes_bottom = params.at[jnp.asarray([12, 13, 14, 15, 16, 17])].get()
            phases_bottom = params.at[jnp.asarray([18, 19, 20, 21, 22, 23])].get()
            top_actions = simple_sine_controller(amplitudes_top, phases_top, timestep)
            bottom_actions = simple_sine_controller(
                amplitudes_bottom, phases_bottom, timestep
            )

            actions = jnp.zeros(shape=(18,))
            actions = actions.at[jnp.asarray([0, 3, 6, 9, 12, 15])].set(
                top_actions * (jnp.pi / 8) * (180 / jnp.pi)
            )
            actions = actions.at[jnp.asarray([1, 4, 7, 10, 13, 16])].set(
                bottom_actions * (jnp.pi / 4) * (180 / jnp.pi)
            )
            actions = actions.at[jnp.asarray([2, 5, 8, 11, 14, 17])].set(
                -bottom_actions * (jnp.pi / 4) * (180 / jnp.pi)
            )
            return actions

        inference_fn = jax.jit(inference)

    # Init environment
    env = tasks.create(
        env_name_brax,
        episode_length=episode_length,
        reset_noise_scale=params_std,
    )

    # Init policy structure
    class PolicyStructure(jnp.ndarray):
        @staticmethod
        def apply(params: Genotype, state: EnvState, timestep: int) -> jnp.ndarray:
            return inference_fn(params, state, timestep)

    # Init population of controllers
    def init_policies_fn(size: int, random_key: RNGKey) -> Tuple[jnp.ndarray, RNGKey]:
        random_key, subkey = jax.random.split(random_key)
        init_policies = jax.random.uniform(
            random_key, shape=(size, dim_control), minval=-1, maxval=1
        )
        return init_policies, random_key

    # Define the fonction to play a step with the policy in the environment
    def control_play_step_fn(
        env_state: EnvState,
        policy_params: Genotype,
        random_key: RNGKey,
        timestep: int,
        env: Any,
    ) -> Tuple[EnvState, Genotype, RNGKey, Transition, int]:
        """
        Play an environment step and return the updated state and the transition.
        """

        actions = inference_fn(policy_params, env_state, timestep)
        next_state = env.step(env_state, actions)

        transition = DCRLTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=env_state.info["state_descriptor"],
            next_state_desc=next_state.info["state_descriptor"],
            desc=jnp.zeros(
                env.behavior_descriptor_length,
            )
            * jnp.nan,
            desc_prime=jnp.zeros(
                env.behavior_descriptor_length,
            )
            * jnp.nan,
        )

        timestep += 1
        return next_state, policy_params, random_key, transition, timestep

    play_step_fn = partial(
        control_play_step_fn,
        env=env,
    )

    # Prepare the scoring function
    bd_extraction_fn = tasks.behavior_descriptor_extractor[env_name_brax]

    if deterministic:

        # Create the initial environment states
        random_key, subkey = jax.random.split(random_key)
        init_state = env.reset(subkey)

        # Define the function to deterministically reset the environment
        def deterministic_reset(key: RNGKey, init_state: EnvState) -> EnvState:
            return init_state

        play_reset_fn = partial(deterministic_reset, init_state=init_state)

    else:

        # Define the function to stochastically reset the environment
        play_reset_fn = partial(env.reset)

    # Use stochastic scoring function
    scoring_fn = partial(
        reset_based_scoring_function_time_brax_envs,
        episode_length=episode_length,
        play_reset_fn=play_reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
        fit_std=fit_std,
        desc_std=desc_std,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = tasks.reward_offset[env_name_brax]
    qd_offset = reward_offset * episode_length

    # Get number descriptor dimensions
    num_descriptors = env.behavior_descriptor_length

    # Get min and max bd
    min_bd, max_bd = env.behavior_descriptor_limits

    # Get min and max genotypes
    hard_limit_genotype = False
    min_genotype = -3
    max_genotype = 3

    # Add noise values to name
    new_env_name = env_name
    if fit_std == 0 and desc_std == 0 and params_std == 0:
        new_env_name += "_nonoise"
    else:
        new_env_name += f"_fit{fit_std}" f"_desc{desc_std}" f"_params{params_std}"

    # Return of control env
    return (
        new_env_name,
        env,
        play_reset_fn,
        control_play_step_fn,
        scoring_fn,
        reevaluation_function,
        incell_reevaluation_function,
        PolicyStructure,
        PolicyStructure,
        init_policies_fn,
        hard_limit_genotype,
        min_genotype,
        max_genotype,
        min_bd,
        max_bd,
        qd_offset,
        num_descriptors,
        delta_fitness,
        delta_reproducibility,
        random_key,
    )
