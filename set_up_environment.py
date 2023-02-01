from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from brax.envs import State as EnvState
from qdax import environments
from qdax.core.neuroevolution.buffers.buffer import QDTransition, Transition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.tasks.arm import noisy_arm_scoring_function
from qdax.tasks.brax_envs import reset_based_scoring_function_brax_envs
from qdax.tasks.standard_functions import rastrigin, sphere
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from tasks import behavior_descriptor_extractor, create
from tasks import reward_offset as reward_offset_list
from tasks.hexapod_env import reset_based_scoring_function_time

ENV_NEUROEVOLUTION = [
    "ant_uni",
    "anttrap",
    "hopper_uni",
    "walker2d_uni",
    "halfcheetah",
    "humanoid_uni",
    "ant_omni",
    "humanoid_omni",
    "antmaze",
    "hexapod_omni",
]
ENV_CONTROL = [
    "hexapod_sin_omni",
]
ENV_OPTIMISATION = [
    "rastrigin",
    "sphere",
    "arm",
]

# Environments list
ENV_LIST = ENV_NEUROEVOLUTION + ENV_CONTROL + ENV_OPTIMISATION


def rastrigin_scoring_function(
    params: Genotype,
    random_key: RNGKey,
    fit_variance: float,
    desc_variance: float,
    params_variance: float,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Evaluate policies contained in params in parallel on the Rastrigin task.

    Args:
        params: genotype of the individuals to evaluate.
        random_key
        fit_variance: variance of fitness noise (paper value: 0.05).
        desc_variance: variance of descriptor noise (paper value: 0.01).
        params_variance: variance of parameters noise.

    Returns:
        fitnesses: fitnesses of individuals in params.
        descriptors: descriptors of individuals in params.
        infos: unused additional informations.
        random_key
    """

    random_key, f_subkey, d_subkey, p_subkey = jax.random.split(random_key, num=4)

    # Add noise to the parameters
    params = params + jax.random.normal(p_subkey, shape=params.shape) * params_variance

    # Evaluate
    fitnesses, descriptors = jax.vmap(rastrigin)(params)

    # Add noise
    fitnesses = (
        fitnesses + jax.random.normal(f_subkey, shape=fitnesses.shape) * fit_variance
    )
    descriptors = (
        descriptors
        + jax.random.normal(d_subkey, shape=descriptors.shape) * desc_variance
    )

    return fitnesses, descriptors, {}, random_key


def sphere_scoring_function(
    params: Genotype,
    random_key: RNGKey,
    fit_variance: float,
    desc_variance: float,
    params_variance: float,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Evaluate policies contained in params in parallel on the Sphere task.

    Args:
        params: genotype of the individuals to evaluate.
        random_key
        fit_variance: variance of fitness noise (paper value: 0.05).
        desc_variance: variance of descriptor noise (paper value: 0.01).
        params_variance: variance of parameters noise.

    Returns:
        fitnesses: fitnesses of individuals in params.
        descriptors: descriptors of individuals in params.
        infos: unused additional informations.
        random_key
    """

    random_key, f_subkey, d_subkey, p_subkey = jax.random.split(random_key, num=4)

    # Add noise to the parameters
    params = params + jax.random.normal(p_subkey, shape=params.shape) * params_variance

    # Evaluate
    fitnesses, descriptors = jax.vmap(sphere)(params)

    # Add noise
    fitnesses = (
        fitnesses + jax.random.normal(f_subkey, shape=fitnesses.shape) * fit_variance
    )
    descriptors = (
        descriptors
        + jax.random.normal(d_subkey, shape=descriptors.shape) * desc_variance
    )

    return fitnesses, descriptors, {}, random_key


def set_up_neuroevolution(
    deterministic: bool,
    env_name: str,
    episode_length: int,
    batch_size: int,
    policy_hidden_layer_sizes: Tuple,
    random_key: RNGKey,
) -> Tuple[
    Any,
    Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    Any,
    Genotype,
    float,
    float,
    jnp.ndarray,
    jnp.ndarray,
    float,
    int,
    RNGKey,
]:

    # Init environment
    env = environments.create(env_name, episode_length=episode_length)

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=batch_size)
    fake_batch = jnp.zeros(shape=(batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Define the fonction to play a step with the policy in the environment
    def play_step_fn(
        env_state: EnvState,
        policy_params: Genotype,
        random_key: RNGKey,
    ) -> Tuple[EnvState, Genotype, RNGKey, Transition]:

        """
        Play an environment step and return the updated state and the transition.
        """

        actions = policy_network.apply(policy_params, env_state.obs)
        state_desc = env_state.info["state_descriptor"]
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=state_desc,
            next_state_desc=next_state.info["state_descriptor"],
        )

        return next_state, policy_params, random_key, transition

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name]

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
    reward_offset = environments.reward_offset[env_name]
    qd_offset = reward_offset * episode_length

    # Get number descriptor dimensions
    num_descriptors = env.behavior_descriptor_length

    # Get min and max bd
    min_bd, max_bd = env.behavior_descriptor_limits

    # Get min and max genotypes (used only for random search)
    min_genotype = -5
    max_genotype = 5

    return (
        env,
        scoring_fn,
        policy_network,
        init_variables,
        min_genotype,
        max_genotype,
        min_bd,
        max_bd,
        qd_offset,
        num_descriptors,
        random_key,
    )


def set_up_control(
    deterministic: bool,
    env_name: str,
    episode_length: int,
    batch_size: int,
    random_key: RNGKey,
) -> Tuple[
    Any,
    Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    Any,
    Genotype,
    float,
    float,
    jnp.ndarray,
    jnp.ndarray,
    float,
    int,
    RNGKey,
]:

    # Set up necessary depending on type of task
    if env_name == "hexapod_sin_omni":
        dim_control = 24
        env_name_brax = "hexapod_omni"

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
    env = create(env_name_brax, episode_length=episode_length)

    # Init policy structure
    class PolicyStructure(jnp.ndarray):
        @staticmethod
        def apply(params: Genotype, state: EnvState, timestep: int) -> jnp.ndarray:
            return inference_fn(params, state, timestep)

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    init_policies = jax.random.uniform(
        random_key, shape=(batch_size, dim_control), minval=-1, maxval=1
    )

    # Define the fonction to play a step with the policy in the environment
    def play_step_fn(
        env_state: EnvState,
        policy_params: Genotype,
        random_key: RNGKey,
        timestep: int,
    ) -> Tuple[EnvState, Genotype, RNGKey, Transition, int]:
        """
        Play an environment step and return the updated state and the transition.
        """

        actions = inference_fn(policy_params, env_state, timestep)
        next_state = env.step(env_state, actions)

        transition = QDTransition(
            obs=env_state.obs,
            next_obs=next_state.obs,
            rewards=next_state.reward,
            dones=next_state.done,
            actions=actions,
            truncations=next_state.info["truncation"],
            state_desc=env_state.info["state_descriptor"],
            next_state_desc=next_state.info["state_descriptor"],
        )

        timestep += 1
        return next_state, policy_params, random_key, transition, timestep

    # Prepare the scoring function
    bd_extraction_fn = behavior_descriptor_extractor[env_name_brax]

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
        reset_based_scoring_function_time,
        episode_length=episode_length,
        play_reset_fn=play_reset_fn,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = reward_offset_list[env_name_brax]
    qd_offset = reward_offset * episode_length

    # Get number descriptor dimensions
    num_descriptors = env.behavior_descriptor_length

    # Get min and max bd
    min_bd, max_bd = env.behavior_descriptor_limits

    # Get min and max genotypes (used only for random search)
    min_genotype = -1
    max_genotype = 1

    return (
        env,
        scoring_fn,
        PolicyStructure,
        init_policies,
        min_genotype,
        max_genotype,
        min_bd,
        max_bd,
        qd_offset,
        num_descriptors,
        random_key,
    )


def set_up_optimisation(
    deterministic: bool,
    env_name: str,
    batch_size: int,
    policy_hidden_layer_sizes: Tuple,
    random_key: RNGKey,
) -> Tuple[
    Any,
    Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    Any,
    Genotype,
    float,
    float,
    jnp.ndarray,
    jnp.ndarray,
    float,
    int,
    RNGKey,
]:

    # Set the noise values
    if deterministic:
        optimisation_fit_variance = 0.0
        optimisation_desc_variance = 0.0
        params_variance = 0.0
    else:
        optimisation_fit_variance = 0.01
        optimisation_desc_variance = 0.01
        params_variance = 0

    # Define the scoring function
    if env_name == "sphere":
        scoring_fn = partial(
            sphere_scoring_function,
            fit_variance=optimisation_fit_variance,
            desc_variance=optimisation_desc_variance,
            params_variance=params_variance,
        )
    elif env_name == "rastrigin":
        scoring_fn = partial(
            rastrigin_scoring_function,
            fit_variance=optimisation_fit_variance,
            desc_variance=optimisation_desc_variance,
            params_variance=params_variance,
        )
    elif env_name == "arm":
        scoring_fn = partial(
            noisy_arm_scoring_function,
            fit_variance=optimisation_fit_variance,
            desc_variance=optimisation_desc_variance,
            params_variance=params_variance,
        )

    # Init population of controllers
    if len(policy_hidden_layer_sizes) > 1:
        print(
            "\n!!!WARNING!!! For optimisation functions,",
            "only the first element of policy_hidden_layer_sizes:",
            policy_hidden_layer_sizes[0],
            "is used as genotype dimension.",
        )
    init_policies = jax.random.uniform(
        random_key, shape=(batch_size, policy_hidden_layer_sizes[0]), minval=0, maxval=1
    )

    # Get minimum reward value to make sure qd_score are positive
    if env_name == "sphere":
        qd_offset = 50 * policy_hidden_layer_sizes[0]
    elif env_name == "rastrigin":
        qd_offset = 50 + 50 * policy_hidden_layer_sizes[0]
    elif env_name == "arm":
        qd_offset = 1

    # Get number descriptor dimensions
    num_descriptors = 2

    # Get min and max bd
    min_bd = jnp.array([0.0, 0.0])
    max_bd = jnp.array([1.0, 1.0])

    # Get min and max genotypes (used only for random search)
    min_genotype = 0
    max_genotype = 1

    return (
        None,
        scoring_fn,
        None,
        init_policies,
        min_genotype,
        max_genotype,
        min_bd,
        max_bd,
        qd_offset,
        num_descriptors,
        random_key,
    )


def set_up_environment(
    deterministic: bool,
    env_name: str,
    episode_length: int,
    batch_size: int,
    policy_hidden_layer_sizes: Tuple,
    random_key: RNGKey,
) -> Tuple[
    Any,
    Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    Any,
    Genotype,
    float,
    float,
    jnp.ndarray,
    jnp.ndarray,
    float,
    int,
    RNGKey,
]:

    if env_name in ENV_NEUROEVOLUTION:
        return set_up_neuroevolution(
            deterministic=deterministic,
            env_name=env_name,
            episode_length=episode_length,
            batch_size=batch_size,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            random_key=random_key,
        )
    elif env_name in ENV_CONTROL:
        return set_up_control(
            deterministic=deterministic,
            env_name=env_name,
            episode_length=episode_length,
            batch_size=batch_size,
            random_key=random_key,
        )
    elif env_name in ENV_OPTIMISATION:
        return set_up_optimisation(
            deterministic=deterministic,
            env_name=env_name,
            batch_size=batch_size,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            random_key=random_key,
        )
    else:
        assert 0, "!!!ERROR!!! Env in none of the categories."
