from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from brax.envs import State as EnvState
from qdax import environments
from qdax.core.neuroevolution.buffers.buffer import Transition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.types import Genotype, RNGKey

from tasks.arm import (
    ArmBimodalGaussianDesc,
    ArmBimodalGaussianFitness,
    ArmGaussianDescBiVarianceNoise,
    ArmGaussianDescFitPropVarianceNoise,
    ArmGaussianNoise,
    ArmSelectedJointsNoise,
)
from tasks.brax_envs import reset_based_scoring_function_brax_envs
from tasks.direct_mapping import (
    deceptive_direct_mapping_scoring_function,
    no_trade_off_direct_mapping_scoring_function,
    perfect_trade_off_direct_mapping_scoring_function,
    sharp_peak_direct_mapping_scoring_function,
)
from tasks.hexapod_env import reset_based_scoring_function_time_brax_envs
from tasks.optimisation_problems import (
    rastrigin_scoring_function,
    sphere_scoring_function,
)

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
    "hexapod_control_omni",
    "hexapod_trap",
]
ENV_CONTROL = [
    "hexapod_sin_omni",
    "hexapod_control_sin_omni",
]
ENV_OPTIMISATION = [
    "rastrigin",
    "sphere",
    "arm_gaussian_fit",
    "arm_gaussian_desc",
    "arm_gaussian_desc_bi_variance",
    "arm_gaussian_desc_fitprop_variance",
    "arm_multi_modal_fit",
    "arm_multi_modal_desc",
    "arm_selected_gaussian_params",
    "direct_mapping_no_trade_off",
    "direct_mapping_perfect_trade_off",
    "direct_mapping_sharp_peak",
    "direct_mapping_deceptive",
    "direct_mapping_perfect_trade_off_0.02",
    "direct_mapping_sharp_peak_bigger_0.2",
    "direct_mapping_sharp_peak_smaller_0.02",
    "direct_mapping_deceptive_0.1",
]

# Environments list
ENV_LIST = ENV_NEUROEVOLUTION + ENV_CONTROL + ENV_OPTIMISATION


def set_up_neuroevolution(
    deterministic: bool,
    env_name: str,
    episode_length: int,
    params_std: float,
    batch_size: int,
    policy_hidden_layer_sizes: Tuple,
    random_key: RNGKey,
    gaussian_vel: bool,
    gaussian_pos: bool,
    delta_fitness: float,
    delta_reproducibility: float,
) -> Tuple:

    # Init environment
    env = environments.create(
        env_name,
        episode_length=episode_length,
        gaussian_vel=gaussian_vel,
        gaussian_pos=gaussian_pos,
        reset_noise_scale=params_std,
    )

    # Init policy network
    policy_layer_sizes = policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    def init_policies_fn(size: int, random_key: RNGKey) -> Tuple[jnp.ndarray, RNGKey]:
        random_key, subkey = jax.random.split(random_key)
        keys = jax.random.split(subkey, num=size)
        fake_batch = jnp.zeros(shape=(size, env.observation_size))
        init_policies = jax.vmap(policy_network.init)(keys, fake_batch)
        return init_policies, random_key

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

        transition = Transition(
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

    # Get min and max genotypes
    hard_limit_genotype = False
    min_genotype = -2
    max_genotype = 2

    # Add noise values to name
    new_env_name = env_name
    if gaussian_vel:
        new_env_name += "_velnormal"
    if gaussian_pos:
        new_env_name += "_posnormal"

    # Return of neuroevolution env
    return (
        new_env_name,
        env,
        scoring_fn,
        policy_network,
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


def set_up_control(
    deterministic: bool,
    env_name: str,
    episode_length: int,
    params_std: float,
    fit_std: float,
    desc_std: float,
    batch_size: int,
    random_key: RNGKey,
    gaussian_vel: bool,
    gaussian_pos: bool,
    delta_fitness: float,
    delta_reproducibility: float,
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
    env = environments.create(
        env_name_brax,
        episode_length=episode_length,
        gaussian_vel=gaussian_vel,
        gaussian_pos=gaussian_pos,
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

        transition = Transition(
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
        )

        timestep += 1
        return next_state, policy_params, random_key, transition, timestep

    # Prepare the scoring function
    bd_extraction_fn = environments.behavior_descriptor_extractor[env_name_brax]

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
    reward_offset = environments.reward_offset[env_name_brax]
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
        scoring_fn,
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


def set_up_optimisation(
    deterministic: bool,
    env_name: str,
    fit_std: float,
    desc_std: float,
    params_std: float,
    batch_size: int,
    policy_hidden_layer_sizes: Tuple,
    delta_fitness: float,
    delta_reproducibility: float,
    random_key: RNGKey,
) -> Tuple:

    if len(policy_hidden_layer_sizes) > 1:
        print(
            "\n!!!WARNING!!! For optimisation functions,",
            "only the first element of policy_hidden_layer_sizes:",
            policy_hidden_layer_sizes[0],
            "is used as genotype dimension.",
        )

    # If deterministic, set all noise params to 0
    if deterministic:
        fit_std = 0
        desc_std = 0
        params_std = 0

    # Sphere
    if env_name == "sphere":

        scoring_fn = partial(
            sphere_scoring_function,
            fit_std=fit_std,
            desc_std=desc_std,
            params_std=params_std,
        )
        qd_offset = 50 * policy_hidden_layer_sizes[0]

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Rastrigin
    elif env_name == "rastrigin":

        scoring_fn = partial(
            rastrigin_scoring_function,
            fit_std=fit_std,
            desc_std=desc_std,
            params_std=params_std,
        )
        qd_offset = 50 + 50 * policy_hidden_layer_sizes[0]

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Direct mapping with no trade-off
    elif env_name == "direct_mapping_no_trade_off":

        if policy_hidden_layer_sizes[0] != 3:
            print("!!!WARNING!!! Genotype is always dimension 3 for direct_mapping.")
            policy_hidden_layer_sizes = (3,)

        scoring_fn = partial(
            no_trade_off_direct_mapping_scoring_function,
            desc_std=desc_std,
        )
        qd_offset = 0

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jnp.abs(
                jax.random.normal(
                    random_key, shape=(size, policy_hidden_layer_sizes[0])
                )
                * 0.1
            )
            return init_policies, random_key

    # Direct mapping with perfect trade-off
    elif env_name == "direct_mapping_perfect_trade_off":

        if policy_hidden_layer_sizes[0] != 3:
            print("!!!WARNING!!! Genotype is always dimension 3 for direct_mapping.")
            policy_hidden_layer_sizes = (3,)

        scoring_fn = partial(
            perfect_trade_off_direct_mapping_scoring_function,
            desc_std=desc_std,
        )
        qd_offset = 0

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jnp.abs(
                jax.random.normal(
                    random_key, shape=(size, policy_hidden_layer_sizes[0])
                )
                * 0.1
            )
            return init_policies, random_key

    # Direct mapping with sharp peak
    elif env_name == "direct_mapping_sharp_peak":

        if policy_hidden_layer_sizes[0] != 3:
            print("!!!WARNING!!! Genotype is always dimension 3 for direct_mapping.")
            policy_hidden_layer_sizes = (3,)

        scoring_fn = partial(
            sharp_peak_direct_mapping_scoring_function,
            desc_std=desc_std,
        )
        qd_offset = 0

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jnp.abs(
                jax.random.normal(
                    random_key, shape=(size, policy_hidden_layer_sizes[0])
                )
                * 0.1
            )
            return init_policies, random_key

    # Direct mapping with deceptive
    elif env_name == "direct_mapping_deceptive":

        if policy_hidden_layer_sizes[0] != 3:
            print("!!!WARNING!!! Genotype is always dimension 3 for direct_mapping.")
            policy_hidden_layer_sizes = (3,)

        scoring_fn = partial(
            deceptive_direct_mapping_scoring_function,
            desc_std=desc_std,
        )
        qd_offset = 0

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jnp.abs(
                jax.random.normal(
                    random_key, shape=(size, policy_hidden_layer_sizes[0])
                )
                * 0.1
            )
            return init_policies, random_key

    # Arm with Gaussian fit noise
    elif env_name == "arm_gaussian_fit":

        if fit_std <= 0:
            print("\n!!!WARNING!!! Invalid std, using default values.")
            fit_std = 0.1

        print(f"Using Gaussian noise on fitness with var {fit_std}")

        env = ArmGaussianNoise(
            fit_std=fit_std,
            desc_std=0.0,
            params_std=0.0,
        )
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 1 + 2 * fit_std

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Arm with Gaussian desc noise
    elif env_name == "arm_gaussian_desc":

        if desc_std <= 0:
            print("\n!!!WARNING!!! Invalid std, using default values.")
            desc_std = 0.01

        print(f"Using Gaussian noise on des with var {desc_std}")

        env = ArmGaussianNoise(
            fit_std=0.0,
            desc_std=desc_std,
            params_std=0.0,
        )
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 1

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Arm with Multimodal fit noise
    elif env_name == "arm_multi_modal_fit":

        if fit_std <= 0 or desc_std <= 0:
            print("\n!!!WARNING!!! Invalid std, using default values.")
            fit_std = 0.01
            desc_std = 0.01

        # Values from UQD Benchmark paper to simplify
        proba_mode_1_fit = 0.85
        mean_fitness_2 = -1
        print("Using multi-modal Gaussian noise on fitness with")
        print(f"proba_mode_1 {proba_mode_1_fit}")
        print(f"var {fit_std}")
        print(f"mean_fitness_2 {mean_fitness_2}")

        env = ArmBimodalGaussianFitness(
            proba_mode_1=proba_mode_1_fit,
            fit_std_1=fit_std,
            fit_std_2=desc_std,
            mean_fitness_2=mean_fitness_2,
        )  # type: ignore
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 1 - mean_fitness_2 + 2 * fit_std

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Arm with Multimodal desc noise
    elif env_name == "arm_multi_modal_desc":

        if fit_std <= 0 or desc_std <= 0:
            print("\n!!!WARNING!!! Invalid std, using default values.")
            fit_std = 0.01
            desc_std = 0.01

        # Values from UQD Benchmark paper to simplify
        proba_mode_1_desc = 0.85
        mean_desc_2 = [1.0, 1.0]
        desc_std_1 = [desc_std, desc_std]
        desc_std_2 = [fit_std, fit_std]
        print("Using multi-modal Gaussian noise on descriptor")
        print(f"proba_mode_1 {proba_mode_1_desc}")
        print(f"var {desc_std_1}")
        print(f"mean_desc_2 {mean_desc_2}")
        print(f"var {desc_std_2}")

        env = ArmBimodalGaussianDesc(
            proba_mode_1=proba_mode_1_desc,
            desc_std_1=desc_std_1,
            desc_std_2=desc_std_2,
            mean_desc_2=mean_desc_2,
        )  # type: ignore
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 1

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Arm with Gaussian param noise
    elif env_name == "arm_selected_gaussian_params":

        if params_std <= 0:
            print("\n!!!WARNING!!! Invalid std, using default values.")
            params_std = 0.1

        # Values from UQD Benchmark paper to simplify
        selected_indexes_noise = [6]
        print("Using Gaussian noise on selected params")
        print(f"noise with var {params_std} on indexes {selected_indexes_noise}")

        env = ArmSelectedJointsNoise(
            selected_indexes=jnp.asarray(selected_indexes_noise),
            params_std=params_std,
        )  # type: ignore
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 5

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Arm with bi-std Gaussian desc noise
    elif env_name == "arm_gaussian_desc_bi_variance":

        if fit_std <= 0 or desc_std <= 0:
            print("\n!!!WARNING!!! Invalid std, using default values.")
            fit_std = 0.01
            desc_std = 0.1

        desc_std_1 = [fit_std, fit_std]
        desc_std_2 = [desc_std, desc_std]
        print("Using Gaussian noise on descriptor with 2 choices of std")
        print(f"{desc_std_1} and {desc_std_2}")
        print("Using fitness of 0 for this task")

        env = ArmGaussianDescBiVarianceNoise(
            desc_std_1=desc_std_1,
            desc_std_2=desc_std_2,
        )  # type: ignore
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 0

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Arm with prop-std Gaussian desc noise
    elif env_name == "arm_gaussian_desc_fitprop_variance":

        # Values from UQD Benchmark paper to simplify
        prop_factors = [0.1, 0.1]
        print("Using Gaussian noise on descriptor with fitness-proportional std")
        print(f"prop_factors {prop_factors}")
        print("Using fitness of 0 for this task")

        env = ArmGaussianDescFitPropVarianceNoise(prop_factors=prop_factors)  # type: ignore
        scoring_fn = env.scoring_fn  # type: ignore
        qd_offset = 0

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jax.random.uniform(
                random_key,
                shape=(size, policy_hidden_layer_sizes[0]),
                minval=0,
                maxval=1,
            )
            return init_policies, random_key

    # Direct mapping with perfect trade-off and parameter 0.02
    elif env_name == "direct_mapping_perfect_trade_off_0.02":

        if policy_hidden_layer_sizes[0] != 3:
            print("!!!WARNING!!! Genotype is always dimension 3 for direct_mapping.")
            policy_hidden_layer_sizes = (3,)

        print(
            "!!!WARNING!!! For direct_mapping_perfect_trade_off_0.02, desc_std is set to 0.2 and delta_fitness and delta_reproducibility to 0.02."
        )
        delta_fitness = 0.02
        delta_reproducibility = 0.02

        scoring_fn = partial(
            perfect_trade_off_direct_mapping_scoring_function,
            desc_std=0.2,
        )
        qd_offset = 0

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jnp.abs(
                jax.random.normal(
                    random_key, shape=(batch_size, policy_hidden_layer_sizes[0])
                )
                * 0.1
            )
            return init_policies, random_key

    # Direct mapping with sharp peak and bigger of 0.2
    elif env_name == "direct_mapping_sharp_peak_bigger_0.2":

        if policy_hidden_layer_sizes[0] != 3:
            print("!!!WARNING!!! Genotype is always dimension 3 for direct_mapping.")
            policy_hidden_layer_sizes = (3,)

        print(
            "!!!WARNING!!! For direct_mapping_sharp_peak_bigger_0.2, desc_std is set to 0.05, delta_fitness to 0.2 and delta_reproducibility to 0.02."
        )
        delta_fitness = 0.2
        delta_reproducibility = 0.02

        scoring_fn = partial(
            sharp_peak_direct_mapping_scoring_function,
            desc_std=0.05,
        )
        qd_offset = 0

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jnp.abs(
                jax.random.normal(
                    random_key, shape=(batch_size, policy_hidden_layer_sizes[0])
                )
                * 0.1
            )
            return init_policies, random_key

    # Direct mapping with sharp peak and smaller of 0.02
    elif env_name == "direct_mapping_sharp_peak_smaller_0.02":

        if policy_hidden_layer_sizes[0] != 3:
            print("!!!WARNING!!! Genotype is always dimension 3 for direct_mapping.")
            policy_hidden_layer_sizes = (3,)

        print(
            "!!!WARNING!!! For direct_mapping_sharp_peak_smaller_0.02, desc_std is set to 0.01 and delta_fitness and delta_reproducibility to 0.01."
        )
        delta_fitness = 0.02
        delta_reproducibility = 0.02

        scoring_fn = partial(
            sharp_peak_direct_mapping_scoring_function,
            desc_std=0.05,
        )
        qd_offset = 0

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jnp.abs(
                jax.random.normal(
                    random_key, shape=(batch_size, policy_hidden_layer_sizes[0])
                )
                * 0.1
            )
            return init_policies, random_key

    # Direct mapping with deceptive of 0.1
    elif env_name == "direct_mapping_deceptive_0.1":

        if policy_hidden_layer_sizes[0] != 3:
            print("!!!WARNING!!! Genotype is always dimension 3 for direct_mapping.")
            policy_hidden_layer_sizes = (3,)

        print(
            "!!!WARNING!!! For direct_mapping_deceptive_0.1, desc_std is set to 0.1 and delta_fitness and delta_reproducibility to 0.05."
        )
        delta_fitness = 0.05
        delta_reproducibility = 0.05

        scoring_fn = partial(
            deceptive_direct_mapping_scoring_function,
            desc_std=0.1,
        )
        qd_offset = 0

        def init_policies_fn(
            size: int, random_key: RNGKey
        ) -> Tuple[jnp.ndarray, RNGKey]:
            random_key, subkey = jax.random.split(random_key)
            init_policies = jnp.abs(
                jax.random.normal(
                    random_key, shape=(batch_size, policy_hidden_layer_sizes[0])
                )
                * 0.1
            )
            return init_policies, random_key

    # Get number descriptor dimensions
    num_descriptors = 2

    # Get min and max bd
    min_bd = jnp.array([0.0, 0.0])
    max_bd = jnp.array([1.0, 1.0])

    # Get min and max genotypes
    hard_limit_genotype = True
    min_genotype = 0
    max_genotype = 1

    # Add noise values to name
    new_env_name = env_name
    if fit_std == 0 and desc_std == 0 and params_std == 0:
        new_env_name += "_nonoise"
    else:
        new_env_name += f"_fit{fit_std}" f"_desc{desc_std}" f"_params{params_std}"

    # Return of optimisation env
    return (
        new_env_name,
        None,
        scoring_fn,
        None,
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


def set_up_environment(
    deterministic: bool,
    env_name: str,
    episode_length: int,
    fit_std: float,
    desc_std: float,
    params_std: float,
    batch_size: int,
    policy_hidden_layer_sizes: Tuple,
    random_key: RNGKey,
    gaussian_vel: bool,
    gaussian_pos: bool,
    delta_fitness: float,
    delta_reproducibility: float,
) -> Tuple:
    assert env_name in ENV_LIST, "\n!!!ERROR!!! Invalid env name:" + env_name

    if env_name in ENV_NEUROEVOLUTION:
        return set_up_neuroevolution(
            deterministic=deterministic,
            env_name=env_name,
            episode_length=episode_length,
            params_std=params_std,
            batch_size=batch_size,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            random_key=random_key,
            gaussian_vel=gaussian_vel,
            gaussian_pos=gaussian_pos,
            delta_fitness=delta_fitness,
            delta_reproducibility=delta_reproducibility,
        )
    elif env_name in ENV_CONTROL:
        return set_up_control(
            deterministic=deterministic,
            env_name=env_name,
            episode_length=episode_length,
            fit_std=fit_std,
            desc_std=desc_std,
            params_std=params_std,
            batch_size=batch_size,
            random_key=random_key,
            gaussian_vel=gaussian_vel,
            gaussian_pos=gaussian_pos,
            delta_fitness=delta_fitness,
            delta_reproducibility=delta_reproducibility,
        )
    elif env_name in ENV_OPTIMISATION:
        return set_up_optimisation(
            deterministic=deterministic,
            env_name=env_name,
            fit_std=fit_std,
            desc_std=desc_std,
            params_std=params_std,
            batch_size=batch_size,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            delta_fitness=delta_fitness,
            delta_reproducibility=delta_reproducibility,
            random_key=random_key,
        )
    else:
        assert 0, "\n!!!ERROR!!! Env in none of the categories."
