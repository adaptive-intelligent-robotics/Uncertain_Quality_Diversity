from typing import Tuple

from qdax.custom_types import RNGKey

ENV_NEUROEVOLUTION = [
    "ant_uni",
    "anttrap",
    "hopper_uni",
    "walker2d_uni",
    "halfcheetah_uni",
    "humanoid_uni",
    "ant_omni",
    "humanoid_omni",
    "antmaze",
    "hexapod_omni",
    "hexapod_control_omni",
    "hexapod_trap",
    "swimmer_uni",
]
ENV_CONTROL = [
    "hexapod_sin_omni",
    "hexapod_control_sin_omni",
]
ENV_OPTIMISATION = [
    "rastrigin",
    "sphere",
    "arm_gaussian",
    "arm_gaussian_fit",
    "arm_gaussian_desc",
    "arm_gaussian_desc_bi_variance",
    "arm_gaussian_desc_fitprop_variance",
    "arm_multi_modal_fit",
    "arm_multi_modal_desc",
    "arm_selected_gaussian_params",
    "direct_mapping_perfect_trade_off_0.02",
    "direct_mapping_sharp_peak_bigger_0.2",
    "direct_mapping_sharp_peak_smaller_0.02",
    "direct_mapping_deceptive_0.1",
    "direct_mapping_no_trade_off",
    "direct_mapping_perfect_trade_off",
    "direct_mapping_sharp_peak",
    "direct_mapping_deceptive",
]

# Environments list
ENV_LIST = ENV_NEUROEVOLUTION + ENV_CONTROL + ENV_OPTIMISATION
ENV_RL_LIST = ENV_NEUROEVOLUTION
ENV_TIMESTEP_LIST = ENV_NEUROEVOLUTION + ENV_CONTROL


def set_up_environment(
    env_name: str,
    episode_length: int,
    batch_size: int,
    policy_hidden_layer_sizes: Tuple,
    random_key: RNGKey,
    deterministic: bool = False,
    domain_randomisation: bool = False,
    fit_std: float = 0.0,
    desc_std: float = 0.0,
    params_std: float = 0.0,
    delta_fitness: float = 0.0,
    delta_reproducibility: float = 0.0,
) -> Tuple:
    assert env_name in ENV_LIST, "\n!!!ERROR!!! Invalid env name:" + env_name

    if env_name in ENV_NEUROEVOLUTION:

        from environments_manager.set_up_brax_v1 import set_up_brax_v1

        return set_up_brax_v1(  # type: ignore
            env_name=env_name,
            episode_length=episode_length,
            batch_size=batch_size,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            random_key=random_key,
            deterministic=deterministic,
            params_std=params_std,
            delta_fitness=delta_fitness,
            delta_reproducibility=delta_reproducibility,
        )

    elif env_name in ENV_CONTROL:

        from environments_manager.set_up_brax_v1 import set_up_brax_v1_control

        return set_up_brax_v1_control(  # type: ignore
            env_name=env_name,
            episode_length=episode_length,
            batch_size=batch_size,
            random_key=random_key,
            deterministic=deterministic,
            fit_std=fit_std,
            desc_std=desc_std,
            params_std=params_std,
            delta_fitness=delta_fitness,
            delta_reproducibility=delta_reproducibility,
        )

    elif env_name in ENV_OPTIMISATION:

        from environments_manager.set_up_optimisation import set_up_optimisation

        return set_up_optimisation(  # type: ignore
            env_name=env_name,
            batch_size=batch_size,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            random_key=random_key,
            deterministic=deterministic,
            fit_std=fit_std,
            desc_std=desc_std,
            params_std=params_std,
            delta_fitness=delta_fitness,
            delta_reproducibility=delta_reproducibility,
        )

    else:
        assert 0, "\n!!!ERROR!!! Env in none of the categories."
