import functools
from typing import Any, Callable, List, Optional, Union

import jax.numpy as jnp
from brax.v1.envs import Env, _envs
from brax.v1.envs import env as brax_env
from brax.v1.envs.wrappers import (
    AutoResetWrapper,
    EpisodeWrapper,
    EvalWrapper,
    VectorWrapper,
)
from qdax.environments.base_wrappers import QDEnv, StateDescriptorResetWrapper
from qdax.environments.bd_extractors import (
    get_feet_contact_proportion,
    get_final_xy_position,
)
from qdax.environments.exploration_wrappers import MazeWrapper, TrapWrapper
from qdax.environments.humanoidtrap import HumanoidTrap
from qdax.environments.init_state_wrapper import FixedInitialStateWrapper
from qdax.environments.pointmaze import PointMaze
from qdax.environments.wrappers import CompletedEvalWrapper

from tasks.hexapod import HexapodAngleDiff, HexapodControl
from tasks.locomotion_wrappers import (
    FeetContactWrapper,
    NoForwardRewardWrapper,
    XYPositionWrapper,
)

# experimentally determinated offset (except for antmaze)
# should be sufficient to have only positive rewards but no guarantee
reward_offset = {
    "pointmaze": 2.3431,
    "anttrap": 3.38,
    "humanoidtrap": 0.0,
    "antnotrap": 3.38,
    "antmaze": 40.32,
    "ant_omni": 3.0,
    "humanoid_omni": 0.0,
    "ant_uni": 3.24,
    "humanoid_uni": 0.0,
    "halfcheetah_uni": 9.231,
    "hopper_uni": 0.9,
    "walker2d_uni": 1.413,
    "hexapod_omni": 3.6,
    "hexapod_control_omni": 0.0,
    "hexapod_trap": 3.38,
}

behavior_descriptor_extractor = {
    "pointmaze": get_final_xy_position,
    "anttrap": get_final_xy_position,
    "humanoidtrap": get_final_xy_position,
    "antnotrap": get_final_xy_position,
    "antmaze": get_final_xy_position,
    "ant_omni": get_final_xy_position,
    "humanoid_omni": get_final_xy_position,
    "ant_uni": get_feet_contact_proportion,
    "humanoid_uni": get_feet_contact_proportion,
    "halfcheetah_uni": get_feet_contact_proportion,
    "hopper_uni": get_feet_contact_proportion,
    "walker2d_uni": get_feet_contact_proportion,
    "hexapod_omni": get_final_xy_position,
    "hexapod_control_omni": get_final_xy_position,
    "hexapod_trap": get_final_xy_position,
}

_qdax_envs = {
    "pointmaze": PointMaze,
    "humanoid_w_trap": HumanoidTrap,
}

_qdax_custom_envs = {
    "anttrap": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, TrapWrapper],
        "kwargs": [{"minval": [0.0, -8.0], "maxval": [30.0, 8.0]}, {}],
    },
    "humanoidtrap": {
        "env": "humanoid_w_trap",
        "wrappers": [XYPositionWrapper],
        "kwargs": [{"minval": [0.0, -8.0], "maxval": [30.0, 8.0]}],
    },
    "antnotrap": {
        "env": "ant",
        "wrappers": [XYPositionWrapper],
        "kwargs": [{"minval": [0.0, -8.0], "maxval": [70.0, 8.0]}],
    },
    "antmaze": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, MazeWrapper],
        "kwargs": [{"minval": [-5.0, -5.0], "maxval": [40.0, 40.0]}, {}],
    },
    "ant_omni": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, NoForwardRewardWrapper],
        "kwargs": [{"minval": [-30.0, -30.0], "maxval": [30.0, 30.0]}, {}],
    },
    "humanoid_omni": {
        "env": "humanoid",
        "wrappers": [XYPositionWrapper, NoForwardRewardWrapper],
        "kwargs": [{"minval": [-30.0, -30.0], "maxval": [30.0, 30.0]}, {}],
    },
    "ant_uni": {"env": "ant", "wrappers": [FeetContactWrapper], "kwargs": [{}, {}]},
    "humanoid_uni": {
        "env": "humanoid",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
    "halfcheetah_uni": {
        "env": "halfcheetah",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
    "hopper_uni": {
        "env": "hopper",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
    "walker2d_uni": {
        "env": "walker2d",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}, {}],
    },
    "hexapod_omni": {
        "env": "hexapod",
        "wrappers": [XYPositionWrapper],
        "kwargs": [{"minval": [-2.0, -2.0], "maxval": [2.0, 2.0]}],
    },
    "hexapod_control_omni": {
        "env": "hexapod_control",
        "wrappers": [XYPositionWrapper],
        "kwargs": [{"minval": [-2.0, -2.0], "maxval": [2.0, 2.0]}],
    },
    "hexapod_trap": {
        "env": "hexapod_control",
        "wrappers": [XYPositionWrapper, TrapWrapper],
        "kwargs": [{"minval": [0.0, -8.0], "maxval": [30.0, 8.0]}, {}],
    },
}


class NoResetWrapper(brax_env.Wrapper):
    """Useless wrapper that just avoid breaking all interface when not using Autoreset."""

    def reset(self, rng: jnp.ndarray) -> brax_env.State:
        state = self.env.reset(rng)
        state.info["first_qp"] = state.qp
        state.info["first_obs"] = state.obs
        return state

    def step(self, state: brax_env.State, action: jnp.ndarray) -> brax_env.State:
        state = self.env.step(state, action)
        return state


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    eval_metrics: bool = False,
    fixed_init_state: bool = False,
    qdax_wrappers_kwargs: Optional[List] = None,
    reset_noise_scale: float = 0,
    **kwargs: Any,
) -> Union[Env, QDEnv]:
    """Creates an Env with a specified brax system.
    Please use namespace to avoid confusion between this function and
    brax.envs.create.
    """

    if env_name in _envs.keys():
        env = _envs[env_name](legacy_spring=True, **kwargs)
    elif env_name in _qdax_envs.keys():
        env = _qdax_envs[env_name](**kwargs)
    elif env_name in _qdax_custom_envs.keys():
        base_env_name = _qdax_custom_envs[env_name]["env"]
        if base_env_name == "hexapod":
            env = HexapodAngleDiff(
                legacy_spring=True,
                **kwargs,
                reset_noise_scale=reset_noise_scale,
            )
        elif base_env_name == "hexapod_control":
            env = HexapodControl(
                legacy_spring=True,
                **kwargs,
                reset_noise_scale=reset_noise_scale,
            )
        elif base_env_name in _envs.keys():
            env = _envs[base_env_name](legacy_spring=True, **kwargs)
        elif base_env_name in _qdax_envs.keys():
            env = _qdax_envs[base_env_name](**kwargs)  # type: ignore
    else:
        raise NotImplementedError("This environment name does not exist!")

    if env_name in _qdax_custom_envs.keys():
        # roll with qdax wrappers
        wrappers = _qdax_custom_envs[env_name]["wrappers"]
        if qdax_wrappers_kwargs is None:
            kwargs_list = _qdax_custom_envs[env_name]["kwargs"]
        else:
            kwargs_list = qdax_wrappers_kwargs
        for wrapper, kwargs in zip(wrappers, kwargs_list):  # type: ignore
            print("Applying wrapper", wrapper)
            env = wrapper(env, base_env_name, **kwargs)  # type: ignore

    if episode_length is not None:
        print("Applying wrapper EpisodeWrapper")
        env = EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        print("Applying wrapper VectorWrapper")
        env = VectorWrapper(env, batch_size)
    if fixed_init_state:
        # retrieve the base env
        if env_name not in _qdax_custom_envs.keys():
            base_env_name = env_name
        # wrap the env
        print("Applying wrapper FixedInitialStateWrapper")
        env = FixedInitialStateWrapper(env, base_env_name=base_env_name)  # type: ignore
    if auto_reset:
        print("Applying wrapper AutoresetWrapper")
        env = AutoResetWrapper(env)
        if env_name in _qdax_custom_envs.keys():
            print("Applying wrapper StateDescriptorResetWrapper")
            env = StateDescriptorResetWrapper(env)
    else:
        print("Applying wrapper NoResetWrapper")
        env = NoResetWrapper(env)
    if eval_metrics:
        print("Applying wrapper EvalWrapper")
        env = EvalWrapper(env)
        print("Applying wrapper CompletedEvalWrapper")
        env = CompletedEvalWrapper(env)

    return env


def create_fn(env_name: str, **kwargs: Any) -> Callable[..., Env]:
    """Returns a function that when called, creates an Env.
    Please use namespace to avoid confusion between this function and
    brax.envs.create_fn.
    """
    return functools.partial(create, env_name, **kwargs)
