from typing import Any

import brax
import jax
import jax.numpy as jnp
from brax import jumpy as jp
from brax.envs import env
from brax.math import quat_to_euler


class Hexapod(env.Env):
    """Hexapod environement with no reward"""

    def __init__(
        self,
        legacy_spring: bool = False,
        reset_noise_scale: float = 0.1,
        gaussian_vel: bool = False,
        gaussian_pos: bool = False,
        **kwargs: Any,
    ) -> None:
        config = _SYSTEM_CONFIG_SPRING if legacy_spring else _SYSTEM_CONFIG
        super().__init__(config=config, **kwargs)
        self.done = False
        self._reset_noise_scale = reset_noise_scale
        self._gaussian_vel = gaussian_vel
        self._gaussian_pos = gaussian_pos

    def reset(self, rng: jp.ndarray) -> env.State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2 = jp.random_split(rng, 3)
        if self._gaussian_pos:
            qpos = self.sys.default_angle() + self._gaussian_noise(rng1)
        else:
            qpos = self.sys.default_angle() + self._noise(rng1)
        if self._gaussian_pos:
            qvel = self._gaussian_noise(rng2)
        else:
            qvel = self._noise(rng2)
        qp = self.sys.default_qp(joint_angle=qpos, joint_velocity=qvel)
        info = self.sys.info(qp)
        obs = self._get_obs(qp, info)
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_ctrl_cost": zero,
            "reward_contact_cost": zero,
            "reward_forward": zero,
            "reward_survive": zero,
        }

        state = env.State(qp, obs, reward, done, metrics)
        state.info["bd"] = jp.zeros((2))
        self.done = False
        return state

    def _angle_dist(self, a: jp.ndarray, b: jp.ndarray) -> jp.ndarray:
        theta = b - a
        theta = jnp.where(theta < -jnp.pi, theta + 2 * jnp.pi, theta)
        theta = jnp.where(theta > jnp.pi, theta - 2 * jnp.pi, theta)

        theta = jnp.where(theta < -jnp.pi, theta + 2 * jnp.pi, theta)
        theta = jnp.where(theta > jnp.pi, theta - 2 * jnp.pi, theta)
        return theta

    def step(self, state: env.State, action: jp.ndarray) -> env.State:
        """Run one timestep of the environment's dynamics."""

        qp, info = self.sys.step(state.qp, action)
        obs = self._get_obs(qp, info)
        state.info["bd"] = qp.pos[0, 0:2]

        # axis, angles = quat_to_axis_angle(obs.at[3:7].get())
        # z_rot = axis.at[2].get()*angles
        com_rot = quat_to_euler(qp.rot[0])
        z_rot = com_rot[2]

        ang_x, ang_y = qp.pos[0, 0], qp.pos[0, 1]
        b_matrix = jnp.sqrt(
            (ang_x / 2.0) * (ang_x / 2.0) + (ang_y / 2.0) * (ang_y / 2.0)
        )
        alpha = jnp.arctan2(ang_y, ang_x)
        a_matrix = b_matrix / jnp.cos(alpha)
        beta = jnp.arctan2(ang_y, ang_x - a_matrix)

        beta = jnp.where(ang_x < 0, beta - jnp.pi, beta)
        beta = jnp.where(beta < -jnp.pi, beta + 2 * jnp.pi, beta)
        beta = jnp.where(beta > jnp.pi, beta - 2 * jnp.pi, beta)

        beta = jnp.where(beta < -jnp.pi, beta + 2 * jnp.pi, beta)
        beta = jnp.where(beta > jnp.pi, beta - 2 * jnp.pi, beta)

        beta = jnp.where(beta < -jnp.pi, beta + 2 * jnp.pi, beta)
        beta = jnp.where(beta > jnp.pi, beta - 2 * jnp.pi, beta)

        angle_diff = jnp.abs(self._angle_dist(beta, z_rot))

        x_before = state.qp.pos[0, 0]
        x_after = qp.pos[0, 0]
        forward_reward = (x_after - x_before) / self.sys.config.dt
        ctrl_cost = 0.5 * jp.sum(jp.square(action))
        contact_cost = 0.5 * 1e-3 * jp.sum(jp.square(jp.clip(info.contact.vel, -1, 1)))
        survive_reward = jp.float32(1)

        reward = (
            -angle_diff
        )  # forward_reward - ctrl_cost - contact_cost + survive_reward
        # reward = jp.float32(0)
        # jnp.where condition, Where True, yield x, otherwise yield y.
        done = jp.where(qp.pos[0, 2] < 0.05, x=jp.float32(1), y=jp.float32(0))
        done = jp.where(qp.pos[0, 2] > 0.19, x=jp.float32(1), y=done)

        # self.done = jp.where((done == 1), True, self.done)
        # reward = jp.where(self.done, jp.float32(-10),- angle_diff)
        # reward = jp.float32(0)

        state.metrics.update(
            reward_ctrl_cost=ctrl_cost,
            reward_contact_cost=contact_cost,
            reward_forward=forward_reward,
            reward_survive=survive_reward,
        )

        return state.replace(qp=qp, obs=obs, reward=reward, done=done)

    def _get_obs(self, qp: brax.QP, info: brax.Info) -> jp.ndarray:
        """Observe hexapod body position and velocities."""
        # some pre-processing to pull joint angles and velocities
        joint_angle, joint_vel = self.sys.joints[0].angle_vel(qp)

        com_quat = qp.rot[0]
        com_euler = quat_to_euler(com_quat)

        # qpos:
        # X,Y,Z of the torso (3,)
        # orientation of the torso as quaternion (4,)
        # joint angles (18,)
        qpos = [qp.pos[0, :], com_euler, joint_angle]

        # qvel:
        # velcotiy of the torso (3,)
        # angular velocity of the torso (3,)
        # joint angle velocities 18,)
        qvel = [qp.vel[0], qp.ang[0], joint_vel]

        # external contact forces:
        # delta velocity (3,), delta ang (3,) * 10 bodies in the system
        # Note that mujoco has 4 extra bodies tucked inside the Torso that Brax
        # ignores
        cfrc = [jp.clip(info.contact.vel, -1, 1), jp.clip(info.contact.ang, -1, 1)]
        # flatten bottom dimension
        cfrc = [jp.reshape(x, x.shape[:-2] + (-1,)) for x in cfrc]

        return jp.concatenate(qpos + qvel)

    def _noise(self, rng: jp.ndarray) -> jp.ndarray:
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        return jp.random_uniform(rng, (self.sys.num_joint_dof,), low, hi)

    def _gaussian_noise(self, rng: jp.ndarray) -> jp.ndarray:
        return (
            jax.random.normal(rng, (self.sys.num_joint_dof,)) * self._reset_noise_scale
        )


_SYSTEM_CONFIG = """
bodies {
  name: "base_link"
  colliders {
    position {
    }
    rotation {
    }
    box {
      halfsize {
        x: 0.114
        y: 0.0905
        z: 0.021
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 8.0
}
bodies {
  name: "leg_0_1"
  colliders {
    position {
      x: 0.019779144
      y: -0.019818814
    }
    rotation {
      z: -45.0574
    }
    box {
      halfsize {
        x: 0.0305
        y: 0.0205
        z: 0.0205
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_0_2"
  colliders {
    position {
      x: 0.02860912
      y: -0.0286665
    }
    rotation {
      z: 135.0
    }
    box {
      halfsize {
        x: 0.07
        y: 0.0212
        z: 0.01425
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_0_3"
  colliders {
    position {
      x: 0.027549522
      y: -0.02760478
      z: -0.05195
    }
    rotation {
      z: -45.0574
    }
    box {
      halfsize {
        x: 0.013
        y: 0.013
        z: 0.059
      }
    }
  }
  colliders {
    position {
      x: 0.027549522
      y: -0.02760478
      z: -0.11095
    }
    rotation {
      z: -45.0574
    }
    sphere {
      radius: 0.025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_1_1"
  colliders {
    position {
      x: -1.0284974e-07
      y: -0.028
    }
    rotation {
      z: -90.00021
    }
    box {
      halfsize {
        x: 0.0305
        y: 0.0205
        z: 0.0205
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_1_2"
  colliders {
    position {
      x: -1.4876481e-07
      y: -0.0405
    }
    rotation {
      z: -90.00021
    }
    box {
      halfsize {
        x: 0.07
        y: 0.0212
        z: 0.01425
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_1_3"
  colliders {
    position {
      x: -1.43255e-07
      y: -0.039
      z: -0.05195
    }
    rotation {
      z: -90.00021
    }
    box {
      halfsize {
        x: 0.013
        y: 0.013
        z: 0.059
      }
    }
  }
  colliders {
    position {
      x: -1.43255e-07
      y: -0.039
      z: -0.11095
    }
    rotation {
      z: -90.00021
    }
    sphere {
      radius: 0.025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_2_1"
  colliders {
    position {
      x: -0.019799098
      y: -0.01979888
    }
    rotation {
      z: -135.00032
    }
    box {
      halfsize {
        x: 0.0305
        y: 0.0205
        z: 0.0205
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_2_2"
  colliders {
    position {
      x: -0.028637983
      y: -0.028637666
    }
    rotation {
      z: -135.00032
    }
    box {
      halfsize {
        x: 0.07
        y: 0.0212
        z: 0.01425
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_2_3"
  colliders {
    position {
      x: -0.027577316
      y: -0.027577013
      z: -0.05195
    }
    rotation {
      z: -135.00032
    }
    box {
      halfsize {
        x: 0.013
        y: 0.013
        z: 0.059
      }
    }
  }
  colliders {
    position {
      x: -0.027577316
      y: -0.027577013
      z: -0.11095
    }
    rotation {
      z: -135.00032
    }
    sphere {
      radius: 0.025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_3_1"
  colliders {
    position {
      x: -0.019707816
      y: 0.019889748
    }
    rotation {
      z: 134.73676
    }
    box {
      halfsize {
        x: 0.0305
        y: 0.0205
        z: 0.0205
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_3_2"
  colliders {
    position {
      x: -0.028505947
      y: 0.028769098
    }
    rotation {
      z: 134.73676
    }
    box {
      halfsize {
        x: 0.07
        y: 0.0212
        z: 0.01425
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_3_3"
  colliders {
    position {
      x: -0.02745017
      y: 0.027703576
      z: -0.05195
    }
    rotation {
      z: 134.73676
    }
    box {
      halfsize {
        x: 0.013
        y: 0.013
        z: 0.059
      }
    }
  }
  colliders {
    position {
      x: -0.02745017
      y: 0.027703576
      z: -0.11095
    }
    rotation {
      z: 134.73676
    }
    sphere {
      radius: 0.025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_4_1"
  colliders {
    position {
      x: -1.0284974e-07
      y: 0.028
    }
    rotation {
      z: 90.00021
    }
    box {
      halfsize {
        x: 0.0305
        y: 0.0205
        z: 0.0205
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_4_2"
  colliders {
    position {
      x: -1.4876481e-07
      y: 0.0405
    }
    rotation {
      z: 90.00021
    }
    box {
      halfsize {
        x: 0.07
        y: 0.0212
        z: 0.01425
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_4_3"
  colliders {
    position {
      x: -1.43255e-07
      y: 0.039
      z: -0.05195
    }
    rotation {
      z: 90.00021
    }
    box {
      halfsize {
        x: 0.013
        y: 0.013
        z: 0.059
      }
    }
  }
  colliders {
    position {
      x: -1.43255e-07
      y: 0.039
      z: -0.11095
    }
    rotation {
      z: 90.00021
    }
    sphere {
      radius: 0.025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_5_1"
  colliders {
    position {
      x: 0.019798953
      y: 0.019799026
    }
    rotation {
      z: 45.000107
    }
    box {
      halfsize {
        x: 0.0305
        y: 0.0205
        z: 0.0205
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_5_2"
  colliders {
    position {
      x: 0.028637772
      y: 0.028637877
    }
    rotation {
      z: 45.000107
    }
    box {
      halfsize {
        x: 0.07
        y: 0.0212
        z: 0.01425
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_5_3"
  colliders {
    position {
      x: 0.027577113
      y: 0.027577216
      z: -0.05195
    }
    rotation {
      z: 45.000107
    }
    box {
      halfsize {
        x: 0.013
        y: 0.013
        z: 0.059
      }
    }
  }
  colliders {
    position {
      x: 0.027577113
      y: 0.027577216
      z: -0.11095
    }
    rotation {
      z: 45.000107
    }
    sphere {
      radius: 0.025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "floor"
  colliders {
    plane {
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    all: true
  }
}
joints {
  name: "body_leg_0"
  parent: "base_link"
  child: "leg_0_1"
  parent_offset {
    x: 0.093
    y: -0.053
  }
  rotation { y: -90
  }
  angular_damping: 20.0
  angle_limit {
    min: -22.5
    max: 22.5
  }
}
joints {
  name: "leg_0_1_2"
  parent: "leg_0_1"
  child: "leg_0_2"
  parent_offset {
    x: 0.039558288
    y: -0.03963763
  }
  rotation { z: 45.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
}
joints {
  name: "leg_0_2_3"
  parent: "leg_0_2"
  child: "leg_0_3"
  parent_offset {
    x: 0.05721824
    y: -0.057333
  }
  rotation { z: 45
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
}
joints {
  name: "body_leg_1"
  parent: "base_link"
  child: "leg_1_1"
  parent_offset {
    y: -0.075
  }
  rotation { y: -90
  }
  angular_damping: 20.0
  angle_limit {
    min: -22.5
    max: 22.5
  }
}
joints {
  name: "leg_1_1_2"

  parent: "leg_1_1"
  child: "leg_1_2"
  parent_offset {
    x: -2.0569948e-07
    y: -0.056
  }
  rotation {
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }

}
joints {
  name: "leg_1_2_3"

  parent: "leg_1_2"
  child: "leg_1_3"
  parent_offset {
    x: -2.9752962e-07
    y: -0.081
  }
  rotation {
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
}
joints {
  name: "body_leg_2"
  parent: "base_link"
  child: "leg_2_1"
  parent_offset {
    x: -0.093
    y: -0.053
  }
  rotation { y: -90
  }
  angular_damping: 20.0
  angle_limit {
    min: -22.5
    max: 22.5
  }
}
joints {
  name: "leg_2_1_2"
  parent: "leg_2_1"
  child: "leg_2_2"
  parent_offset {
    x: -0.039598197
    y: -0.03959776
  }
  rotation { z: -45
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
}
joints {
  name: "leg_2_2_3"
  parent: "leg_2_2"
  child: "leg_2_3"
  parent_offset {
    x: -0.057275966
    y: -0.057275333
  }
  rotation {z:-45.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
}
joints {
  name: "body_leg_3"
  parent: "base_link"
  child: "leg_3_1"
  parent_offset {
    x: -0.093
    y: 0.053
  }
  rotation { y: -90
  }
  angular_damping: 20.0
  angle_limit {
    min: -22.5
    max: 22.5
  }
}
joints {
  name: "leg_3_1_2"

  parent: "leg_3_1"
  child: "leg_3_2"
  parent_offset {
    x: -0.03941563
    y: 0.039779495
  }
  rotation { z: 45
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }

}
joints {
  name: "leg_3_2_3"

  parent: "leg_3_2"
  child: "leg_3_3"
  parent_offset {
    x: -0.057011895
    y: 0.057538196
  }
  rotation { z:45
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }

}
joints {
  name: "body_leg_4"
  parent: "base_link"
  child: "leg_4_1"
  parent_offset {
    y: 0.075
  }
  rotation { y: -90
  }
  angular_damping: 20.0
  angle_limit {
    min: -22.5
    max: 22.5
  }

}
joints {
  name: "leg_4_1_2"
  parent: "leg_4_1"
  child: "leg_4_2"
  parent_offset {
    x: -2.0569948e-07
    y: 0.056
  }
  rotation {
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }

}
joints {
  name: "leg_4_2_3"
  parent: "leg_4_2"
  child: "leg_4_3"
  parent_offset {
    x: -2.9752962e-07
    y: 0.081
  }
  rotation {
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
}
joints {
  name: "body_leg_5"
  parent: "base_link"
  child: "leg_5_1"
  parent_offset {
    x: 0.093
    y: 0.053
  }
  rotation { y: -90
  }
  angular_damping: 20.0
  angle_limit {
    min: -22.5
    max: 22.5
  }
}
joints {
  name: "leg_5_1_2"
  parent: "leg_5_1"
  child: "leg_5_2"
  parent_offset {
    x: 0.039597906
    y: 0.03959805
  }
  rotation { z:-45.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
}
joints {
  name: "leg_5_2_3"
  parent: "leg_5_2"
  child: "leg_5_3"
  parent_offset {
    x: 0.057275545
    y: 0.057275753
  }
  rotation { z:-45
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
}
actuators {
  name: "body_leg_0"
  joint: "body_leg_0"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_0_1_2"
  joint: "leg_0_1_2"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_0_2_3"
  joint: "leg_0_2_3"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "body_leg_1"
  joint: "body_leg_1"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_1_1_2"
  joint: "leg_1_1_2"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_1_2_3"
  joint: "leg_1_2_3"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "body_leg_2"
  joint: "body_leg_2"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_2_1_2"
  joint: "leg_2_1_2"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_2_2_3"
  joint: "leg_2_2_3"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "body_leg_3"
  joint: "body_leg_3"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_3_1_2"
  joint: "leg_3_1_2"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_3_2_3"
  joint: "leg_3_2_3"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "body_leg_4"
  joint: "body_leg_4"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_4_1_2"
  joint: "leg_4_1_2"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_4_2_3"
  joint: "leg_4_2_3"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "body_leg_5"
  joint: "body_leg_5"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_5_1_2"
  joint: "leg_5_1_2"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_5_2_3"
  joint: "leg_5_2_3"
  strength: 200.0
  angle {
  }
}
friction: 0.6
gravity: { z: -9.8 }
angular_damping: -0.05
collide_include {
  first: "base_link"
  second: "floor"
}
collide_include {
  first: "leg_0_3"
  second: "floor"
}
collide_include {
  first: "leg_1_3"
  second: "floor"
}
collide_include {
  first: "leg_2_3"
  second: "floor"
}
collide_include {
  first: "leg_3_3"
  second: "floor"
}
collide_include {
  first: "leg_4_3"
  second: "floor"
}
collide_include {
  first: "leg_5_3"
  second: "floor"
}
dt: 0.03
substeps: 10
dynamics_mode: "pbd"
"""


_SYSTEM_CONFIG_SPRING = """
bodies {
  name: "base_link"
  colliders {
    position {
    }
    rotation {
    }
    box {
      halfsize {
        x: 0.114
        y: 0.0905
        z: 0.021
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 5.0
}
bodies {
  name: "leg_0_1"
  colliders {
    position {
      x: 0.019779144
      y: -0.019818814
    }
    rotation {
      z: -45.0574
    }
    box {
      halfsize {
        x: 0.0305
        y: 0.0205
        z: 0.0205
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_0_2"
  colliders {
    position {
      x: 0.02860912
      y: -0.0286665
    }
    rotation {
      z: 135.0
    }
    box {
      halfsize {
        x: 0.07
        y: 0.0212
        z: 0.01425
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_0_3"
  colliders {
    position {
      x: 0.027549522
      y: -0.02760478
      z: -0.05195
    }
    rotation {
      z: -45.0574
    }
    box {
      halfsize {
        x: 0.013
        y: 0.013
        z: 0.059
      }
    }
  }
  colliders {
    position {
      x: 0.027549522
      y: -0.02760478
      z: -0.11095
    }
    rotation {
      z: -45.0574
    }
    sphere {
      radius: 0.025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_1_1"
  colliders {
    position {
      x: -1.0284974e-07
      y: -0.028
    }
    rotation {
      z: -90.00021
    }
    box {
      halfsize {
        x: 0.0305
        y: 0.0205
        z: 0.0205
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_1_2"
  colliders {
    position {
      x: -1.4876481e-07
      y: -0.0405
    }
    rotation {
      z: -90.00021
    }
    box {
      halfsize {
        x: 0.07
        y: 0.0212
        z: 0.01425
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_1_3"
  colliders {
    position {
      x: -1.43255e-07
      y: -0.039
      z: -0.05195
    }
    rotation {
      z: -90.00021
    }
    box {
      halfsize {
        x: 0.013
        y: 0.013
        z: 0.059
      }
    }
  }
  colliders {
    position {
      x: -1.43255e-07
      y: -0.039
      z: -0.11095
    }
    rotation {
      z: -90.00021
    }
    sphere {
      radius: 0.025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_2_1"
  colliders {
    position {
      x: -0.019799098
      y: -0.01979888
    }
    rotation {
      z: -135.00032
    }
    box {
      halfsize {
        x: 0.0305
        y: 0.0205
        z: 0.0205
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_2_2"
  colliders {
    position {
      x: -0.028637983
      y: -0.028637666
    }
    rotation {
      z: -135.00032
    }
    box {
      halfsize {
        x: 0.07
        y: 0.0212
        z: 0.01425
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_2_3"
  colliders {
    position {
      x: -0.027577316
      y: -0.027577013
      z: -0.05195
    }
    rotation {
      z: -135.00032
    }
    box {
      halfsize {
        x: 0.013
        y: 0.013
        z: 0.059
      }
    }
  }
  colliders {
    position {
      x: -0.027577316
      y: -0.027577013
      z: -0.11095
    }
    rotation {
      z: -135.00032
    }
    sphere {
      radius: 0.025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_3_1"
  colliders {
    position {
      x: -0.019707816
      y: 0.019889748
    }
    rotation {
      z: 134.73676
    }
    box {
      halfsize {
        x: 0.0305
        y: 0.0205
        z: 0.0205
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_3_2"
  colliders {
    position {
      x: -0.028505947
      y: 0.028769098
    }
    rotation {
      z: 134.73676
    }
    box {
      halfsize {
        x: 0.07
        y: 0.0212
        z: 0.01425
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_3_3"
  colliders {
    position {
      x: -0.02745017
      y: 0.027703576
      z: -0.05195
    }
    rotation {
      z: 134.73676
    }
    box {
      halfsize {
        x: 0.013
        y: 0.013
        z: 0.059
      }
    }
  }
  colliders {
    position {
      x: -0.02745017
      y: 0.027703576
      z: -0.11095
    }
    rotation {
      z: 134.73676
    }
    sphere {
      radius: 0.025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_4_1"
  colliders {
    position {
      x: -1.0284974e-07
      y: 0.028
    }
    rotation {
      z: 90.00021
    }
    box {
      halfsize {
        x: 0.0305
        y: 0.0205
        z: 0.0205
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_4_2"
  colliders {
    position {
      x: -1.4876481e-07
      y: 0.0405
    }
    rotation {
      z: 90.00021
    }
    box {
      halfsize {
        x: 0.07
        y: 0.0212
        z: 0.01425
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_4_3"
  colliders {
    position {
      x: -1.43255e-07
      y: 0.039
      z: -0.05195
    }
    rotation {
      z: 90.00021
    }
    box {
      halfsize {
        x: 0.013
        y: 0.013
        z: 0.059
      }
    }
  }
  colliders {
    position {
      x: -1.43255e-07
      y: 0.039
      z: -0.11095
    }
    rotation {
      z: 90.00021
    }
    sphere {
      radius: 0.025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_5_1"
  colliders {
    position {
      x: 0.019798953
      y: 0.019799026
    }
    rotation {
      z: 45.000107
    }
    box {
      halfsize {
        x: 0.0305
        y: 0.0205
        z: 0.0205
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_5_2"
  colliders {
    position {
      x: 0.028637772
      y: 0.028637877
    }
    rotation {
      z: 45.000107
    }
    box {
      halfsize {
        x: 0.07
        y: 0.0212
        z: 0.01425
      }
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "leg_5_3"
  colliders {
    position {
      x: 0.027577113
      y: 0.027577216
      z: -0.05195
    }
    rotation {
      z: 45.000107
    }
    box {
      halfsize {
        x: 0.013
        y: 0.013
        z: 0.059
      }
    }
  }
  colliders {
    position {
      x: 0.027577113
      y: 0.027577216
      z: -0.11095
    }
    rotation {
      z: 45.000107
    }
    sphere {
      radius: 0.025
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
}
bodies {
  name: "floor"
  colliders {
    plane {
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen {
    all: true
  }
}
joints {
  name: "body_leg_0"
  stiffness: 18000.0
  parent: "base_link"
  child: "leg_0_1"
  parent_offset {
    x: 0.093
    y: -0.053
  }
  rotation { y: -90
  }
  angular_damping: 20.0
  angle_limit {
    min: -22.5
    max: 22.5
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "leg_0_1_2"
  stiffness: 18000.0
  parent: "leg_0_1"
  child: "leg_0_2"
  parent_offset {
    x: 0.039558288
    y: -0.03963763
  }
  rotation { z: 45.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "leg_0_2_3"
  stiffness: 18000.0
  parent: "leg_0_2"
  child: "leg_0_3"
  parent_offset {
    x: 0.05721824
    y: -0.057333
  }
  rotation { z: 45
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "body_leg_1"
  stiffness: 18000.0
  parent: "base_link"
  child: "leg_1_1"
  parent_offset {
    y: -0.075
  }
  rotation { y: -90
  }
  angular_damping: 20.0
  angle_limit {
    min: -22.5
    max: 22.5
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "leg_1_1_2"
  stiffness: 18000.0
  parent: "leg_1_1"
  child: "leg_1_2"
  parent_offset {
    x: -2.0569948e-07
    y: -0.056
  }
  rotation {
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "leg_1_2_3"
  stiffness: 18000.0
  parent: "leg_1_2"
  child: "leg_1_3"
  parent_offset {
    x: -2.9752962e-07
    y: -0.081
  }
  rotation {
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "body_leg_2"
  stiffness: 18000.0
  parent: "base_link"
  child: "leg_2_1"
  parent_offset {
    x: -0.093
    y: -0.053
  }
  rotation { y: -90
  }
  angular_damping: 20.0
  angle_limit {
    min: -22.5
    max: 22.5
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "leg_2_1_2"
  stiffness: 18000.0
  parent: "leg_2_1"
  child: "leg_2_2"
  parent_offset {
    x: -0.039598197
    y: -0.03959776
  }
  rotation { z: -45
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "leg_2_2_3"
  stiffness: 18000.0
  parent: "leg_2_2"
  child: "leg_2_3"
  parent_offset {
    x: -0.057275966
    y: -0.057275333
  }
  rotation {z:-45.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "body_leg_3"
  stiffness: 18000.0
  parent: "base_link"
  child: "leg_3_1"
  parent_offset {
    x: -0.093
    y: 0.053
  }
  rotation { y: -90
  }
  angular_damping: 20.0
  angle_limit {
    min: -22.5
    max: 22.5
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "leg_3_1_2"
  stiffness: 18000.0
  parent: "leg_3_1"
  child: "leg_3_2"
  parent_offset {
    x: -0.03941563
    y: 0.039779495
  }
  rotation { z: 45
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "leg_3_2_3"
  stiffness: 18000.0
  parent: "leg_3_2"
  child: "leg_3_3"
  parent_offset {
    x: -0.057011895
    y: 0.057538196
  }
  rotation { z:45
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "body_leg_4"
  stiffness: 18000.0
  parent: "base_link"
  child: "leg_4_1"
  parent_offset {
    y: 0.075
  }
  rotation { y: -90
  }
  angular_damping: 20.0
  angle_limit {
    min: -22.5
    max: 22.5
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "leg_4_1_2"
  stiffness: 18000.0
  parent: "leg_4_1"
  child: "leg_4_2"
  parent_offset {
    x: -2.0569948e-07
    y: 0.056
  }
  rotation {
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "leg_4_2_3"
  stiffness: 18000.0
  parent: "leg_4_2"
  child: "leg_4_3"
  parent_offset {
    x: -2.9752962e-07
    y: 0.081
  }
  rotation {
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "body_leg_5"
  stiffness: 18000.0
  parent: "base_link"
  child: "leg_5_1"
  parent_offset {
    x: 0.093
    y: 0.053
  }
  rotation { y: -90
  }
  angular_damping: 20.0
  angle_limit {
    min: -22.5
    max: 22.5
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "leg_5_1_2"
  stiffness: 18000.0
  parent: "leg_5_1"
  child: "leg_5_2"
  parent_offset {
    x: 0.039597906
    y: 0.03959805
  }
  rotation { z:-45.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
joints {
  name: "leg_5_2_3"
  stiffness: 18000.0
  parent: "leg_5_2"
  child: "leg_5_3"
  parent_offset {
    x: 0.057275545
    y: 0.057275753
  }
  rotation { z:-45
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  limit_strength: 1300.0
  spring_damping: 10.0
}
actuators {
  name: "body_leg_0"
  joint: "body_leg_0"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_0_1_2"
  joint: "leg_0_1_2"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_0_2_3"
  joint: "leg_0_2_3"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "body_leg_1"
  joint: "body_leg_1"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_1_1_2"
  joint: "leg_1_1_2"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_1_2_3"
  joint: "leg_1_2_3"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "body_leg_2"
  joint: "body_leg_2"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_2_1_2"
  joint: "leg_2_1_2"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_2_2_3"
  joint: "leg_2_2_3"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "body_leg_3"
  joint: "body_leg_3"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_3_1_2"
  joint: "leg_3_1_2"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_3_2_3"
  joint: "leg_3_2_3"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "body_leg_4"
  joint: "body_leg_4"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_4_1_2"
  joint: "leg_4_1_2"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_4_2_3"
  joint: "leg_4_2_3"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "body_leg_5"
  joint: "body_leg_5"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_5_1_2"
  joint: "leg_5_1_2"
  strength: 200.0
  angle {
  }
}
actuators {
  name: "leg_5_2_3"
  joint: "leg_5_2_3"
  strength: 200.0
  angle {
  }
}
friction: 0.6
gravity: { z: -9.8 }
angular_damping: -0.05
baumgarte_erp: 0.1
collide_include {
  first: "base_link"
  second: "floor"
}
collide_include {
  first: "leg_0_3"
  second: "floor"
}
collide_include {
  first: "leg_1_3"
  second: "floor"
}
collide_include {
  first: "leg_2_3"
  second: "floor"
}
collide_include {
  first: "leg_3_3"
  second: "floor"
}
collide_include {
  first: "leg_4_3"
  second: "floor"
}
collide_include {
  first: "leg_5_3"
  second: "floor"
}
dt: 0.02
substeps: 5
dynamics_mode: "legacy_spring"
"""
