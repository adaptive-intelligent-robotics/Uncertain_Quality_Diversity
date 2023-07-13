import abc
from typing import List, Tuple

import jax
import jax.numpy as jnp

from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


def arm(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Compute the fitness and BD of one individual in the Planar Arm task.
    Based on the Planar Arm implementation in fast_map_elites
    (https://github.com/hucebot/fast_map-elites).

    Args:
        params: genotype of the individual to evaluate, corresponding to
            the normalised angles for each DoF of the arm.
            Params should be between [0, 1].

    Returns:
        f: the fitness of the individual, given as the variance of the angles.
        bd: the bd of the individual, given as the [x, y] position of the
            end-effector of the arm.
            BD is normalized to [0, 1] regardless of the num of DoF.
            Arm is centered at 0.5, 0.5.
    """

    x = jnp.clip(params, 0, 1)
    size = params.shape[0]

    f = jnp.sqrt(jnp.mean(jnp.square(x - jnp.mean(x))))

    # Compute the end-effector position - forward kinemateics
    cum_angles = jnp.cumsum(2 * jnp.pi * x - jnp.pi)
    x_pos = jnp.sum(jnp.cos(cum_angles)) / (2 * size) + 0.5
    y_pos = jnp.sum(jnp.sin(cum_angles)) / (2 * size) + 0.5

    return -f, jnp.array([x_pos, y_pos])


def arm_scoring_function(
    params: Genotype,
    random_key: RNGKey,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Evaluate policies contained in params in parallel.
    """
    fitnesses, descriptors = jax.vmap(arm)(params)

    return (
        fitnesses,
        descriptors,
        {},
        random_key,
    )


def noisy_arm_scoring_function(
    params: Genotype,
    random_key: RNGKey,
    fit_variance: float,
    desc_variance: float,
    params_variance: float,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Evaluate policies contained in params in parallel.
    """

    random_key, f_subkey, d_subkey, p_subkey = jax.random.split(random_key, num=4)

    # Add noise to the parameters
    params = params + jax.random.normal(p_subkey, shape=params.shape) * params_variance

    # Evaluate
    fitnesses, descriptors = jax.vmap(arm)(params)

    # Add noise to the fitnesses and descriptors
    fitnesses = (
        fitnesses + jax.random.normal(f_subkey, shape=fitnesses.shape) * fit_variance
    )
    descriptors = (
        descriptors
        + jax.random.normal(d_subkey, shape=descriptors.shape) * desc_variance
    )

    return (
        fitnesses,
        descriptors,
        {},
        random_key,
    )


class ArmNoisy(abc.ABC):
    def scoring_fn(self, params, random_key):
        random_key, subkey = jax.random.split(random_key)
        params_noisy = self.apply_noise_on_params(params, subkey)

        random_key, subkey = jax.random.split(random_key)
        fit, desc, _, _ = arm_scoring_function(params_noisy, subkey)

        random_key, subkey = jax.random.split(random_key)
        new_fit = self.apply_noise_on_fitness(params_noisy, fit, desc, subkey)

        random_key, subkey = jax.random.split(random_key)
        new_desc = self.apply_noise_on_desc(params_noisy, fit, desc, subkey)

        return new_fit, new_desc, {}, random_key

    def apply_noise_on_params(self, params, random_key):
        return params

    def apply_noise_on_fitness(self, params, fitness, desc, random_key):
        return fitness

    def apply_noise_on_desc(self, params, fitness, desc, random_key):
        return desc

    @property
    def behavior_descriptor_length(self) -> int:
        return 2

    @property
    def behavior_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return ([0, 0], [1, 1])


class ArmGaussianNoise(ArmNoisy):
    def __init__(self, fit_variance, desc_variance, params_variance):
        self.fit_variance = fit_variance
        self.desc_variance = desc_variance
        self.params_variance = params_variance

    def apply_noise_on_params(self, params, random_key):
        return (
            params
            + jax.random.normal(random_key, shape=params.shape) * self.params_variance
        )

    def apply_noise_on_fitness(self, params, fitness, desc, random_key):
        return (
            fitness
            + jax.random.normal(random_key, shape=fitness.shape) * self.fit_variance
        )

    def apply_noise_on_desc(self, params, fitness, desc, random_key):
        return (
            desc + jax.random.normal(random_key, shape=desc.shape) * self.desc_variance
        )


class ArmBimodalGaussianFitness(ArmNoisy):
    """
    Fitness is a bimodal gaussian distribution.

    The first mode is a gaussian distribution with mean 0 and variance
    `fit_variance_1`.
    The second mode is a gaussian distribution with mean `mean_fitness_2`
    and variance `fit_variance_2`.
    The probability of using the first mode is `proba_mode_1`.
    """

    def __init__(self, proba_mode_1, fit_variance_1, fit_variance_2, mean_fitness_2):
        assert 0 <= proba_mode_1 <= 1

        self.proba_mode_1 = proba_mode_1
        self.fit_variance_1 = fit_variance_1
        self.fit_variance_2 = fit_variance_2
        self.mean_fitness_2 = mean_fitness_2

    def get_noise(self, random_key):
        def use_mode_1(_random_key):
            _random_key, _subkey = jax.random.split(_random_key)
            return jax.random.normal(_subkey) * self.fit_variance_1

        def use_mode_2(_random_key):
            _random_key, _subkey = jax.random.split(_random_key)
            return (
                jax.random.normal(_subkey) * self.fit_variance_2 + self.mean_fitness_2
            )

        random_key, subkey_1, subkey_2 = jax.random.split(random_key, 3)
        return jax.lax.cond(
            jax.random.uniform(subkey_1) < self.proba_mode_1,
            use_mode_1,
            use_mode_2,
            subkey_2,
        )

    def apply_noise_on_fitness(self, _, fitness, desc, random_key):
        batch_size = desc.shape[0]
        random_key, *subkeys = jax.random.split(random_key, batch_size + 1)
        subkeys = jnp.asarray(subkeys)

        noise = jax.vmap(self.get_noise)(subkeys)

        return fitness + noise


class ArmBimodalGaussianDesc(ArmNoisy):
    """
    Descriptor follows a bimodal gaussian distribution.

    The first mode is a gaussian distribution with mean 0 and variance
    `fit_variance_1`.
    The second mode is a gaussian distribution with mean `mean_fitness_2`
    and variance `fit_variance_2`.
    The probability of using the first mode is `proba_mode_1`.
    """

    def __init__(self, proba_mode_1, desc_variances_1, desc_variances_2, mean_desc_2):
        assert 0 <= proba_mode_1 <= 1

        self.proba_mode_1 = proba_mode_1
        self.desc_variances_1 = jnp.asarray(
            desc_variances_1, dtype=jnp.float32
        )  # should be a vector of length 2 (for directions x and y)
        self.desc_variances_2 = jnp.asarray(
            desc_variances_2, dtype=jnp.float32
        )  # should be a vector of length 2
        self.mean_fitness_2 = jnp.asarray(
            mean_desc_2, dtype=jnp.float32
        )  # should be a vector of length 2

    def get_noise(self, random_key):
        def use_mode_1(_random_key):
            _random_key, _subkey = jax.random.split(_random_key)
            cov_1 = jnp.power(
                jnp.diag(self.desc_variances_1), 2.0
            )  # squared values because we provide std values instead of variances
            return jax.random.multivariate_normal(_subkey, jnp.zeros(2), cov_1)

        def use_mode_2(_random_key):
            _random_key, _subkey = jax.random.split(_random_key)
            cov_2 = jnp.power(
                jnp.diag(self.desc_variances_2), 2.0
            )  # squared values because we provide std values instead of variances
            return jax.random.multivariate_normal(_subkey, self.mean_fitness_2, cov_2)

        random_key, subkey_1, subkey_2 = jax.random.split(random_key, 3)
        return jax.lax.cond(
            jax.random.uniform(subkey_1) < self.proba_mode_1,
            use_mode_1,
            use_mode_2,
            subkey_2,
        )

    def apply_noise_on_desc(self, _1, _2, desc, random_key):
        batch_size = desc.shape[0]
        random_key, *subkeys = jax.random.split(random_key, batch_size + 1)
        subkeys = jnp.asarray(subkeys)

        noise = jax.vmap(self.get_noise)(subkeys)

        return desc + noise


class ArmSelectedJointsNoise(ArmNoisy):
    def __init__(self, selected_indexes: jnp.ndarray, params_variance: float, no_fitness=True):
        self.selected_indexes = jnp.asarray(selected_indexes, dtype=jnp.int32)
        self.params_variance = params_variance
        self.no_fitness = no_fitness

    def get_noise(self, random_key):
        return (
            jax.random.normal(random_key, shape=(len(self.selected_indexes),))
            * self.params_variance
        )

    def apply_noise_on_params(self, params, random_key):
        batch_size = params.shape[0]
        random_key, *subkeys = jax.random.split(random_key, batch_size + 1)
        subkeys = jnp.asarray(subkeys)

        noise = jax.vmap(self.get_noise)(subkeys)
        new_params = params.at[:, self.selected_indexes].set(
            params[:, self.selected_indexes] + noise
        )

        return new_params

    def apply_noise_on_fitness(self, params, fitness, desc, random_key):
        return jnp.zeros_like(fitness) if self.no_fitness else fitness


class ArmGaussianDescBiVarianceNoise(ArmNoisy):
    def __init__(self, desc_variances_1, desc_variances_2, no_fitness=True):
        self.desc_variances_1 = jnp.asarray(
            desc_variances_1, dtype=jnp.float32
        )  # should be a vector of length 2 (for directions x and y)
        self.desc_variances_2 = jnp.asarray(
            desc_variances_2, dtype=jnp.float32
        )  # should be a vector of length 2
        self.no_fitness = no_fitness

    def apply_noise_on_params(self, params, random_key):
        return params

    def apply_noise_on_fitness(self, params, fitness, desc, random_key):
        return jnp.zeros_like(fitness) if self.no_fitness else fitness

    def _apply_noise(self, param, random_key) -> None:
        variances = jnp.where(
            jnp.prod(param) >= 0,
            self.desc_variances_1,
            self.desc_variances_2,
        )
        _random_key, _subkey = jax.random.split(random_key)
        cov_1 = jnp.power(
            jnp.diag(variances), 2.0
        )  # squared values because we provide std values instead of variances
        return jax.random.multivariate_normal(_subkey, jnp.zeros(2), cov_1)

    def apply_noise_on_desc(self, params, fitness, desc, random_key):
        batch_size = desc.shape[0]
        random_key, *subkeys = jax.random.split(random_key, batch_size + 1)
        subkeys = jnp.asarray(subkeys)
        return desc + jax.vmap(self._apply_noise)(params, subkeys)


class ArmGaussianDescFitPropVarianceNoise(ArmNoisy):
    def __init__(self, prop_factors, no_fitness=True):
        self.prop_factors = jnp.asarray(
            prop_factors, dtype=jnp.float32
        )  # should be a vector of length 2 (for directions x and y)
        self.min_fitness = -0.5
        self.max_fitness = 0
        self.no_fitness = no_fitness

    def apply_noise_on_params(self, params, random_key):
        return params

    def apply_noise_on_fitness(self, params, fitness, desc, random_key):
        return jnp.zeros_like(fitness) if self.no_fitness else fitness

    def _apply_noise(self, fitness, random_key) -> None:
        variances = (
            self.prop_factors
            * (1 - (jnp.clip(fitness, self.min_fitness, self.max_fitness) - self.min_fitness)
            / (self.max_fitness - self.min_fitness))
        )
        _random_key, _subkey = jax.random.split(random_key)
        cov_1 = jnp.power(
            jnp.diag(variances), 2.0
        )  # squared values because we provide std values instead of variances
        return jax.random.multivariate_normal(_subkey, jnp.zeros(2), cov_1)

    def apply_noise_on_desc(self, params, fitness, desc, random_key):
        batch_size = desc.shape[0]
        random_key, *subkeys = jax.random.split(random_key, batch_size + 1)
        subkeys = jnp.asarray(subkeys)
        return desc + jax.vmap(self._apply_noise)(fitness, subkeys)


def test():
    random_key = jax.random.PRNGKey(0)
    env = ArmSelectedJointsNoise(jnp.array([0, 2, 4]), 0.1)
    params = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    random_key, subkey = jax.random.split(random_key, 2)
    print(env.apply_noise_on_params(params, subkey))


if __name__ == "__main__":
    test()
