from typing import List, Tuple

import jax
import jax.numpy as jnp
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


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
    fit_std: float,
    desc_std: float,
    params_std: float,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Evaluate policies contained in params in parallel.
    """

    random_key, f_subkey, d_subkey, p_subkey = jax.random.split(random_key, num=4)

    # Add noise to the parameters
    params = params + jax.random.normal(p_subkey, shape=params.shape) * params_std

    # Evaluate
    fitnesses, descriptors = jax.vmap(arm)(params)

    # Add noise to the fitnesses and descriptors
    fitnesses = fitnesses + jax.random.normal(f_subkey, shape=fitnesses.shape) * fit_std
    descriptors = (
        descriptors + jax.random.normal(d_subkey, shape=descriptors.shape) * desc_std
    )

    return (
        fitnesses,
        descriptors,
        {},
        random_key,
    )


class ArmNoisy:
    def scoring_fn(
        self, params: Genotype, random_key: RNGKey
    ) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        random_key, subkey = jax.random.split(random_key)
        params_noisy = self.apply_noise_on_params(params, subkey)

        random_key, subkey = jax.random.split(random_key)
        fit, desc, _, _ = arm_scoring_function(params_noisy, subkey)

        random_key, subkey = jax.random.split(random_key)
        new_fit = self.apply_noise_on_fitness(params_noisy, fit, desc, subkey)

        random_key, subkey = jax.random.split(random_key)
        new_desc = self.apply_noise_on_desc(params_noisy, fit, desc, subkey)

        return new_fit, new_desc, {}, random_key

    def get_std(
        self, param: Genotype, fitness: Fitness, desc: Descriptor
    ) -> jnp.ndarray:
        return jnp.zeros((1,))

    def apply_noise_on_params(self, params: Genotype, random_key: RNGKey) -> Genotype:
        return params

    def apply_noise_on_fitness(
        self, params: Genotype, fitness: Fitness, desc: Descriptor, random_key: RNGKey
    ) -> Fitness:
        return fitness

    def apply_noise_on_desc(
        self, params: Genotype, fitness: Fitness, desc: Descriptor, random_key: RNGKey
    ) -> Descriptor:
        return desc

    @property
    def behavior_descriptor_length(self) -> int:
        return 2

    @property
    def behavior_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return ([0, 0], [1, 1])


class ArmGaussianNoise(ArmNoisy):
    def __init__(self, fit_std: float, desc_std: float, params_std: float):
        self.fit_std = fit_std
        self.desc_std = desc_std
        self.params_std = params_std

    def get_std(
        self, param: Genotype, fitness: Fitness, desc: Descriptor
    ) -> jnp.ndarray:
        return jnp.asrray([self.fit_std, self.desc_std, self.params_std])

    def apply_noise_on_params(self, params: Genotype, random_key: RNGKey) -> Genotype:
        return (
            params + jax.random.normal(random_key, shape=params.shape) * self.params_std
        )

    def apply_noise_on_fitness(
        self, params: Genotype, fitness: Fitness, desc: Descriptor, random_key: RNGKey
    ) -> Fitness:
        return (
            fitness + jax.random.normal(random_key, shape=fitness.shape) * self.fit_std
        )

    def apply_noise_on_desc(
        self, params: Genotype, fitness: Fitness, desc: Descriptor, random_key: RNGKey
    ) -> Descriptor:
        return desc + jax.random.normal(random_key, shape=desc.shape) * self.desc_std


class ArmBimodalGaussianFitness(ArmNoisy):
    """
    Fitness is a bimodal gaussian distribution.

    The first mode is a gaussian distribution with mean 0 and std
    `fit_std_1`.
    The second mode is a gaussian distribution with mean `mean_fitness_2`
    and std `fit_std_2`.
    The probability of using the first mode is `proba_mode_1`.
    """

    def __init__(
        self,
        proba_mode_1: float,
        fit_std_1: float,
        fit_std_2: float,
        mean_fitness_2: float,
    ) -> None:
        assert 0 <= proba_mode_1 <= 1

        self.proba_mode_1 = proba_mode_1
        self.fit_std_1 = fit_std_1
        self.fit_std_2 = fit_std_2
        self.mean_fitness_2 = mean_fitness_2

    def get_std(
        self, param: Genotype, fitness: Fitness, desc: Descriptor
    ) -> jnp.ndarray:
        return jnp.asrray([self.fit_std_1, self.fit_std_2])

    def get_noise(self, random_key: RNGKey) -> jnp.ndarray:
        def use_mode_1(_random_key: RNGKey) -> jnp.ndarray:
            _random_key, _subkey = jax.random.split(_random_key)
            return jax.random.normal(_subkey) * self.fit_std_1

        def use_mode_2(_random_key: RNGKey) -> jnp.ndarray:
            _random_key, _subkey = jax.random.split(_random_key)
            return jax.random.normal(_subkey) * self.fit_std_2 + self.mean_fitness_2

        random_key, subkey_1, subkey_2 = jax.random.split(random_key, 3)
        return jax.lax.cond(
            jax.random.uniform(subkey_1) < self.proba_mode_1,
            use_mode_1,
            use_mode_2,
            subkey_2,
        )

    def apply_noise_on_fitness(
        self, params: Genotype, fitness: Fitness, desc: Descriptor, random_key: RNGKey
    ) -> Fitness:
        batch_size = desc.shape[0]
        random_key, *subkeys = jax.random.split(random_key, batch_size + 1)
        subkeys = jnp.asarray(subkeys)

        noise = jax.vmap(self.get_noise)(subkeys)

        return fitness + noise


class ArmBimodalGaussianDesc(ArmNoisy):
    """
    Descriptor follows a bimodal gaussian distribution.

    The first mode is a gaussian distribution with mean 0 and std
    `fit_std_1`.
    The second mode is a gaussian distribution with mean `mean_fitness_2`
    and std `fit_std_2`.
    The probability of using the first mode is `proba_mode_1`.
    """

    def __init__(
        self,
        proba_mode_1: float,
        desc_std_1: List,
        desc_std_2: List,
        mean_desc_2: List,
    ) -> None:
        assert 0 <= proba_mode_1 <= 1

        self.proba_mode_1 = proba_mode_1
        self.desc_std_1 = jnp.asarray(
            desc_std_1, dtype=jnp.float32
        )  # should be a vector of length 2 (for directions x and y)
        self.desc_std_2 = jnp.asarray(
            desc_std_2, dtype=jnp.float32
        )  # should be a vector of length 2
        self.mean_fitness_2 = jnp.asarray(
            mean_desc_2, dtype=jnp.float32
        )  # should be a vector of length 2

    def get_std(
        self, param: Genotype, fitness: Fitness, desc: Descriptor
    ) -> jnp.ndarray:
        return jnp.asrray([self.desc_std_1, self.desc_std_2])

    def get_noise(self, random_key: RNGKey) -> jnp.ndarray:
        def use_mode_1(_random_key: RNGKey) -> jnp.ndarray:
            _random_key, _subkey = jax.random.split(_random_key)
            cov_1 = jnp.power(
                jnp.diag(self.desc_std_1), 2.0
            )  # squared values because we provide std values instead of variances
            return jax.random.multivariate_normal(_subkey, jnp.zeros(2), cov_1)

        def use_mode_2(_random_key: RNGKey) -> jnp.ndarray:
            _random_key, _subkey = jax.random.split(_random_key)
            cov_2 = jnp.power(
                jnp.diag(self.desc_std_2), 2.0
            )  # squared values because we provide std values instead of variances
            return jax.random.multivariate_normal(_subkey, self.mean_fitness_2, cov_2)

        random_key, subkey_1, subkey_2 = jax.random.split(random_key, 3)
        return jax.lax.cond(
            jax.random.uniform(subkey_1) < self.proba_mode_1,
            use_mode_1,
            use_mode_2,
            subkey_2,
        )

    def apply_noise_on_desc(
        self, params: Genotype, fitness: Fitness, desc: Descriptor, random_key: RNGKey
    ) -> Descriptor:
        batch_size = desc.shape[0]
        random_key, *subkeys = jax.random.split(random_key, batch_size + 1)
        subkeys = jnp.asarray(subkeys)

        noise = jax.vmap(self.get_noise)(subkeys)

        return desc + noise


class ArmSelectedJointsNoise(ArmNoisy):
    def __init__(
        self,
        selected_indexes: jnp.ndarray,
        params_std: float,
        no_fitness: bool = True,
    ) -> None:
        self.selected_indexes = jnp.asarray(selected_indexes, dtype=jnp.int32)
        self.params_std = params_std
        self.no_fitness = no_fitness

    def get_std(
        self, param: Genotype, fitness: Fitness, desc: Descriptor
    ) -> jnp.ndarray:
        return jnp.asrray([self.params_std])

    def get_noise(self, random_key: RNGKey) -> jnp.ndarray:
        return (
            jax.random.normal(random_key, shape=(len(self.selected_indexes),))
            * self.params_std
        )

    def apply_noise_on_params(self, params: Genotype, random_key: RNGKey) -> Genotype:
        batch_size = params.shape[0]
        random_key, *subkeys = jax.random.split(random_key, batch_size + 1)
        subkeys = jnp.asarray(subkeys)

        noise = jax.vmap(self.get_noise)(subkeys)
        new_params = params.at[:, self.selected_indexes].set(
            params[:, self.selected_indexes] + noise
        )

        return new_params

    def apply_noise_on_fitness(
        self, params: Genotype, fitness: Fitness, desc: Descriptor, random_key: RNGKey
    ) -> Fitness:
        return jnp.zeros_like(fitness) if self.no_fitness else fitness


class ArmGaussianDescBiVarianceNoise(ArmNoisy):
    def __init__(
        self, desc_std_1: List, desc_std_2: List, no_fitness: bool = True
    ) -> None:
        self.desc_std_1 = jnp.asarray(
            desc_std_1, dtype=jnp.float32
        )  # should be a vector of length 2 (for directions x and y)
        self.desc_std_2 = jnp.asarray(
            desc_std_2, dtype=jnp.float32
        )  # should be a vector of length 2
        self.no_fitness = no_fitness

    def get_std(
        self, param: Genotype, fitness: Fitness, desc: Descriptor
    ) -> jnp.ndarray:
        return jnp.where(
            jnp.prod(param - 0.5) >= 0,
            self.desc_std_1,
            self.desc_std_2,
        )

    def apply_noise_on_params(self, params: Genotype, random_key: RNGKey) -> Genotype:
        return params

    def apply_noise_on_fitness(
        self, params: Genotype, fitness: Fitness, desc: Descriptor, random_key: RNGKey
    ) -> Fitness:
        return jnp.zeros_like(fitness) if self.no_fitness else fitness

    def _apply_noise(
        self, param: Genotype, fitness: Fitness, desc: Descriptor, random_key: RNGKey
    ) -> Genotype:
        std = self.get_std(param=param, fitness=fitness, desc=desc)
        _random_key, _subkey = jax.random.split(random_key)
        cov_1 = jnp.power(
            jnp.diag(std), 2.0
        )  # squared values because we provide std values instead of std
        return jax.random.multivariate_normal(_subkey, jnp.zeros(2), cov_1)

    def apply_noise_on_desc(
        self, params: Genotype, fitness: Fitness, desc: Descriptor, random_key: RNGKey
    ) -> Descriptor:
        batch_size = desc.shape[0]
        random_key, *subkeys = jax.random.split(random_key, batch_size + 1)
        subkeys = jnp.asarray(subkeys)
        return desc + jax.vmap(self._apply_noise)(params, fitness, desc, subkeys)


class ArmGaussianDescFitPropVarianceNoise(ArmNoisy):
    def __init__(self, prop_factors: List, no_fitness: bool = True) -> None:
        self.prop_factors = jnp.asarray(
            prop_factors, dtype=jnp.float32
        )  # should be a vector of length 2 (for directions x and y)
        self.min_fitness = -0.5
        self.max_fitness = 0
        self.no_fitness = no_fitness

    def get_std(
        self, param: Genotype, fitness: Fitness, desc: Descriptor
    ) -> jnp.ndarray:
        return self.prop_factors * (
            1
            - (jnp.clip(fitness, self.min_fitness, self.max_fitness) - self.min_fitness)
            / (self.max_fitness - self.min_fitness)
        )

    def apply_noise_on_params(self, params: Genotype, random_key: RNGKey) -> Genotype:
        return params

    def apply_noise_on_fitness(
        self, params: Genotype, fitness: Fitness, desc: Descriptor, random_key: RNGKey
    ) -> Fitness:
        return jnp.zeros_like(fitness) if self.no_fitness else fitness

    def _apply_noise(
        self, params: Genotype, fitness: Fitness, desc: Descriptor, random_key: RNGKey
    ) -> Fitness:
        std = self.get_std(param=params, fitness=fitness, desc=desc)
        _random_key, _subkey = jax.random.split(random_key)
        cov_1 = jnp.power(
            jnp.diag(std), 2.0
        )  # squared values because we provide std values instead of variances
        return jax.random.multivariate_normal(_subkey, jnp.zeros(2), cov_1)

    def apply_noise_on_desc(
        self, params: Genotype, fitness: Fitness, desc: Descriptor, random_key: RNGKey
    ) -> Descriptor:
        batch_size = desc.shape[0]
        random_key, *subkeys = jax.random.split(random_key, batch_size + 1)
        subkeys = jnp.asarray(subkeys)
        return desc + jax.vmap(self._apply_noise)(params, fitness, desc, subkeys)
