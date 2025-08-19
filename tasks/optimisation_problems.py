import math
from typing import Tuple

import jax
import jax.numpy as jnp
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


def rastrigin(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Compute the fitness and BD of one individual in the Rastrigin task.
    The genotype value are clip to [0, 1] for consistency with the
    original problem definition.

    Args:
        params: genotype of the individual to evaluate, corresponding to
            the input of the Rastrigin function.

    Returns:
        f: the fitness of the individual, given as the Rastrigin value of params.
        bd: the bd of the individual, given as the two first params values.
    """
    x = jnp.clip(params, 0, 1)

    x = x * 10 - 5  # scaling to [-5, 5]
    f = 10 * x.shape[0] + (x * x - 10 * jnp.cos(2 * math.pi * x)).sum()
    return -f, jnp.array([x[0], x[1]])


def rastrigin_scoring_function(
    params: Genotype,
    random_key: RNGKey,
    fit_std: float,
    desc_std: float,
    params_std: float,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Evaluate policies contained in params in parallel on the Rastrigin task.

    Args:
        params: genotype of the individuals to evaluate.
        random_key
        fit_std: std of fitness noise (paper value: 0.05).
        desc_std: std of descriptor noise (paper value: 0.01).
        params_std: std of parameters noise.

    Returns:
        fitnesses: fitnesses of individuals in params.
        descriptors: descriptors of individuals in params.
        infos: unused additional informations.
        random_key
    """

    random_key, f_subkey, d_subkey, p_subkey = jax.random.split(random_key, num=4)

    # Add noise to the parameters
    params = params + jax.random.normal(p_subkey, shape=params.shape) * params_std

    # Evaluate
    fitnesses, descriptors = jax.vmap(rastrigin)(params)

    # Add noise
    fitnesses = fitnesses + jax.random.normal(f_subkey, shape=fitnesses.shape) * fit_std
    descriptors = (
        descriptors + jax.random.normal(d_subkey, shape=descriptors.shape) * desc_std
    )

    return fitnesses, descriptors, {}, random_key


def sphere(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Compute the fitness and BD of one individual in the Sphere task.
    The genotype value are clip to [0, 1] for consistency with the
    original problem definition.

    Args:
        params: genotype of the individual to evaluate, corresponding to
            the input of the Sphere function.

    Returns:
        f: the fitness of the individual, given as the Sphere value of params.
        bd: the bd of the individual, given as the two first params values.
    """
    x = jnp.clip(params, 0, 1)

    x = x * 10 - 5  # scaling to [-5, 5]
    f = (x * x).sum()
    return -f, jnp.array([x[0], x[1]])


def sphere_scoring_function(
    params: Genotype,
    random_key: RNGKey,
    fit_std: float,
    desc_std: float,
    params_std: float,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Evaluate policies contained in params in parallel on the Sphere task.

    Args:
        params: genotype of the individuals to evaluate.
        random_key
        fit_std: std of fitness noise (paper value: 0.05).
        desc_std: std of descriptor noise (paper value: 0.01).
        params_std: std of parameters noise.

    Returns:
        fitnesses: fitnesses of individuals in params.
        descriptors: descriptors of individuals in params.
        infos: unused additional informations.
        random_key
    """

    random_key, f_subkey, d_subkey, p_subkey = jax.random.split(random_key, num=4)

    # Add noise to the parameters
    params = params + jax.random.normal(p_subkey, shape=params.shape) * params_std

    # Evaluate
    fitnesses, descriptors = jax.vmap(sphere)(params)

    # Add noise
    fitnesses = fitnesses + jax.random.normal(f_subkey, shape=fitnesses.shape) * fit_std
    descriptors = (
        descriptors + jax.random.normal(d_subkey, shape=descriptors.shape) * desc_std
    )

    return fitnesses, descriptors, {}, random_key
