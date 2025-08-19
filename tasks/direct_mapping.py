from typing import Tuple

import jax
import jax.numpy as jnp
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey


def direct_mapping(params: Genotype) -> Tuple[Fitness, Descriptor]:
    """
    Compute the fitness and BD of one individual in the Direct Mapping task.

    Args:
        params: genotype of the individual to evaluate, corresponding
            to the descriptor followed by the fitness.
            Params should be between [0, 1].

    Returns:
        f: the fitness of the individual, given by the genotype.
        bd: the bd of the individual, given by the genotype.
    """

    x = jnp.clip(params, 0, 1)
    return x[2], x[0:2]


def direct_mapping_scoring_function(
    params: Genotype,
    random_key: RNGKey,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Evaluate policies contained in params in parallel.
    """
    fitnesses, descriptors = jax.vmap(direct_mapping)(params)

    return (
        fitnesses,
        descriptors,
        {},
        random_key,
    )


def noisy_direct_mapping_scoring_function(
    params: Genotype,
    random_key: RNGKey,
    fit_std: float,
    desc_std: float,
    params_std: float,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Evaluate policies contained in params in parallel.
    Args:
        params
        random_key
        fit_std: std of the gaussian noise on fitness
        desc_std: std of the gaussian noise on descriptor
        params_std: std of the gaussian noise on params
    """

    random_key, f_subkey, d_subkey, p_subkey = jax.random.split(random_key, num=4)

    # Add noise to the parameters
    params = params + jax.random.normal(p_subkey, shape=params.shape) * params_std

    # Evaluate
    fitnesses, descriptors = jax.vmap(direct_mapping)(params)

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


def no_trade_off_direct_mapping_scoring_function(
    params: Genotype,
    random_key: RNGKey,
    desc_std: float,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Evaluate policies contained in params in parallel, with all the same
    std of the gaussian noise.
    Args:
        params
        random_key
        desc_std: std of the gaussian noise on descriptor
    """

    random_key, subkey = jax.random.split(random_key)

    # Evaluate
    fitnesses, descriptors = jax.vmap(direct_mapping)(params)

    # Add noise to the descriptors
    descriptors = (
        descriptors + jax.random.normal(subkey, shape=descriptors.shape) * desc_std
    )

    return (
        fitnesses,
        descriptors,
        {},
        random_key,
    )


def perfect_trade_off_direct_mapping_scoring_function(
    params: Genotype,
    random_key: RNGKey,
    desc_std: float,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Evaluate policies contained in params in parallel, with std of the
    gaussian noise on descriptor exactly equal to the fitness.
    Args:
        params
        random_key
        desc_std: maximum std on descriptor used for normalisation.
    """

    random_key, subkey = jax.random.split(random_key)

    # Evaluate
    fitnesses, descriptors = jax.vmap(direct_mapping)(params)

    # Normalise the fitness to a raisonable std range
    fitnesses = fitnesses * desc_std

    # Add noise to the descriptors
    extended_fitnesses = jnp.repeat(
        jnp.expand_dims(fitnesses, axis=1), descriptors.shape[1], axis=1
    )
    descriptors = descriptors + jnp.multiply(
        jax.random.normal(subkey, shape=descriptors.shape), extended_fitnesses
    )

    return (
        fitnesses,
        descriptors,
        {},
        random_key,
    )


def sharp_peak_direct_mapping_scoring_function(
    params: Genotype,
    random_key: RNGKey,
    desc_std: float,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Evaluate policies contained in params in parallel, adding a large std
    only to the 10% highest-fitness individual.
    Args:
        params
        random_key
        desc_std: std for 10% highest-fitness individual.
    """

    random_key, subkey = jax.random.split(random_key)

    # Evaluate
    fitnesses, descriptors = jax.vmap(direct_mapping)(params)

    # Add noise to the descriptors, only when 10% highest-fitness individual
    extended_fitnesses = jnp.repeat(
        jnp.expand_dims(fitnesses, axis=1), descriptors.shape[1], axis=1
    )
    descriptors = jnp.where(
        extended_fitnesses < 0.9,
        descriptors,
        descriptors + jax.random.normal(subkey, shape=descriptors.shape) * desc_std,
    )

    return (
        fitnesses,
        descriptors,
        {},
        random_key,
    )


def deceptive_direct_mapping_scoring_function(
    params: Genotype,
    random_key: RNGKey,
    desc_std: float,
) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    """
    Evaluate policies contained in params in parallel, adding a large std
    to all individuals within range [60%, 70%] and [90%, 100%].
    Args:
        params
        random_key
        desc_std: std for individual who have noise.
    """

    random_key, subkey = jax.random.split(random_key)

    # Evaluate
    fitnesses, descriptors = jax.vmap(direct_mapping)(params)

    # Add noise to the descriptors, only when 10% highest-fitness individual
    extended_fitnesses = jnp.repeat(
        jnp.expand_dims(fitnesses, axis=1), descriptors.shape[1], axis=1
    )
    descriptors = jnp.where(
        jnp.logical_or(
            extended_fitnesses < 0.6,
            extended_fitnesses > 0.7,
        ),
        descriptors,
        descriptors + jax.random.normal(subkey, shape=descriptors.shape) * desc_std,
    )

    return (
        fitnesses,
        descriptors,
        {},
        random_key,
    )
