from functools import partial
from typing import Any, Callable, Tuple

import jax
from qdax.core.emitters.emitter import Emitter
from qdax.core.emitters.mutation_operators import isoline_variation, polynomial_mutation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from core.emitters.random_emitter import RandomEmitter

# Emitter list
EMITTER_LIST = [
    "Random",
    "Mixing",
]

# Metrics type returned by each emitter
USAGE_EMITTER = [
    "Random",
    "Mixing",
]

# Mutation list
MUTATION_LIST = [
    "isoline",
    "polynomial",
]


def get_evals_per_offspring(args: Any) -> int:
    """Get the number of samples spent on each offspring."""

    # Base is the number of samples
    evals_per_offspring = max(args.num_samples, 1)

    return evals_per_offspring  # type: ignore


def set_up_emitter(
    emitter_name: str,
    container_name: str,
    num_iterations: int,
    batch_size: int,
    env: int,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    num_descriptors: int,
    num_centroids: int,
    hard_limit_genotype: bool,
    min_genotype: float,
    max_genotype: float,
    policy_structure: Any,
    init_policies: Genotype,
    mutation: str,
    iso_sigma: float,
    line_sigma: float,
    proportion_mutation: float,
    eta: float,
) -> Emitter:

    # Input check
    assert emitter_name in EMITTER_LIST, "\n!!!ERROR!!! Invalid emitter:" + emitter_name
    assert mutation in MUTATION_LIST, "\n!!!ERROR!!! Invalid mutation:" + mutation

    # Define mutation
    if mutation == "isoline":
        variation_fn = partial(
            isoline_variation,
            iso_sigma=iso_sigma,
            line_sigma=line_sigma,
            minval=min_genotype if hard_limit_genotype else None,
            maxval=max_genotype if hard_limit_genotype else None,
        )
        mutation_fn = None
        variation_percentage = 1.0
    elif mutation == "polynomial":
        variation_fn = None
        mutation_fn = partial(
            polynomial_mutation,
            proportion_to_mutate=proportion_mutation,
            eta=eta,
            minval=min_genotype,
            maxval=max_genotype,
        )
        variation_percentage = 0.0
    else:
        assert 0, "!!!ERROR!!! Undefined mutation."

    # Define emitter
    if emitter_name == "Random":
        emitter = RandomEmitter(
            batch_size=batch_size,
            genotypes=jax.tree_map(lambda x: x[0], init_policies),
            min_genotype=min_genotype,
            max_genotype=max_genotype,
            behavior_descriptor_length=num_descriptors,
        )
    elif emitter_name == "Mixing":
        emitter = MixingEmitter(
            mutation_fn=mutation_fn,
            variation_fn=variation_fn,
            variation_percentage=variation_percentage,
            batch_size=batch_size,
        )

    return emitter
