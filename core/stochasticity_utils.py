from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from core.sampling import (
    dummy_extra_scores_extractor,
    median,
    multi_sample_scoring_function,
    std,
)


@partial(
    jax.jit,
    static_argnames=(
        "scoring_fn",
        "num_reevals",
        "scan_size",
        "fitness_extractor",
        "fitness_reproducibility_extractor",
        "descriptor_extractor",
        "descriptor_reproducibility_extractor",
        "extra_scores_extractor",
    ),
)
def reevaluation_function(
    repertoire: MapElitesRepertoire,
    random_key: RNGKey,
    metric_repertoire: MapElitesRepertoire,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    num_reevals: int,
    scan_size: int,
    fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray] = median,
    fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray] = std,
    descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray] = median,
    descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray] = std,
    extra_scores_extractor: Callable[
        [ExtraScores, int], ExtraScores
    ] = dummy_extra_scores_extractor,
) -> Tuple[
    MapElitesRepertoire,
    MapElitesRepertoire,
    MapElitesRepertoire,
    MapElitesRepertoire,
    MapElitesRepertoire,
    MapElitesRepertoire,
    MapElitesRepertoire,
    RNGKey,
]:
    """
    Perform reevaluation of a repertoire in stochastic applications.

    Args:
        repertoire: repertoire to reevaluate.
        metric_repertoire: repertoire used to compute reeval stats, allow to use a
            different type of container than the one from the algorithm (in most cases
            just set to the same as repertoire).
        random_key: JAX random key.
        scoring_fn: scoring function used for evaluation.
        num_reevals: number of samples to generate for each individual.
        scan_size: allow to split the reevaluations in multiple batch in case the
            memory is limited.
                fitness_extractor: function to extract the final fitness from
            multiple samples of the same policy.
                fitness_reproducibility_extractor: function to extract the fitness
            reproducibility from multiple samples of the same policy.
        descriptor_extractor: function to extract the final descriptor from
            multiple samples of the same policy.
                descriptor_reproducibility_extractor: function to extract the descriptor
            reproducibility from multiple samples of the same policy.
        extra_scores_extractor: function to extract the extra_scores from
            multiple samples of the same policy.
    Returns:
        A container with reevaluated fitness and descriptor.
        A container with reevaluated fitness only.
        A container with reevaluated descriptor only.
        A non-reevaluated container with reproducibility in fitness.
        A reevaluated container with reproducibility in fitness.
        A non-reevaluated container with reproducibility in descriptor.
        A reevaluated container with reproducibility in descriptor.
        A random key.
    """

    # If no reevaluations, return copies of the original container
    if num_reevals == 0:
        return (
            repertoire,
            repertoire,
            repertoire,
            repertoire,
            repertoire,
            repertoire,
            repertoire,
            random_key,
        )

    policies_params = repertoire.genotypes

    # If no need for scan, call the sampling function
    if scan_size == 0:
        (
            all_fitnesses,
            all_descriptors,
            all_extra_scores,
            random_key,
        ) = multi_sample_scoring_function(
            policies_params=policies_params,
            random_key=random_key,
            scoring_fn=scoring_fn,
            num_samples=num_reevals,
        )
    else:
        num_loops = num_reevals // scan_size

        def _sampling_scan(
            random_key: RNGKey,
            unused: Tuple[()],
        ) -> Tuple[Tuple[RNGKey], Tuple[Fitness, Descriptor, ExtraScores]]:
            (
                all_fitnesses,
                all_descriptors,
                all_extra_scores,
                random_key,
            ) = multi_sample_scoring_function(
                policies_params=policies_params,
                random_key=random_key,
                scoring_fn=scoring_fn,
                num_samples=scan_size,
            )
            return (random_key), (
                all_fitnesses,
                all_descriptors,
                all_extra_scores,
            )

        (random_key), (
            all_fitnesses,
            all_descriptors,
            all_extra_scores,
        ) = jax.lax.scan(_sampling_scan, (random_key), (), length=num_loops)
        all_fitnesses = jnp.hstack(all_fitnesses)
        all_descriptors = jnp.hstack(all_descriptors)

    # Extract the final scores
    extra_scores = extra_scores_extractor(all_extra_scores, num_reevals)
    fitnesses = fitness_extractor(all_fitnesses)
    fitnesses_reproducibility = fitness_reproducibility_extractor(all_fitnesses)
    descriptors = descriptor_extractor(all_descriptors)
    descriptors_reproducibility = descriptor_reproducibility_extractor(all_descriptors)

    # WARNING: in the case of descriptors_reproducibility, take average over dimensions
    descriptors_reproducibility = jnp.average(descriptors_reproducibility, axis=-1)

    # Set -inf fitness for all unexisting indivs
    fitnesses = jnp.where(repertoire.fitnesses == -jnp.inf, -jnp.inf, fitnesses)
    fitnesses_reproducibility = jnp.where(
        repertoire.fitnesses == -jnp.inf, -jnp.inf, fitnesses_reproducibility
    )
    descriptors_reproducibility = jnp.where(
        repertoire.fitnesses == -jnp.inf, -jnp.inf, descriptors_reproducibility
    )

    # Fill-in reeval repertoire
    reeval_repertoire = metric_repertoire.empty()
    reeval_repertoire = reeval_repertoire.add(
        repertoire.genotypes,
        descriptors,
        fitnesses,
        extra_scores,
    )

    # Fill-in fit_reeval repertoire
    fit_reeval_repertoire = metric_repertoire.empty()
    fit_reeval_repertoire = fit_reeval_repertoire.add(
        repertoire.genotypes,
        repertoire.descriptors,
        fitnesses,
        extra_scores,
    )

    # Fill-in desc_reeval repertoire
    desc_reeval_repertoire = metric_repertoire.empty()
    desc_reeval_repertoire = desc_reeval_repertoire.add(
        repertoire.genotypes,
        descriptors,
        repertoire.fitnesses,
        extra_scores,
    )

    # Fill-in fit_reproducibility repertoire
    fit_reproducibility_repertoire = metric_repertoire.empty()
    fit_reproducibility_repertoire = fit_reproducibility_repertoire.add(
        repertoire.genotypes,
        repertoire.descriptors,
        fitnesses_reproducibility,
        extra_scores,
    )

    # Fill-in reeval_fit_reproducibility repertoire
    reeval_fit_reproducibility_repertoire = metric_repertoire.empty()
    reeval_fit_reproducibility_repertoire = reeval_fit_reproducibility_repertoire.add(
        repertoire.genotypes,
        descriptors,
        fitnesses_reproducibility,
        extra_scores,
    )

    # Fill-in desc_reproducibility repertoire
    desc_reproducibility_repertoire = metric_repertoire.empty()
    desc_reproducibility_repertoire = desc_reproducibility_repertoire.add(
        repertoire.genotypes,
        repertoire.descriptors,
        descriptors_reproducibility,
        extra_scores,
    )

    # Fill-in reeval_desc_reproducibility repertoire
    reeval_desc_reproducibility_repertoire = metric_repertoire.empty()
    reeval_desc_reproducibility_repertoire = reeval_desc_reproducibility_repertoire.add(
        repertoire.genotypes,
        descriptors,
        descriptors_reproducibility,
        extra_scores,
    )

    return (
        reeval_repertoire,
        fit_reeval_repertoire,
        desc_reeval_repertoire,
        fit_reproducibility_repertoire,
        reeval_fit_reproducibility_repertoire,
        desc_reproducibility_repertoire,
        reeval_desc_reproducibility_repertoire,
        random_key,
    )
