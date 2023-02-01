from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from core.containers.mapelites_repertoire import MapElitesRepertoire


@partial(jax.jit, static_argnames=("num_samples"))
def dummy_extra_scores_extractor(
    extra_scores: ExtraScores,
    num_samples: int,
) -> ExtraScores:
    """
    Extract the final extra scores of a policy from multiple samples of
    the same policy in the environment.
    This Dummy implementation just return the full concatenate extra_score
    of all samples without extra computation.

    Args:
        extra_scores: extra scores of the samples
        num_samples: the number of samples used

    Returns:
        the new extra scores after extraction
    """
    return extra_scores


@partial(
    jax.jit,
    static_argnames=(
        "scoring_fn",
        "num_samples",
        "extra_scores_extractor",
        "use_median",
    ),
)
def sampling(
    policies_params: Genotype,
    random_key: RNGKey,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    num_samples: int,
    extra_scores_extractor: Callable[
        [ExtraScores, int], ExtraScores
    ] = dummy_extra_scores_extractor,
    use_median: bool = False,
) -> Tuple[Fitness, Fitness, Descriptor, Descriptor, ExtraScores, RNGKey]:
    """
    Wrap scoring_function to perform sampling.

    Args:
        policies_params: policies to evaluate
        random_key
        scoring_fn: scoring function used for evaluation
        num_samples
        extra_scores_extractor: function to extract the extra_scores from
            multiple samples of the same policy.
        use_median: use the median instead of average to compute final score.

    Returns:
        The new fitness and descriptor of the individuals
        The fitness and descriptor variance of the individuals
        The extra_score extract from samples with extra_scores_extractor
        A new random key
    """

    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=num_samples)

    # evaluate
    sample_scoring_fn = jax.vmap(scoring_fn, (None, 0), 1)
    all_fitnesses, all_descriptors, all_extra_scores, _ = sample_scoring_fn(
        policies_params, keys
    )

    # compute new results
    if use_median:
        descriptors = jnp.median(all_descriptors, axis=1)
        fitnesses = jnp.median(all_fitnesses, axis=1)
    else:
        descriptors = jnp.average(all_descriptors, axis=1)
        fitnesses = jnp.average(all_fitnesses, axis=1)

    # compute variance
    descriptors_var = jnp.mean(jnp.nanstd(all_descriptors, axis=1), axis=1)
    fitnesses_var = jnp.nanstd(all_fitnesses, axis=1)

    # extract extra scores and add number of evaluations to it
    extra_scores = extra_scores_extractor(all_extra_scores, num_samples)
    extra_scores["num_evaluations"] = jnp.full(fitnesses.shape[0], num_samples)

    return (
        fitnesses,
        fitnesses_var,
        descriptors,
        descriptors_var,
        extra_scores,
        random_key,
    )


@partial(
    jax.jit,
    static_argnames=(
        "scoring_fn",
        "num_reevals",
        "extra_scores_extractor",
        "use_median",
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
    extra_scores_extractor: Callable[
        [ExtraScores, int], ExtraScores
    ] = dummy_extra_scores_extractor,
    use_median: bool = False,
) -> Tuple[
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
        random_key
        scoring_fn
        num_reevals
        extra_scores_extractor: function to average the extra_scores of the samples.
        use_median: use the median instead of average to compute final score.
    Returns:
        The reevaluated container and a new random key.
    """

    if num_reevals == 0:
        return repertoire, repertoire, repertoire, repertoire, repertoire, random_key

    # Eval
    (
        fitnesses,
        fitnesses_var,
        descriptors,
        descriptors_var,
        extra_scores,
        random_key,
    ) = sampling(
        policies_params=repertoire.genotypes,
        random_key=random_key,
        scoring_fn=scoring_fn,
        num_samples=num_reevals,
        extra_scores_extractor=extra_scores_extractor,
        use_median=use_median,
    )

    # Set -inf fitness for all unexisting indivs
    fitnesses = jnp.where(repertoire.fitnesses == -jnp.inf, -jnp.inf, fitnesses)
    fitnesses_var = jnp.where(repertoire.fitnesses == -jnp.inf, -jnp.inf, fitnesses_var)
    descriptors_var = jnp.where(
        repertoire.fitnesses == -jnp.inf, -jnp.inf, descriptors_var
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

    # Fill-in fit_var repertoire
    fit_var_repertoire = metric_repertoire.empty()
    fit_var_repertoire = fit_var_repertoire.add(
        repertoire.genotypes,
        repertoire.descriptors,
        fitnesses_var,
        extra_scores,
    )

    # Fill-in desc_var repertoire
    desc_var_repertoire = metric_repertoire.empty()
    desc_var_repertoire = desc_var_repertoire.add(
        repertoire.genotypes,
        repertoire.descriptors,
        descriptors_var,
        extra_scores,
    )
    return (
        reeval_repertoire,
        fit_reeval_repertoire,
        desc_reeval_repertoire,
        fit_var_repertoire,
        desc_var_repertoire,
        random_key,
    )
