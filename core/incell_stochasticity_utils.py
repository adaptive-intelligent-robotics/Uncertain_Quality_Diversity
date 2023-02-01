from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Metrics, RNGKey

from core.containers.mapelites_repertoire import MapElitesRepertoire
from core.stochasticity_utils import dummy_extra_scores_extractor


@partial(
    jax.jit,
    static_argnames=(
        "scoring_fn",
        "num_reevals",
        "extra_scores_extractor",
        "use_median",
    ),
)
def _incell_reevaluation_function(
    random_key: RNGKey,
    repertoire: MapElitesRepertoire,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    num_reevals: int,
    extra_scores_extractor: Callable[
        [ExtraScores, int], ExtraScores
    ] = dummy_extra_scores_extractor,
    use_median: bool = False,
) -> Tuple[RNGKey, Genotype, Descriptor, Descriptor, Fitness, Fitness, ExtraScores]:
    """
    Perform reevaluation of a repertoire in stochastic applications.

    Args:
        random_key
        repertoire: repertoire to reevaluate.
        scoring_fn
        num_reevals
        extra_scores_extractor: function to average the extra_scores of the samples.
        use_median: use the median instead of average to compute final metric.
    Returns:
        The results of the reevaluation and a new random key.
    """

    # Get genotypes
    num_centroids = repertoire.centroids.shape[0]
    (
        all_genotypes,
        all_repertoire_fitnesses,
        _,
        random_key,
    ) = repertoire.sample_all_cells(random_key, num_reevals)

    # Eval
    all_fitnesses, all_descriptors, all_extra_scores, random_key = scoring_fn(
        all_genotypes, random_key
    )

    # Set -inf fitness for all unexisting indivs
    all_fitnesses = jnp.where(
        all_repertoire_fitnesses == -jnp.inf, -jnp.inf, all_fitnesses
    )

    # Take genotypes in repertoire.genotypes
    genotypes = repertoire.genotypes

    # Average fitnesses and descriptors
    all_fitnesses = jnp.reshape(all_fitnesses, (num_centroids, num_reevals))
    all_descriptors = jnp.reshape(all_descriptors, (num_centroids, num_reevals, -1))
    if use_median:
        descriptors = jnp.median(all_descriptors, axis=1)
        fitnesses = jnp.median(all_fitnesses, axis=1)
    else:
        descriptors = jnp.average(all_descriptors, axis=1)
        fitnesses = jnp.average(all_fitnesses, axis=1)

    # Compute variance
    descriptors_var = jnp.mean(jnp.nanstd(all_descriptors, axis=1), axis=1)
    fitnesses_var = jnp.nanstd(all_fitnesses, axis=1)

    # Set -inf fitness for all unexisting indivs
    fitnesses = jnp.where(repertoire.fitnesses == -jnp.inf, -jnp.inf, fitnesses)
    fitnesses_var = jnp.where(repertoire.fitnesses == -jnp.inf, -jnp.inf, fitnesses_var)
    descriptors_var = jnp.where(
        repertoire.fitnesses == -jnp.inf, -jnp.inf, descriptors_var
    )

    # Extract extra scores
    extra_scores = extra_scores_extractor(all_extra_scores, num_reevals)

    return (
        random_key,
        genotypes,
        descriptors,
        descriptors_var,
        fitnesses,
        fitnesses_var,
        extra_scores,
    )


@partial(
    jax.jit,
    static_argnames=(
        "scoring_fn",
        "depth",
        "num_reevals",
        "extra_scores_extractor",
        "use_median",
    ),
)
def incell_reevaluation_function(
    repertoire: MapElitesRepertoire,
    random_key: RNGKey,
    metric_repertoire: MapElitesRepertoire,
    scoring_fn: Callable[
        [Genotype, RNGKey],
        Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
    ],
    depth: int,
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
        depth: depth of the grid
        num_reevals
        extra_scores_extractor: function to average the extra_scores of the samples.
        use_median: use the median instead of average to compute final metric.
    Returns:
        The reevaluated container and a new random key.
    """

    if num_reevals == 0:
        return repertoire, repertoire, repertoire, repertoire, repertoire, random_key

    (
        random_key,
        genotypes,
        descriptors,
        descriptors_var,
        fitnesses,
        fitnesses_var,
        extra_scores,
    ) = _incell_reevaluation_function(
        random_key=random_key,
        repertoire=repertoire,
        scoring_fn=scoring_fn,
        num_reevals=num_reevals,
        extra_scores_extractor=extra_scores_extractor,
        use_median=use_median,
    )

    # Fill-in new repertoire
    reeval_repertoire = metric_repertoire.empty()
    reeval_repertoire = reeval_repertoire.add(
        genotypes,
        descriptors,
        fitnesses,
        extra_scores,
    )

    # Compute in-cell metrics for other repertoires
    _, original_descriptors, original_fitnesses, random_key = _incell_metrics(
        repertoire,
        random_key,
        depth,
        use_median,
    )

    # Fill-in fit_reeval repertoire
    fit_reeval_repertoire = metric_repertoire.empty()
    fit_reeval_repertoire = fit_reeval_repertoire.add(
        genotypes,
        original_descriptors,
        fitnesses,
        extra_scores,
    )

    # Fill-in desc_reeval repertoire
    desc_reeval_repertoire = metric_repertoire.empty()
    desc_reeval_repertoire = desc_reeval_repertoire.add(
        genotypes,
        descriptors,
        original_fitnesses,
        extra_scores,
    )

    # Fill-in fit_var repertoire
    fit_var_repertoire = metric_repertoire.empty()
    fit_var_repertoire = fit_var_repertoire.add(
        genotypes,
        original_descriptors,
        fitnesses_var,
        extra_scores,
    )

    # Fill-in desc_var repertoire
    desc_var_repertoire = metric_repertoire.empty()
    desc_var_repertoire = desc_var_repertoire.add(
        genotypes,
        original_descriptors,
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


@partial(
    jax.jit,
    static_argnames=(
        "depth",
        "use_median",
        "use_closer",
    ),
)
def _incell_metrics(
    repertoire: MapElitesRepertoire,
    random_key: RNGKey,
    depth: int,
    use_median: bool = False,
    use_closer: bool = False,
) -> Tuple[Genotype, Fitness, Descriptor, RNGKey]:
    """
    Compute cell-metrics in stochastic applications.

    Args:
        repertoire: repertoire to reevaluate.
        random_key
        depth: depth of the repertoire
        use_median: use the median instead of average to compute final metric.
        use_closer: use the individuals closest to median instead of average
            to compute final metric (WARNING: overriding use_median)
    Returns:
        The cell-metrics and a new random key
    """

    # Sample depth fitnesses and descriptors for each cell
    num_centroids = repertoire.centroids.shape[0]
    _, all_fitnesses, all_descriptors, random_key = repertoire.sample_all_cells(
        random_key, depth
    )

    # Take genotypes in repertoire.genotypes
    genotypes = repertoire.genotypes

    # Average fitnesses and descriptors per cell for metrics computation
    all_fitnesses = jnp.reshape(all_fitnesses, (num_centroids, depth))
    all_descriptors = jnp.reshape(all_descriptors, (num_centroids, depth, -1))
    if use_closer:

        def one_dim_closer_median(values: jnp.ndarray) -> jnp.ndarray:
            def distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
                return jnp.sqrt(jnp.sum(jnp.square(x - y)))

            distances = jax.vmap(
                jax.vmap(partial(distance), in_axes=(None, 0)), in_axes=(0, None)
            )(values, values)
            return values[jnp.argmin(jnp.mean(distances, axis=0))]

        descriptors = jax.vmap(one_dim_closer_median)(all_descriptors)
        fitnesses = jax.vmap(one_dim_closer_median)(all_fitnesses)
    elif use_median:
        descriptors = jnp.median(all_descriptors, axis=1)
        fitnesses = jnp.median(all_fitnesses, axis=1)
    else:
        descriptors = jnp.average(all_descriptors, axis=1)
        fitnesses = jnp.average(all_fitnesses, axis=1)

    return genotypes, descriptors, fitnesses, random_key


@partial(
    jax.jit,
    static_argnames=(
        "metrics_function",
        "depth",
        "use_median",
        "use_closer",
    ),
)
def metrics_incell_wrapper(
    repertoire: MapElitesRepertoire,
    random_key: RNGKey,
    metrics_function: Callable[[MapElitesRepertoire], Metrics],
    depth: int,
    use_median: bool = False,
    use_closer: bool = False,
) -> Tuple[Metrics, RNGKey]:
    """
    Perform evaluation of a repertoire in stochastic applications.

    Args:
        repertoire: repertoire to reevaluate.
        random_key
        metrics_function: function to compute metrics of a repertoire
        depth: depth of the repertoire
        use_median: use the median instead of average to compute final metric.
        use_closer: use the individuals closest to median instead of average
            to compute final metric (WARNING: overriding use_median)
    Returns:
        The metrics container and a new random key.
    """

    # Compute the in-cell metrics
    genotypes, descriptors, fitnesses, random_key = _incell_metrics(
        repertoire,
        random_key,
        depth,
        use_median,
        use_closer,
    )

    # Fill-in new repertoire
    metrics_repertoire = repertoire.empty()
    metrics_repertoire = metrics_repertoire.add(
        genotypes,
        descriptors,
        fitnesses,
        {},
    )

    metrics = metrics_function(metrics_repertoire)

    return metrics, random_key


@partial(
    jax.jit,
    static_argnames=(
        "metrics_function",
        "depth",
        "use_median",
        "use_closer",
    ),
)
def metrics_incell_random_wrapper(
    repertoire: MapElitesRepertoire,
    random_key: RNGKey,
    metrics_function: Callable[[MapElitesRepertoire], Metrics],
    depth: int,
    use_median: bool = False,
    use_closer: bool = False,
) -> Tuple[Metrics, RNGKey]:
    """
    Perform evaluation of a repertoire in stochastic applications.

    Args:
        repertoire: repertoire to reevaluate.
        random_key
        metrics_function: function to compute metrics of a repertoire
        depth: depth of the repertoire
        use_median: use the median instead of average to compute final metric.
        use_closer: use the individuals closest to median instead of average
            to compute final metric (WARNING: overriding use_median)
    Returns:
        The metrics container and a new random key.
    """

    # Compute the in-cell metrics
    genotypes, descriptors, fitnesses, random_key = _incell_metrics(
        repertoire,
        random_key,
        depth,
        use_median,
        use_closer,
    )

    # Fill-in new repertoire
    metrics_repertoire = repertoire.empty()
    metrics_repertoire, random_key = metrics_repertoire.add(
        genotypes,
        descriptors,
        fitnesses,
        {},
        random_key,
    )

    metrics = metrics_function(metrics_repertoire)

    return metrics, random_key
