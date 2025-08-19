from __future__ import annotations

from typing import Callable, Optional, Tuple

import jax.numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

from core.containers.adaptive_sampling_repertoire import AdaptiveSamplingRepertoire


class AdaptiveSampling:
    """
    Core elements of the Archive-Sampling algorithm.
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[Repertoire], Metrics],
        depth: int,
        max_number_evals: int,
        sampling_size: int,
        num_descriptors: int,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._depth = depth
        self._sampling_size = sampling_size
        self._num_descriptors = num_descriptors
        self._max_number_evals = max_number_evals
        self._fitness_extractor = fitness_extractor
        self._fitness_reproducibility_extractor = fitness_reproducibility_extractor
        self._descriptor_extractor = descriptor_extractor
        self._descriptor_reproducibility_extractor = (
            descriptor_reproducibility_extractor
        )

    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[AdaptiveSamplingRepertoire, Optional[EmitterState], RNGKey]:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            An initialized MAP-Elite repertoire with the initial state of the emitter,
            and a random key.
        """

        # init the repertoire
        repertoire = AdaptiveSamplingRepertoire.init(
            genotypes=genotypes,
            centroids=centroids,
            depth=self._depth,
            num_evals=self._max_number_evals,
            fitness_extractor=self._fitness_extractor,
            fitness_reproducibility_extractor=self._fitness_reproducibility_extractor,
            descriptor_extractor=self._descriptor_extractor,
            descriptor_reproducibility_extractor=self._descriptor_reproducibility_extractor,
            random_key=random_key,
            scoring_fn=self._scoring_function,
            maximum_evals_per_generation=self._sampling_size,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            random_key=random_key,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=jnp.zeros((self._emitter._batch_size,)),
            descriptors=jnp.zeros((self._emitter._batch_size, self._num_descriptors)),
            extra_scores={},
        )

        return repertoire, emitter_state, random_key

    def update(
        self,
        repertoire: AdaptiveSamplingRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[AdaptiveSamplingRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.


        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """

        # generate offsprings with the emitter
        genotypes, extra_info, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # add genotypes in the repertoire
        repertoire = repertoire.add(
            batch_of_genotypes=genotypes,
            fitness_extractor=self._fitness_extractor,
            fitness_reproducibility_extractor=self._fitness_reproducibility_extractor,
            descriptor_extractor=self._descriptor_extractor,
            descriptor_reproducibility_extractor=self._descriptor_reproducibility_extractor,
            scoring_fn=self._scoring_function,
        )

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=jnp.zeros((self._emitter._batch_size,)),
            descriptors=jnp.zeros((self._emitter._batch_size, self._num_descriptors)),
            extra_scores={},
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key
