from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.map_elites import MAPElites
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

from core.containers.mels_repertoire import MELSRepertoire, _mode, get_cells_indices
from core.sampling import multi_sample_scoring_function


class MELS(MAPElites):
    """Core elements of the MAP-Elites Low-Spread algorithm.

    Most methods in this class are inherited from MAPElites.

    The same scoring function can be passed into both MAPElites and this class.
    We have overridden __init__ such that it takes in the scoring function and
    wraps it such that every solution is evaluated `num_samples` times.

    We also overrode the init method to use the MELSRepertoire instead of
    MapElitesRepertoire.
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MELSRepertoire], Metrics],
        num_samples: int,
    ) -> None:
        self._scoring_function = partial(
            multi_sample_scoring_function,
            scoring_fn=scoring_function,
            num_samples=num_samples,
        )
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._num_samples = num_samples

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[MELSRepertoire, Optional[EmitterState], RNGKey]:
        """Initialize a MAP-Elites Low-Spread repertoire with an initial
        population of genotypes. Requires the definition of centroids that can
        be computed with any method such as CVT or Euclidean mapping.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tessellation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            A tuple of (initialized MAP-Elites Low-Spread repertoire, initial emitter
            state, JAX random key).
        """
        # score initial genotypes
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # init the repertoire
        repertoire = MELSRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=genotypes, random_key=random_key
        )

        # update emitter state
        batch_size, num_samples, _ = descriptors.shape
        cells = get_cells_indices(
            descriptors.reshape(batch_size * num_samples, -1), repertoire.centroids
        ).reshape((batch_size, num_samples))
        cells = jax.vmap(_mode)(cells)[:, None]
        extracted_descriptors = jnp.take_along_axis(repertoire.centroids, cells, axis=0)
        extracted_fitnesses = fitnesses.mean(axis=-1, keepdims=True)
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=extracted_fitnesses,
            descriptors=extracted_descriptors,
            extra_scores=extra_scores,
        )
        return repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], Metrics, RNGKey]:
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
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # scores the offsprings
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        # add genotypes in the repertoire
        repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        # update emitter state after scoring is made
        batch_size, num_samples, _ = descriptors.shape
        cells = get_cells_indices(
            descriptors.reshape(batch_size * num_samples, -1), repertoire.centroids
        ).reshape((batch_size, num_samples))
        cells = jax.vmap(_mode)(cells)[:, None]
        extracted_descriptors = jnp.take_along_axis(repertoire.centroids, cells, axis=0)
        extracted_fitnesses = fitnesses.mean(axis=-1, keepdims=True)
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=extracted_fitnesses,
            descriptors=extracted_descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key
