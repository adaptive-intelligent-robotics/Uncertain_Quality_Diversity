from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
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

from core.containers.mapelites_delta_reprod_repertoire import (
    MapElitesDeltaReprodRepertoire,
)


class MAPElitesDeltaReprod(MAPElites):
    """Core elements of the MAP-Elites Reproducibility algorithm."""

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesDeltaReprodRepertoire], Metrics],
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        use_weighting: bool,
        delta_fitness: float,
        delta_reproducibility: float,
        rho: float,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._fitness_extractor = fitness_extractor
        self._fitness_reproducibility_extractor = fitness_reproducibility_extractor
        self._descriptor_extractor = descriptor_extractor
        self._descriptor_reproducibility_extractor = (
            descriptor_reproducibility_extractor
        )
        self._use_weighting = use_weighting
        self._delta_fitness = delta_fitness
        self._delta_reproducibility = delta_reproducibility
        self._rho = rho

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[MapElitesDeltaReprodRepertoire, Optional[EmitterState], RNGKey]:
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
        # jax.debug.print("descriptors {x}", x=descriptors)

        # init the repertoire
        repertoire = MapElitesDeltaReprodRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            centroids=centroids,
            extra_scores=extra_scores,
            fitness_extractor=self._fitness_extractor,
            fitness_reproducibility_extractor=self._fitness_reproducibility_extractor,
            descriptor_extractor=self._descriptor_extractor,
            descriptor_reproducibility_extractor=self._descriptor_reproducibility_extractor,
            use_weighting=self._use_weighting,
            delta_fitness=self._delta_fitness,
            delta_reproducibility=self._delta_reproducibility,
            rho=self._rho,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=genotypes, random_key=random_key
        )

        # update emitter state
        extracted_descriptors = self._descriptor_extractor(descriptors)
        extracted_fitnesses = self._fitness_extractor(fitnesses)
        # jax.debug.print("extracted_descriptors {x}", x=extracted_descriptors)
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
        repertoire: MapElitesDeltaReprodRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[MapElitesDeltaReprodRepertoire, Optional[EmitterState], Metrics, RNGKey]:
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
        # jax.debug.print("descriptors {x}", x=descriptors)

        # add genotypes in the repertoire
        repertoire = repertoire.add(
            batch_of_genotypes=genotypes,
            batch_of_descriptors=descriptors,
            batch_of_fitnesses=fitnesses,
            batch_of_extra_scores=extra_scores,
            fitness_extractor=self._fitness_extractor,
            fitness_reproducibility_extractor=self._fitness_reproducibility_extractor,
            descriptor_extractor=self._descriptor_extractor,
            descriptor_reproducibility_extractor=self._descriptor_reproducibility_extractor,
            use_weighting=self._use_weighting,
            delta_fitness=self._delta_fitness,
            delta_reproducibility=self._delta_reproducibility,
            rho=self._rho,
        )

        # update emitter state after scoring is made
        extracted_descriptors = self._descriptor_extractor(descriptors)
        # jax.debug.print("extracted_descriptors {x}", x=extracted_descriptors)
        extracted_fitnesses = self._fitness_extractor(fitnesses)
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
