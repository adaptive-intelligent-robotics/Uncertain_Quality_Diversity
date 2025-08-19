"""Core components of the MAP-Elites Low-Spread algorithm."""
from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.map_elites import MAPElites
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

from core.containers.mome_reprod_biased_repertoire import MOMEReprodBiasedRepertoire
from core.containers.mome_reprod_repertoire import MOMEReprodRepertoire
from core.sampling import multi_sample_scoring_function


class MOMEReprod(MAPElites):
    """Core elements of the MAP-Elites Reproducibility algorithm."""

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MOMEReprodRepertoire], Metrics],
        num_samples: int,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        biased_sampling: bool = False,
        pareto_front_max_length: int = 50,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._num_samples = num_samples
        self._fitness_extractor = fitness_extractor
        self._fitness_reproducibility_extractor = fitness_reproducibility_extractor
        self._descriptor_extractor = descriptor_extractor
        self._descriptor_reproducibility_extractor = (
            descriptor_reproducibility_extractor
        )
        self._biased_sampling = biased_sampling
        self._pareto_front_max_length = pareto_front_max_length

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[MOMEReprodRepertoire, Optional[EmitterState], RNGKey]:
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
        (
            fitnesses,
            descriptors,
            extra_scores,
            random_key,
        ) = multi_sample_scoring_function(
            genotypes, random_key, self._scoring_function, self._num_samples
        )

        if self._biased_sampling:
            repertoire = MOMEReprodBiasedRepertoire.init(
                genotypes=genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors,
                centroids=centroids,
                extra_scores=extra_scores,
                fitness_extractor=self._fitness_extractor,
                fitness_reproducibility_extractor=self._fitness_reproducibility_extractor,
                descriptor_extractor=self._descriptor_extractor,
                descriptor_reproducibility_extractor=self._descriptor_reproducibility_extractor,
                pareto_front_max_length=self._pareto_front_max_length,
            )
        else:
            repertoire = MOMEReprodRepertoire.init(
                genotypes=genotypes,
                fitnesses=fitnesses,
                descriptors=descriptors,
                centroids=centroids,
                extra_scores=extra_scores,
                fitness_extractor=self._fitness_extractor,
                fitness_reproducibility_extractor=self._fitness_reproducibility_extractor,
                descriptor_extractor=self._descriptor_extractor,
                descriptor_reproducibility_extractor=self._descriptor_reproducibility_extractor,
                pareto_front_max_length=self._pareto_front_max_length,
            )

        # get initial state of the emitter
        extracted_descriptors = self._descriptor_extractor(descriptors)
        extracted_fitnesses = self._fitness_extractor(fitnesses)
        emitter_state, random_key = self._emitter.init(
            random_key=random_key,
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
        repertoire: MOMEReprodRepertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[MOMEReprodRepertoire, Optional[EmitterState], Metrics, RNGKey]:
        """
        Performs one iteration of the MAP-Elites algorithm.
        1. A batch of genotypes is sampled in the repertoire and the genotypes
            are copied.
        2. The copies are mutated and crossed-over
        3. The obtained offsprings are scored and then added to the repertoire.


        Args:
            repertoire: the MOME repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Returns:
            the updated MOME repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """

        # generate offsprings with the emitter
        genotypes, _, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # scores the offsprings
        (
            fitnesses,
            descriptors,
            extra_scores,
            random_key,
        ) = multi_sample_scoring_function(
            genotypes, random_key, self._scoring_function, self._num_samples
        )

        # add genotypes in the repertoire
        repertoire = repertoire.add(
            batch_of_genotypes=genotypes,
            batch_of_all_descriptors=descriptors,
            batch_of_all_fitnesses=fitnesses,
            batch_of_extra_scores=extra_scores,
            fitness_extractor=self._fitness_extractor,
            fitness_reproducibility_extractor=self._fitness_reproducibility_extractor,
            descriptor_extractor=self._descriptor_extractor,
            descriptor_reproducibility_extractor=self._descriptor_reproducibility_extractor,
        )

        # update emitter state after scoring is made
        extracted_descriptors = self._descriptor_extractor(descriptors)
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
