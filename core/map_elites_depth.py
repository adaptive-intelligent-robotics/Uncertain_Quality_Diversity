"""Core components of the MAP-Elites algorithm."""
from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
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

from core.containers.mapelites_depth_repertoire import DepthMapElitesRepertoire
from core.sampling import sampling


class MAPElitesDepth(MAPElites):
    """
    Core elements of the MAP-Elites algorithm with depth.
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        depth: int,
        num_samples: int,
        fitness_extractor: Callable[[Fitness], Fitness],
        descriptor_extractor: Callable[[Descriptor], Descriptor],
    ) -> None:
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._depth = depth

        self._scoring_function = partial(
            sampling,
            scoring_fn=scoring_function,
            num_samples=num_samples,
            fitness_extractor=fitness_extractor,
            descriptor_extractor=descriptor_extractor,
        )

    @partial(jax.jit, static_argnames=("self"))
    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey]:
        """
        Initialize a deep Map-Elites grid with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            initialized deep MAP-Elite repertoire with the initial state of the emitter.
        """
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            genotypes, random_key
        )

        repertoire = DepthMapElitesRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            centroids=centroids,
            depth=self._depth,
        )

        # get initial state of the emitter
        emit_genotypes = jax.tree_util.tree_map(
            lambda x: x.at[: self._emitter.batch_size].get(),
            genotypes,
        )
        emit_fitnesses = fitnesses.at[: self._emitter.batch_size].get()
        emit_descriptors = descriptors.at[: self._emitter.batch_size].get()
        emit_extra_scores = jax.tree_util.tree_map(
            lambda x: x.at[: self._emitter.batch_size].get(),
            extra_scores,
        )
        emitter_state, random_key = self._emitter.init(
            random_key=random_key,
            repertoire=repertoire,
            genotypes=emit_genotypes,
            fitnesses=emit_fitnesses,
            descriptors=emit_descriptors,
            extra_scores=emit_extra_scores,
        )

        return repertoire, emitter_state, random_key
