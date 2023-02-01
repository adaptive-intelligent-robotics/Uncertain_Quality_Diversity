"""Core components of the MAP-Elites algorithm."""
from __future__ import annotations

import os
from functools import partial
from typing import Optional, Tuple

import jax
from qdax.core.emitters.emitter import EmitterState
from qdax.types import Centroid, Genotype, Metrics, RNGKey

from core.containers.deep_grid_repertoire import DeepGridRepertoire
from core.containers.mapelites_repertoire import MapElitesRepertoire
from core.incell_stochasticity_utils import metrics_incell_random_wrapper
from core.map_elites_depth import MAPElitesDepth

# Limit CPU usage for HPC
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=4"
)


class DeepGrid(MAPElitesDepth):
    """
    Core elements of the MAP-Elites algorithm with depth.
    """

    @partial(jax.jit, static_argnames=("self"))
    def init(
        self,
        init_genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey]:
        """
        Initialize a deep Map-Elites grid with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Args:
            init_genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            initialized deep MAP-Elite repertoire with the initial state of the emitter.
        """
        fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
            init_genotypes, random_key
        )

        repertoire, random_key = DeepGridRepertoire.init(
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            centroids=centroids,
            depth=self._depth,
            random_key=random_key,
        )
        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=init_genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=init_genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self"))
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
        repertoire, random_key = repertoire.add(
            genotypes, descriptors, fitnesses, extra_scores, random_key
        )

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics, random_key = metrics_incell_random_wrapper(
            repertoire=repertoire,
            random_key=random_key,
            metrics_function=self._metrics_function,
            depth=self._depth,
        )

        return repertoire, emitter_state, metrics, random_key
