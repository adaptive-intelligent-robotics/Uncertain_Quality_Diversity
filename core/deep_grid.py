"""Core components of the MAP-Elites algorithm."""
from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import EmitterState
from qdax.custom_types import Centroid, Genotype, RNGKey

from core.containers.deep_grid_repertoire import DeepGridRepertoire
from core.map_elites_depth import MAPElitesDepth


class DeepGrid(MAPElitesDepth):
    """
    Core elements of Deep-Grid.
    """

    @partial(jax.jit, static_argnames=("self"))
    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[Repertoire, Optional[EmitterState], RNGKey]:
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

        repertoire = DeepGridRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            centroids=centroids,
            depth=self._depth,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            random_key=random_key,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state, random_key
