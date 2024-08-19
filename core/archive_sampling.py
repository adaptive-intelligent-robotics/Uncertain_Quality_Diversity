from __future__ import annotations

from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from qdax.core.emitters.emitter import EmitterState
from qdax.core.containers.repertoire import Repertoire
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

from core.containers.evaluations_depth_repertoire import EvaluationsDeepMapElitesRepertoire
from core.map_elites_depth import MAPElitesDepth


class ArchiveSampling(MAPElitesDepth):
    """
    Core elements of the Archive-Sampling algorithm.
    """

    @partial(jax.jit, static_argnames=("self",))
    def _scoring_repertoire_offspring(
        self,
        repertoire: Repertoire,
        genotypes: Genotype,
        random_key: RNGKey,
    ) -> Tuple[Genotype, Fitness, Descriptor, ExtraScores, RNGKey]:
        """
        Evaluate the offspring and the repertoire content all together.
        Input:
            repertoire
            genotypes: offspring to evaluate
            random_key
        Returns:
            genotypes: concatenated genotypes of repertoire and offspring
            fitnesses: corresponding fitnesses
            descriptors: corresponding descriptors
            extra_scores: corresponding extra_scores
        """

        # get the content of the repertoire
        num_centroids = repertoire.centroids.shape[0]
        num_indivs = num_centroids * self._depth
        batch_size = jax.tree_util.tree_leaves(genotypes)[0].shape[0]
        total_size = num_indivs + batch_size
        repertoire_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (num_indivs,) + x.shape[2:]),
            repertoire.genotypes_depth,
        )
        repertoire_fitnesses = jnp.reshape(
            repertoire.fitnesses_depth,
            (num_indivs,),
        )
        repertoire_descriptors = jnp.reshape(
            repertoire.descriptors_depth,
            (num_indivs, -1),
        )
        repertoire_evaluations = jnp.reshape(
            repertoire.evaluations_depth,
            (num_indivs,),
        )

        # concatenate to offspring
        all_genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            repertoire_genotypes,
            genotypes,
        )

        # evaluate
        (
            new_eval_fitnesses,
            new_eval_descriptors,
            extra_scores,
            random_key,
        ) = self._scoring_function(all_genotypes, random_key)

        # filter empty cells in repertoire
        new_eval_weights = jnp.concatenate(
            [repertoire_fitnesses > -jnp.inf, jnp.full(batch_size, 1)],
            axis=0,
        )
        new_eval_fitnesses = jnp.where(new_eval_weights, new_eval_fitnesses, -jnp.inf)

        # define concatenated array for the averaging
        previous_eval_fitnesses = jnp.concatenate(
            [
                repertoire_fitnesses,
                new_eval_fitnesses[num_indivs:],
            ],
            axis=0,
        )
        previous_eval_descriptors = jnp.concatenate(
            [
                repertoire_descriptors,
                new_eval_descriptors[num_indivs:],
            ],
            axis=0,
        )
        previous_eval_weights = jnp.concatenate(
            [repertoire_evaluations, jnp.full(batch_size, 0)],
            axis=0,
        )

        # compute average fitnesses for repertoire indivs
        all_fitnesses = jnp.concatenate(
            [new_eval_fitnesses, previous_eval_fitnesses], axis=0
        )
        weights = jnp.concatenate([new_eval_weights, previous_eval_weights], axis=0)
        all_fitnesses = jnp.reshape(all_fitnesses, (2, total_size))
        weights = jnp.reshape(weights, (2, total_size))
        fitnesses = jnp.average(all_fitnesses, axis=0, weights=weights).squeeze()
        fitnesses = jnp.where(fitnesses != fitnesses, -jnp.inf, fitnesses)  # filter nan

        # compute average descriptors for repertoire indivs
        all_descriptors = jnp.concatenate(
            [new_eval_descriptors, previous_eval_descriptors], axis=0
        )
        weights = jnp.concatenate([new_eval_weights, previous_eval_weights], axis=0)
        weights = jnp.repeat(
            weights,
            new_eval_descriptors.shape[1],
            total_repeat_length=2 * total_size * new_eval_descriptors.shape[1],
        )
        all_descriptors = jnp.reshape(all_descriptors, (2, total_size, -1))
        weights = jnp.reshape(weights, (2, total_size, -1))
        descriptors = jnp.average(all_descriptors, axis=0, weights=weights).squeeze()
        descriptors = jnp.where(
            descriptors != descriptors, 0, descriptors
        )  # filter nan

        # Set up number of evaluations
        extra_scores["num_evaluations"] = previous_eval_weights + 1

        return all_genotypes, fitnesses, descriptors, extra_scores, random_key

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[Repertoire, Optional[EmitterState], RNGKey]:
        """
        Initialize a Map-Elites grid with an initial population of genotypes. Requires
        the definition of centroids that can be computed with any method such as
        CVT or Euclidean mapping.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            random_key: a random key used for stochastic operations.

        Returns:
            an initialized MAP-Elite repertoire with the initial state of the emitter.
        """

        # num_samples evaluations of each indiv
        (
            fitnesses,
            descriptors,
            extra_scores,
            random_key,
        ) = self._scoring_function(genotypes, random_key)
        extra_scores["num_evaluations"] = jnp.full(fitnesses.shape[0], 1)

        # init repertoire
        repertoire = EvaluationsDeepMapElitesRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            centroids=centroids,
            depth=self._depth,
        )
        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=genotypes,
            random_key=random_key,
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
        )

        return repertoire, emitter_state, random_key

    @partial(jax.jit, static_argnames=("self",))
    def update(
        self,
        repertoire: Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Repertoire, Optional[EmitterState], Metrics, RNGKey]:
        """
        Performs one iteration of the Archive-Sampling algorithm, re-evaluating
        the content of the repertoire before each generation.

        Args:
            repertoire: the MAP-Elites repertoire
            emitter_state: state of the emitter
            random_key: a jax PRNG random key

        Results:
            the updated MAP-Elites repertoire
            the updated (if needed) emitter state
            metrics about the updated repertoire
            a new jax PRNG key
        """

        # generate offsprings with the emitter
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # evaluate all individuals already in the archive and offspring
        (
            all_genotypes,
            fitnesses,
            descriptors,
            extra_scores,
            random_key,
        ) = self._scoring_repertoire_offspring(repertoire, genotypes, random_key)

        # empty repertoire
        total_evaluations = repertoire.total_evaluations
        repertoire = repertoire.empty()

        # add everything back to the archive
        repertoire = repertoire.add(all_genotypes, descriptors, fitnesses, extra_scores)

        # set up the total number of evaluations
        total_indivs = jax.tree_util.tree_leaves(all_genotypes)[0].shape[0]
        repertoire = repertoire.set_total_evaluations(total_evaluations + total_indivs)

        # update emitter state after scoring is made
        batch_size = jax.tree_util.tree_leaves(genotypes)[0].shape[0]
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=fitnesses[total_indivs - batch_size :],
            descriptors=descriptors[total_indivs - batch_size :],
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key
