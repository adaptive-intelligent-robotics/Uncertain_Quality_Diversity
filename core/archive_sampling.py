from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

from core.containers.archive_sampling_repertoire import ArchiveSamplingRepertoire
from core.map_elites_depth import MAPElitesDepth
from core.sampling import multi_sample_scoring_function


class ArchiveSampling(MAPElitesDepth):
    """
    Core elements of the Archive-Sampling algorithm storing all replications to allow any estimator.
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        depth: int,
        num_iterations: int,
        num_samples: int,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._depth = depth
        self._max_total_evals = num_iterations + num_samples
        self._offspring_num_samples = num_samples
        self._fitness_extractor = fitness_extractor
        self._fitness_reproducibility_extractor = fitness_reproducibility_extractor
        self._descriptor_extractor = descriptor_extractor
        self._descriptor_reproducibility_extractor = (
            descriptor_reproducibility_extractor
        )

    @partial(jax.jit, static_argnames=("self",))
    def _add_repertoire(
        self,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        descriptors: Descriptor,
        fitnesses: Fitness,
        extra_scores: ExtraScores,
    ) -> MapElitesRepertoire:
        print("Adding in ArchiveSampling")

        return repertoire.add(
            batch_of_genotypes=genotypes,
            batch_of_all_descriptors=descriptors,
            batch_of_all_fitnesses=fitnesses,
            batch_of_extra_scores=extra_scores,
            fitness_extractor=self._fitness_extractor,
            fitness_reproducibility_extractor=self._fitness_reproducibility_extractor,
            descriptor_extractor=self._descriptor_extractor,
            descriptor_reproducibility_extractor=self._descriptor_reproducibility_extractor,
        )

    @partial(jax.jit, static_argnames=("self",))
    def _init_repertoire(
        self,
        genotypes: Genotype,
        descriptors: Descriptor,
        fitnesses: Fitness,
        extra_scores: ExtraScores,
        centroids: Centroid,
    ) -> MapElitesRepertoire:
        print("Initialisating in ArchiveSampling")

        return ArchiveSamplingRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            centroids=centroids,
            depth=self._depth,
            num_evals=self._max_total_evals,
            fitness_extractor=self._fitness_extractor,
            fitness_reproducibility_extractor=self._fitness_reproducibility_extractor,
            descriptor_extractor=self._descriptor_extractor,
            descriptor_reproducibility_extractor=self._descriptor_reproducibility_extractor,
        )

    @partial(jax.jit, static_argnames=("self", "num_samples"))
    def _scoring_repertoire_offspring(
        self,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        num_samples: int,
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

        # evaluate the new offspring
        batch_size = jax.tree_util.tree_leaves(genotypes)[0].shape[0]
        (fitnesses, descriptors, _, random_key,) = multi_sample_scoring_function(
            genotypes, random_key, self._scoring_function, num_samples
        )

        # extand to the good shape
        fitnesses = jnp.pad(
            fitnesses,
            ((0, 0), (0, self._max_total_evals - num_samples)),
            "constant",
            constant_values=jnp.nan,
        )
        descriptors = jnp.pad(
            descriptors,
            ((0, 0), (0, self._max_total_evals - num_samples), (0, 0)),
            "constant",
            constant_values=jnp.nan,
        )

        # re-evaluate one time the content of the repertoire
        (
            repertoire_fitnesses,
            repertoire_descriptors,
            _,
            random_key,
        ) = self._scoring_function(repertoire.genotypes_depth, random_key)

        # filter empty cells in repertoire
        repertoire_fitnesses = jnp.where(
            repertoire.fitnesses_depth > -jnp.inf, repertoire_fitnesses, -jnp.inf
        )

        # add to the existing evaluations in the archive
        repertoire_fitnesses = jax.lax.dynamic_update_slice(
            repertoire.fitnesses_depth_all,
            jnp.expand_dims(repertoire_fitnesses, axis=1),
            (0, self._max_total_evals - 1),
        )
        repertoire_descriptors = jax.lax.dynamic_update_slice(
            repertoire.descriptors_depth_all,
            jnp.expand_dims(repertoire_descriptors, axis=1),
            (0, self._max_total_evals - 1, 0),
        )

        # sort everything to put the nan at the end before the next loop
        index = jnp.argsort(repertoire_fitnesses, axis=1)
        repertoire_fitnesses = jnp.take_along_axis(repertoire_fitnesses, index, axis=1)
        index = jnp.repeat(
            jnp.expand_dims(index, axis=2), repertoire_descriptors.shape[2], axis=2
        )
        repertoire_descriptors = jnp.take_along_axis(
            repertoire_descriptors, index, axis=1
        )

        # set up number of evaluations
        extra_scores = {}
        extra_scores["num_evaluations"] = jnp.concatenate(
            [
                repertoire.evaluations_depth + 1,
                num_samples * jnp.ones((batch_size)),
            ],
            axis=0,
        )

        # finally concatenate everything for addition
        all_genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            genotypes,
            repertoire.genotypes_depth,
        )
        all_fitnesses = jnp.concatenate([fitnesses, repertoire_fitnesses], axis=0)
        all_descriptors = jnp.concatenate([descriptors, repertoire_descriptors], axis=0)

        return all_genotypes, all_fitnesses, all_descriptors, extra_scores, random_key

    @partial(jax.jit, static_argnames=("self",))
    def init(
        self,
        genotypes: Genotype,
        centroids: Centroid,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, Optional[EmitterState], RNGKey]:
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
        ) = multi_sample_scoring_function(
            genotypes, random_key, self._scoring_function, self._offspring_num_samples
        )
        extra_scores["num_evaluations"] = jnp.full(
            fitnesses.shape[0], self._offspring_num_samples
        )

        # extend all evaluations vector to teh good shape with jnp.nan
        fitnesses = jnp.pad(
            fitnesses,
            ((0, 0), (0, self._max_total_evals - self._offspring_num_samples)),
            "constant",
            constant_values=jnp.nan,
        )
        descriptors = jnp.pad(
            descriptors,
            ((0, 0), (0, self._max_total_evals - self._offspring_num_samples), (0, 0)),
            "constant",
            constant_values=jnp.nan,
        )

        # init repertoire
        repertoire = self._init_repertoire(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            centroids=centroids,
        )

        # get initial state of the emitter
        emitter_state, random_key = self._emitter.init(
            init_genotypes=genotypes, random_key=random_key
        )

        # update emitter state
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=self._fitness_extractor(fitnesses),
            descriptors=self._descriptor_extractor(descriptors),
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
        ) = self._scoring_repertoire_offspring(
            repertoire, genotypes, self._offspring_num_samples, random_key
        )

        # empty repertoire
        total_evaluations = repertoire.total_evaluations
        repertoire = repertoire.empty()

        # add everything back to the archive
        repertoire = self._add_repertoire(
            repertoire=repertoire,
            genotypes=all_genotypes,
            descriptors=descriptors,
            fitnesses=fitnesses,
            extra_scores=extra_scores,
        )

        # set up the total number of evaluations
        total_indivs = jax.tree_util.tree_leaves(all_genotypes)[0].shape[0]
        repertoire = repertoire.set_total_evaluations(total_evaluations + total_indivs)

        # update emitter state after scoring is made
        batch_size = jax.tree_util.tree_leaves(genotypes)[0].shape[0]
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=self._fitness_extractor(fitnesses[total_indivs - batch_size :]),
            descriptors=self._descriptor_extractor(
                descriptors[total_indivs - batch_size :]
            ),
            extra_scores=extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key
