from __future__ import annotations

from functools import partial
from typing import Callable, Optional, Tuple

import jax
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

from core.containers.archive_sampling_repertoire import ArchiveSamplingRepertoire
from core.sampling import multi_sample_scoring_function


class ArchiveSampling:
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
        num_samples: int,
        repertoire_num_samples: int,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._depth = depth
        self._max_number_evals = max_number_evals
        self._num_samples = num_samples
        self._repertoire_num_samples = repertoire_num_samples
        self._fitness_extractor = fitness_extractor
        self._fitness_reproducibility_extractor = fitness_reproducibility_extractor
        self._descriptor_extractor = descriptor_extractor
        self._descriptor_reproducibility_extractor = (
            descriptor_reproducibility_extractor
        )

    @partial(jax.jit, static_argnames=("self",))
    def _add_repertoire(
        self,
        repertoire: Repertoire,
        genotypes: Genotype,
        descriptors: Descriptor,
        fitnesses: Fitness,
        extra_scores: ExtraScores,
    ) -> Repertoire:

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
        random_key: RNGKey,
    ) -> Repertoire:

        return ArchiveSamplingRepertoire.init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            centroids=centroids,
            depth=self._depth,
            num_evals=self._max_number_evals,
            fitness_extractor=self._fitness_extractor,
            fitness_reproducibility_extractor=self._fitness_reproducibility_extractor,
            descriptor_extractor=self._descriptor_extractor,
            descriptor_reproducibility_extractor=self._descriptor_reproducibility_extractor,
        )

    @partial(jax.jit, static_argnames=("self"))
    def _extract_repertoire(
        self,
        repertoire: Repertoire,
        random_key: RNGKey,
    ) -> Tuple[Repertoire, Genotype, Fitness, Descriptor, RNGKey]:
        """
        Extract the part of the repertoire that needs to be evaluated.
        The assumption is that these are removed from the repertoire.

        Input:
            repertoire
            random_key
        Returns:
            repertoire: repertoire after extraction
            extract_genotypes: the genotypes extracted from the repertoire
            extract_fitnesses: the corresponding fitnesses, also extracted
            extract_descriptors: the corresponding descriptors, also extracted
            random_key
        """

        # Extract the content of the archive
        num_centroids = repertoire.centroids.shape[0]
        extract_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (num_centroids * self._depth,) + x.shape[2:]),
            repertoire.genotypes_depth,
        )
        extract_fitnesses = jnp.reshape(
            repertoire.fitnesses_depth_all,
            (num_centroids * self._depth, self._max_number_evals),
        )
        extract_descriptors = jnp.reshape(
            repertoire.descriptors_depth_all,
            (num_centroids * self._depth, self._max_number_evals, -1),
        )

        # Empty the repertoire
        repertoire = repertoire.empty()

        return (
            repertoire,
            extract_genotypes,
            extract_fitnesses,
            extract_descriptors,
            random_key,
        )

    @partial(jax.jit, static_argnames=("self", "num_samples", "repertoire_num_samples"))
    def _evaluate(
        self,
        emit_genotypes: Genotype,
        extract_genotypes: Genotype,
        extract_fitnesses: Fitness,
        extract_descriptors: Descriptor,
        num_samples: int,
        repertoire_num_samples: int,
        random_key: RNGKey,
    ) -> Tuple[
        Genotype,
        Fitness,
        Descriptor,
        ExtraScores,
        Fitness,
        Descriptor,
        ExtraScores,
        RNGKey,
    ]:
        """
        Evaluate emited and extracted genotypes all together and centralise
        all the evaluations.
        Input:
            emit_genotypes
            extract_genotypes
            extract_fitnesses
            extract_descriptors
            random_key
        Returns:
            genotypes: concatenated genotypes
            fitnesses: corresponding fitnesses
            descriptors: corresponding descriptors
            extra_scores: corresponding extra_scores
            emit_fitnesses
            emit_descriptors
            emit_extra_scores
            random_key
        """

        # Create a genotypes to match num_samples and repertoire_num_samples evaluations
        evaluation_genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate(
                [
                    jnp.repeat(x, repertoire_num_samples, axis=0),
                    jnp.repeat(y, num_samples, axis=0),
                ],
                axis=0,
            ),
            extract_genotypes,
            emit_genotypes,
        )

        # Evaluate num_samples times
        (
            new_fitnesses,
            new_descriptors,
            extra_scores,
            random_key,
        ) = self._scoring_function(evaluation_genotypes, random_key)

        # Get fitness for emit in correct shape
        new_emit_fitnesses = new_fitnesses[-self._emitter.batch_size * num_samples :]
        new_emit_fitnesses = jnp.reshape(
            new_emit_fitnesses, (self._emitter.batch_size, num_samples)
        )
        new_emit_descriptors = new_descriptors[
            -self._emitter.batch_size * num_samples :
        ]
        new_emit_descriptors = jnp.reshape(
            new_emit_descriptors,
            (self._emitter.batch_size, num_samples, new_emit_descriptors.shape[-1]),
        )
        new_emit_fitnesses = jnp.pad(
            new_emit_fitnesses,
            ((0, 0), (0, self._max_number_evals - self._num_samples)),
            "constant",
            constant_values=jnp.nan,
        )
        new_emit_descriptors = jnp.pad(
            new_emit_descriptors,
            ((0, 0), (0, self._max_number_evals - self._num_samples), (0, 0)),
            "constant",
            constant_values=jnp.nan,
        )

        # Concatenate the results of the new evaluations of extract with the old ones
        new_extract_fitnesses = new_fitnesses[: -self._emitter.batch_size * num_samples]
        new_extract_fitnesses = jnp.reshape(
            new_extract_fitnesses, (-1, repertoire_num_samples)
        )
        new_extract_fitnesses = jnp.concatenate(
            [
                new_extract_fitnesses,
                extract_fitnesses.at[:, :-repertoire_num_samples].get(),
            ],
            axis=1,
        )
        new_extract_descriptors = new_descriptors[
            : -self._emitter.batch_size * num_samples
        ]
        new_extract_descriptors = jnp.reshape(
            new_extract_descriptors,
            (-1, repertoire_num_samples, new_extract_descriptors.shape[-1]),
        )
        new_extract_descriptors = jnp.concatenate(
            [
                new_extract_descriptors,
                extract_descriptors.at[:, :-repertoire_num_samples].get(),
            ],
            axis=1,
        )

        # Concatenate emit and extract evaluations
        genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            extract_genotypes,
            emit_genotypes,
        )
        fitnesses = jnp.concatenate(
            [
                new_extract_fitnesses,
                new_emit_fitnesses,
            ],
            axis=0,
        )
        descriptors = jnp.concatenate(
            [
                new_extract_descriptors,
                new_emit_descriptors,
            ],
            axis=0,
        )

        # Separate results of emitter solutions to update the emitter state
        emit_fitnesses = self._fitness_extractor(new_emit_fitnesses)
        emit_descriptors = self._descriptor_extractor(new_emit_descriptors)
        emit_extra_scores = jax.tree_util.tree_map(
            lambda x: x.at[-self._emitter.batch_size * num_samples :].get(),
            extra_scores,
        )
        emit_extra_scores = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x,
                (
                    self._emitter.batch_size,
                    num_samples,
                )
                + x.shape[1:],
            ),
            emit_extra_scores,
        )
        emit_extra_scores = jax.tree_util.tree_map(
            lambda x: x.at[:, 0].get(),
            emit_extra_scores,
        )

        return (
            genotypes,
            fitnesses,
            descriptors,
            extra_scores,
            emit_fitnesses,
            emit_descriptors,
            emit_extra_scores,
            random_key,
        )

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

        # Evaluate self._num_samples times
        (
            fitnesses,
            descriptors,
            extra_scores,
            random_key,
        ) = multi_sample_scoring_function(
            genotypes, random_key, self._scoring_function, self._num_samples
        )

        # Extend results to the good shape with jnp.nan
        fitnesses = jnp.pad(
            fitnesses,
            ((0, 0), (0, self._max_number_evals - self._num_samples)),
            "constant",
            constant_values=jnp.nan,
        )
        descriptors = jnp.pad(
            descriptors,
            ((0, 0), (0, self._max_number_evals - self._num_samples), (0, 0)),
            "constant",
            constant_values=jnp.nan,
        )

        # Init repertoire
        random_key, subkey = jax.random.split(random_key)
        repertoire = self._init_repertoire(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            centroids=centroids,
            random_key=subkey,
        )

        # Init emitter
        emit_genotypes = jax.tree_util.tree_map(
            lambda x: x.at[-self._emitter.batch_size :].get(), genotypes
        )
        emit_fitnesses = self._fitness_extractor(
            fitnesses.at[-self._emitter.batch_size :].get()
        )
        emit_descriptors = self._descriptor_extractor(
            descriptors.at[-self._emitter.batch_size :].get()
        )
        emit_extra_scores = jax.tree_util.tree_map(
            lambda x: x.at[-self._emitter.batch_size :].get(),
            extra_scores,
        )
        emit_extra_scores = jax.tree_util.tree_map(
            lambda x: x.at[:, 0].get(),
            emit_extra_scores,
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

        # Generate offsprings with the emitter
        emit_genotypes, extra_info, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # Extract from the repertoire
        (
            repertoire,
            extract_genotypes,
            extract_fitnesses,
            extract_descriptors,
            random_key,
        ) = self._extract_repertoire(repertoire, random_key)

        # Evaluate
        (
            genotypes,
            fitnesses,
            descriptors,
            extra_scores,
            emit_fitnesses,
            emit_descriptors,
            emit_extra_scores,
            random_key,
        ) = self._evaluate(
            emit_genotypes=emit_genotypes,
            extract_genotypes=extract_genotypes,
            extract_fitnesses=extract_fitnesses,
            extract_descriptors=extract_descriptors,
            num_samples=self._num_samples,
            repertoire_num_samples=self._repertoire_num_samples,
            random_key=random_key,
        )

        # Add everything back to the archive
        repertoire = self._add_repertoire(
            repertoire=repertoire,
            genotypes=genotypes,
            descriptors=descriptors,
            fitnesses=fitnesses,
            extra_scores=extra_scores,
        )

        # Update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=emit_genotypes,
            fitnesses=emit_fitnesses,
            descriptors=emit_descriptors,
            extra_scores={**emit_extra_scores, **extra_info},
        )

        # Update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key
