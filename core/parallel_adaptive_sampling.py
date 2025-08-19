"""Core components of the MAP-Elites algorithm."""
from __future__ import annotations

import os
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.custom_types import (
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

from core.archive_sampling import ArchiveSampling


class ParallelAdaptiveSampling(ArchiveSampling):
    """
    Core elements of the Extended Adaptive Sampling algorithm storing all replications to allow any estimator.
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        depth: int,
        max_number_evals: int,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        sampling_size: int,
        batch_size: int,
        max_num_samples: int,
        use_evals: str,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._depth = depth
        self._max_number_evals = max_number_evals
        self._fitness_extractor = fitness_extractor
        self._fitness_reproducibility_extractor = fitness_reproducibility_extractor
        self._descriptor_extractor = descriptor_extractor
        self._descriptor_reproducibility_extractor = (
            descriptor_reproducibility_extractor
        )
        self._sampling_size = sampling_size

        # Number of evaluations selection mode
        valid_evals = ["max", "min", "mean", "median"]
        assert (
            use_evals in valid_evals
        ), "!!!ERROR!!! Unvalid eval-selection method, should be in" + str(valid_evals)
        self._use_evals = use_evals
        assert max_num_samples > 0, "!!!ERROR!!! max_num_samples should be > 0."
        self._max_num_samples = max_num_samples

        # Archive Sampling required attributes
        self._num_samples = 1
        self._repertoire_num_samples = 1

        # Externally used attributes
        self.num_samples = 1
        self.batch_size = batch_size

    @partial(jax.jit, static_argnames=("self", "num_samples", "batch_size"))
    def _sub_update(
        self,
        num_samples: int,
        batch_size: int,
        repertoire: MapElitesRepertoire,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, EmitterState, Metrics, RNGKey]:
        """
        Performs one iteration of the jitable part of Parallel-Adaptive-Sampling.

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

        #####################
        # 1. Emit offspring #

        # generate offsprings with the emitter
        genotypes, extra_info, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # only keep batch-size of the offspring
        genotypes = jax.tree_util.tree_map(lambda x: x[:batch_size], genotypes)

        #############################################
        # 2. Evaluate offspring and archive content #

        # evaluate all individuals already in the archive and offspring
        (
            repertoire,
            extract_genotypes,
            extract_fitnesses,
            extract_descriptors,
            random_key,
        ) = self._extract_repertoire(repertoire=repertoire, random_key=random_key)
        (
            all_genotypes,
            all_fitnesses,
            all_descriptors,
            all_extra_scores,
            emit_fitnesses,
            emit_descriptors,
            emit_extra_scores,
            random_key,
        ) = self._evaluate(
            emit_genotypes=genotypes,
            extract_genotypes=extract_genotypes,
            extract_fitnesses=extract_fitnesses,
            extract_descriptors=extract_descriptors,
            num_samples=num_samples,
            repertoire_num_samples=1,
            random_key=random_key,
        )

        ##############################
        # 3. Add back to the archive #

        # empty repertoire
        repertoire = repertoire.empty()

        # add everything back to the archive
        repertoire = self._add_repertoire(
            repertoire=repertoire,
            genotypes=all_genotypes,
            descriptors=all_descriptors,
            fitnesses=all_fitnesses,
            extra_scores=all_extra_scores,
        )

        # set up the total number of evaluations
        total_indivs = (
            batch_size * num_samples
            + jax.tree_util.tree_leaves(repertoire.genotypes_depth)[0].shape[0]
        )

        ############################
        # 4. Perform final updates #

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=emit_fitnesses,
            descriptors=emit_descriptors,
            extra_scores={**emit_extra_scores, **extra_info},
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key

    def update(
        self,
        repertoire: MapElitesRepertoire,
        emitter_state: EmitterState,
        random_key: RNGKey,
    ) -> Tuple[MapElitesRepertoire, EmitterState, Metrics, RNGKey]:
        """
        !!!WARNING!!! Un-jitable as it is now

        Performs one iteration of the Extended Adaptive Sampling algorithm.

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

        # Limit CPU usage as not jited (for HPC)
        os.environ["XLA_FLAGS"] = (
            "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=4"
        )

        # Chose number of samples for the offspring
        evaluations_depth = jnp.sum(
            jnp.logical_not(jnp.isnan(repertoire.fitnesses_depth_all)),
            axis=2,
        )
        if jnp.sum(evaluations_depth > 0) == 0:
            num_evaluations = 1
        elif self._use_evals == "max":
            num_evaluations = int(jnp.nanmax(evaluations_depth))
        elif self._use_evals == "min":
            num_evaluations = int(jnp.nanmin(evaluations_depth[evaluations_depth > 0]))
        elif self._use_evals == "mean":
            num_evaluations = int(jnp.nanmean(evaluations_depth[evaluations_depth > 0]))
        elif self._use_evals == "median":
            num_evaluations = int(
                jnp.nanmedian(evaluations_depth[evaluations_depth > 0])
            )
        num_samples = max(1, min(self._max_num_samples, num_evaluations))

        # Remove part of the offspring (or samples) to match samples per generation
        num_indivs = jax.tree_util.tree_leaves(repertoire.genotypes_depth)[0].shape[0]
        if self._sampling_size > 0:
            batch_size = (self._sampling_size - num_indivs) // num_samples
            while batch_size < 1 and num_samples > 1:
                num_samples -= 1
                batch_size = (self._sampling_size - num_indivs) // num_samples
        else:
            batch_size = self.batch_size
        assert batch_size > 0 and num_samples > 0, (
            "!!!ERROR!!! batch_size: "
            + str(batch_size)
            + ", num_samples:"
            + str(num_samples)
        )

        # Final number of samples and batch_size
        self.num_samples = num_samples
        self.batch_size = batch_size

        # Call usual update using these values
        repertoire, emitter_state, metrics, random_key = self._sub_update(
            num_samples, batch_size, repertoire, emitter_state, random_key
        )
        return repertoire, emitter_state, metrics, random_key
