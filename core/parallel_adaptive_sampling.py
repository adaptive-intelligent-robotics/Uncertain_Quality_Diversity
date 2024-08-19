"""Core components of the MAP-Elites algorithm."""
from __future__ import annotations

import os
from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.containers.repertoire import Repertoire
from qdax.types import Descriptor, ExtraScores, Fitness, Genotype, Metrics, RNGKey

from core.archive_sampling import ArchiveSampling

class ParallelAdaptiveSampling(ArchiveSampling):
    """
    Core elements of the Extended Adaptive Sampling algorithm.

    Args:
        scoring_function: a function that takes a batch of genotypes and compute
            their fitnesses and descriptors.
        emitter: an emitter is used to suggest offsprings given a MAPELites
            repertoire. It has two compulsory functions. A function that takes
            emits a new population, and a function that update the internal state
            of the emitter.
        metrics_function: a function that takes a MAP-Elites repertoire and compute
            any useful metric to track its evolution
        max_num_samples: maximal number of samples for all indivs (if 0 then not set).
        depth: depth of the repertoire.
        sampling_size: if greater than 0, cap the number of samples allowed per
            generation (not counting sampling spent on archive reevaluation)
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[Repertoire], Metrics],
        depth: int,
        sampling_size: int = 0,
        batch_size: int = 0,
        max_num_samples: int = 0,
        use_evals: str = "max",
        archive_out_sampling: bool = False,
    ) -> None:
        self._scoring_function = scoring_function
        self._emitter = emitter
        self._metrics_function = metrics_function
        self._max_num_samples = max_num_samples if max_num_samples != 0 else jnp.inf
        self._depth = depth
        self._sampling_size = sampling_size
        self._archive_out_sampling = archive_out_sampling

        # Number of evaluations selection mode
        valid_evals = ["max", "min", "mean", "median"]
        assert (
            use_evals in valid_evals
        ), "!!!ERROR!!! Unvalid eval-selection method, should be in" + str(valid_evals)
        self._use_evals = use_evals

        # Externally used attributes
        self.num_samples = 1
        self.batch_size = batch_size

    @partial(jax.jit, static_argnames=("self", "num_samples", "batch_size"))
    def _sub_update(
        self,
        num_samples: int,
        batch_size: int,
        repertoire: Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Repertoire, Optional[EmitterState], Metrics, RNGKey]:
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
        genotypes, random_key = self._emitter.emit(
            repertoire, emitter_state, random_key
        )

        # only keep batch-size of the offspring
        genotypes = jax.tree_util.tree_map(lambda x: x[:batch_size], genotypes)

        #############################################
        # 2. Evaluate offspring and archive content #

        # copy offspring num_samples times
        sample_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x, num_samples, axis=0),
            genotypes,
        )

        # evaluate all individuals already in the archive and offspring
        (
            _,
            fitnesses,
            descriptors,
            extra_scores,
            random_key,
        ) = self._scoring_repertoire_offspring(repertoire, sample_genotypes, random_key)

        #########################################
        # 3. Compute average perf for offspring #

        # final genotypes
        final_genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([jnp.reshape(x, y.shape), y], axis=0),
            repertoire.genotypes_depth,
            genotypes,
        )
        num_indivs = jax.tree_util.tree_leaves(repertoire.genotypes_depth)[0].shape[0]

        if num_samples > 1:

            # final fitnesses
            offpsring_fitnesses = fitnesses[num_indivs:]
            offpsring_fitnesses = jnp.reshape(
                offpsring_fitnesses, (batch_size, num_samples)
            )
            offpsring_fitnesses = jnp.average(offpsring_fitnesses, axis=1)
            final_fitnesses = jnp.concatenate(
                [fitnesses[:num_indivs], offpsring_fitnesses], axis=0
            )
            assert final_fitnesses.shape[0] == batch_size + num_indivs

            # final descriptors
            offpsring_descriptors = descriptors[num_indivs:]
            offpsring_descriptors = jnp.reshape(
                offpsring_descriptors, (batch_size, num_samples, -1)
            )
            offpsring_descriptors = jnp.average(offpsring_descriptors, axis=1)
            final_descriptors = jnp.concatenate(
                [descriptors[:num_indivs], offpsring_descriptors], axis=0
            )
            assert final_descriptors.shape[0] == batch_size + num_indivs

            # final extra scores
            # WARNING assuming dummy extra_scores extractor for non num_evaluations
            final_extra_scores = extra_scores
            final_extra_scores["num_evaluations"] = jnp.concatenate(
                [
                    jnp.full(batch_size, num_samples),
                    extra_scores["num_evaluations"][:num_indivs],
                ],
                axis=0,
            )

        else:
            final_fitnesses = fitnesses
            final_descriptors = descriptors
            final_extra_scores = extra_scores

        ##############################
        # 4. Add back to the archive #

        # empty repertoire
        total_evaluations = repertoire.total_evaluations
        repertoire = repertoire.empty()

        # add everything back to the archive
        repertoire = repertoire.add(
            final_genotypes, final_descriptors, final_fitnesses, final_extra_scores
        )

        # set up the total number of evaluations
        total_indivs = batch_size * num_samples + num_indivs
        repertoire = repertoire.set_total_evaluations(total_evaluations + total_indivs)

        ############################
        # 5. Perform final updates #

        # update emitter state after scoring is made
        emitter_state = self._emitter.state_update(
            emitter_state=emitter_state,
            repertoire=repertoire,
            genotypes=genotypes,
            fitnesses=final_fitnesses[-batch_size:],
            descriptors=final_descriptors[-batch_size:],
            extra_scores=final_extra_scores,
        )

        # update the metrics
        metrics = self._metrics_function(repertoire)

        return repertoire, emitter_state, metrics, random_key

    def update(
        self,
        repertoire: Repertoire,
        emitter_state: Optional[EmitterState],
        random_key: RNGKey,
    ) -> Tuple[Repertoire, Optional[EmitterState], Metrics, RNGKey]:
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
        if jnp.sum(repertoire.evaluations_depth > 0) == 0:
            num_evaluations = 1
        elif self._use_evals == "max":
            num_evaluations = int(jnp.nanmax(repertoire.evaluations_depth))
        elif self._use_evals == "min":
            num_evaluations = int(
                jnp.nanmin(
                    repertoire.evaluations_depth[repertoire.evaluations_depth > 0]
                )
            )
        elif self._use_evals == "mean":
            num_evaluations = int(
                jnp.nanmean(
                    repertoire.evaluations_depth[repertoire.evaluations_depth > 0]
                )
            )
        elif self._use_evals == "median":
            num_evaluations = int(
                jnp.nanmedian(
                    repertoire.evaluations_depth[repertoire.evaluations_depth > 0]
                )
            )
        num_samples = max(1, min(self._max_num_samples, num_evaluations))

        # Remove part of the offspring (or samples) to match samples per generation
        if self._archive_out_sampling:
            num_indivs = 0
        else:
            num_indivs = jax.tree_util.tree_leaves(repertoire.genotypes_depth)[0].shape[
                0
            ]
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
