from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter
from qdax.types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

from core.archive_sampling import ArchiveSampling
from core.containers.archive_sampling_delta_reprod_repertoire import (
    ArchiveSamplingDeltaReprodRepertoire,
)


class ArchiveSamplingDeltaReprod(ArchiveSampling):
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
        use_weighting: bool,
        delta_fitness: float,
        delta_reproducibility: float,
        rho: float,
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
        self._use_weighting = use_weighting
        self._delta_fitness = delta_fitness
        self._delta_reproducibility = delta_reproducibility
        self._rho = rho

    @partial(jax.jit, static_argnames=("self",))
    def _add_repertoire(
        self,
        repertoire: MapElitesRepertoire,
        genotypes: Genotype,
        descriptors: Descriptor,
        fitnesses: Fitness,
        extra_scores: ExtraScores,
    ) -> MapElitesRepertoire:
        print("Adding in ArchiveSamplingDeltaReprod")

        return repertoire.add(
            batch_of_genotypes=genotypes,
            batch_of_all_descriptors=descriptors,
            batch_of_all_fitnesses=fitnesses,
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

    @partial(jax.jit, static_argnames=("self",))
    def _init_repertoire(
        self,
        genotypes: Genotype,
        descriptors: Descriptor,
        fitnesses: Fitness,
        extra_scores: ExtraScores,
        centroids: Centroid,
    ) -> MapElitesRepertoire:
        print("Initialising in ArchiveSamplingDeltaReprod")

        return ArchiveSamplingDeltaReprodRepertoire.init(
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
            use_weighting=self._use_weighting,
            delta_fitness=self._delta_fitness,
            delta_reproducibility=self._delta_reproducibility,
            rho=self._rho,
        )
