from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

from core.archive_sampling import ArchiveSampling
from core.containers.archive_sampling_weighted_repertoire import (
    ArchiveSamplingWeightedRepertoire,
)


class ArchiveSamplingWeighted(ArchiveSampling):
    """
    Core elements of the Archive-Sampling with Delta Reproducibility algorithm.
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
        delta_fitness: float,
        delta_reproducibility: float,
        rho: float,
    ) -> None:
        super().__init__(
            scoring_function=scoring_function,
            emitter=emitter,
            metrics_function=metrics_function,
            depth=depth,
            max_number_evals=max_number_evals,
            num_samples=num_samples,
            repertoire_num_samples=repertoire_num_samples,
            fitness_extractor=fitness_extractor,
            fitness_reproducibility_extractor=fitness_reproducibility_extractor,
            descriptor_extractor=descriptor_extractor,
            descriptor_reproducibility_extractor=descriptor_reproducibility_extractor,
        )
        self._delta_fitness = delta_fitness
        self._delta_reproducibility = delta_reproducibility
        self._rho = rho

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
        random_key: RNGKey,
    ) -> Repertoire:

        return ArchiveSamplingWeightedRepertoire.init(
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
            delta_fitness=self._delta_fitness,
            delta_reproducibility=self._delta_reproducibility,
            rho=self._rho,
        )
