"""Core components of the MAP-Elites algorithm."""
from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import jax.numpy as jnp
from qdax.core.containers.repertoire import Repertoire
from qdax.core.emitters.emitter import Emitter
from qdax.custom_types import (
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

from core.archive_sampling import ArchiveSampling
from core.sampling import sampling_descriptor_reproducibility


class ArchiveSamplingReprod(ArchiveSampling):
    """
    Core elements of Archive-Sampling optimising
    reproducibilty only and no fitness.
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

        self._scoring_function = partial(
            sampling_descriptor_reproducibility,
            scoring_fn=scoring_function,
            num_samples=num_samples,
            descriptor_extractor=descriptor_extractor,
            descriptor_reproducibility_extractor=descriptor_reproducibility_extractor,
        )
