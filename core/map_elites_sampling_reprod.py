"""Core components of the MAP-Elites algorithm."""
from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter
from qdax.core.map_elites import MAPElites
from qdax.custom_types import (
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    Metrics,
    RNGKey,
)

from core.sampling import sampling_descriptor_reproducibility


class MAPElitesSamplingReprod(MAPElites):
    """
    Core elements of the MAP-Elites algorithm with sampling, optimising
    reproducibilty only and no fitness.
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[MapElitesRepertoire], Metrics],
        num_samples: int,
        descriptor_extractor: Callable[[Descriptor], Descriptor],
        descriptor_reproducibility_extractor: Callable[[Descriptor], Fitness],
    ) -> None:
        self._emitter = emitter
        self._metrics_function = metrics_function

        self._scoring_function = partial(
            sampling_descriptor_reproducibility,
            scoring_fn=scoring_function,
            num_samples=num_samples,
            descriptor_extractor=descriptor_extractor,
            descriptor_reproducibility_extractor=descriptor_reproducibility_extractor,
        )
