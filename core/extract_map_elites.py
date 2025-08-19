from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import jax
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


class ExtractMAPElites(ArchiveSampling):
    """
    Core elements of the Extract-MAP-Elites algorithm.
    """

    def __init__(
        self,
        scoring_function: Callable[
            [Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]
        ],
        emitter: Emitter,
        metrics_function: Callable[[Repertoire], Metrics],
        depth: int,
        batch_size: int,
        emit_batch_size: int,
        max_number_evals: int,
        num_samples: int,
        repertoire_num_samples: int,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        extract_type: str,
    ) -> None:
        self._emit_batch_size = emit_batch_size
        self._extract_batch_size = batch_size - emit_batch_size
        self._extract_type = extract_type
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

        # Extract with prob proportional to ranking
        if self._extract_type == "proportional":
            return repertoire.extract_prop(random_key, self._extract_batch_size, type_prop="exponential")  # type: ignore
        if self._extract_type == "proportional_harmonic":
            return repertoire.extract_prop(random_key, self._extract_batch_size, type_prop="harmonic")  # type: ignore
        if self._extract_type == "proportional_linear":
            return repertoire.extract_prop(random_key, self._extract_batch_size, type_prop="linear")  # type: ignore

        # Extract with uniform prob
        elif self._extract_type == "uniform":
            return repertoire.extract_uniform(random_key, self._extract_batch_size)  # type: ignore

        else:
            assert 0, "!!!ERROR!!! Undefined extract type."
