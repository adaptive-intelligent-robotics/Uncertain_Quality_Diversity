"""This file contains the class to define the repertoire used to
store individuals in the Multi-Objective MAP-Elites algorithm as
well as several variants."""

from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import get_cells_indices
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype

from core.containers.mome_x_repertoire import MOMEXRepertoire


class MOMEXReprodRepertoire(MOMEXRepertoire):
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: ExtraScores,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> MOMEXReprodRepertoire:

        batch_size, num_samples = batch_of_fitnesses.shape

        # Compute batch of reproducibility
        # WARNING: in the case of descriptors_reproducibility, take average over dimensions
        batch_of_reproducibilities = descriptor_reproducibility_extractor(
            batch_of_descriptors
        )
        batch_of_reproducibilities = jnp.average(batch_of_reproducibilities, axis=-1)

        # WARNING: takes the negative as the extractor is actually returning the inverse of reproducibility
        batch_of_reproducibilities = -batch_of_reproducibilities
        batch_of_reproducibilities = jnp.expand_dims(
            batch_of_reproducibilities, axis=-1
        )

        # Compute batch of descriptor
        batch_of_descriptors = descriptor_extractor(batch_of_descriptors)

        # Compute batch of fitness
        batch_of_fitnesses = jnp.expand_dims(
            fitness_extractor(batch_of_fitnesses), axis=-1
        )

        # Stack task fitness and reproducibility to get MOME fitness scores
        batch_of_fitnesses = jnp.hstack(
            (batch_of_fitnesses, batch_of_reproducibilities)
        )

        # get the indices that corresponds to the descriptors in the repertoire
        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
        batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)

        def _add_one(
            carry: MOMEXReprodRepertoire,
            data: Tuple[Genotype, Descriptor, Fitness, jnp.ndarray],
        ) -> Tuple[MOMEXReprodRepertoire, Any]:
            # unwrap data
            genotype, descriptors, fitness, index = data

            index = index.astype(jnp.int32)

            # get cell data
            cell_genotype = jax.tree_util.tree_map(
                lambda x: (x[index]).squeeze(axis=0), carry.genotypes
            )
            cell_fitness = (carry.fitnesses[index]).squeeze(axis=0)
            cell_descriptor = (carry.descriptors[index]).squeeze(axis=0)
            cell_mask = jnp.any(cell_fitness == -jnp.inf, axis=-1)

            # get new data
            new_batch_of_fitnesses = jnp.expand_dims(fitness, axis=0)
            new_batch_of_genotypes = jax.tree_util.tree_map(
                lambda x: jnp.expand_dims(x, axis=0), genotype
            )
            new_batch_of_descriptors = jnp.expand_dims(descriptors, axis=0)
            new_mask = jnp.zeros(shape=(1,), dtype=bool)

            # update pareto front
            (
                cell_fitness,
                cell_genotype,
                cell_descriptor,
                cell_mask,
            ) = self._update_masked_pareto_front(
                pareto_front_fitnesses=cell_fitness,
                pareto_front_genotypes=cell_genotype,
                pareto_front_descriptors=cell_descriptor,
                mask=cell_mask,
                new_batch_of_fitnesses=new_batch_of_fitnesses,
                new_batch_of_genotypes=new_batch_of_genotypes,
                new_batch_of_descriptors=new_batch_of_descriptors,
                new_mask=new_mask,
            )

            # update cell fitness
            cell_fitness = cell_fitness - jnp.inf * jnp.expand_dims(cell_mask, axis=-1)

            # update grid
            new_genotypes = jax.tree_util.tree_map(
                lambda x, y: x.at[index].set(y), carry.genotypes, cell_genotype
            )
            new_fitnesses = carry.fitnesses.at[index].set(cell_fitness)
            new_descriptors = carry.descriptors.at[index].set(cell_descriptor)
            carry = carry.replace(  # type: ignore
                genotypes=new_genotypes,
                descriptors=new_descriptors,
                fitnesses=new_fitnesses,
            )

            # return new grid
            return carry, ()

        # scan the addition operation for all the data
        new_repertoire, _ = jax.lax.scan(
            _add_one,
            self,
            (
                batch_of_genotypes,
                batch_of_descriptors,
                batch_of_fitnesses,
                batch_of_indices,
            ),
        )
        return new_repertoire

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        centroids: Centroid,
        extra_scores: ExtraScores,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        pareto_front_max_length: int,
    ) -> MOMEXReprodRepertoire:

        # get dimensions
        num_criteria = 2  # always fitness and reproducibility
        num_descriptors = descriptors.shape[-1]
        num_centroids = centroids.shape[0]

        # create default values
        default_fitnesses = -jnp.inf * jnp.ones(
            shape=(num_centroids, pareto_front_max_length, num_criteria)
        )
        default_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.zeros(
                shape=(
                    num_centroids,
                    pareto_front_max_length,
                )
                + x.shape[1:]
            ),
            genotypes,
        )
        default_descriptors = jnp.zeros(
            shape=(num_centroids, pareto_front_max_length, num_descriptors)
        )

        # create repertoire with default values
        repertoire = MOMEXReprodRepertoire(  # type: ignore
            genotypes=default_genotypes,
            fitnesses=default_fitnesses,
            descriptors=default_descriptors,
            centroids=centroids,
        )

        # add first batch of individuals in the repertoire
        new_repertoire = repertoire.add(
            batch_of_genotypes=genotypes,
            batch_of_descriptors=descriptors,
            batch_of_fitnesses=fitnesses,
            batch_of_extra_scores=extra_scores,
            fitness_extractor=fitness_extractor,
            fitness_reproducibility_extractor=fitness_reproducibility_extractor,
            descriptor_extractor=descriptor_extractor,
            descriptor_reproducibility_extractor=descriptor_reproducibility_extractor,
        )

        return new_repertoire  # type: ignore

    @jax.jit
    def empty(self) -> MOMEXReprodRepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty MapElitesReproducibilityRepertoire
        """

        new_fitnesses = jnp.full_like(self.fitnesses, -jnp.inf)
        new_descriptors = jnp.zeros_like(self.descriptors)
        new_genotypes = jax.tree_map(lambda x: jnp.zeros_like(x), self.genotypes)
        return MOMEXReprodRepertoire(
            genotypes=new_genotypes,
            fitnesses=new_fitnesses,
            descriptors=new_descriptors,
            centroids=self.centroids,
        )
