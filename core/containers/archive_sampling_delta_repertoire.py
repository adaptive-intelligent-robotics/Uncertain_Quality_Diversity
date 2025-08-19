from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import get_cells_indices
from qdax.custom_types import Centroid, Descriptor, ExtraScores, Fitness, Genotype

from core.containers.archive_sampling_repertoire import ArchiveSamplingRepertoire


class ArchiveSamplingDeltaRepertoire(ArchiveSamplingRepertoire):
    """
    Class for a deep repertoire that also stores all past evaluations of each indiv.

    Args:
        reproducibilities: an array that contains the fitness of best solutions in
            each cell of the repertoire, ordered by centroids.
            The array shape is (num_centroids,).
        reproducibilities_depth: an array that contains the fitness of all
            solutions in each cell of the repertoire, ordered by centroids.
            The array shape is (num_centroids * depth).
    """

    reproducibilities: Fitness
    reproducibilities_depth: Fitness

    def save(self, path: str = "./") -> None:
        """Saves the grid on disk in the form of .npy files.

        Flattens the genotypes to store it with .npy format. Supposes that
        a user will have access to the reconstruction function when loading
        the genotypes.

        Args:
            path: Path where the data will be saved. Defaults to "./".
        """

        super().save(path=path)
        jnp.save(path + "reproducibilities.npy", self.reproducibilities)
        jnp.save(path + "reproducibilities_depth.npy", self.reproducibilities_depth)

    @classmethod
    def load(
        cls, reconstruction_fn: Callable, path: str = "./"
    ) -> ArchiveSamplingDeltaRepertoire:
        """Loads a MAP Elites Grid.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            A MAP Elites Repertoire.
        """

        repertoire = super().load(reconstruction_fn=reconstruction_fn, path=path)
        reproducibilities = jnp.load(path + "reproducibilities.npy")
        reproducibilities_depth = jnp.load(path + "reproducibilities_depth.npy")

        return ArchiveSamplingDeltaRepertoire(
            genotypes=repertoire.genotypes,
            genotypes_depth=repertoire.genotypes_depth,
            fitnesses=repertoire.fitnesses,
            fitnesses_depth=repertoire.fitnesses_depth,
            fitnesses_depth_all=repertoire.fitnesses_depth_all,
            reproducibilities=reproducibilities,
            reproducibilities_depth=reproducibilities_depth,
            descriptors=repertoire.descriptors,
            descriptors_depth=repertoire.descriptors_depth,
            descriptors_depth_all=repertoire.descriptors_depth_all,
            centroids=repertoire.centroids,
        )

    @partial(
        jax.jit,
        static_argnames=(
            "fitness_extractor",
            "fitness_reproducibility_extractor",
            "descriptor_extractor",
            "descriptor_reproducibility_extractor",
            "delta_fitness",
            "delta_reproducibility",
        ),
    )
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_all_descriptors: Descriptor,
        batch_of_all_fitnesses: Fitness,
        batch_of_extra_scores: ExtraScores,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        delta_fitness: float,
        delta_reproducibility: float,
    ) -> ArchiveSamplingDeltaRepertoire:
        """
        Add a batch of elements to the repertoire.
        WARNING: This addition makes the hypothesis that batch_of_all_descriptors
        and batch_of_all_fitnesses are already dimensions num_evals.

        Args:
            batch_of_genotypes: a batch of genotypes to be added to the repertoire.
                Similarly to the self.genotypes argument, this is a PyTree in which
                the leaves have a shape (batch_size, num_features)
            batch_of_descriptors: an array that contains the descriptors of the
                aforementioned genotypes. Its shape is (batch_size, num_descriptors)
            batch_of_fitnesses: an array that contains the fitnesses of the
            batch_of_extra_scores: unused tree that contains the extra_scores of
                aforementioned genotypes. Its shape is (batch_size,)

        Returns:
            The updated MAP-Elites repertoire.
        """

        num_centroids = self.fitnesses_depth.shape[0]
        depth = self.fitnesses_depth.shape[1]
        out_of_bound = max(
            num_centroids * depth,
            batch_of_all_fitnesses.shape[0],
        )

        # Compute batch of descriptor
        batch_of_descriptors = descriptor_extractor(batch_of_all_descriptors)

        # Compute batch of fitness
        batch_of_fitnesses = fitness_extractor(batch_of_all_fitnesses)
        batch_of_fitnesses = jnp.where(
            jnp.isnan(batch_of_fitnesses), -jnp.inf, batch_of_fitnesses
        )

        # Compute batch of reproducibility
        # WARNING: in the case of descriptors_reproducibility, take average over dimensions
        batch_of_reproducibilities = descriptor_reproducibility_extractor(
            batch_of_all_descriptors
        )
        batch_of_reproducibilities = jnp.average(batch_of_reproducibilities, axis=-1)

        # WARNING: takes the negative as the extractor is actually returning the inverse of reproducibility
        batch_of_reproducibilities = -batch_of_reproducibilities
        batch_of_reproducibilities = jnp.where(
            jnp.isnan(batch_of_reproducibilities), -jnp.inf, batch_of_reproducibilities
        )

        # Compute indices
        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)

        # Filter dead individuals
        batch_of_reproducibilities = jnp.where(
            batch_of_fitnesses > -jnp.inf,
            batch_of_reproducibilities,
            -jnp.inf,
        )
        batch_of_indices = jnp.where(
            batch_of_fitnesses > -jnp.inf,
            batch_of_indices,
            out_of_bound,
        )

        # Compare to elites one by one

        def _compare_elite_one(
            carry: Tuple,
            unused: Tuple,
        ) -> Tuple:

            # unwrap data
            (
                cell_genotypes,
                cell_descriptors,
                cell_all_descriptors,
                cell_fitnesses,
                cell_all_fitnesses,
                cell_reproducibilities,
                genotype,
                descriptor,
                all_descriptor,
                fitness,
                all_fitness,
                reproducibility,
                elite_index,
            ) = carry

            # get elite values
            elite_fitness = cell_fitnesses[elite_index]
            elite_reproducibility = cell_reproducibilities[elite_index]

            # all criterions
            fitness_only_condition = fitness > (elite_fitness + delta_fitness)
            fitness_condition = jnp.logical_and(
                fitness >= elite_fitness,
                reproducibility >= elite_reproducibility,
            )
            reproducibility_condition = jnp.logical_and(
                fitness > (elite_fitness - delta_fitness),
                reproducibility >= (elite_reproducibility + delta_reproducibility),
            )

            # final addition condition
            final_condition = jnp.logical_or(
                fitness_only_condition,
                jnp.logical_or(fitness_condition, reproducibility_condition),
            )

            # update elite to add
            new_genotype = jax.tree_util.tree_map(
                lambda x, y: jnp.where(
                    final_condition,
                    x.at[elite_index].get(),
                    y,
                ),
                cell_genotypes,
                genotype,
            )
            new_descriptor = jnp.where(
                final_condition, cell_descriptors.at[elite_index].get(), descriptor
            )
            new_all_descriptor = jnp.where(
                final_condition,
                cell_all_descriptors.at[elite_index].get(),
                all_descriptor,
            )
            new_fitness = jnp.where(
                final_condition, cell_fitnesses.at[elite_index].get(), fitness
            )
            new_all_fitness = jnp.where(
                final_condition, cell_all_fitnesses.at[elite_index].get(), all_fitness
            )
            new_reproduciblity = jnp.where(
                final_condition,
                cell_reproducibilities.at[elite_index].get(),
                reproducibility,
            )
            new_elite_index = elite_index + 1

            # update grid
            elite_index = jnp.where(final_condition, elite_index, num_centroids)
            new_cell_genotypes = jax.tree_util.tree_map(
                lambda x, y: x.at[elite_index].set(y), cell_genotypes, genotype
            )
            new_cell_descriptors = cell_descriptors.at[elite_index].set(descriptor)
            new_cell_all_descriptors = cell_all_descriptors.at[elite_index].set(
                all_descriptor
            )
            new_cell_fitnesses = cell_fitnesses.at[elite_index].set(fitness)
            new_cell_all_fitnesses = cell_all_fitnesses.at[elite_index].set(all_fitness)
            new_cell_reproducibilities = cell_reproducibilities.at[elite_index].set(
                reproducibility
            )

            # return new grid
            return (  # type: ignore
                new_cell_genotypes,
                new_cell_descriptors,
                new_cell_all_descriptors,
                new_cell_fitnesses,
                new_cell_all_fitnesses,
                new_cell_reproducibilities,
                new_genotype,
                new_descriptor,
                new_all_descriptor,
                new_fitness,
                new_all_fitness,
                new_reproduciblity,
                new_elite_index,
            ), ()

        # Add individuals one by one
        def _add_one(
            carry: Tuple,
            data: Tuple,
        ) -> Tuple:

            # unwrap data
            (
                genotypes_depth,
                descriptors_depth,
                descriptors_depth_all,
                fitnesses_depth,
                fitnesses_depth_all,
                reproducibilities_depth,
            ) = carry
            (
                genotype,
                descriptor,
                all_descriptor,
                fitness,
                all_fitness,
                reproducibility,
                index,
            ) = data

            # get cell values
            cell_genotypes = jax.tree_util.tree_map(
                lambda x: x.at[index].get(), genotypes_depth
            )
            cell_descriptors = descriptors_depth.at[index].get()
            cell_all_descriptors = descriptors_depth_all.at[index].get()
            cell_fitnesses = fitnesses_depth.at[index].get()
            cell_all_fitnesses = fitnesses_depth_all.at[index].get()
            cell_reproducibilities = reproducibilities_depth.at[index].get()

            # compare to elites one by one
            (
                new_cell_genotypes,
                new_cell_descriptors,
                new_cell_all_descriptors,
                new_cell_fitnesses,
                new_cell_all_fitnesses,
                new_cell_reproducibilities,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ), _ = jax.lax.scan(
                _compare_elite_one,
                (
                    cell_genotypes,
                    cell_descriptors,
                    cell_all_descriptors,
                    cell_fitnesses,
                    cell_all_fitnesses,
                    cell_reproducibilities,
                    genotype,
                    descriptor,
                    all_descriptor,
                    fitness,
                    all_fitness,
                    reproducibility,
                    0,
                ),
                (),
                length=depth,
            )

            # update depth grid
            new_genotypes_depth = jax.tree_util.tree_map(
                lambda x, y: x.at[index].set(y), genotypes_depth, new_cell_genotypes
            )
            new_descriptors_depth = descriptors_depth.at[index].set(
                new_cell_descriptors
            )
            new_descriptors_depth_all = descriptors_depth_all.at[index].set(
                new_cell_all_descriptors
            )
            new_fitnesses_depth = fitnesses_depth.at[index].set(new_cell_fitnesses)
            new_fitnesses_depth_all = fitnesses_depth_all.at[index].set(
                new_cell_all_fitnesses
            )
            new_reproducibilities_depth = reproducibilities_depth.at[index].set(
                new_cell_reproducibilities,
            )

            # return new grid
            return (
                new_genotypes_depth,
                new_descriptors_depth,
                new_descriptors_depth_all,
                new_fitnesses_depth,
                new_fitnesses_depth_all,
                new_reproducibilities_depth,
            ), ()

        # scan the addition operation for all the individuals
        (
            genotypes_depth,
            descriptors_depth,
            descriptors_depth_all,
            fitnesses_depth,
            fitnesses_depth_all,
            reproducibilities_depth,
        ), _ = jax.lax.scan(
            _add_one,
            (
                self.genotypes_depth,
                self.descriptors_depth,
                self.descriptors_depth_all,
                self.fitnesses_depth,
                self.fitnesses_depth_all,
                self.reproducibilities_depth,
            ),
            (
                batch_of_genotypes,
                batch_of_descriptors,
                batch_of_all_descriptors,
                batch_of_fitnesses,
                batch_of_all_fitnesses,
                batch_of_reproducibilities,
                batch_of_indices,
            ),
        )

        # update repertoire
        genotypes = jax.tree_util.tree_map(
            lambda x: x.at[:, 0].get(),
            genotypes_depth,
        )
        descriptors = descriptors_depth.at[:, 0].get()
        fitnesses = fitnesses_depth.at[:, 0].get()
        reproducibilities = reproducibilities_depth.at[:, 0].get()

        new_repertoire = self.replace(
            genotypes=genotypes,
            descriptors=descriptors,
            fitnesses=fitnesses,
            reproducibilities=reproducibilities,
            genotypes_depth=genotypes_depth,
            descriptors_depth=descriptors_depth,
            descriptors_depth_all=descriptors_depth_all,
            fitnesses_depth=fitnesses_depth,
            fitnesses_depth_all=fitnesses_depth_all,
            reproducibilities_depth=reproducibilities_depth,
        )

        return new_repertoire  # type: ignore

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
        centroids: Centroid,
        depth: int,
        num_evals: int,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        delta_fitness: float,
        delta_reproducibility: float,
    ) -> ArchiveSamplingDeltaRepertoire:
        """
        Initialize a Map-Elites repertoire with an initial population of genotypes.
        Requires the definition of centroids that can be computed with any method
        such as CVT or Euclidean mapping.

        Note: this function has been kept outside of the object MapElites, so it can
        be called easily called from other modules.

        Args:
            genotypes: initial genotypes, pytree in which leaves
                have shape (batch_size, num_features)
            fitnesses: fitness of the initial genotypes of shape (batch_size,)
            descriptors: descriptors of the initial genotypes
                of shape (batch_size, num_descriptors)
            extra_scores: unused extra_scores of the initial genotypes
            centroids: tesselation centroids of shape (batch_size, num_descriptors)
            depth
            num_evals

        Returns:
            an initialized MAP-Elite repertoire
        """

        # Initialize grid with default values
        num_centroids = centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)
        default_fitnesses_depth = -jnp.inf * jnp.ones(shape=(num_centroids, depth))
        default_fitnesses_depth_all = jnp.nan * jnp.ones(
            shape=(num_centroids, depth, num_evals)
        )
        default_genotypes = jax.tree_map(
            lambda x: jnp.zeros(shape=(num_centroids,) + x.shape[1:]),
            genotypes,
        )
        default_genotypes_depth = jax.tree_map(
            lambda x: jnp.zeros(
                shape=(
                    num_centroids,
                    depth,
                )
                + x.shape[1:]
            ),
            genotypes,
        )
        default_descriptors = jnp.zeros(shape=(num_centroids, centroids.shape[-1]))
        default_descriptors_depth = jnp.zeros(
            shape=(num_centroids, depth, centroids.shape[-1])
        )
        default_descriptors_depth_all = jnp.nan * jnp.ones(
            shape=(num_centroids, depth, num_evals, centroids.shape[-1])
        )
        default_reproducibilities = -jnp.inf * jnp.ones(shape=num_centroids)
        default_reproducibilities_depth = -jnp.inf * jnp.ones(
            shape=(num_centroids, depth)
        )

        repertoire = ArchiveSamplingDeltaRepertoire(
            genotypes=default_genotypes,
            genotypes_depth=default_genotypes_depth,
            fitnesses=default_fitnesses,
            fitnesses_depth=default_fitnesses_depth,
            fitnesses_depth_all=default_fitnesses_depth_all,
            reproducibilities=default_reproducibilities,
            reproducibilities_depth=default_reproducibilities_depth,
            descriptors=default_descriptors,
            descriptors_depth=default_descriptors_depth,
            descriptors_depth_all=default_descriptors_depth_all,
            centroids=centroids,
        )

        # Add initial values to the grid
        new_repertoire = repertoire.add(
            batch_of_genotypes=genotypes,
            batch_of_all_descriptors=descriptors,
            batch_of_all_fitnesses=fitnesses,
            batch_of_extra_scores=extra_scores,
            fitness_extractor=fitness_extractor,
            fitness_reproducibility_extractor=fitness_reproducibility_extractor,
            descriptor_extractor=descriptor_extractor,
            descriptor_reproducibility_extractor=descriptor_reproducibility_extractor,
            delta_fitness=delta_fitness,
            delta_reproducibility=delta_reproducibility,
        )

        return new_repertoire  # type: ignore

    @jax.jit
    def empty(self) -> ArchiveSamplingDeltaRepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty ArchiveSamplingDeltaRepertoire
        """

        repertoire = super().empty()
        new_reproducibilities = jnp.full_like(self.reproducibilities, -jnp.inf)
        new_reproducibilities_depth = jnp.full_like(
            self.reproducibilities_depth, -jnp.inf
        )

        return ArchiveSamplingDeltaRepertoire(
            genotypes=repertoire.genotypes,
            genotypes_depth=repertoire.genotypes_depth,
            fitnesses=repertoire.fitnesses,
            fitnesses_depth=repertoire.fitnesses_depth,
            fitnesses_depth_all=repertoire.fitnesses_depth_all,
            reproducibilities=new_reproducibilities,
            reproducibilities_depth=new_reproducibilities_depth,
            descriptors=repertoire.descriptors,
            descriptors_depth=repertoire.descriptors_depth,
            descriptors_depth_all=repertoire.descriptors_depth_all,
            centroids=repertoire.centroids,
        )
