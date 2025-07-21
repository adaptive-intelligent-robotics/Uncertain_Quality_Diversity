from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import get_cells_indices
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype

from core.containers.archive_sampling_repertoire import ArchiveSamplingRepertoire


class ArchiveSamplingDeltaReprodRepertoire(ArchiveSamplingRepertoire):
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
    max_reproducibility: jnp.ndarray
    max_fitness: jnp.ndarray

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
        jnp.save(path + "max_reproducibility.npy", self.max_reproducibility)
        jnp.save(path + "max_fitness.npy", self.max_fitness)

    @classmethod
    def load(
        cls, reconstruction_fn: Callable, path: str = "./"
    ) -> ArchiveSamplingDeltaReprodRepertoire:
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
        max_reproducibility = jnp.load(path + "max_reproducibility.npy")
        max_fitness = jnp.load(path + "max_fitness.npy")

        return ArchiveSamplingDeltaReprodRepertoire(
            genotypes=repertoire.genotypes,
            genotypes_depth=repertoire.genotypes_depth,
            fitnesses=repertoire.fitnesses,
            fitnesses_depth=repertoire.fitnesses_depth,
            fitnesses_depth_all=repertoire.fitnesses_depth_all,
            reproducibilities=reproducibilities,
            reproducibilities_depth=reproducibilities_depth,
            descriptors=repertoire.descriptors,
            descriptors_depth_all=repertoire.descriptors_depth_all,
            evaluations_depth=repertoire.evaluations_depth,
            total_evaluations=repertoire.total_evaluations,
            centroids=repertoire.centroids,
            dims=repertoire.dims,
            max_reproducibility=max_reproducibility,
            max_fitness=max_fitness,
        )

    @partial(
        jax.jit,
        static_argnames=(
            "fitness_extractor",
            "fitness_reproducibility_extractor",
            "descriptor_extractor",
            "descriptor_reproducibility_extractor",
            "use_weighting",
            "delta_fitness",
            "delta_reproducibility",
            "rho",
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
        use_weighting: bool,
        delta_fitness: float,
        delta_reproducibility: float,
        rho: float,
    ) -> ArchiveSamplingDeltaReprodRepertoire:
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

        num_centroids = self.centroids.shape[0]
        depth = self.dims.shape[0]
        out_of_bound = num_centroids * depth  # Index of non-added individuals

        # Compute batch of descriptor
        batch_of_descriptors = descriptor_extractor(batch_of_all_descriptors)

        # Compute batch of fitness
        batch_of_fitnesses = fitness_extractor(batch_of_all_fitnesses)

        # Compute batch of reproducibility
        # WARNING: in the case of descriptors_reproducibility, take average over dimensions
        batch_of_reproducibilities = descriptor_reproducibility_extractor(
            batch_of_all_descriptors
        )
        batch_of_reproducibilities = jnp.average(batch_of_reproducibilities, axis=-1)

        # WARNING: takes the negative as the extractor is actually returning the inverse of reproducibility
        batch_of_reproducibilities = -batch_of_reproducibilities

        # Compute indices
        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)

        # Filter dead individuals
        batch_of_indices = jnp.where(
            batch_of_fitnesses > -jnp.inf,
            batch_of_indices,
            out_of_bound,
        )

        # Compute new maximum reproducibility and fitness
        new_max_reproducibility = jnp.max(batch_of_reproducibilities)
        new_max_reproducibility = jnp.maximum(
            new_max_reproducibility, self.max_reproducibility
        )
        new_max_fitness = jnp.max(batch_of_fitnesses)
        new_max_fitness = jnp.maximum(new_max_fitness, self.max_fitness)
        new_repertoire = self.replace(
            max_reproducibility=new_max_reproducibility,
            max_fitness=new_max_fitness,
        )

        # If using weighting
        if use_weighting:

            # Compute the weight based on the delta
            weight_fitness = 1
            weight_reproducibility = (delta_fitness + rho) / (
                delta_reproducibility + rho
            )

            # Compute comparison metrics for new individuals
            batch_of_comparison_metrics = (
                weight_fitness * batch_of_fitnesses
                + weight_reproducibility * batch_of_reproducibilities
            )
            batch_of_comparison_metrics = jnp.where(
                jnp.isnan(batch_of_comparison_metrics),
                -jnp.inf,
                batch_of_comparison_metrics,
            )

            # Compute comparison metrics for repertoire content
            repertoire_comparison_metrics = (
                weight_fitness * new_repertoire.fitnesses
                + weight_reproducibility * new_repertoire.reproducibilities
            )
            repertoire_comparison_metrics = jnp.where(
                new_repertoire.fitnesses > -jnp.inf,
                repertoire_comparison_metrics,
                -jnp.inf,
            )
            current_comparison_metrics = jnp.take_along_axis(
                repertoire_comparison_metrics, batch_of_indices, 0
            )

            # Get final indices of individuals addded to top layer of the grid
            # (i.e. best indivs added in: genotypes, fitnesses, descriptors)
            best_comparison_metrics = jax.ops.segment_max(
                batch_of_comparison_metrics,
                batch_of_indices,
                num_segments=num_centroids,
            )
            filter_comparison_metrics = jnp.where(
                best_comparison_metrics[batch_of_indices]
                == batch_of_comparison_metrics,
                batch_of_comparison_metrics,
                -jnp.inf,
            )
            final_batch_of_max_indices = jnp.where(
                filter_comparison_metrics > current_comparison_metrics,
                batch_of_indices,
                out_of_bound,
            )

            # Compute comparison metrics for repertoire depth content
            repertoire_comparison_metrics_depth = (
                weight_fitness * new_repertoire.fitnesses_depth
                + weight_reproducibility * new_repertoire.reproducibilities_depth
            )
            repertoire_comparison_metrics_depth = jnp.where(
                new_repertoire.fitnesses_depth > -jnp.inf,
                repertoire_comparison_metrics_depth,
                -jnp.inf,
            )

            # Get final indices of individuals added to the depth of the grid
            # (i.e. indivs in: genotypes_depth, fitnesses_depth_all, descriptors_depth_all)
            final_batch_of_indices = self._place_indivs(
                batch_of_indices=batch_of_indices,
                batch_of_comparison_metrics=batch_of_comparison_metrics,
                comparison_metrics_depth=repertoire_comparison_metrics_depth,
            )

            # Create new grid
            new_grid_genotypes_depth = jax.tree_map(
                lambda grid_genotypes, new_genotypes: grid_genotypes.at[
                    final_batch_of_indices
                ].set(new_genotypes),
                new_repertoire.genotypes_depth,
                batch_of_genotypes,
            )
            new_grid_genotypes = jax.tree_map(
                lambda grid_genotypes, new_genotypes: grid_genotypes.at[
                    final_batch_of_max_indices
                ].set(new_genotypes),
                new_repertoire.genotypes,
                batch_of_genotypes,
            )

            # Compute new fitness and descriptors
            new_fitnesses = new_repertoire.fitnesses.at[final_batch_of_max_indices].set(
                batch_of_fitnesses
            )
            new_fitnesses_depth = new_repertoire.fitnesses_depth.at[
                final_batch_of_indices
            ].set(batch_of_fitnesses)
            new_fitnesses_depth_all = new_repertoire.fitnesses_depth_all.at[
                final_batch_of_indices
            ].set(batch_of_all_fitnesses)
            new_descriptors = new_repertoire.descriptors.at[
                final_batch_of_max_indices
            ].set(batch_of_descriptors)
            new_descriptors_depth_all = new_repertoire.descriptors_depth_all.at[
                final_batch_of_indices
            ].set(batch_of_all_descriptors)
            new_reproducibilities = new_repertoire.reproducibilities.at[
                final_batch_of_max_indices
            ].set(batch_of_reproducibilities)
            new_reproducibilities_depth = new_repertoire.reproducibilities_depth.at[
                final_batch_of_indices
            ].set(batch_of_reproducibilities)

            # Compute new evaluations
            batch_of_evaluations = batch_of_extra_scores["num_evaluations"]
            new_evaluations_depth = new_repertoire.evaluations_depth.at[
                final_batch_of_indices
            ].set(batch_of_evaluations)
            new_total_evaluations = new_repertoire.total_evaluations + jnp.sum(
                batch_of_evaluations
            )

            return new_repertoire.replace(  # type: ignore
                genotypes=new_grid_genotypes,
                genotypes_depth=new_grid_genotypes_depth,
                fitnesses=new_fitnesses,
                fitnesses_depth=new_fitnesses_depth,
                fitnesses_depth_all=new_fitnesses_depth_all,
                reproducibilities=new_reproducibilities,
                reproducibilities_depth=new_reproducibilities_depth,
                descriptors=new_descriptors,
                descriptors_depth_all=new_descriptors_depth_all,
                evaluations_depth=new_evaluations_depth,
                total_evaluations=new_total_evaluations,
            )

        # If using delta version, compare to elites one by one
        def _compare_elite_one(
            carry: Tuple,
            unused: Tuple,
        ) -> Tuple:

            # unwrap data
            (
                cell_genotypes,
                cell_all_descriptors,
                cell_fitnesses,
                cell_all_fitnesses,
                cell_reproducibilities,
                cell_evaluations,
                genotype,
                all_descriptor,
                fitness,
                all_fitness,
                reproducibility,
                evaluation,
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
            new_evaluation = jnp.where(
                final_condition, cell_evaluations.at[elite_index].get(), evaluation
            )
            new_elite_index = elite_index + 1

            # update grid
            elite_index = jnp.where(final_condition, elite_index, num_centroids)
            new_cell_genotypes = jax.tree_util.tree_map(
                lambda x, y: x.at[elite_index].set(y), cell_genotypes, genotype
            )
            new_cell_all_descriptors = cell_all_descriptors.at[elite_index].set(
                all_descriptor
            )
            new_cell_fitnesses = cell_fitnesses.at[elite_index].set(fitness)
            new_cell_all_fitnesses = cell_all_fitnesses.at[elite_index].set(all_fitness)
            new_cell_reproducibilities = cell_reproducibilities.at[elite_index].set(
                reproducibility
            )
            new_cell_evaluations = cell_evaluations.at[elite_index].set(evaluation)

            # return new grid
            return (  # type: ignore
                new_cell_genotypes,
                new_cell_all_descriptors,
                new_cell_fitnesses,
                new_cell_all_fitnesses,
                new_cell_reproducibilities,
                new_cell_evaluations,
                new_genotype,
                new_all_descriptor,
                new_fitness,
                new_all_fitness,
                new_reproduciblity,
                new_evaluation,
                new_elite_index,
            ), ()

        # If using delta version, add individuals one by one
        def _add_one(
            carry: Tuple,
            data: Tuple,
        ) -> Tuple:

            # unwrap data
            (
                reshape_genotypes,
                reshape_all_descriptors,
                reshape_fitnesses,
                reshape_all_fitnesses,
                reshape_reproducibilities,
                reshape_evaluations,
            ) = carry
            (
                genotype,
                descriptor,
                all_descriptor,
                fitness,
                all_fitness,
                reproducibility,
                evaluation,
                index,
            ) = data

            # get cell values
            cell_genotypes = jax.tree_util.tree_map(
                lambda x: x.at[index].get(), reshape_genotypes
            )
            cell_all_descriptors = reshape_all_descriptors.at[index].get()
            cell_fitnesses = reshape_fitnesses.at[index].get()
            cell_all_fitnesses = reshape_all_fitnesses.at[index].get()
            cell_reproducibilities = reshape_reproducibilities.at[index].get()
            cell_evaluations = reshape_evaluations.at[index].get()

            # compare to elites one by one
            (
                new_cell_genotypes,
                new_cell_all_descriptors,
                new_cell_fitnesses,
                new_cell_all_fitnesses,
                new_cell_reproducibilities,
                new_cell_evaluations,
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
                    cell_all_descriptors,
                    cell_fitnesses,
                    cell_all_fitnesses,
                    cell_reproducibilities,
                    cell_evaluations,
                    genotype,
                    all_descriptor,
                    fitness,
                    all_fitness,
                    reproducibility,
                    evaluation,
                    0,
                ),
                (),
                length=depth,
            )

            # update depth grid
            new_reshape_genotypes = jax.tree_util.tree_map(
                lambda x, y: x.at[index].set(y), reshape_genotypes, new_cell_genotypes
            )
            new_reshape_all_descriptors = reshape_all_descriptors.at[index].set(
                new_cell_all_descriptors
            )
            new_reshape_fitnesses = reshape_fitnesses.at[index].set(new_cell_fitnesses)
            new_reshape_all_fitnesses = reshape_all_fitnesses.at[index].set(
                new_cell_all_fitnesses
            )
            new_reshape_reproducibilities = reshape_reproducibilities.at[index].set(
                new_cell_reproducibilities,
            )
            new_reshape_evaluations = reshape_evaluations.at[index].set(
                new_cell_evaluations
            )

            # return new grid
            return (
                new_reshape_genotypes,
                new_reshape_all_descriptors,
                new_reshape_fitnesses,
                new_reshape_all_fitnesses,
                new_reshape_reproducibilities,
                new_reshape_evaluations,
            ), ()

        # reshape grid element to scan
        reshape_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (num_centroids, depth) + x.shape[1:]),
            new_repertoire.genotypes_depth,
        )
        reshape_all_descriptors = jnp.reshape(
            new_repertoire.descriptors_depth_all,
            (num_centroids, depth) + new_repertoire.descriptors_depth_all.shape[1:],
        )
        reshape_fitnesses = jnp.reshape(
            new_repertoire.fitnesses_depth, (num_centroids, depth)
        )
        reshape_all_fitnesses = jnp.reshape(
            new_repertoire.fitnesses_depth_all,
            (num_centroids, depth) + new_repertoire.fitnesses_depth_all.shape[1:],
        )
        reshape_reproducibilities = jnp.reshape(
            new_repertoire.reproducibilities_depth, (num_centroids, depth)
        )
        reshape_evaluations = jnp.reshape(
            new_repertoire.evaluations_depth, (num_centroids, depth)
        )

        # scan the addition operation for all the individuals
        batch_of_evaluations = batch_of_extra_scores["num_evaluations"]
        (
            reshape_genotypes,
            reshape_all_descriptors,
            reshape_fitnesses,
            reshape_all_fitnesses,
            reshape_reproducibilities,
            reshape_evaluations,
        ), _ = jax.lax.scan(
            _add_one,
            (
                reshape_genotypes,
                reshape_all_descriptors,
                reshape_fitnesses,
                reshape_all_fitnesses,
                reshape_reproducibilities,
                reshape_evaluations,
            ),
            (
                batch_of_genotypes,
                batch_of_descriptors,
                batch_of_all_descriptors,
                batch_of_fitnesses,
                batch_of_all_fitnesses,
                batch_of_reproducibilities,
                batch_of_evaluations,
                batch_of_indices,
            ),
        )

        # update repertoire
        new_genotypes = jax.tree_util.tree_map(
            lambda x: x.at[:, 0].get(),
            reshape_genotypes,
        )
        new_descriptors = descriptor_extractor(reshape_all_descriptors.at[:, 0].get())
        new_fitnesses = reshape_fitnesses.at[:, 0].get()
        new_reproducibilities = reshape_reproducibilities.at[:, 0].get()
        new_total_evaluations = new_repertoire.total_evaluations + jnp.sum(
            batch_of_evaluations
        )

        # update depth repertoire
        new_genotypes_depth = jax.tree_util.tree_map(
            lambda x, y: jnp.reshape(x, y.shape),
            reshape_genotypes,
            new_repertoire.genotypes_depth,
        )
        new_descriptors_depth_all = jnp.reshape(
            reshape_all_descriptors,
            new_repertoire.descriptors_depth_all.shape,
        )
        new_fitnesses_depth = jnp.reshape(
            reshape_fitnesses, new_repertoire.fitnesses_depth.shape
        )
        new_fitnesses_depth_all = jnp.reshape(
            reshape_all_fitnesses,
            new_repertoire.fitnesses_depth_all.shape,
        )
        new_reproducibilities_depth = jnp.reshape(
            reshape_reproducibilities, new_repertoire.reproducibilities_depth.shape
        )
        new_evaluations_depth = jnp.reshape(
            reshape_evaluations,
            new_repertoire.evaluations_depth.shape,
        )
        new_repertoire = new_repertoire.replace(
            genotypes=new_genotypes,
            descriptors=new_descriptors,
            fitnesses=new_fitnesses,
            reproducibilities=new_reproducibilities,
            genotypes_depth=new_genotypes_depth,
            descriptors_depth_all=new_descriptors_depth_all,
            fitnesses_depth=new_fitnesses_depth,
            fitnesses_depth_all=new_fitnesses_depth_all,
            reproducibilities_depth=new_reproducibilities_depth,
            evaluations_depth=new_evaluations_depth,
            total_evaluations=new_total_evaluations,
        )

        return new_repertoire  # type: ignore

    @jax.jit
    def set_total_evaluations(
        self, total_evaluations: int
    ) -> ArchiveSamplingDeltaReprodRepertoire:
        """Set up current number of evaluations."""
        return self.replace(total_evaluations=total_evaluations)  # type: ignore

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
        use_weighting: bool,
        delta_fitness: float,
        delta_reproducibility: float,
        rho: float,
    ) -> ArchiveSamplingDeltaReprodRepertoire:
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
        repertoire = super().init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            centroids=centroids,
            depth=depth,
            num_evals=num_evals,
            fitness_extractor=fitness_extractor,
            fitness_reproducibility_extractor=fitness_reproducibility_extractor,
            descriptor_extractor=descriptor_extractor,
            descriptor_reproducibility_extractor=descriptor_reproducibility_extractor,
        )
        num_centroids = centroids.shape[0]
        default_reproducibilities = -jnp.inf * jnp.ones(shape=num_centroids)
        default_reproducibilities_depth = -jnp.inf * jnp.ones(
            shape=(num_centroids * depth)
        )
        default_max_reproducibility = -jnp.inf * jnp.ones((1))
        default_max_fitness = -jnp.inf * jnp.ones((1))

        repertoire = ArchiveSamplingDeltaReprodRepertoire(
            genotypes=repertoire.genotypes,
            genotypes_depth=repertoire.genotypes_depth,
            fitnesses=repertoire.fitnesses,
            fitnesses_depth=repertoire.fitnesses_depth,
            fitnesses_depth_all=repertoire.fitnesses_depth_all,
            reproducibilities=default_reproducibilities,
            reproducibilities_depth=default_reproducibilities_depth,
            descriptors=repertoire.descriptors,
            descriptors_depth_all=repertoire.descriptors_depth_all,
            evaluations_depth=repertoire.evaluations_depth,
            total_evaluations=repertoire.total_evaluations,
            centroids=repertoire.centroids,
            dims=repertoire.dims,
            max_reproducibility=default_max_reproducibility,
            max_fitness=default_max_fitness,
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
            use_weighting=use_weighting,
            delta_fitness=delta_fitness,
            delta_reproducibility=delta_reproducibility,
            rho=rho,
        )

        return new_repertoire  # type: ignore

    @jax.jit
    def empty(self) -> ArchiveSamplingDeltaReprodRepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty ArchiveSamplingDeltaReprodRepertoire
        """

        repertoire = super().empty()
        new_reproducibilities = jnp.full_like(self.reproducibilities, -jnp.inf)
        new_reproducibilities_depth = jnp.full_like(
            self.reproducibilities_depth, -jnp.inf
        )
        new_max_reproducibility = -jnp.inf * jnp.ones_like(self.max_reproducibility)
        new_max_fitness = -jnp.inf * jnp.ones_like(self.max_fitness)

        return ArchiveSamplingDeltaReprodRepertoire(
            genotypes=repertoire.genotypes,
            genotypes_depth=repertoire.genotypes_depth,
            fitnesses=repertoire.fitnesses,
            fitnesses_depth=repertoire.fitnesses_depth,
            fitnesses_depth_all=repertoire.fitnesses_depth_all,
            reproducibilities=new_reproducibilities,
            reproducibilities_depth=new_reproducibilities_depth,
            descriptors=repertoire.descriptors,
            descriptors_depth_all=repertoire.descriptors_depth_all,
            evaluations_depth=repertoire.evaluations_depth,
            total_evaluations=repertoire.total_evaluations,
            centroids=repertoire.centroids,
            dims=repertoire.dims,
            max_reproducibility=new_max_reproducibility,
            max_fitness=new_max_fitness,
        )
