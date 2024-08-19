from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from qdax.core.containers.mapelites_repertoire import get_cells_indices
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype

from core.containers.depth_repertoire import DeepMapElitesRepertoire

class EvaluationsDeepMapElitesRepertoire(DeepMapElitesRepertoire):
    """
    Class for the deep repertoire in Map Elites storing evaluations.

    Args:
        evaluations_depth: an array that contains the number of evaluations of
            individuals in the grid.
        total_evaluations: the total number of evaluations spent.
    """

    evaluations_depth: jnp.ndarray
    total_evaluations: int

    def save(self, path: str = "./") -> None:
        """Saves the grid on disk in the form of .npy files.

        Flattens the genotypes to store it with .npy format. Supposes that
        a user will have access to the reconstruction function when loading
        the genotypes.

        Args:
            path: Path where the data will be saved. Defaults to "./".
        """

        def flatten_genotype(genotype: Genotype) -> jnp.ndarray:
            flatten_genotype, _ = ravel_pytree(genotype)
            return flatten_genotype

        # flatten all the genotypes
        flat_genotypes = jax.vmap(flatten_genotype)(self.genotypes)
        flat_genotypes_depth = jax.vmap(flatten_genotype)(self.genotypes_depth)

        # save data
        jnp.save(path + "genotypes.npy", flat_genotypes)
        jnp.save(path + "genotypes_depth.npy", flat_genotypes_depth)
        jnp.save(path + "fitnesses.npy", self.fitnesses)
        jnp.save(path + "fitnesses_depth.npy", self.fitnesses_depth)
        jnp.save(path + "descriptors.npy", self.descriptors)
        jnp.save(path + "descriptors_depth.npy", self.descriptors_depth)
        jnp.save(path + "evaluations_depth.npy", self.evaluations_depth)
        jnp.save(path + "total_evaluations.npy", self.total_evaluations)
        jnp.save(path + "centroids.npy", self.centroids)
        jnp.save(path + "dims.npy", self.dims)

    @classmethod
    def load(
        cls, reconstruction_fn: Callable, path: str = "./"
    ) -> EvaluationsDeepMapElitesRepertoire:
        """Loads a MAP Elites Grid.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            A MAP Elites Repertoire.
        """

        flat_genotypes = jnp.load(path + "genotypes.npy")
        genotypes = jax.vmap(reconstruction_fn)(flat_genotypes)
        flat_genotypes_depth = jnp.load(path + "genotypes_depth.npy")
        genotypes_depth = jax.vmap(reconstruction_fn)(flat_genotypes_depth)

        fitnesses = jnp.load(path + "fitnesses.npy")
        fitnesses_depth = jnp.load(path + "fitnesses_depth.npy")
        descriptors = jnp.load(path + "descriptors.npy")
        descriptors_depth = jnp.load(path + "descriptors_depth.npy")
        evaluations_depth = jnp.load(path + "evaluations_depth.npy")
        total_evaluations = jnp.load(path + "total_evaluations.npy")
        centroids = jnp.load(path + "centroids.npy")
        dims = jnp.load(path + "dims.npy")

        return EvaluationsDeepMapElitesRepertoire(
            genotypes=genotypes,
            genotypes_depth=genotypes_depth,
            fitnesses=fitnesses,
            fitnesses_depth=fitnesses_depth,
            descriptors=descriptors,
            descriptors_depth=descriptors_depth,
            evaluations_depth=evaluations_depth,
            total_evaluations=total_evaluations,
            centroids=centroids,
            dims=dims,
        )

    @jax.jit
    def set_total_evaluations(
        self, total_evaluations: int
    ) -> EvaluationsDeepMapElitesRepertoire:
        """Set up current number of evaluations."""
        return self.replace(total_evaluations=total_evaluations)  # type: ignore

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: ExtraScores,
    ) -> EvaluationsDeepMapElitesRepertoire:
        """
        Add a batch of elements to the repertoire.

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

        out_of_bound = max(
            self.dims.shape[0] * self.centroids.shape[0], batch_of_fitnesses.shape[0]
        )
        num_centroids = self.centroids.shape[0]
        depth = self.dims.shape[0]

        # Get indices
        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)

        # Filter dead individuals
        batch_of_indices = jnp.where(
            batch_of_fitnesses > -jnp.inf,
            batch_of_indices,
            out_of_bound,
        )

        # Get evaluations
        batch_of_evaluations = batch_of_extra_scores["num_evaluations"]

        @jax.jit
        def _add_per_cell(
            cell_idx: jnp.ndarray,
            cell_genotype_depth: Genotype,
            cell_fitnesses_depth: Fitness,
            cell_descriptors_depth: Descriptor,
            cell_evaluations_depth: jnp.ndarray,
        ) -> Tuple[
            Genotype, Fitness, Descriptor, jnp.ndarray, Genotype, Fitness, Descriptor
        ]:
            """
            For a given cell with index cell_idx, filter candidate
            indivs for this cell, and add them to it, reordering so
            highest-fitness individuals are first.

            Args:
              cell_idx: cell index
              cell_genotype_depth: genotype in the cell
              cell_fitnesses_depth: fitnesses in the cell
              cell_descriptors_depth: descriptors in the cell
              cell_evaluations_depth: evaluations in the cell

            Returns:
              new_cell_genotype_depth
              new_cell_fitnesses_depth
              new_cell_descriptors_depth
              new_cell_evaluations_depth
              new_cell_genotype: genotype in the top layer of the cell
              new_cell_fitnesses: fitnesses in the top layer of the cell
              new_cell_descriptors: descriptors in the top layer of the cell
            """

            # Order existing and candidate indivs by fitness
            candidate_fitnesses = jnp.where(
                batch_of_indices == cell_idx, batch_of_fitnesses, -jnp.inf
            )
            all_fitnesses = jnp.concatenate(
                [cell_fitnesses_depth, candidate_fitnesses],
                axis=0,
            )
            _, final_indices = jax.lax.top_k(all_fitnesses, depth)

            # First, move around existing indivs to follow order
            cell_indices = jnp.where(
                final_indices < depth,
                final_indices,
                out_of_bound,
            )
            new_cell_genotype_depth = jax.tree_map(
                lambda x: x.at[cell_indices].get(),
                cell_genotype_depth,
            )
            new_cell_fitnesses_depth = cell_fitnesses_depth.at[cell_indices].get()
            new_cell_descriptors_depth = cell_descriptors_depth.at[cell_indices].get()
            new_cell_evaluations_depth = cell_evaluations_depth.at[cell_indices].get()

            # Second, add the candidate indivs
            candidate_indices = jnp.where(
                final_indices >= depth,
                final_indices - depth,
                out_of_bound,
            )
            depth_indices = jnp.where(
                candidate_indices < out_of_bound,
                jnp.arange(0, depth, step=1),
                out_of_bound,
            )
            new_cell_genotype_depth = jax.tree_map(
                lambda x, y: x.at[depth_indices].set(y[candidate_indices]),
                new_cell_genotype_depth,
                batch_of_genotypes,
            )
            new_cell_fitnesses_depth = new_cell_fitnesses_depth.at[depth_indices].set(
                batch_of_fitnesses[candidate_indices]
            )
            new_cell_descriptors_depth = new_cell_descriptors_depth.at[
                depth_indices
            ].set(batch_of_descriptors[candidate_indices])
            new_cell_evaluations_depth = new_cell_evaluations_depth.at[
                depth_indices
            ].set(batch_of_evaluations[candidate_indices])

            # Also return the top layer of the grid
            new_cell_genotype = jax.tree_map(
                lambda x: x.at[0].get(),
                new_cell_genotype_depth,
            )
            new_cell_fitnesses = new_cell_fitnesses_depth.at[0].get()
            new_cell_descriptors = new_cell_descriptors_depth.at[0].get()

            # Return the updated cell
            return (
                new_cell_genotype_depth,
                new_cell_fitnesses_depth,
                new_cell_descriptors_depth,
                new_cell_evaluations_depth,
                new_cell_genotype,
                new_cell_fitnesses,
                new_cell_descriptors,
            )

        # Add individuals cell by cell
        (
            new_genotype_depth,
            new_fitnesses_depth,
            new_descriptors_depth,
            new_evaluations_depth,
            new_genotype,
            new_fitnesses,
            new_descriptors,
        ) = jax.vmap(_add_per_cell)(
            jnp.arange(0, num_centroids, step=1),
            self.genotypes_depth,
            self.fitnesses_depth,
            self.descriptors_depth,
            self.evaluations_depth,
        )

        # Compute new total evaluations
        new_total_evaluations = self.total_evaluations + jnp.sum(batch_of_evaluations)

        return self.replace(  # type:ignore
            genotypes=new_genotype,
            genotypes_depth=new_genotype_depth,
            fitnesses=new_fitnesses,
            fitnesses_depth=new_fitnesses_depth,
            descriptors=new_descriptors,
            descriptors_depth=new_descriptors_depth,
            evaluations_depth=new_evaluations_depth,
            total_evaluations=new_total_evaluations,
        )

    @classmethod
    def init(
        cls,
        genotypes: Genotype,
        fitnesses: Fitness,
        descriptors: Descriptor,
        extra_scores: ExtraScores,
        centroids: Centroid,
        depth: int,
    ) -> EvaluationsDeepMapElitesRepertoire:
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
            centroids: tesselation centroids of shape (batch_size, num_descriptors)

        Returns:
            an initialized MAP-Elite repertoire
        """

        # Initialize grid with default values
        num_centroids = centroids.shape[0]
        default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)
        default_fitnesses_depth = -jnp.inf * jnp.ones(shape=(num_centroids, depth))
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
        default_evaluations_depth = jnp.zeros(shape=(num_centroids, depth))
        default_total_evaluations = 0
        dims = jnp.zeros(shape=(depth))

        repertoire = EvaluationsDeepMapElitesRepertoire(
            genotypes=default_genotypes,
            genotypes_depth=default_genotypes_depth,
            fitnesses=default_fitnesses,
            fitnesses_depth=default_fitnesses_depth,
            descriptors=default_descriptors,
            descriptors_depth=default_descriptors_depth,
            evaluations_depth=default_evaluations_depth,
            total_evaluations=default_total_evaluations,
            centroids=centroids,
            dims=dims,
        )

        # Add initial values to the grid
        new_repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        return new_repertoire  # type: ignore

    @jax.jit
    def empty(self) -> EvaluationsDeepMapElitesRepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty EvaluationsDeepMapElitesRepertoire
        """

        new_fitnesses = jnp.full_like(self.fitnesses, -jnp.inf)
        new_fitnesses_depth = jnp.full_like(self.fitnesses_depth, -jnp.inf)
        new_descriptors = jnp.zeros_like(self.descriptors)
        new_descriptors_depth = jnp.zeros_like(self.descriptors_depth)
        new_genotypes = jax.tree_map(lambda x: jnp.zeros_like(x), self.genotypes)
        new_genotypes_depth = jax.tree_map(
            lambda x: jnp.zeros_like(x), self.genotypes_depth
        )
        new_evaluations_depth = jnp.zeros_like(self.evaluations_depth)
        new_total_evaluations = 0
        return EvaluationsDeepMapElitesRepertoire(
            genotypes=new_genotypes,
            genotypes_depth=new_genotypes_depth,
            fitnesses=new_fitnesses,
            fitnesses_depth=new_fitnesses_depth,
            descriptors=new_descriptors,
            descriptors_depth=new_descriptors_depth,
            evaluations_depth=new_evaluations_depth,
            total_evaluations=new_total_evaluations,
            centroids=self.centroids,
            dims=self.dims,
        )
