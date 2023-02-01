from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import get_cells_indices
from qdax.types import Centroid, Descriptor, ExtraScores, Fitness, Genotype

from core.containers.depth_repertoire import DeepMapElitesRepertoire


class EvaluationDepthRepertoire(DeepMapElitesRepertoire):
    """
    Class for the repertoire of the extended adaptive sampling algorithm. Simply
    add a track of the number of evaluations of each indivs stored in extra scores.

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

        super().save(path=path)
        jnp.save(path + "evaluations_depth.npy", self.evaluations_depth)
        jnp.save(path + "total_evaluations.npy", self.total_evaluations)

    @classmethod
    def load(
        cls, reconstruction_fn: Callable, path: str = "./"
    ) -> EvaluationDepthRepertoire:
        """Loads a MAP Elites Grid.

        Args:
            reconstruction_fn: Function to reconstruct a PyTree
                from a flat array.
            path: Path where the data is saved. Defaults to "./".

        Returns:
            A MAP Elites Repertoire.
        """

        repertoire = super().load(reconstruction_fn=reconstruction_fn, path=path)
        evaluations_depth = jnp.load(path + "evaluations_depth.npy")
        total_evaluations = jnp.load(path + "total_evaluations.npy")

        return EvaluationDepthRepertoire(
            genotypes=repertoire.genotypes,
            genotypes_depth=repertoire.genotypes_depth,
            fitnesses=repertoire.fitnesses,
            fitnesses_depth=repertoire.fitnesses_depth,
            descriptors=repertoire.descriptors,
            descriptors_depth=repertoire.descriptors_depth,
            evaluations_depth=evaluations_depth,
            total_evaluations=total_evaluations,
            centroids=repertoire.centroids,
            dims=repertoire.dims,
        )

    @jax.jit
    def set_total_evaluations(
        self, total_evaluations: int
    ) -> EvaluationDepthRepertoire:
        """Set up current number of evaluations."""
        return self.replace(total_evaluations=total_evaluations)  # type: ignore

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: ExtraScores,
    ) -> EvaluationDepthRepertoire:
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

        out_of_bound = (
            self.dims.shape[0] * self.centroids.shape[0]
        )  # Index of non-added individuals

        # Get indices for given descriptors
        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)

        # Filter dead individuals
        batch_of_indices = jnp.where(
            batch_of_fitnesses > -jnp.inf,
            batch_of_indices,
            out_of_bound,
        )

        # Get final indices of individuals addded to top layer of the grid
        # (i.e. best indivs added in: genotypes, fitnesses, descriptors)
        best_fitnesses = jax.ops.segment_max(
            batch_of_fitnesses,
            batch_of_indices,
            num_segments=self.centroids.shape[0],
        )
        filter_fitnesses = jnp.where(
            best_fitnesses[batch_of_indices] == batch_of_fitnesses,
            batch_of_fitnesses,
            -jnp.inf,
        )
        current_fitnesses = jnp.take_along_axis(self.fitnesses, batch_of_indices, 0)
        final_batch_of_max_indices = jnp.where(
            filter_fitnesses > current_fitnesses,
            batch_of_indices,
            out_of_bound,
        )

        # Get final indices of individuals added to the depth of the grid
        # (i.e. indivs in: genotypes_depth, fitnesses_depth, descriptors_depth)
        final_batch_of_indices = self._place_indivs(
            batch_of_indices, batch_of_fitnesses
        )

        # Create new grid
        new_grid_genotypes_depth = jax.tree_map(
            lambda grid_genotypes, new_genotypes: grid_genotypes.at[
                final_batch_of_indices
            ].set(new_genotypes),
            self.genotypes_depth,
            batch_of_genotypes,
        )
        new_grid_genotypes = jax.tree_map(
            lambda grid_genotypes, new_genotypes: grid_genotypes.at[
                final_batch_of_max_indices
            ].set(new_genotypes),
            self.genotypes,
            batch_of_genotypes,
        )

        # Compute new fitness and descriptors
        new_fitnesses = self.fitnesses.at[final_batch_of_max_indices].set(
            batch_of_fitnesses
        )
        new_fitnesses_depth = self.fitnesses_depth.at[final_batch_of_indices].set(
            batch_of_fitnesses
        )
        new_descriptors = self.descriptors.at[final_batch_of_max_indices].set(
            batch_of_descriptors
        )
        new_descriptors_depth = self.descriptors_depth.at[final_batch_of_indices].set(
            batch_of_descriptors
        )

        # Compute new evaluations
        batch_of_evaluations = batch_of_extra_scores["num_evaluations"]
        new_evaluations_depth = self.evaluations_depth.at[final_batch_of_indices].set(
            batch_of_evaluations
        )
        new_total_evaluations = self.total_evaluations + jnp.sum(batch_of_evaluations)

        return EvaluationDepthRepertoire(
            genotypes=new_grid_genotypes,
            genotypes_depth=new_grid_genotypes_depth,
            fitnesses=new_fitnesses.squeeze(),
            fitnesses_depth=new_fitnesses_depth,
            descriptors=new_descriptors.squeeze(),
            descriptors_depth=new_descriptors_depth,
            evaluations_depth=new_evaluations_depth,
            total_evaluations=new_total_evaluations,
            centroids=self.centroids,
            dims=self.dims,
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
    ) -> EvaluationDepthRepertoire:
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
        repertoire = super().init(
            genotypes=genotypes,
            fitnesses=fitnesses,
            descriptors=descriptors,
            extra_scores=extra_scores,
            centroids=centroids,
            depth=depth,
        )
        default_evaluations_depth = jnp.zeros(
            shape=repertoire.dims.shape[0] * repertoire.centroids.shape[0]
        )
        default_total_evaluations = 0

        repertoire = EvaluationDepthRepertoire(
            genotypes=repertoire.genotypes,
            genotypes_depth=repertoire.genotypes_depth,
            fitnesses=repertoire.fitnesses,
            fitnesses_depth=repertoire.fitnesses_depth,
            descriptors=repertoire.descriptors,
            descriptors_depth=repertoire.descriptors_depth,
            evaluations_depth=default_evaluations_depth,
            total_evaluations=default_total_evaluations,
            centroids=repertoire.centroids,
            dims=repertoire.dims,
        )

        # Add initial values to the grid
        new_repertoire = repertoire.add(genotypes, descriptors, fitnesses, extra_scores)

        return new_repertoire  # type: ignore

    @jax.jit
    def empty(self) -> EvaluationDepthRepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty MapElitesRepertoire
        """

        repertoire = super().empty()
        new_evaluations_depth = jnp.zeros_like(self.evaluations_depth)
        new_total_evaluations = 0
        return EvaluationDepthRepertoire(
            genotypes=repertoire.genotypes,
            genotypes_depth=repertoire.genotypes_depth,
            fitnesses=repertoire.fitnesses,
            fitnesses_depth=repertoire.fitnesses_depth,
            descriptors=repertoire.descriptors,
            descriptors_depth=repertoire.descriptors_depth,
            evaluations_depth=new_evaluations_depth,
            total_evaluations=new_total_evaluations,
            centroids=repertoire.centroids,
            dims=repertoire.dims,
        )
