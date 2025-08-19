from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import get_cells_indices
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
)

from core.containers.archive_sampling_repertoire import ArchiveSamplingRepertoire


class ArchiveSamplingWeightedRepertoire(ArchiveSamplingRepertoire):
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
    ) -> ArchiveSamplingWeightedRepertoire:
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

        return ArchiveSamplingWeightedRepertoire(
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
            "delta_fitness",
            "delta_reproducibility",
            "rho",
        ),
    )
    def _compute_weights(
        self,
        delta_fitness: float,
        delta_reproducibility: float,
        rho: float,
    ) -> Tuple[float, float]:
        weight_fitness = 1
        weight_reproducibility = (delta_fitness + rho) / (delta_reproducibility + rho)
        return weight_fitness, weight_reproducibility

    @partial(
        jax.jit,
        static_argnames=(
            "fitness_extractor",
            "fitness_reproducibility_extractor",
            "descriptor_extractor",
            "descriptor_reproducibility_extractor",
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
        delta_fitness: float,
        delta_reproducibility: float,
        rho: float,
    ) -> ArchiveSamplingWeightedRepertoire:
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

        # Compute the weight based on the delta
        weight_fitness, weight_reproducibility = self._compute_weights(
            delta_fitness=delta_fitness,
            delta_reproducibility=delta_reproducibility,
            rho=rho,
        )

        # Compute weighted sum for new individuals
        batch_of_weighted_sums = (
            weight_fitness * batch_of_fitnesses
            + weight_reproducibility * batch_of_reproducibilities
        )
        batch_of_weighted_sums = jnp.where(
            jnp.isnan(batch_of_weighted_sums),
            -jnp.inf,
            batch_of_weighted_sums,
        )
        batch_of_weighted_sums = jnp.where(
            batch_of_fitnesses > -jnp.inf,
            batch_of_weighted_sums,
            -jnp.inf,
        )

        @jax.jit
        def _add_per_cell(
            cell_idx: jnp.ndarray,
            cell_genotypes_depth: Genotype,
            cell_fitnesses_depth: Fitness,
            cell_fitnesses_depth_all: Fitness,
            cell_descriptors_depth: Descriptor,
            cell_descriptors_depth_all: Descriptor,
            cell_reproducibilities_depth: jnp.ndarray,
        ) -> Tuple[
            Genotype,
            Fitness,
            Fitness,
            Descriptor,
            Descriptor,
            jnp.ndarray,
            Genotype,
            Fitness,
            Descriptor,
            jnp.ndarray,
        ]:
            """
            For a given cell with index cell_idx, filter candidate
            indivs for this cell, and add them to it, reordering so
            highest-fitness individuals are first.

            Args:
              cell_idx: cell index
              cell_genotypes_depth: genotype in the cell
              cell_fitnesses_depth: fitnesses in the cell
              cell_fitnesses_depth_all
              cell_descriptors_depth: descriptors in the cell
              cell_descriptors_depth_all
              cell_reproducibilities_depth

            Returns:
              new_cell_genotypes_depth
              new_cell_fitnesses_depth
              new_cell_fitnesses_depth_all
              new_cell_descriptors_depth
              new_cell_descriptors_depth_all
              new_cell_reproducibilities_depth
              new_cell_genotype: genotype in the top layer of the cell
              new_cell_fitnesses: fitnesses in the top layer of the cell
              new_cell_descriptors: descriptors in the top layer of the cell
              new_cell_reproducibilities: reproducibilities in the top layer of the cell
            """

            # Order existing and candidate indivs by weighted sum
            candidate_weighted_sums = jnp.where(
                batch_of_indices == cell_idx, batch_of_weighted_sums, -jnp.inf
            )
            cell_weighted_sums_depth = jnp.where(
                cell_fitnesses_depth > -jnp.inf,
                weight_fitness * cell_fitnesses_depth
                + weight_reproducibility * cell_reproducibilities_depth,
                -jnp.inf,
            )
            all_weighted_sums = jnp.concatenate(
                [cell_weighted_sums_depth, candidate_weighted_sums],
                axis=0,
            )
            _, final_indices = jax.lax.top_k(all_weighted_sums, depth)

            # First, move around existing indivs to follow order
            cell_indices = jnp.where(
                final_indices < depth,
                final_indices,
                out_of_bound,
            )
            new_cell_genotypes_depth = jax.tree_map(
                lambda x: x.at[cell_indices].get(),
                cell_genotypes_depth,
            )
            new_cell_fitnesses_depth = cell_fitnesses_depth.at[cell_indices].get()
            new_cell_fitnesses_depth_all = cell_fitnesses_depth_all.at[
                cell_indices
            ].get()
            new_cell_descriptors_depth = cell_descriptors_depth.at[cell_indices].get()
            new_cell_descriptors_depth_all = cell_descriptors_depth_all.at[
                cell_indices
            ].get()
            new_cell_reproducibilities_depth = cell_reproducibilities_depth.at[
                cell_indices
            ].get()

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
            new_cell_genotypes_depth = jax.tree_map(
                lambda x, y: x.at[depth_indices].set(y[candidate_indices]),
                new_cell_genotypes_depth,
                batch_of_genotypes,
            )
            new_cell_fitnesses_depth = new_cell_fitnesses_depth.at[depth_indices].set(
                batch_of_fitnesses[candidate_indices]
            )
            new_cell_fitnesses_depth_all = new_cell_fitnesses_depth_all.at[
                depth_indices
            ].set(batch_of_all_fitnesses[candidate_indices])
            new_cell_descriptors_depth = new_cell_descriptors_depth.at[
                depth_indices
            ].set(batch_of_descriptors[candidate_indices])
            new_cell_descriptors_depth_all = new_cell_descriptors_depth_all.at[
                depth_indices
            ].set(batch_of_all_descriptors[candidate_indices])
            new_cell_reproducibilities_depth = new_cell_reproducibilities_depth.at[
                depth_indices
            ].set(batch_of_reproducibilities[candidate_indices])

            # Also return the top layer of the grid
            new_cell_genotype = jax.tree_map(
                lambda x: x.at[0].get(),
                new_cell_genotypes_depth,
            )
            new_cell_fitnesses = new_cell_fitnesses_depth.at[0].get()
            new_cell_descriptors = new_cell_descriptors_depth.at[0].get()
            new_cell_reproducibilities = new_cell_reproducibilities_depth.at[0].get()

            # Return the updated cell
            return (
                new_cell_genotypes_depth,
                new_cell_fitnesses_depth,
                new_cell_fitnesses_depth_all,
                new_cell_descriptors_depth,
                new_cell_descriptors_depth_all,
                new_cell_reproducibilities_depth,
                new_cell_genotype,
                new_cell_fitnesses,
                new_cell_descriptors,
                new_cell_reproducibilities,
            )

        # Add individuals cell by cell
        (
            new_genotypes_depth,
            new_fitnesses_depth,
            new_fitnesses_depth_all,
            new_descriptors_depth,
            new_descriptors_depth_all,
            new_reproducibilities_depth,
            new_genotype,
            new_fitnesses,
            new_descriptors,
            new_reproducibilities,
        ) = jax.vmap(_add_per_cell)(
            jnp.arange(0, num_centroids, step=1),
            self.genotypes_depth,
            self.fitnesses_depth,
            self.fitnesses_depth_all,
            self.descriptors_depth,
            self.descriptors_depth_all,
            self.reproducibilities_depth,
        )

        return self.replace(  # type:ignore
            genotypes=new_genotype,
            genotypes_depth=new_genotypes_depth,
            fitnesses=new_fitnesses,
            fitnesses_depth=new_fitnesses_depth,
            fitnesses_depth_all=new_fitnesses_depth_all,
            descriptors=new_descriptors,
            descriptors_depth=new_descriptors_depth,
            descriptors_depth_all=new_descriptors_depth_all,
            reproducibilities=new_reproducibilities,
            reproducibilities_depth=new_reproducibilities_depth,
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
        num_evals: int,
        fitness_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        fitness_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        descriptor_reproducibility_extractor: Callable[[jnp.ndarray], jnp.ndarray],
        delta_fitness: float,
        delta_reproducibility: float,
        rho: float,
    ) -> ArchiveSamplingWeightedRepertoire:
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

        repertoire = ArchiveSamplingWeightedRepertoire(
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
            rho=rho,
        )

        return new_repertoire  # type: ignore

    @jax.jit
    def empty(self) -> ArchiveSamplingWeightedRepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty ArchiveSamplingWeightedRepertoire
        """

        repertoire = super().empty()
        new_reproducibilities = jnp.full_like(self.reproducibilities, -jnp.inf)
        new_reproducibilities_depth = jnp.full_like(
            self.reproducibilities_depth, -jnp.inf
        )

        return ArchiveSamplingWeightedRepertoire(
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

    @partial(
        jax.jit,
        static_argnames=(
            "delta_fitness",
            "delta_reproducibility",
            "rho",
        ),
    )
    def _order_repertoire(
        self,
        rows: jnp.ndarray,
        cols: jnp.ndarray,
        random_key: RNGKey,
        delta_fitness: float,
        delta_reproducibility: float,
        rho: float,
    ) -> Tuple[ArchiveSamplingWeightedRepertoire, RNGKey]:
        """
        Re-order repertoire following extraction.

        Args:
            random_key: a jax PRNG random key
            rows, cols: position of extracted individuals

        Returns:
            repertoire: the new repertoire
            random_key: an updated jax PRNG random key
        """

        # Remove extracted individuals from all grids
        new_genotypes_depth = jax.tree_util.tree_map(
            lambda x: x.at[rows, cols].set(0),
            self.genotypes_depth,
        )
        new_fitnesses_depth = self.fitnesses_depth.at[rows, cols].set(-jnp.inf)
        new_fitnesses_depth_all = self.fitnesses_depth_all.at[rows, cols].set(jnp.nan)
        new_reproducibilities_depth = self.reproducibilities_depth.at[rows, cols].set(
            -jnp.inf
        )
        new_descriptors_depth = self.descriptors_depth.at[rows, cols].set(0)
        new_descriptors_depth_all = self.descriptors_depth_all.at[rows, cols].set(
            jnp.nan
        )

        # Compute the weight based on the delta
        weight_fitness, weight_reproducibility = self._compute_weights(
            delta_fitness=delta_fitness,
            delta_reproducibility=delta_reproducibility,
            rho=rho,
        )

        def re_order_cell(
            genotypes_depth_cell: Genotype,
            fitnesses_depth_cell: Fitness,
            fitnesses_depth_all_cell: Fitness,
            reproducibilities_depth_cell: Fitness,
            descriptors_depth_cell: Descriptor,
            descriptors_depth_all_cell: Descriptor,
        ) -> Tuple[
            Genotype,
            Fitness,
            Fitness,
            jnp.ndarray,
            jnp.ndarray,
            Descriptor,
            Descriptor,
            Genotype,
            Fitness,
            Descriptor,
        ]:
            """
            Re-order a cell after extraction. Put highest fitness first and
            empty slot at the end.

            Inputs:
                genotypes_depth_cell: current genotypes of the cell
                fitnesses_depth_cell
                fitnesses_depth_all_cell
                reproducibilities_depth_cell
                descriptors_depth_cell
                descriptors_depth_all_cell

            Returns:
                genotypes_depth_cell: new genotypes of the cell
                fitnesses_depth_cell
                fitnesses_depth_all_cell
                reproducibilities_depth_cell
                descriptors_depth_cell
                descriptors_depth_all_cell
                genotypes_cell: new top layer of the cell
                fitnesses_cell
                reproducibilities_cell
                descriptors_cell
            """

            weighted_sums = (
                weight_fitness * fitnesses_depth_cell
                + weight_reproducibility * reproducibilities_depth_cell
            )

            # Get re-ordering index for given cell
            index = jnp.argsort(weighted_sums)
            index = index[::-1]

            # Re-order given cell
            genotypes_depth_cell = jax.tree_util.tree_map(
                lambda x: x.at[index].get(),
                genotypes_depth_cell,
            )
            fitnesses_depth_cell = fitnesses_depth_cell.at[index].get()
            fitnesses_depth_all_cell = fitnesses_depth_all_cell.at[index].get()
            reproducibilities_depth_cell = reproducibilities_depth_cell.at[index].get()
            descriptors_depth_cell = descriptors_depth_cell.at[index].get()
            descriptors_depth_all_cell = descriptors_depth_all_cell.at[index].get()

            # Get the top layer of the cell
            genotypes_cell = jax.tree_map(
                lambda x: x.at[0].get(),
                genotypes_depth_cell,
            )
            fitnesses_cell = fitnesses_depth_cell.at[0].get()
            reproducibilities_cell = reproducibilities_depth_cell.at[0].get()
            descriptors_cell = descriptors_depth_cell.at[0].get()

            return (
                genotypes_depth_cell,
                fitnesses_depth_cell,
                fitnesses_depth_all_cell,
                reproducibilities_depth_cell,
                descriptors_depth_cell,
                descriptors_depth_all_cell,
                genotypes_cell,
                fitnesses_cell,
                reproducibilities_cell,
                descriptors_cell,
            )

        # Re-order to put extracted individuals at the end of each cell
        (
            new_genotypes_depth,
            new_fitnesses_depth,
            new_fitnesses_depth_all,
            new_reproducibilities_depth,
            new_descriptors_depth,
            new_descriptors_depth_all,
            new_genotypes,
            new_fitnesses,
            new_reproducibilities,
            new_descriptors,
        ) = jax.vmap(re_order_cell)(
            new_genotypes_depth,
            new_fitnesses_depth,
            new_fitnesses_depth_all,
            new_reproducibilities_depth,
            new_descriptors_depth,
            new_descriptors_depth_all,
        )

        # Create the new repertoire
        repertoire = self.replace(
            genotypes=new_genotypes,
            genotypes_depth=new_genotypes_depth,
            fitnesses=new_fitnesses,
            fitnesses_depth=new_fitnesses_depth,
            fitnesses_depth_all=new_fitnesses_depth_all,
            reproducibilities=new_reproducibilities,
            reproducibilities_depth=new_reproducibilities_depth,
            descriptors=new_descriptors,
            descriptors_depth=new_descriptors_depth,
            descriptors_depth_all=new_descriptors_depth_all,
        )

        return repertoire, random_key

    @partial(
        jax.jit,
        static_argnames=(
            "num_samples",
            "delta_fitness",
            "delta_reproducibility",
            "rho",
        ),
    )
    def extract_uniform(
        self,
        random_key: RNGKey,
        num_samples: int,
        delta_fitness: float,
        delta_reproducibility: float,
        rho: float,
    ) -> Tuple[
        ArchiveSamplingWeightedRepertoire, Genotype, Fitness, Descriptor, RNGKey
    ]:
        """
        Extract num_samples random element from the grid.
        Extract means that they are removed from the grid when sampled.

        Args:
            random_key: a jax PRNG random key
            num_samples: the number of elements to be sampled

        Returns:
            repertoire: the new repertoire
            extract_genotypes: extracted genotypes
            extract_fitnesses: all fitnesses of the extracted genotypes
            extract_descriptors: all descriptors of the extracted genotypes
            random_key: an updated jax PRNG random key
        """

        num_centroids = self.fitnesses_depth.shape[0]
        depth = self.fitnesses_depth.shape[1]

        # Set probability for each individual to be sampled
        reshape_fitnesses_depth = self.fitnesses_depth.flatten()
        p = reshape_fitnesses_depth > -jnp.inf

        # Extract num_samples indivs
        p = p / jnp.sum(p)
        random_key, subkey = jax.random.split(random_key)
        indices = jax.random.choice(
            subkey,
            num_centroids * depth,
            shape=(num_samples,),
            p=p,
            replace=False,
        )
        rows, cols = jnp.divmod(indices, depth)
        rows = rows.astype(int)
        cols = cols.astype(int)

        # Extract the final genotypes, fitnesses_all and descriptors_all to return
        extract_genotypes = jax.tree_util.tree_map(
            lambda x: x.at[rows, cols].get(),
            self.genotypes_depth,
        )
        extract_fitnesses_all = self.fitnesses_depth_all.at[rows, cols].get()
        extract_descriptors_all = self.descriptors_depth_all.at[rows, cols].get()

        # Re-order repertoire following extraction
        repertoire, random_key = self._order_repertoire(
            rows=rows,
            cols=cols,
            random_key=random_key,
            delta_fitness=delta_fitness,
            delta_reproducibility=delta_reproducibility,
            rho=rho,
        )

        return (
            repertoire,
            extract_genotypes,
            extract_fitnesses_all,
            extract_descriptors_all,
            random_key,
        )

    @partial(
        jax.jit,
        static_argnames=(
            "num_samples",
            "delta_fitness",
            "delta_reproducibility",
            "rho",
            "type_prop",
        ),
    )
    def extract_prop(
        self,
        random_key: RNGKey,
        num_samples: int,
        delta_fitness: float,
        delta_reproducibility: float,
        rho: float,
        type_prop: str = "exponential",
    ) -> Tuple[
        ArchiveSamplingWeightedRepertoire, Genotype, Fitness, Descriptor, RNGKey
    ]:
        """
        Extract num_samples random element from the grid.
        Extract means that they are removed from the grid when sampled.

        Args:
            random_key: a jax PRNG random key
            num_samples: the number of elements to be sampled

        Returns:
            repertoire: the new repertoire
            extract_genotypes: extracted genotypes
            extract_fitnesses: all fitnesses of the extracted genotypes
            extract_descriptors: all descriptors of the extracted genotypes
            random_key: an updated jax PRNG random key
        """

        num_centroids = self.fitnesses_depth.shape[0]
        depth = self.fitnesses_depth.shape[1]

        # Set probability for each individual to be sampled
        reshape_fitnesses_depth = self.fitnesses_depth.flatten()
        if type_prop == "exponential":
            p = jnp.exp(-jnp.arange(0, depth))
        elif type_prop == "linear":
            p = depth - jnp.arange(0, depth)
        elif type_prop == "harmonic":
            p = 1.0 / (jnp.arange(depth) + 1)
        else:
            assert 0, "!!!ERROR!!! Not implemented type_prop."
        p = jnp.repeat(jnp.expand_dims(p, axis=0), num_centroids, axis=0)
        p = p.flatten()
        p = jnp.where(
            reshape_fitnesses_depth > -jnp.inf,
            p,
            0,
        )

        # Extract num_samples indivs
        p = p / jnp.sum(p)
        random_key, subkey = jax.random.split(random_key)
        indices = jax.random.choice(
            subkey,
            num_centroids * depth,
            shape=(num_samples,),
            p=p,
            replace=False,
        )
        rows, cols = jnp.divmod(indices, depth)
        rows = rows.astype(int)
        cols = cols.astype(int)

        # Extract the final genotypes, fitnesses_all and descriptors_all to return
        extract_genotypes = jax.tree_util.tree_map(
            lambda x: x.at[rows, cols].get(),
            self.genotypes_depth,
        )
        extract_fitnesses_all = self.fitnesses_depth_all.at[rows, cols].get()
        extract_descriptors_all = self.descriptors_depth_all.at[rows, cols].get()

        # Re-order repertoire following extraction
        repertoire, random_key = self._order_repertoire(
            rows=rows,
            cols=cols,
            random_key=random_key,
            delta_fitness=delta_fitness,
            delta_reproducibility=delta_reproducibility,
            rho=rho,
        )

        return (
            repertoire,
            extract_genotypes,
            extract_fitnesses_all,
            extract_descriptors_all,
            random_key,
        )

    @partial(
        jax.jit,
        static_argnames=(
            "num_layers",
            "delta_fitness",
            "delta_reproducibility",
            "rho",
        ),
    )
    def extract_top_layer(
        self,
        random_key: RNGKey,
        num_layers: int,
        delta_fitness: float,
        delta_reproducibility: float,
        rho: float,
    ) -> Tuple[
        ArchiveSamplingWeightedRepertoire, Genotype, Fitness, Descriptor, RNGKey
    ]:
        """
        Extract the num_layers top layers from the grid.
        Extract means that they are removed from the grid when sampled.

        Args:
            random_key: a jax PRNG random key
            num_layers: the number of top layers to be sampled

        Returns:
            repertoire: the new repertoire
            extract_genotypes: extracted genotypes
            extract_fitnesses: all fitnesses of the extracted genotypes
            extract_descriptors: all descriptors of the extracted genotypes
            random_key: an updated jax PRNG random key
        """

        num_centroids = self.centroids.shape[0]

        # Extract the top num_layers
        extract_genotypes = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x.at[:, :num_layers].get(),
                (num_layers * num_centroids,) + x.shape[2:],
            ),
            self.genotypes_depth,
        )
        extract_fitnesses_all = jnp.reshape(
            self.fitnesses_depth_all.at[:, :num_layers].get(),
            (num_layers * num_centroids, self.fitnesses_depth_all.shape[2]),
        )
        extract_descriptors_all = jnp.reshape(
            self.descriptors_depth_all.at[:, :num_layers].get(),
            (num_layers * num_centroids,) + self.descriptors_depth_all.shape[2:],
        )

        # Set the top num_layers to -jnp.inf
        genotypes_depth = jax.tree_util.tree_map(
            lambda x: x.at[:, :num_layers].set(0),
            self.genotypes_depth,
        )
        fitnesses_depth = self.fitnesses_depth.at[:, :num_layers].set(-jnp.inf)
        fitnesses_depth_all = self.fitnesses_depth_all.at[:, :num_layers].set(jnp.nan)
        descriptors_depth = self.descriptors_depth.at[:, :num_layers].set(0)
        descriptors_depth_all = self.descriptors_depth_all.at[:, :num_layers].set(
            jnp.nan
        )

        # Roll the layers best fitnesses at the top
        genotypes_depth = jax.tree_util.tree_map(
            lambda x: jnp.roll(x, -num_layers, axis=1),
            genotypes_depth,
        )
        fitnesses_depth = jnp.roll(fitnesses_depth, -num_layers, axis=1)
        fitnesses_depth_all = jnp.roll(fitnesses_depth_all, -num_layers, axis=1)
        descriptors_depth = jnp.roll(descriptors_depth, -num_layers, axis=1)
        descriptors_depth_all = jnp.roll(descriptors_depth_all, -num_layers, axis=1)

        # Create top layer of the grid from depth
        genotypes = jax.tree_map(lambda x: x.at[:, 0].get(), genotypes_depth)
        fitnesses = fitnesses_depth.at[:, 0].get()
        descriptors = descriptors_depth.at[:, 0].get()

        # Create the new repertoire
        repertoire = self.replace(
            genotypes=genotypes,
            genotypes_depth=genotypes_depth,
            fitnesses=fitnesses,
            fitnesses_depth=fitnesses_depth,
            fitnesses_depth_all=fitnesses_depth_all,
            descriptors=descriptors,
            descriptors_depth=descriptors_depth,
            descriptors_depth_all=descriptors_depth_all,
        )

        return (
            repertoire,
            extract_genotypes,
            extract_fitnesses_all,
            extract_descriptors_all,
            random_key,
        )

    @partial(
        jax.jit,
        static_argnames=(
            "num_layers",
            "delta_fitness",
            "delta_reproducibility",
            "rho",
        ),
    )
    def extract_top_layer_fillin(
        self,
        random_key: RNGKey,
        num_layers: int,
        delta_fitness: float,
        delta_reproducibility: float,
        rho: float,
    ) -> Tuple[
        ArchiveSamplingWeightedRepertoire, Genotype, Fitness, Descriptor, RNGKey
    ]:
        """
         Extract the num_layers top layers from the grid.
         Extract means that they are removed from the grid when sampled.

         Args:
             random_key: a jax PRNG random key
             num_layers: the number of top layers to be sampled

         Returns:
        repertoire: the new repertoire
             extract_genotypes: extracted genotypes
             extract_fitnesses: all fitnesses of the extracted genotypes
             extract_descriptors: all descriptors of the extracted genotypes
             random_key: an updated jax PRNG random key
        """

        num_centroids = self.fitnesses_depth.shape[0]
        depth = self.fitnesses_depth.shape[1]
        out_of_bound = num_centroids * depth

        # Extract the top num_layers
        (
            repertoire,
            extract_genotypes,
            extract_fitnesses_all,
            extract_descriptors_all,
            random_key,
        ) = self.extract_top_layer(random_key=random_key, num_layers=num_layers)

        # Set probability for each individual to be sampled
        reshape_fitnesses_depth = repertoire.fitnesses_depth.flatten()
        p = reshape_fitnesses_depth > -jnp.inf

        # Extract num_samples indivs
        p = p / jnp.sum(p)
        random_key, subkey = jax.random.split(random_key)
        indices = jax.random.choice(
            subkey,
            num_centroids * depth,
            shape=(num_layers * num_centroids,),
            p=p,
            replace=False,
        )
        rows, cols = jnp.divmod(indices, depth)
        rows = rows.astype(int)
        cols = cols.astype(int)

        # Only extract where nothing was already extracted
        cond = jnp.any(jnp.logical_not(jnp.isnan(extract_fitnesses_all)), axis=1)
        rows = jnp.where(cond, out_of_bound, rows)
        cols = jnp.where(cond, out_of_bound, cols)
        extract_genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.where(
                jnp.reshape(cond, cond.shape + (1,) * (len(x.shape) - len(cond.shape))),
                x,
                y.at[rows, cols].get(),
            ),
            extract_genotypes,
            repertoire.genotypes_depth,
        )
        extract_fitnesses_all = jnp.where(
            jnp.reshape(cond, cond.shape + (1,)),
            extract_fitnesses_all,
            repertoire.fitnesses_depth_all.at[rows, cols].get(),
        )
        extract_descriptors_all = jnp.where(
            jnp.reshape(
                cond,
                cond.shape
                + (
                    1,
                    1,
                ),
            ),
            extract_descriptors_all,
            repertoire.descriptors_depth_all.at[rows, cols].get(),
        )

        # Re-order repertoire following extraction
        repertoire, random_key = self._order_repertoire(
            rows=rows,
            cols=cols,
            random_key=random_key,
            delta_fitness=delta_fitness,
            delta_reproducibility=delta_reproducibility,
            rho=rho,
        )

        return (
            repertoire,
            extract_genotypes,
            extract_fitnesses_all,
            extract_descriptors_all,
            random_key,
        )
