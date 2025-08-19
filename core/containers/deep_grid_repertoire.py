from __future__ import annotations

from functools import partial
from typing import Callable, Tuple

import flax
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from qdax.core.containers.mapelites_repertoire import get_cells_indices
from qdax.custom_types import (
    Centroid,
    Descriptor,
    ExtraScores,
    Fitness,
    Genotype,
    RNGKey,
)

from core.containers.mapelites_depth_repertoire import DepthMapElitesRepertoire


class DeepGridRepertoire(flax.struct.PyTreeNode):
    """
    Class for the repertoire of Deep-Grid.

    Args:
        genotypes: a PyTree containing the genotypes of the best solutions ordered
            by the centroids. Each leaf has a shape (num_centroids, num_features). The
            PyTree can be a simple Jax array or a more complex nested structure such
            as to represent parameters of neural network in Flax.
        genotypes_depth: a PyTree containing all the genotypes ordered by the centroids.
            Each leaf has a shape (num_centroids * depth, num_features). The PyTree
            can be a simple Jax array or a more complex nested structure such as to
            represent parameters of neural network in Flax.
        fitnesses: an array that contains the fitness of best solutions in each cell of
            the repertoire, ordered by centroids. The array shape is (num_centroids,).
        fitnesses_depth: an array that contains the fitness of all solutions in each
            cell of the repertoire, ordered by centroids. The array shape
            is (num_centroids * depth).
        descriptors: an array that contains the descriptors of best solutions in each
            cell of the repertoire, ordered by centroids. The array shape
            is (num_centroids, num_descriptors).
        descriptors_depth: an array that contains the descriptors of all solutions in
            each cell of the repertoire, ordered by centroids. The array shape
            is (num_centroids * depth, num_descriptors).
        seniorities_depth: an array that contains the seniorities of all solutions in each
            cell of the repertoire, ordered by centroids. The array shape is
            (num_centroids * depth).
        centroids: an array the contains the centroids of the tesselation. The array
            shape is (num_centroids, num_descriptors).
    """

    genotypes: Genotype
    genotypes_depth: Genotype
    fitnesses: Fitness
    fitnesses_depth: Fitness
    descriptors: Descriptor
    descriptors_depth: Descriptor
    seniorities_depth: jnp.ndarray
    centroids: Centroid

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
        jnp.save(path + "seniorities_depth.npy", self.seniorities_depth)
        jnp.save(path + "centroids.npy", self.centroids)

    @classmethod
    def load(
        cls, reconstruction_fn: Callable, path: str = "./"
    ) -> DepthMapElitesRepertoire:
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
        seniorities_depth = jnp.load(path + "seniorities_depth.npy")
        centroids = jnp.load(path + "centroids.npy")

        return DeepGridRepertoire(
            genotypes=genotypes,
            genotypes_depth=genotypes_depth,
            fitnesses=fitnesses,
            fitnesses_depth=fitnesses_depth,
            descriptors=descriptors,
            descriptors_depth=descriptors_depth,
            seniorities_depth=seniorities_depth,
            centroids=centroids,
        )

    @partial(jax.jit, static_argnames=("num_samples",))
    def _in_cell_sample(
        self,
        cell_fitnesses: Fitness,
        random_key: RNGKey,
        num_samples: int,
    ) -> jnp.ndarray:
        """
        Return the index of an indiv of the cell, fitness proportionally. Applied
        using apply_along_axis on all considered cells.
        """

        # Normalise all fitnesses in cell
        cell_empty = cell_fitnesses == -jnp.inf
        max_fitness = jnp.max(cell_fitnesses)
        min_fitness = jnp.min(jnp.where(cell_empty, jnp.inf, cell_fitnesses))
        norm_fitnesses = jnp.where(
            cell_empty,
            0,
            (cell_fitnesses - min_fitness) / (max_fitness - min_fitness),
        )

        # Compute probabilities accounting for cases where all are min_fitness
        p = jnp.where(
            jnp.sum(norm_fitnesses, keepdims=True) == 0,
            (1.0 - cell_empty) / jnp.sum(1.0 - cell_empty),
            norm_fitnesses / jnp.sum(norm_fitnesses),
        )

        # Sample following probabilities
        indices_indivs = jnp.arange(0, cell_fitnesses.shape[0], step=1)
        final_indices = jax.random.choice(
            random_key, indices_indivs, shape=(num_samples,), p=p
        )
        return final_indices

    @partial(jax.jit, static_argnames=("num_samples",))
    def _sample_indices(
        self, random_key: RNGKey, num_samples: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray, RNGKey]:
        """
        Sample elements in the grid.
        For each sample, choose a random cell and return a random individual of this
        cell fitness-proportionally.

        Args:
            random_key: a jax PRNG random key
            num_samples: the number of elements to be sampled

        Returns:
            samples: a batch of genotypes sampled in the repertoire
            random_key: an updated jax PRNG random key
        """

        num_centroids = self.fitnesses_depth.shape[0]

        # Split keys
        random_key, sub_key = jax.random.split(random_key)
        cell_key, indiv_key = jax.random.split(sub_key)
        indivs_keys = jax.random.split(indiv_key, num=num_samples)

        # Pick num_samples random non-empty cell
        grid_empty = self.fitnesses == -jnp.inf
        p = (1.0 - grid_empty) / jnp.sum(1.0 - grid_empty)
        indices_cells = jnp.arange(0, num_centroids, step=1)
        sample_cells = jax.random.choice(
            cell_key, indices_cells, shape=(num_samples,), p=p
        )

        # Pick one indiv per sampled cell fitness-proportionaly
        in_cell_sample = partial(self._in_cell_sample, num_samples=1)
        sample_indivs = jax.vmap(in_cell_sample)(
            self.fitnesses_depth.at[sample_cells].get(),
            indivs_keys,
        )

        # Get corresponding indices
        return sample_cells, sample_indivs, random_key

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample(self, random_key: RNGKey, num_samples: int) -> Tuple[Genotype, RNGKey]:
        """
        Sample elements in the grid.
        For each sample, choose a random cell and return a random individual of this
        cell fitness-proportionally.

        Args:
            random_key: a jax PRNG random key
            num_samples: the number of elements to be sampled

        Returns:
            samples: a batch of genotypes sampled in the repertoire
            random_key: an updated jax PRNG random key
        """

        def get_final_samples(indexes: jnp.ndarray, genotypes: Genotype) -> Genotype:
            return jax.tree_util.tree_map(lambda x: x.at[indexes].get(), genotypes)

        sample_cells, sample_indivs, random_key = self._sample_indices(
            random_key, num_samples
        )
        samples = jax.tree_map(
            lambda x: x.at[sample_cells].get(),
            self.genotypes_depth,
        )
        samples = jax.vmap(get_final_samples)(sample_indivs.squeeze(axis=-1), samples)
        return samples, random_key

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample_with_descs(
        self, random_key: RNGKey, num_samples: int
    ) -> Tuple[Genotype, Descriptor, RNGKey]:
        """Sample elements in the repertoire and return both their
        genotypes, descriptors and fitnesses.

        Args:
            random_key: a jax PRNG random key
            num_samples: the number of elements to be sampled

        Returns:
            samples: a batch of genotypes sampled in the repertoire
            descriptors: the corresponding descriptors
            random_key: an updated jax PRNG random key
        """

        def get_final_samples(
            indexes: jnp.ndarray, genotypes: Genotype, descriptors: Descriptor
        ) -> Tuple[Genotype, Descriptor]:
            return (
                jax.tree_util.tree_map(lambda x: x.at[indexes].get(), genotypes),
                descriptors.at[indexes].get(),
            )

        sample_cells, sample_indivs, random_key = self._sample_indices(
            random_key, num_samples
        )
        samples = jax.tree_util.tree_map(
            lambda x: x.at[sample_cells].get(),
            self.genotypes_depth,
        )
        descriptors = self.descriptors_depth.at[sample_cells, sample_indivs].get()
        samples, descriptors = jax.vmap(get_final_samples)(
            sample_indivs.squeeze(axis=-1), samples, descriptors
        )

        return samples, descriptors, random_key

    @partial(jax.jit, static_argnames=("num_samples",))
    def sample_all_cells(
        self, random_key: RNGKey, num_samples: int
    ) -> Tuple[Genotype, Fitness, Descriptor, RNGKey]:
        """
        Sample num_samples indivs from each cells using in-cell selector.
        Args:
           random_key
           num_samples: number of samples per cell.
        Returns:
            genotypes: genotypes of samples indivs.
            fitnesses: associated fitnesses.
            descriptor: associated descriptors.
            random_key
        """

        num_centroids = self.fitnesses_depth.shape[0]

        # Split keys
        random_key, sub_key = jax.random.split(random_key)
        indivs_keys = jax.random.split(sub_key, num=num_centroids)

        # Pick num_samples indivs per cell fitness-proportionaly
        in_cell_sample = partial(self._in_cell_sample, num_samples=num_samples)
        sample_indivs = jax.vmap(in_cell_sample)(
            self.fitnesses_depth,
            indivs_keys,
        )

        # Get corresponding genotypes, fitnesses and descriptors
        def get_final_samples(
            indexes: jnp.ndarray,
            genotypes: Genotype,
            fitnesses: Fitness,
            descriptors: Descriptor,
        ) -> Tuple[Genotype, Fitness, Descriptor]:
            return (
                jax.tree_util.tree_map(lambda x: x.at[indexes].get(), genotypes),
                fitnesses.at[indexes].get(),
                descriptors.at[indexes].get(),
            )

        genotypes, fitnesses, descriptors = jax.vmap(get_final_samples)(
            sample_indivs,
            self.genotypes_depth,
            self.fitnesses_depth,
            self.descriptors_depth,
        )

        return genotypes, fitnesses, descriptors, random_key

    @jax.jit
    def add(
        self,
        batch_of_genotypes: Genotype,
        batch_of_descriptors: Descriptor,
        batch_of_fitnesses: Fitness,
        batch_of_extra_scores: ExtraScores,
    ) -> Tuple[DeepGridRepertoire, RNGKey]:
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

        num_centroids = self.fitnesses_depth.shape[0]
        depth = self.fitnesses_depth.shape[1]
        batch_size = batch_of_fitnesses.shape[0]
        out_of_bound = max(
            num_centroids * depth,
            batch_size,
        )

        # Get indices
        batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)

        # Create an array with max seniorities for candidate solutions
        batch_of_seniorities = jnp.full_like(batch_of_fitnesses, depth + 1)

        # Filter dead individuals
        batch_of_indices = jnp.where(
            batch_of_fitnesses > -jnp.inf,
            batch_of_indices,
            out_of_bound,
        )
        batch_of_seniorities = jnp.where(
            batch_of_fitnesses > -jnp.inf,
            batch_of_seniorities,
            out_of_bound,
        )

        @jax.jit
        def _add_per_cell(
            cell_idx: jnp.ndarray,
            cell_genotypes_depth: Genotype,
            cell_fitnesses_depth: Fitness,
            cell_descriptors_depth: Descriptor,
            cell_seniorities_depth: jnp.ndarray,
        ) -> Tuple[
            Genotype,
            Fitness,
            Descriptor,
            jnp.ndarray,
            Genotype,
            Fitness,
            Descriptor,
        ]:
            """
            For a given cell with index cell_idx, filter candidate
            indivs for this cell, and add them to it, reordering so
            highest-fitness individuals are first.

            Args:
              cell_idx: cell index
              cell_genotypes_depth: genotype in the cell
              cell_fitnesses_depth: fitnesses in the cell
              cell_descriptors_depth: descriptors in the cell
              cell_seniorities_depth

            Returns:
              new_cell_genotypes_depth
              new_cell_fitnesses_depth
              new_cell_descriptors_depth
              cell_seniorities_depth
              new_cell_genotype: genotype in the top layer of the cell
              new_cell_fitnesses: fitnesses in the top layer of the cell
              new_cell_descriptors: descriptors in the top layer of the cell
            """

            # Set seniorities and fitnesses of -jnp.inf for the one that don't belong to this cell
            candidate_seniorities = jnp.where(
                batch_of_indices == cell_idx, batch_of_seniorities, -jnp.inf
            )
            candidate_fitnesses = jnp.where(
                batch_of_indices == cell_idx, batch_of_fitnesses, -jnp.inf
            )
            all_seniorities = jnp.concatenate(
                [cell_seniorities_depth, candidate_seniorities],
                axis=0,
            )
            all_fitnesses = jnp.concatenate(
                [cell_fitnesses_depth, candidate_fitnesses],
                axis=0,
            )
            # previous_all_seniorities = all_seniorities

            # Keep only more recent solutions
            _, kept_indices = jax.lax.top_k(all_seniorities, depth)
            mask = jnp.zeros_like(all_fitnesses, dtype=bool).at[kept_indices].set(True)
            all_seniorities = jnp.where(
                mask,
                all_seniorities,
                -jnp.inf,
            )
            all_fitnesses = jnp.where(
                mask,
                all_fitnesses,
                -jnp.inf,
            )
            # jax.debug.print("Seniorities {x} {y}", x=all_seniorities, y=all_seniorities)

            # Order existing and candidate indivs by fitness
            _, final_indices = jax.lax.top_k(all_fitnesses, depth)

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
            new_cell_descriptors_depth = cell_descriptors_depth.at[cell_indices].get()
            new_cell_seniorities_depth = cell_seniorities_depth.at[cell_indices].get()

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
            new_cell_descriptors_depth = new_cell_descriptors_depth.at[
                depth_indices
            ].set(batch_of_descriptors[candidate_indices])
            new_cell_seniorities_depth = new_cell_seniorities_depth.at[
                depth_indices
            ].set(batch_of_seniorities[candidate_indices])

            # Third, offset all seniorities by the current min value of the cell
            # jax.debug.print(
            #    "Cell before {x} {y}",
            #    x=new_cell_seniorities_depth,
            #    y=new_cell_fitnesses_depth,
            # )
            min_seniorities = jnp.nanmin(
                jnp.where(
                    new_cell_seniorities_depth > -jnp.inf,
                    new_cell_seniorities_depth,
                    jnp.inf,
                )
            )
            new_cell_seniorities_depth = new_cell_seniorities_depth - min_seniorities
            # jax.debug.print("Cell after {x}", x=new_cell_seniorities_depth)

            # Also return the top layer of the grid
            new_cell_genotype = jax.tree_map(
                lambda x: x.at[0].get(),
                new_cell_genotypes_depth,
            )
            new_cell_fitnesses = new_cell_fitnesses_depth.at[0].get()
            new_cell_descriptors = new_cell_descriptors_depth.at[0].get()

            # Return the updated cell
            return (
                new_cell_genotypes_depth,
                new_cell_fitnesses_depth,
                new_cell_descriptors_depth,
                new_cell_seniorities_depth,
                new_cell_genotype,
                new_cell_fitnesses,
                new_cell_descriptors,
            )

        # Add individuals cell by cell
        (
            new_genotypes_depth,
            new_fitnesses_depth,
            new_descriptors_depth,
            new_seniorities_depth,
            new_genotypes,
            new_fitnesses,
            new_descriptors,
        ) = jax.vmap(_add_per_cell)(
            jnp.arange(0, num_centroids, step=1),
            self.genotypes_depth,
            self.fitnesses_depth,
            self.descriptors_depth,
            self.seniorities_depth,
        )

        return self.replace(  # type: ignore
            genotypes=new_genotypes,
            genotypes_depth=new_genotypes_depth,
            fitnesses=new_fitnesses,
            fitnesses_depth=new_fitnesses_depth,
            descriptors=new_descriptors,
            descriptors_depth=new_descriptors_depth,
            seniorities_depth=new_seniorities_depth,
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
    ) -> DeepGridRepertoire:
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
        default_seniorities_depth = -jnp.inf * jnp.ones(shape=(num_centroids, depth))

        repertoire = DeepGridRepertoire(
            genotypes=default_genotypes,
            genotypes_depth=default_genotypes_depth,
            fitnesses=default_fitnesses,
            fitnesses_depth=default_fitnesses_depth,
            descriptors=default_descriptors,
            descriptors_depth=default_descriptors_depth,
            seniorities_depth=default_seniorities_depth,
            centroids=centroids,
        )

        # Add initial values to the grid
        new_repertoire = repertoire.add(
            batch_of_genotypes=genotypes,
            batch_of_descriptors=descriptors,
            batch_of_fitnesses=fitnesses,
            batch_of_extra_scores=extra_scores,
        )

        return new_repertoire  # type: ignore

    @jax.jit
    def empty(self) -> DeepGridRepertoire:
        """
        Empty the grid from all existing individuals.

        Returns:
            An empty DeepGridRepertoire
        """

        new_fitnesses = jnp.full_like(self.fitnesses, -jnp.inf)
        new_fitnesses_depth = jnp.full_like(self.fitnesses_depth, -jnp.inf)
        new_descriptors = jnp.zeros_like(self.descriptors)
        new_descriptors_depth = jnp.zeros_like(self.descriptors_depth)
        new_genotypes = jax.tree_map(lambda x: jnp.zeros_like(x), self.genotypes)
        new_genotypes_depth = jax.tree_map(
            lambda x: jnp.zeros_like(x), self.genotypes_depth
        )
        new_seniorities_depth = jnp.full_like(self.seniorities_depth, -jnp.inf)
        return DeepGridRepertoire(
            genotypes=new_genotypes,
            genotypes_depth=new_genotypes_depth,
            fitnesses=new_fitnesses,
            fitnesses_depth=new_fitnesses_depth,
            descriptors=new_descriptors,
            descriptors_depth=new_descriptors_depth,
            seniorities_depth=new_seniorities_depth,
            centroids=self.centroids,
        )

    @jax.jit
    def added_repertoire(
        self,
        genotypes: Genotype,
        descriptors: Descriptor,
    ) -> jnp.ndarray:
        """Compute if the given genotypes have been added to the repertoire in
        corresponding cell.

        Args:
            genotypes: genotypes candidate to addition
            descriptors: corresponding descriptors
        Returns:
            boolean for each genotype
        """
        cells = get_cells_indices(descriptors, self.centroids)
        repertoire_genotypes = jax.tree_util.tree_map(
            lambda x: x[cells], self.genotypes_depth
        )
        genotypes = jax.tree_util.tree_map(
            lambda x, y: jnp.repeat(jnp.expand_dims(x, axis=1), y.shape[1], axis=1),
            genotypes,
            repertoire_genotypes,
        )
        added = jax.tree_util.tree_map(
            lambda x, y: jnp.equal(x, y),
            genotypes,
            repertoire_genotypes,
        )
        added = jax.tree_util.tree_map(lambda x: jnp.any(x, axis=1), added)
        added = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (descriptors.shape[0], -1)), added
        )
        added = jax.tree_util.tree_map(lambda x: jnp.all(x, axis=1), added)
        final_added = jnp.array(jax.tree_util.tree_leaves(added))
        final_added = jnp.all(final_added, axis=0)
        return final_added
